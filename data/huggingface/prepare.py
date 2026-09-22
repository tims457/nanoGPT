"""Prepare Hugging Face text as nanoGPT uint16 tokens; OpenWebText is downloaded first."""

import argparse
from contextlib import ExitStack, closing
from itertools import islice
import json
from pathlib import Path
import pickle
from tempfile import TemporaryDirectory

from datasets import load_dataset_builder
import fsspec
import numpy as np
import pyarrow.parquet as pq
import tiktoken


OPENWEBTEXT = 'Skylion007/openwebtext'
OPENWEBTEXT_REVISION = '433fe0f44ed7894fea29c08b3202aa348ccc6369'


def downloaded_rows(builder, split, text_column):
    dataset = builder.as_dataset(split=split, in_memory=False)
    if text_column not in dataset.column_names:
        raise ValueError(f'Missing text column {text_column!r}; available: {dataset.column_names}')
    yield from dataset.select_columns([text_column])


def stream_rows(builder, split, text_column):
    if builder.info.builder_name != 'parquet':
        yield from builder.as_streaming_dataset(split=split)
        return
    if split not in builder.config.data_files:
        raise ValueError(f'Missing split {split!r}; available: {list(builder.config.data_files)}')
    # Bounded reads often stop midway through a remote file. Reading synchronously
    # keeps Arrow workers from retaining Python file handles during shutdown.
    for path in builder.config.data_files[split]:
        with fsspec.open(path, 'rb') as source:
            with pq.ParquetFile(source, pre_buffer=False) as parquet:
                if text_column not in parquet.schema_arrow.names:
                    raise ValueError(f'Missing text column {text_column!r}; available: {parquet.schema_arrow.names}')
                for batch in parquet.iter_batches(batch_size=128, columns=[text_column], use_threads=False):
                    yield from batch.to_pylist()


def write_split(rows, path, encoding, text_column, max_examples, block_size):
    """Bound rows read (including blank rows), and hold only one document in memory."""
    examples = tokens = 0
    with path.open('wb') as output:
        for row in islice(rows, max_examples):
            if text_column not in row:
                raise ValueError(f"Missing text column {text_column!r}; available: {list(row)}")
            text = row[text_column]
            if not isinstance(text, str):
                raise ValueError(f"Column {text_column!r} must contain strings")
            if not text.strip():
                continue
            ids = encoding.encode_ordinary(text) + [encoding.eot_token]
            np.asarray(ids, dtype=np.uint16).tofile(output)
            examples += 1
            tokens += len(ids)
    if tokens <= block_size:
        raise ValueError(
            f"{path.name} has {tokens} tokens; need at least {block_size + 1}. "
            "Increase the example limit or reduce --block-size."
        )
    return {'examples': examples, 'tokens': tokens}


def prepare(args):
    if args.max_train_examples < 1 or args.max_val_examples < 1 or args.block_size < 1:
        raise ValueError('Example limits and --block-size must be positive')
    if args.val_split == args.train_split:
        raise ValueError('Train and validation splits must differ; use --val-split="" for a holdout')
    if args.val_file and not args.val_split:
        raise ValueError('--val-file requires --val-split')
    if args.val_file and not args.train_file:
        raise ValueError('--val-file requires --train-file')
    if args.train_file and args.val_split and not args.val_file:
        raise ValueError('Provide --val-file or use --val-split="" for a holdout')

    name = args.dataset_config
    if name is None and args.dataset == 'Salesforce/wikitext':
        name = 'wikitext-2-raw-v1'
    revision = args.revision
    download = args.download or args.dataset == OPENWEBTEXT
    if args.dataset == OPENWEBTEXT:
        name = name or 'default'
        revision = revision or OPENWEBTEXT_REVISION
    load_kwargs = dict(path=args.dataset, name=name)
    if revision:
        load_kwargs['revision'] = revision
    if args.train_file:
        load_kwargs['data_files'] = {args.train_split: args.train_file}
        if args.val_file:
            load_kwargs['data_files'][args.val_split] = args.val_file

    builder = load_dataset_builder(**load_kwargs)
    if download:
        print(f'Downloading/preparing the full source dataset in {builder.cache_dir}', flush=True)
        builder.download_and_prepare()
    read_rows = downloaded_rows if download else stream_rows
    encoding = tiktoken.get_encoding('gpt2')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        'dataset': args.dataset,
        'dataset_config': name,
        'revision': revision,
        'data_loading': 'downloaded' if download else 'streaming',
        'train_split': args.train_split,
        'val_split': args.val_split or 'prefix_holdout',
        'text_column': args.text_column,
        'train_file': args.train_file,
        'val_file': args.val_file,
        'max_train_examples': args.max_train_examples,
        'max_val_examples': args.max_val_examples,
        'tokenizer': 'gpt2',
        'vocab_size': encoding.n_vocab,
        'dtype': 'uint16',
    }
    # Validate both splits before replacing any previously prepared dataset.
    with TemporaryDirectory(dir=output_dir, prefix='.prepare-') as staging:
        staging = Path(staging)
        # Close partially consumed streams before interpreter shutdown, including
        # their underlying remote files and Arrow readers.
        with ExitStack() as streams:
            train_rows = streams.enter_context(closing(read_rows(builder, args.train_split, args.text_column)))
            if args.val_split:
                val_rows = streams.enter_context(closing(read_rows(builder, args.val_split, args.text_column)))
            else:
                # Reserve the first N source rows, then train on the following rows.
                val_rows = islice(train_rows, args.max_val_examples)
            for split, rows, limit in (
                ('val', val_rows, args.max_val_examples),
                ('train', train_rows, args.max_train_examples),
            ):
                metadata[split] = write_split(
                    rows, staging / f'{split}.bin', encoding,
                    args.text_column, limit, args.block_size,
                )
        (staging / 'meta.json').write_text(json.dumps(metadata, indent=2) + '\n')
        with (staging / 'meta.pkl').open('wb') as output:
            pickle.dump({'vocab_size': encoding.n_vocab, 'tokenizer': 'gpt2'}, output)
        for filename in ('train.bin', 'val.bin', 'meta.pkl', 'meta.json'):
            (staging / filename).replace(output_dir / filename)

    print(json.dumps(metadata, indent=2))
    print(f'Prepared {output_dir}; pass --dataset={output_dir.resolve()} to train.py')
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', default='Salesforce/wikitext', help='Hub ID or local loader (json, text, parquet)')
    parser.add_argument('--dataset-config', help='Defaults to wikitext-2-raw-v1 for WikiText or default for OpenWebText')
    parser.add_argument('--revision', help='Hub commit, tag, or branch; OpenWebText defaults to a pinned Parquet revision')
    parser.add_argument('--download', action='store_true',
                        help='Cache the full source dataset before tokenization (always enabled for OpenWebText)')
    parser.add_argument('--text-column', default='text')
    parser.add_argument('--train-split', default='train')
    parser.add_argument('--val-split', default='validation', help='Empty string reserves the first max-val-examples train rows')
    parser.add_argument('--train-file', help='Local data file; use with --dataset=json, text, or parquet')
    parser.add_argument('--val-file', help='Local validation data file')
    parser.add_argument('--max-train-examples', type=int, default=256, help='Maximum rows to tokenize for training; does not limit full downloads')
    parser.add_argument('--max-val-examples', type=int, default=64, help='Maximum rows to tokenize for validation; does not limit full downloads')
    parser.add_argument('--block-size', type=int, default=32, help='Ensure each split has enough tokens for this context')
    parser.add_argument('--output-dir', default='data/hf_smoke')
    args = parser.parse_args()
    try:
        prepare(args)
    except ValueError as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()
