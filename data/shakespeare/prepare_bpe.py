"""Prepare a small byte-level BPE vocabulary using only the Shakespeare training split."""

import argparse
import hashlib
import json
from pathlib import Path
import pickle
from tempfile import TemporaryDirectory

import numpy as np
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode


ROOT = Path(__file__).resolve().parents[2]


def prepare(input_path, output_dir, vocab_size=512):
    if not 257 <= vocab_size <= 65536:
        raise ValueError('vocab_size must fit 256 bytes plus EOS, and uint16 token IDs')
    text = input_path.read_text(encoding='utf-8')
    boundary = int(len(text) * 0.9)
    splits = {'train': text[:boundary], 'val': text[boundary:]}
    if not all(splits.values()):
        raise ValueError('Input must contain nonempty training and validation splits')

    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, show_progress=False,
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                                  special_tokens=['<|endoftext|>'])
    tokenizer.train_from_iterator(splits['train'].splitlines(keepends=True), trainer=trainer)
    actual_vocab_size = tokenizer.get_vocab_size()
    if actual_vocab_size != vocab_size:
        raise ValueError(f'Only learned {actual_vocab_size} tokens; supply more training text')
    eos_token = tokenizer.token_to_id('<|endoftext|>')
    tokenizer_json = tokenizer.to_str()
    # Preserve individual token bytes for correct incremental UTF-8 decoding in chat.
    byte_decoder = {char: byte for byte, char in bytes_to_unicode().items()}
    token_bytes = [b'' if index == eos_token else bytes(byte_decoder[char] for char in tokenizer.id_to_token(index))
                   for index in range(actual_vocab_size)]
    metadata = {
        'dataset': 'tiny_shakespeare',
        'source_sha256': hashlib.sha256(text.encode('utf-8')).hexdigest(),
        'tokenizer': 'byte_bpe',
        'tokenizer_training_split': 'train only',
        'tokenizer_sha256': hashlib.sha256(tokenizer_json.encode('utf-8')).hexdigest(),
        'vocab_size': actual_vocab_size,
        'dtype': 'uint16',
        'split_method': 'first 90% of characters for training, remaining 10% for validation',
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output_dir, prefix='.prepare-') as staging:
        staging = Path(staging)
        for split, content in splits.items():
            ids = tokenizer.encode(content, add_special_tokens=False).ids
            if tokenizer.decode(ids, skip_special_tokens=False) != content:
                raise ValueError(f'{split} did not round-trip through the tokenizer')
            np.asarray(ids, dtype=np.uint16).tofile(staging / f'{split}.bin')
            metadata[split] = {'tokens': len(ids), 'characters': len(content),
                               'bytes': len(content.encode('utf-8'))}
        (staging / 'tokenizer.json').write_text(tokenizer_json, encoding='utf-8')
        (staging / 'meta.json').write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
        with (staging / 'meta.pkl').open('wb') as destination:
            pickle.dump({'vocab_size': actual_vocab_size, 'tokenizer': 'byte_bpe',
                         'tokenizer_json': tokenizer_json, 'eos_token': eos_token,
                         'token_bytes': token_bytes}, destination)
        for filename in ('train.bin', 'val.bin', 'tokenizer.json', 'meta.json', 'meta.pkl'):
            (staging / filename).replace(output_dir / filename)
    print(json.dumps(metadata, indent=2))
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT / 'data/shakespeare/input.txt')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'data/shakespeare_bpe512')
    parser.add_argument('--vocab-size', type=int, default=512)
    args = parser.parse_args()
    prepare(args.input, args.output_dir, args.vocab_size)


if __name__ == '__main__':
    main()
