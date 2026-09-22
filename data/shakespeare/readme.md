
# tiny shakespeare

Tiny shakespeare, of the good old char-rnn fame :)

After running `prepare.py`:

- train.bin has 301,966 tokens
- val.bin has 36,059 tokens

Preparation also writes `meta.pkl` with the GPT-2 vocabulary size (50,257) and
`meta.json` with the source hash, tokenizer, split method, and token counts for
W&B logging.

From the repository root, prepare and train the 124M-parameter model from scratch
on CUDA:

```sh
uv sync --locked --extra cu128
uv run --locked --extra cu128 python data/shakespeare/prepare.py
uv run --locked --extra cu128 python train.py config/train_shakespeare_gpt2.py
```

The preset uses a 1,024-token context, batch size 4, 1,000 optimizer steps,
dropout 0.2, and AdamW with 100 warmup steps followed by cosine learning-rate
decay from `3e-4` to `3e-5`. Evaluation and checkpointing run every 50 steps and
at completion. W&B logs online to `lm_research`; use `--wandb_mode=offline` to
keep logs local. Checkpoints go to `runs/shakespeare-gpt2/ckpt.pt`.

Resume with the same configuration and add `--init_from=resume`, setting
`--max_iters` above the saved iteration to continue training. The W&B run ID is
restored from the checkpoint. For full-range cosine decay over an extended run,
also adjust `--lr_decay_iters` deliberately.

This is a small-data experiment: 124M parameters greatly exceed the available
302K training tokens. Expect overfitting; it is not a substitute for pretraining
on a large corpus.

The original `config/finetune_shakespeare.py` instead starts from pretrained
GPT-2 XL and fine-tunes for 20 steps at a constant learning rate of `3e-5`.

## 100K-parameter experiment

`config/train_shakespeare_100k.py` trains **100,032 total parameters**, including
token and position embeddings: 4 layers, 2 heads, width 32, and a 1,024-token
context. It uses a 512-token byte-level BPE vocabulary learned only from the
training split. The original 90/10 character split and source text are retained;
the prepared files live separately in `data/shakespeare_bpe512`.

```sh
uv run --locked --extra cu128 python data/shakespeare/prepare_bpe.py
uv run --locked --extra cu128 python train.py config/train_shakespeare_100k.py
uv run --locked --extra cu128 python chat.py --out_dir=runs/shakespeare-100k --raw
```

Prepare `data/shakespeare/input.txt` with the original `prepare.py` first if it
is missing. The smaller run retains the 124M run's batch size, 1,000-step
optimizer schedule, dropout, learning rate, and evaluation intervals. It writes
to `runs/shakespeare-100k/ckpt.pt` and a new W&B run in `lm_research`. Both
`sample.py` and `chat.py` load its tokenizer from the saved dataset metadata.

Perplexity depends on the tokenizer: do not directly compare this run's loss or
perplexity with the GPT-2-tokenized run. Tokens cover fewer characters here, so
the same context and training-token budget cover less text.
