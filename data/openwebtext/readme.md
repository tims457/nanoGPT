
## openwebtext dataset

### Quick 1,000-iteration run on one CUDA GPU

The quick preset trains GPT-2 small from scratch: 12 layers, 12 heads, width 768,
1,024-token context, and **124,439,808 total parameters**. It uses the original
50,257-token GPT-2 BPE vocabulary, with an end-of-text token after each document.

From the repository root:

```sh
uv run --locked --extra cu128 python data/huggingface/prepare.py \
    --dataset=Skylion007/openwebtext --dataset-config=default \
    --revision=433fe0f44ed7894fea29c08b3202aa348ccc6369 \
    --val-split= --max-train-examples=10000 --max-val-examples=1000 \
    --block-size=1024 --output-dir=data/openwebtext_10k

uv run --locked --extra cu128 python train.py config/train_openwebtext_gpt2_1k.py
```

The pinned revision is from the dataset's `refs/convert/parquet` branch. The
loader downloads all 8,013,769 documents to the default shared Hugging Face cache
at `~/.cache/huggingface` before tokenizing; OpenWebText is never streamed.
The first download includes roughly 24 GB of Parquet files plus the prepared
Arrow cache. Later runs and other projects can reuse the cached dataset with the
same dataset ID, config, and revision. From that local copy, the first 1,000
documents become validation and the next 10,000 become training, producing
disjoint document subsets. The example limits bound tokenization, not the source
download. Dataset provenance, loading mode, and exact prepared token counts are
saved in `meta.json` and W&B.

Training uses batch size 4 with no gradient accumulation: 4,096 tokens per update
and 4.096 million tokens across 1,000 optimizer steps. AdamW uses a peak learning
rate of `6e-4`, 100 warmup steps, cosine decay to `6e-5`, weight decay 0.1, and no
dropout. CUDA bfloat16 and `torch.compile` are enabled. Evaluation uses 20 batches
per split every 100 steps and at completion.

Results, loss, perplexity, parameter count, and model configuration are logged to
W&B project `lm_research`. The checkpoint is `runs/openwebtext-gpt2-1k/ckpt.pt`.
Resume the same W&B run by adding `--init_from=resume` and a larger `--max_iters`;
adjust `--lr_decay_iters` deliberately if extending the cosine schedule.

Sample from the result with the terminal interface:

```sh
uv run --locked --extra cu128 python chat.py --out_dir=runs/openwebtext-gpt2-1k --raw
```

### Fresh 10,000-iteration run

Reuse the prepared subset above and start a separate model from scratch:

```sh
uv run --locked --extra cu128 python train.py config/train_openwebtext_gpt2_10k.py
```

This preset retains the same architecture, batch size, optimizer, and 100-step
warmup. Cosine decay spans 10,000 steps, totaling 40.96 million training tokens
(about 3.69 passes over the 11,089,145-token training subset). Evaluation and
checkpointing run every 500 steps and at completion. It creates a separate W&B
run in `lm_research` and saves to `runs/openwebtext-gpt2-10k/ckpt.pt`.

`chat.py` loads the most recently saved run by default. To explicitly select this
model, pass `--out_dir=runs/openwebtext-gpt2-10k`.

### Full dataset preparation

after running `prepare.py` (preprocess) we get:

- train.bin is ~17GB, val.bin ~8.5MB
- train has ~9B tokens (9,035,582,198)
- val has ~4M tokens (4,434,897)

this came from 8,013,769 documents in total.

references:

- OpenAI's WebText dataset is discussed in [GPT-2 paper](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [OpenWebText](https://skylion007.github.io/OpenWebTextCorpus/) dataset
