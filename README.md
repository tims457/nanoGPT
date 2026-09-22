
# nanoGPT

![nanoGPT](assets/nanogpt.jpg)


---

**Update Nov 2025** nanoGPT has a new and improved cousin called [nanochat](https://github.com/karpathy/nanochat). It is very likely you meant to use/find nanochat instead. nanoGPT (this repo) is now very old and deprecated but I will leave it up for posterity.

---

The simplest, fastest repository for training/finetuning medium-sized GPTs. It is a rewrite of [minGPT](https://github.com/karpathy/minGPT) that prioritizes teeth over education. Still under active development, but currently the file `train.py` reproduces GPT-2 (124M) on OpenWebText, running on a single 8XA100 40GB node in about 4 days of training. The code itself is plain and readable: `train.py` is a ~300-line boilerplate training loop and `model.py` a ~300-line GPT model definition, which can optionally load the GPT-2 weights from OpenAI. That's it.

![repro124m](assets/gpt2_124M_loss.png)

Because the code is so simple, it is very easy to hack to your needs, train new models from scratch, or finetune pretrained checkpoints (e.g. biggest one currently available as a starting point would be the GPT-2 1.3B model from OpenAI).

## install

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then run from the repository root:

```sh
uv sync --locked --extra cpu
```

`uv` manages Python (pinned to 3.11 in `.python-version`), `.venv`, and the
dependencies in `uv.lock`. Use `uv run --locked --extra cpu` before the Python
commands below. Keep the same extra on `sync` and `run` so uv keeps the selected
PyTorch build. For CUDA 12.8, use `--extra cu128` instead; without an extra uv uses
the PyPI PyTorch build. CPU and CUDA extras are mutually exclusive. See
[uv's PyTorch guide](https://docs.astral.sh/uv/guides/integration/pytorch/).

Core dependencies:

- [pytorch](https://pytorch.org) <3
- [numpy](https://numpy.org/install/) <3
-  `transformers` for huggingface transformers <3 (to load GPT-2 checkpoints)
-  `datasets` for Hugging Face Hub and local datasets <3
-  `tiktoken` for OpenAI's fast BPE code <3
-  `wandb` for optional logging <3
-  `tqdm` for progress bars <3

## smoke test: Hugging Face datasets and W&B

These commands prepare a small dataset and run **exactly two CPU optimizer
steps**, with a one-layer model, one batch per evaluation, and no compilation:

```sh
uv run --locked --extra cpu python data/huggingface/prepare.py
OMP_NUM_THREADS=1 uv run --locked --extra cpu python train.py config/smoke.py
uv run --locked --extra cpu python sample.py --out_dir=runs/smoke --device=cpu --dtype=float32 --num_samples=1 --max_new_tokens=8
```

Preparation streams at most 256 train rows and 64 validation rows from
[`Salesforce/wikitext`, `wikitext-2-raw-v1`](https://huggingface.co/datasets/Salesforce/wikitext).
It writes `data/hf_smoke/{train.bin,val.bin,meta.pkl,meta.json}` with GPT-2 BPE
tokens, an end-of-text token after each nonblank document, and dataset metadata.
The first invocation needs internet access for the dataset and tokenizer.
Streaming avoids preparing the whole dataset, although a remote reader may
fetch an entire shard or Parquet row group. The smoke config writes its checkpoint
to `runs/smoke/ckpt.pt` and W&B data to `runs/smoke/wandb/`.

Run the repeatable integration smoke tests (local JSON/Parquet fixtures, no Hub dataset
download; the tokenizer must be cached or downloadable):

```sh
uv run --locked --extra cpu python -m unittest discover -s tests -v
```

They check bounded, disjoint data splits, two optimizer updates, offline W&B
metrics, checkpoint loading, evaluation-only resume, and four-token sampling.
They never launch a full training job.

### Other Hugging Face datasets

The preparation command accepts a Hub ID, optional subset/config and revision,
text column, split names, example limits, and output directory. For example:

```sh
uv run --locked --extra cpu python data/huggingface/prepare.py \
  --dataset=Salesforce/wikitext --dataset-config=wikitext-2-raw-v1 \
  --train-split=train --val-split=validation --text-column=text \
  --max-train-examples=128 --max-val-examples=32 \
  --block-size=32 --output-dir=data/my_dataset
uv run --locked --extra cpu python train.py config/smoke.py --dataset=my_dataset
```

Limits count source rows, including blank rows; blank text is not tokenized.
For a dataset with only a train split, pass `--val-split=''`: the first
`--max-val-examples` rows become validation and the next `--max-train-examples`
rows become training. This deterministic prefix holdout does not shuffle the
source. Each output must have more tokens than `--block-size`; set it to the
context size you intend to use. Both limits must be positive and bound
tokenization. Use `--revision=<commit>` to pin a Hub dataset version. `HF_TOKEN`
provides authentication for gated/private datasets. Hugging Face uses its default
shared cache at `~/.cache/huggingface`, which other projects can reuse; this repo
does not override `HF_HOME`. Prepared nanoGPT token files remain in `data/`.
OpenWebText always downloads and prepares the full source dataset before reading
local rows; it never streams. Its default revision is the pinned Parquet commit
`433fe0f44ed7894fea29c08b3202aa348ccc6369`. Other datasets use bounded streaming
unless `--download` is passed. In download mode, example limits only limit
tokenization, not the source download. See the
[Datasets loading guide](https://huggingface.co/docs/datasets/loading).

Local files also use Hugging Face Datasets, for example:

```sh
uv run --locked --extra cpu python data/huggingface/prepare.py \
  --dataset=json --train-file=/path/to/train.jsonl \
  --val-file=/path/to/validation.jsonl --text-column=text
```

### W&B logging

`config/smoke.py` enables W&B in **offline mode** without credentials. It records
configuration and dataset metadata, train/validation loss and perplexity, batch
loss, learning rate, step duration, and tokens per second, with `iter` as the chart axis.
At each evaluation, `train/perplexity` and `val/perplexity` are `exp(loss)` for
their respective splits. Run configuration includes the actual model dimensions
(also after loading or resuming a checkpoint) and `num_params`, the total number
of unique parameters including position embeddings and counting tied weights once.
The default W&B project is `lm_research` in all training presets. Runs
are finalized on completion or training errors. General training still opts in
with `--wandb_log=True`; `--wandb_mode` accepts `online`, `offline`, or `disabled`
and otherwise defaults to `WANDB_MODE` (or `online`).

To send the same two-step smoke run to your W&B account:

```sh
uv run --locked --extra cpu wandb login
uv run --locked --extra cpu python train.py config/smoke.py \
  --wandb_mode=online --wandb_project=lm_research --wandb_entity=YOUR_TEAM
```

Alternatively, supply `WANDB_API_KEY` through your environment; `WANDB_ENTITY`
can set the default team. To upload an existing offline run, use
`uv run --locked --extra cpu wandb sync runs/smoke/wandb/offline-run-...`.
Offline smoke tests do not upload anything. See the
[W&B run API](https://docs.wandb.ai/models/ref/python/experiments/run).

Checkpoints save the W&B run ID, project, entity, display name, and next history
step. With `--init_from=resume`, W&B logging reuses that identity automatically,
even if the checkpoint is moved to another output directory. In online mode it
uses `resume="allow"` to append to the existing run (or create that same ID if an
offline run has not been synced yet). A fresh training run gets a fresh ID.
For example, to continue the smoke checkpoint up to ten total optimizer steps:

```sh
uv run --locked --extra cpu python train.py config/smoke.py \
  --init_from=resume --max_iters=10
```

Reuse your original training configuration and set `--max_iters` above the saved
iteration. Use `--wandb_mode=online` for live logging. The saved project/entity
take precedence on resume so logging stays attached to the original run. Updated
configuration, such as `max_iters`, is recorded in W&B. Temporarily disabling
logging preserves the run identity in later checkpoints.

Offline restarts write separate local segments with the same run ID and a
continuing history counter. Preserve those directories and sync the segments
together when online; the installed W&B CLI syncs files for the same run in start
time order. Online resume also respects the server's latest history counter.
This appends history; it does not erase metrics logged after an older checkpoint.
See [resuming W&B runs](https://docs.wandb.ai/models/runs/resuming) and
[syncing offline runs](https://docs.wandb.ai/models/ref/cli/wandb-sync).

Older checkpoints without W&B metadata still load. To attach one to its existing
run, pass `--wandb_run_id=PREVIOUS_RUN_ID` (or `WANDB_RUN_ID`) with the correct
`--wandb_project` and `--wandb_entity`. Without an ID, training reports that it is
starting a new W&B run. An ID conflicting with one saved in the checkpoint is
rejected rather than redirecting the resumed training to a different run.

## quick start

If you are not a deep learning professional and you just want to feel the magic and get your feet wet, the fastest way to get started is to train a character-level GPT on the works of Shakespeare. First, we download it as a single (1MB) file and turn it from raw text into one large stream of integers:

```sh
python data/shakespeare_char/prepare.py
```

This creates a `train.bin` and `val.bin` in that data directory. Now it is time to train your GPT. The size of it very much depends on the computational resources of your system:

**I have a GPU**. Great, we can quickly train a baby GPT with the settings provided in the [config/train_shakespeare_char.py](config/train_shakespeare_char.py) config file:

```sh
python train.py config/train_shakespeare_char.py
```

If you peek inside it, you'll see that we're training a GPT with a context size of up to 256 characters, 384 feature channels, and it is a 6-layer Transformer with 6 heads in each layer. On one A100 GPU this training run takes about 3 minutes and the best validation loss is 1.4697. Based on the configuration, the model checkpoints are being written into the `--out_dir` directory `runs/shakespeare-char`. So once the training finishes we can sample from the best model by pointing the sampling script at this directory:

```sh
python sample.py --out_dir=runs/shakespeare-char
```

This generates a few samples, for example:

```
ANGELO:
And cowards it be strawn to my bed,
And thrust the gates of my threats,
Because he that ale away, and hang'd
An one with him.

DUKE VINCENTIO:
I thank your eyes against it.

DUKE VINCENTIO:
Then will answer him to save the malm:
And what have you tyrannous shall do this?

DUKE VINCENTIO:
If you have done evils of all disposition
To end his power, the day of thrust for a common men
That I leave, to fight with over-liking
Hasting in a roseman.
```

lol  `¯\_(ツ)_/¯`. Not bad for a character-level model after 3 minutes of training on a GPU. Better results are quite likely obtainable by instead finetuning a pretrained GPT-2 model on this dataset (see finetuning section later).

**I only have a macbook** (or other cheap computer). No worries, we can still train a GPT but we want to dial things down a notch. I recommend getting the bleeding edge PyTorch nightly ([select it here](https://pytorch.org/get-started/locally/) when installing) as it is currently quite likely to make your code more efficient. But even without it, a simple train run could look as follows:

```sh
python train.py config/train_shakespeare_char.py --device=cpu --compile=False --eval_iters=20 --log_interval=1 --block_size=64 --batch_size=12 --n_layer=4 --n_head=4 --n_embd=128 --max_iters=2000 --lr_decay_iters=2000 --dropout=0.0
```

Here, since we are running on CPU instead of GPU we must set both `--device=cpu` and also turn off PyTorch 2.0 compile with `--compile=False`. Then when we evaluate we get a bit more noisy but faster estimate (`--eval_iters=20`, down from 200), our context size is only 64 characters instead of 256, and the batch size only 12 examples per iteration, not 64. We'll also use a much smaller Transformer (4 layers, 4 heads, 128 embedding size), and decrease the number of iterations to 2000 (and correspondingly usually decay the learning rate to around max_iters with `--lr_decay_iters`). Because our network is so small we also ease down on regularization (`--dropout=0.0`). This still runs in about ~3 minutes, but gets us a loss of only 1.88 and therefore also worse samples, but it's still good fun:

```sh
python sample.py --out_dir=runs/shakespeare-char --device=cpu
```
Generates samples like this:

```
GLEORKEN VINGHARD III:
Whell's the couse, the came light gacks,
And the for mought you in Aut fries the not high shee
bot thou the sought bechive in that to doth groan you,
No relving thee post mose the wear
```

Not bad for ~3 minutes on a CPU, for a hint of the right character gestalt. If you're willing to wait longer, feel free to tune the hyperparameters, increase the size of the network, the context length (`--block_size`), the length of training, etc.

Finally, on Apple Silicon Macbooks and with a recent PyTorch version make sure to add `--device=mps` (short for "Metal Performance Shaders"); PyTorch then uses the on-chip GPU that can *significantly* accelerate training (2-3X) and allow you to use larger networks. See [Issue 28](https://github.com/karpathy/nanoGPT/issues/28) for more.

## reproducing GPT-2

For a quick **1,000-iteration run on one CUDA GPU**, use
[`config/train_openwebtext_gpt2_1k.py`](config/train_openwebtext_gpt2_1k.py).
The [OpenWebText quick-start instructions](data/openwebtext/readme.md#quick-1000-iteration-run-on-one-cuda-gpu)
prepare 10,000 training documents plus a separate 1,000-document holdout from a
pinned Hugging Face revision and log the run to W&B project `lm_research`.

To train a fresh model for **10,000 iterations** on the same prepared subset, run
`uv run --locked --extra cu128 python train.py config/train_openwebtext_gpt2_10k.py`.
This saves a separate run in `runs/openwebtext-gpt2-10k`.

A more serious deep learning professional may be more interested in reproducing GPT-2 results. So here we go - we first tokenize the dataset, in this case the [OpenWebText](https://openwebtext2.readthedocs.io/en/latest/), an open reproduction of OpenAI's (private) WebText:

```sh
python data/openwebtext/prepare.py
```

This downloads and tokenizes the [OpenWebText](https://huggingface.co/datasets/openwebtext) dataset. It will create a `train.bin` and `val.bin` which holds the GPT2 BPE token ids in one sequence, stored as raw uint16 bytes. Then we're ready to kick off training. To reproduce GPT-2 (124M) you'll want at least an 8X A100 40GB node and run:

```sh
torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py
```

This will run for about 4 days using PyTorch Distributed Data Parallel (DDP) and go down to loss of ~2.85. Now, a GPT-2 model just evaluated on OWT gets a val loss of about 3.11, but if you finetune it it will come down to ~2.85 territory (due to an apparent domain gap), making the two models ~match.

If you're in a cluster environment and you are blessed with multiple GPU nodes you can make GPU go brrrr e.g. across 2 nodes like:

```sh
# Run on the first (master) node with example IP 123.456.123.456:
torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
# Run on the worker node:
torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
```

It is a good idea to benchmark your interconnect (e.g. iperf3). In particular, if you don't have Infiniband then also prepend `NCCL_IB_DISABLE=1` to the above launches. Your multinode training will work, but most likely _crawl_. By default checkpoints are periodically written to the `--out_dir`. We can sample from the model by simply `python sample.py`.

Finally, to train on a single GPU simply run the `python train.py` script. Have a look at all of its args, the script tries to be very readable, hackable and transparent. You'll most likely want to tune a number of those variables depending on your needs.

## baselines

OpenAI GPT-2 checkpoints allow us to get some baselines in place for openwebtext. We can get the numbers as follows:

```sh
$ python train.py config/eval_gpt2.py
$ python train.py config/eval_gpt2_medium.py
$ python train.py config/eval_gpt2_large.py
$ python train.py config/eval_gpt2_xl.py
```

and observe the following losses on train and val:

| model | params | train loss | val loss |
| ------| ------ | ---------- | -------- |
| gpt2 | 124M         | 3.11  | 3.12     |
| gpt2-medium | 350M  | 2.85  | 2.84     |
| gpt2-large | 774M   | 2.66  | 2.67     |
| gpt2-xl | 1558M     | 2.56  | 2.54     |

However, we have to note that GPT-2 was trained on (closed, never released) WebText, while OpenWebText is just a best-effort open reproduction of this dataset. This means there is a dataset domain gap. Indeed, taking the GPT-2 (124M) checkpoint and finetuning on OWT directly for a while reaches loss down to ~2.85. This then becomes the more appropriate baseline w.r.t. reproduction.

## finetuning

For a CUDA experiment that trains GPT-2 124M **from scratch** on tokenized Tiny
Shakespeare, use [config/train_shakespeare_gpt2.py](config/train_shakespeare_gpt2.py).
See the [dataset setup and launch instructions](data/shakespeare/readme.md).
This small dataset is useful for studying overfitting, not broad pretraining.

Finetuning is no different than training, we just make sure to initialize from a pretrained model and train with a smaller learning rate. For an example of how to finetune a GPT on new text go to `data/shakespeare` and run `prepare.py` to download the tiny shakespeare dataset and render it into a `train.bin` and `val.bin`, using the OpenAI BPE tokenizer from GPT-2. Unlike OpenWebText this will run in seconds. Finetuning can take very little time, e.g. on a single GPU just a few minutes. Run an example finetuning like:

```sh
python train.py config/finetune_shakespeare.py
```

This will load the config parameter overrides in `config/finetune_shakespeare.py` (I didn't tune them much though). Basically, we initialize from a GPT2 checkpoint with `init_from` and train as normal, except shorter and with a small learning rate. If you're running out of memory try decreasing the model size (they are `{'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}`) or possibly decreasing the `block_size` (context length). The best checkpoint (lowest validation loss) will be in the `out_dir` directory, e.g. in `runs/shakespeare` by default, per the config file. You can then run the code in `sample.py --out_dir=runs/shakespeare`:

```
THEODORE:
Thou shalt sell me to the highest bidder: if I die,
I sell thee to the first; if I go mad,
I sell thee to the second; if I
lie, I sell thee to the third; if I slay,
I sell thee to the fourth: so buy or sell,
I tell thee again, thou shalt not sell my
possession.

JULIET:
And if thou steal, thou shalt not sell thyself.

THEODORE:
I do not steal; I sell the stolen goods.

THEODORE:
Thou know'st not what thou sell'st; thou, a woman,
Thou art ever a victim, a thing of no worth:
Thou hast no right, no right, but to be sold.
```

Whoa there, GPT, entering some dark place over there. I didn't really tune the hyperparameters in the config too much, feel free to try!

## sampling / inference

Use the script `sample.py` to sample either from pre-trained GPT-2 models released by OpenAI, or from a model you trained yourself. For example, here is a way to sample from the largest available `gpt2-xl` model:

```sh
python sample.py \
    --init_from=gpt2-xl \
    --start="What is the answer to life, the universe, and everything?" \
    --num_samples=5 --max_new_tokens=100
```

If you'd like to sample from a model you trained, use the `--out_dir` to point the code appropriately. You can also prompt the model with some text from a file, e.g. ```python sample.py --start=FILE:prompt.txt```.

### terminal chat

Training presets save checkpoints and logs in `runs/<run-name>/`. Run `chat.py` to interact with the newest saved checkpoint. From the repository root, using the CUDA environment:

```sh
uv run --locked --extra cu128 python chat.py
```

Replies stream as they are generated. The interface keeps conversation history, drops the oldest complete turns when the context fills, and supports `/reset`, `/help`, and `/exit`. Ctrl+C stops a reply; Ctrl+D exits. By default, chat immediately loads the most recently modified checkpoint in `runs/*/ckpt.pt`. Use `--select` for the interactive chooser, `--list-models` to list checkpoints newest first, or `--out_dir=runs/shakespeare-gpt2` or `--checkpoint=/path/to/ckpt.pt` to select a model explicitly. CUDA is selected automatically when available; `--device=cpu` forces CPU inference. Keep the same PyTorch extra used when installing the environment (`cpu` or `cu128`).

Base models trained on Shakespeare generate text continuations and have not been trained to follow chat instructions. For these models, `--raw` lets you enter a prompt such as `ROMEO:` without adding conversation labels or history:

```sh
uv run --locked --extra cu128 python chat.py --out_dir=runs/shakespeare-gpt2 --raw --max_new_tokens=200
```

Use `--temperature=0` for greedy decoding, or adjust `--temperature` and `--top_k` for sampling (`--top_k=0` disables top-k filtering). GPT-2, byte-level BPE, and character tokenizers are supported. Custom tokenizer checkpoints use `data/<dataset>/meta.pkl` from the saved configuration; pass `--meta=/path/to/meta.pkl` if it has moved. Character prompts must use characters in the model's vocabulary; use `--raw` if that vocabulary cannot represent the chat labels. No additional training or W&B run is started.

For the 100,032-parameter Shakespeare experiment, see [the small-model preset](data/shakespeare/readme.md#100k-parameter-experiment). It uses a 512-token subword vocabulary and saves to `runs/shakespeare-100k`.

## efficiency notes

For simple model benchmarking and profiling, `bench.py` might be useful. It's identical to what happens in the meat of the training loop of `train.py`, but omits much of the other complexities.

Note that the code by default uses [PyTorch 2.0](https://pytorch.org/get-started/pytorch-2.0/). At the time of writing (Dec 29, 2022) this makes `torch.compile()` available in the nightly release. The improvement from the one line of code is noticeable, e.g. cutting down iteration time from ~250ms / iter to 135ms / iter. Nice work PyTorch team!

## todos

- Investigate and add FSDP instead of DDP
- Eval zero-shot perplexities on standard evals (e.g. LAMBADA? HELM? etc.)
- Finetune the finetuning script, I think the hyperparams are not great
- Schedule for linear batch size increase during training
- Incorporate other embeddings (rotary, alibi)
- Separate out the optim buffers from model params in checkpoints I think
- Additional logging around network health (e.g. gradient clip events, magnitudes)
- Few more investigations around better init etc.

## troubleshooting

Note that by default this repo uses PyTorch 2.0 (i.e. `torch.compile`). This is fairly new and experimental, and not yet available on all platforms (e.g. Windows). If you're running into related error messages try to disable this by adding `--compile=False` flag. This will slow down the code but at least it will run.

For some context on this repository, GPT, and language modeling it might be helpful to watch my [Zero To Hero series](https://karpathy.ai/zero-to-hero.html). Specifically, the [GPT video](https://www.youtube.com/watch?v=kCc8FmEb1nY) is popular if you have some prior language modeling context.

For more questions/discussions feel free to stop by **#nanoGPT** on Discord:

[![](https://dcbadge.vercel.app/api/server/3zy8kqD9Cp?compact=true&style=flat)](https://discord.gg/3zy8kqD9Cp)

## acknowledgements

All nanoGPT experiments are powered by GPUs on [Lambda labs](https://lambdalabs.com), my favorite Cloud GPU provider. Thank you Lambda labs for sponsoring nanoGPT!
