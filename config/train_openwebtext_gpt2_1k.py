"""Quick GPT-2 124M pretraining run on a streamed Hugging Face OpenWebText subset."""

out_dir = 'runs/openwebtext-gpt2-1k'
dataset = 'openwebtext_10k'
init_from = 'scratch'

# Original GPT-2 small architecture: 124,439,808 total parameters with GPT-2 BPE.
n_layer = 12
n_head = 12
n_embd = 768
block_size = 1024
bias = True
dropout = 0.0

device = 'cuda'
dtype = 'bfloat16'
compile = True

# Single GPU: 4,096 tokens per update and 4,096,000 tokens over this run.
batch_size = 4
gradient_accumulation_steps = 1
max_iters = 1000

learning_rate = 6e-4
decay_lr = True
warmup_iters = 100
lr_decay_iters = 1000
min_lr = 6e-5
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

eval_interval = 100
eval_iters = 20
log_interval = 10
always_save_checkpoint = True

wandb_log = True
wandb_project = 'lm_research'
wandb_run_name = 'openwebtext-gpt2-124m-1k-scratch'
wandb_run_id = '' # start a new run; --init_from=resume restores the checkpoint ID
wandb_mode = 'online'
