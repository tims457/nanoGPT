"""Train a 100,032-parameter GPT on Tiny Shakespeare with 512 subword tokens."""

out_dir = 'runs/shakespeare-100k'
dataset = 'shakespeare_bpe512'
init_from = 'scratch'

# Includes token/position embeddings; tied input/output weights count once.
# (512 + 1024) * 32 + 4 * (12 * 32**2 + 13 * 32) + 2 * 32 = 100,032.
n_layer = 4
n_head = 2
n_embd = 32
block_size = 1024
bias = True
dropout = 0.2

device = 'cuda'
dtype = 'bfloat16'
compile = True

# Match the previous 124M run's optimizer schedule and 4,096 tokens per step.
batch_size = 4
gradient_accumulation_steps = 1
max_iters = 1000
learning_rate = 3e-4
decay_lr = True
warmup_iters = 100
lr_decay_iters = 1000
min_lr = 3e-5
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

eval_interval = 50
eval_iters = 20
log_interval = 10
always_save_checkpoint = True

wandb_log = True
wandb_project = 'lm_research'
wandb_run_name = 'shakespeare-100k-bpe512-scratch'
wandb_mode = 'online'
