"""Train GPT-2 124M from scratch for 10,000 updates on the prepared OpenWebText subset."""

out_dir = 'runs/openwebtext-gpt2-10k'
dataset = 'openwebtext_10k'
init_from = 'scratch'

n_layer = 12
n_head = 12
n_embd = 768
block_size = 1024
bias = True
dropout = 0.0

device = 'cuda'
dtype = 'bfloat16'
compile = True

# 4,096 tokens per update; 40.96M tokens total (~3.69 passes over this subset).
batch_size = 4
gradient_accumulation_steps = 1
max_iters = 10000

# Retain the 1K run's optimizer and warmup; extend cosine decay to all 10K steps.
learning_rate = 6e-4
decay_lr = True
warmup_iters = 100
lr_decay_iters = 10000
min_lr = 6e-5
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

eval_interval = 500
eval_iters = 20
log_interval = 10
always_save_checkpoint = True

wandb_log = True
wandb_project = 'lm_research'
wandb_run_name = 'openwebtext-gpt2-124m-10k-scratch'
wandb_run_id = ''
wandb_mode = 'online'
