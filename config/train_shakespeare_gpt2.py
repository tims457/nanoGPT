"""Train GPT-2 124M from scratch on GPT-2-tokenized Tiny Shakespeare using CUDA."""

out_dir = 'runs/shakespeare-gpt2'
dataset = 'shakespeare'
init_from = 'scratch'

# GPT-2 small architecture; prepare.py supplies the 50,257-token vocabulary.
n_layer = 12
n_head = 12
n_embd = 768
block_size = 1024
bias = True
dropout = 0.2

device = 'cuda'
dtype = 'bfloat16'
compile = True

# 4,096 tokens per optimizer step; 1,000 steps revisit the training data ~13.6 times.
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
wandb_run_name = 'shakespeare-gpt2-124m-scratch'
wandb_mode = 'online'
