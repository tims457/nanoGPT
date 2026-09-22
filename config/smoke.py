"""Two CPU optimizer steps with Hugging Face data and offline W&B logging."""
out_dir = 'runs/smoke'
dataset = 'hf_smoke'
device = 'cpu'
dtype = 'float32'
compile = False

n_layer = 1
n_head = 2
n_embd = 32
block_size = 32
batch_size = 2
gradient_accumulation_steps = 1
dropout = 0.0

max_iters = 2
eval_interval = 1
eval_iters = 1
log_interval = 1
decay_lr = False
always_save_checkpoint = True

wandb_log = True
wandb_project = 'lm_research'
wandb_run_name = 'hf-smoke'
wandb_mode = 'offline'
