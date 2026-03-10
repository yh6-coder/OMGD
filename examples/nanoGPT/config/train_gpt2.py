# config for training GPT-2 (124M) down to very nice loss of ~2.85 on 1 node of 8X A100 40GB
# launch as the following (e.g. in a screen session) and wait ~5 days:
# $ torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py

init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'
mask_type = 'lisa_wor' # 'none', 'lisa', 'lisa_wor'
warm_up = 1000
warmup_iters = warm_up
seed = 555

# these make the total batch size be ~0.5M
# 12 batch size * 1024 block size * 5 gradaccum * 8 GPUs = 491,520
batch_size = 60 
block_size = 1024
gradient_accumulation_steps = 8

# model settings
n_layer = 12 # 12
n_head = 12 # 12
n_embd = 768 # 768

# this makes total number of tokens be 300B
max_iters = 100000
lr_decay_iters = 100000

# eval stuff
eval_interval = 100 #100
eval_iters = 50 #200, for ddp, //ddp_model_size
log_interval = 10

# weight decay
weight_decay = 1e-1

# ckpt saving
ckpt_interval = 20000