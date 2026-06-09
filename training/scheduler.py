# training/scheduler.py
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

def get_scheduler(optimizer, warmup_steps, total_steps):
    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps)
    return SequentialLR(optimizer, [warmup_scheduler, cosine_scheduler], milestones=[warmup_steps])