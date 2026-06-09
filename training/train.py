# training/train.py

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

def train(model, dataloader, config, device='cuda'):
    optimizer = AdamW(model.parameters(), lr=3e-4,
                      betas=(0.9, 0.95), weight_decay=0.1)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.total_steps)

    model.train()
    step = 0

    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        labels = input_ids[:, 1:].contiguous()

        # Forward pass
        logits = model(input_ids[:, :-1])

        # LM loss
        lm_loss = torch.nn.functional.cross_entropy(
            logits.view(-1, config.vocab_size),
            labels.view(-1)
        )

        # MASSIF regularization (from stored hidden states)
        hidden = model.get_hidden_states()  # (B, T, d)
        if hidden is not None and step > config.warmup_steps:
            reg_loss = massif_regularization_loss(hidden)
        else:
            reg_loss = torch.tensor(0.0, device=device)

        total_loss = lm_loss + reg_loss

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        # Log
        if step % 100 == 0:
            print(f"Step {step} | LM: {lm_loss:.4f} | Reg: {reg_loss:.4f}")

        # MASSIF sweep every 5K steps
        if step % 5000 == 0 and step > 0:
            run_checkpoint_massif_eval(model, config, step)

        step += 1