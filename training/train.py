# training/train.py

import torch
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from architecture.model import MASSIFModel
from architecture.config import MASSIFConfig
from massif.losses import MASSIFRegularizer, MASSIFRegConfig

def load_config(config_path):
    """Load YAML config file."""
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    return config_dict

def create_model(config_dict):
    """Create model from architecture config."""
    arch_config = MASSIFConfig(
        d_model=config_dict.get('d_model', 512),
        n_layers=config_dict.get('n_layers', 12),
        n_heads=config_dict.get('n_heads', 8),
        vocab_size=config_dict.get('vocab_size', 32000),
        max_seq_len=config_dict.get('max_seq_len', 512),
        track_norms=config_dict.get('track_norms', True),
    )
    return MASSIFModel(arch_config)

def train(config_path, dataloader, device='cuda'):
    """Main training loop."""
    # Load config
    config_dict = load_config(config_path)
    
    # Create model
    model = create_model(config_dict).to(device)
    
    # Setup MASSIF regularizer if enabled
    reg_config = config_dict.get('massif_regularization', {})
    if reg_config.get('enabled', False):
        # Parse config into dataclasses
        massif_reg_config = MASSIFRegConfig(
            enabled=reg_config.get('enabled', True),
            warmup_steps=reg_config.get('warmup_steps', 1000),
            # ... parse persistence, norm_stability, curvature
        )
        regularizer = MASSIFRegularizer(massif_reg_config)
    else:
        regularizer = None
    
    # Optimizer
    optimizer = AdamW(model.parameters(), lr=3e-4,
                      betas=(0.9, 0.95), weight_decay=0.1)
    
    # Scheduler
    total_steps = config_dict.get('total_steps', 10000)
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)
    
    model.train()
    step = 0
    
    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        labels = input_ids[:, 1:].contiguous()
        
        # Forward pass
        logits = model(input_ids[:, :-1])
        
        # LM loss
        lm_loss = torch.nn.functional.cross_entropy(
            logits.view(-1, config_dict.get('vocab_size', 32000)),
            labels.view(-1)
        )
        
        # MASSIF regularization
        hidden = model.get_hidden_states()
        if regularizer is not None and hidden is not None:
            reg_losses = regularizer.compute(hidden, step=step)
            reg_loss = reg_losses['total']
        else:
            reg_loss = torch.tensor(0.0, device=device)
        
        total_loss = lm_loss + reg_loss
        
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        # Logging
        if step % 100 == 0:
            print(f"Step {step} | LM: {lm_loss:.4f} | Reg: {reg_loss:.4f}")
        if step % 5000 == 0 and step > 0:
            run_checkpoint_massif_eval(model, config, step, tokenizer=tokenizer, device=device)
        step += 1
        if step >= total_steps:
            break
    
    return model

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    
    # Dummy dataloader for testing
    # Replace with actual dataloader
    dummy_dataloader = []
    train(args.config, dummy_dataloader)