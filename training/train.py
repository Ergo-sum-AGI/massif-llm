# training/train.py
#
# Main training loop for the MASSIF-optimized LLM.
#
# Usage (Colab):
#   !python training/train.py --config configs/skeleton_100m.yaml --max_steps 1000
#
# The script is self-contained: it creates a TinyStories dataloader internally
# when no external dataloader is provided, so it runs immediately without
# additional setup.

import os
import sys
import argparse
import yaml
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# Ensure project root is on the path regardless of where the script is called from
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from architecture.model import MASSIFModel
from architecture.config import MASSIFConfig
from massif.losses import MASSIFRegularizer, MASSIFRegConfig, PersistenceConfig, NormStabilityConfig, CurvatureConfig


# ===========================================================================
# Config loading
# ===========================================================================

def load_config_dict(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def build_massif_config(config_dict: dict) -> MASSIFConfig:
    """Build MASSIFConfig from a flat or nested yaml dict."""
    return MASSIFConfig(
        d_model=config_dict.get('d_model', 512),
        n_layers=config_dict.get('n_layers', 12),
        n_heads=config_dict.get('n_heads', 8),
        vocab_size=config_dict.get('vocab_size', 32000),
        max_seq_len=config_dict.get('max_seq_len', 512),
        track_norms=config_dict.get('track_norms', True),
        total_steps=config_dict.get('total_steps', 10_000),
        warmup_steps=config_dict.get('warmup_steps', 1_000),
        learning_rate=config_dict.get('learning_rate', 3e-4),
        weight_decay=config_dict.get('weight_decay', 0.1),
        batch_size=config_dict.get('batch_size', 16),
        grad_clip=config_dict.get('grad_clip', 1.0),
        massif_reg_enabled=config_dict.get('massif_reg_enabled', True),
        massif_warmup_steps=config_dict.get('massif_warmup_steps', 1_000),
        persist_enabled=config_dict.get('persist_enabled', True),
        persist_lambda=config_dict.get('persist_lambda', 0.01),
        persist_threshold=config_dict.get('persist_threshold', 0.3),
        norm_stab_enabled=config_dict.get('norm_stab_enabled', True),
        norm_stab_lambda=config_dict.get('norm_stab_lambda', 0.005),
        kappa_enabled=config_dict.get('kappa_enabled', True),
        kappa_lambda=config_dict.get('kappa_lambda', 0.01),
        kappa_target=config_dict.get('kappa_target', 1.5),
        massif_eval_every=config_dict.get('massif_eval_every', 5_000),
        massif_n_runs=config_dict.get('massif_n_runs', 50),
        massif_prompt=config_dict.get('massif_prompt', 'I ' * 4),
        massif_temperature=config_dict.get('massif_temperature', 0.5),
        massif_max_tokens=config_dict.get('massif_max_tokens', 30),
        checkpoint_dir=config_dict.get('checkpoint_dir', 'checkpoints'),
        checkpoint_every=config_dict.get('checkpoint_every', 10_000),
        log_every=config_dict.get('log_every', 100),
        run_name=config_dict.get('run_name', 'massif_run'),
    )


def build_regularizer(config: MASSIFConfig) -> MASSIFRegularizer:
    reg_config = MASSIFRegConfig(
        enabled=config.massif_reg_enabled,
        warmup_steps=config.massif_warmup_steps,
        persistence=PersistenceConfig(
            enabled=config.persist_enabled,
            lambda_weight=config.persist_lambda,
            threshold=config.persist_threshold,
            schedule='linear',
        ),
        norm_stability=NormStabilityConfig(
            enabled=config.norm_stab_enabled,
            lambda_weight=config.norm_stab_lambda,
            schedule='constant',
        ),
        curvature=CurvatureConfig(
            enabled=config.kappa_enabled,
            lambda_weight=config.kappa_lambda,
            target_kappa=config.kappa_target,
            schedule='constant',
        ),
    )
    return MASSIFRegularizer(reg_config)


# ===========================================================================
# Dataloader
# ===========================================================================

def get_dataloader(config: MASSIFConfig, tokenizer, max_steps: int):
    """
    Returns a dataloader. Tries FineWeb-Edu first; falls back to TinyStories;
    falls back to a synthetic random-token dataset for pipeline testing.
    """
    try:
        from datasets import load_dataset
        print("Loading TinyStories dataset (streaming)...")
        dataset = load_dataset(
            "roneneldan/TinyStories",
            split="train",
            streaming=True,
        )
        return _wrap_dataset(dataset, tokenizer, config, max_steps)
    except Exception as e:
        print(f"Dataset load failed ({e}). Falling back to synthetic data.")
        return _synthetic_dataloader(config, max_steps)


def _wrap_dataset(dataset, tokenizer, config: MASSIFConfig, max_steps: int):
    """Wrap a HuggingFace streaming dataset into batches of input_ids."""
    seq_len = config.max_seq_len
    batch_size = config.batch_size
    buffer = []

    for example in dataset:
        text = example.get('text', '')
        if not text:
            continue
        ids = tokenizer.encode(text, add_special_tokens=True)
        buffer.extend(ids)

        while len(buffer) >= seq_len + 1:
            chunk = buffer[:seq_len + 1]
            buffer = buffer[seq_len + 1:]
            input_ids = torch.tensor(chunk[:seq_len], dtype=torch.long).unsqueeze(0)

            # Accumulate into batch
            if not hasattr(_wrap_dataset, '_batch'):
                _wrap_dataset._batch = []
            _wrap_dataset._batch.append(input_ids)

            if len(_wrap_dataset._batch) == batch_size:
                batch_tensor = torch.cat(_wrap_dataset._batch, dim=0)
                _wrap_dataset._batch = []
                yield {'input_ids': batch_tensor}

        if hasattr(_wrap_dataset, '_step_count'):
            _wrap_dataset._step_count += 1
            if _wrap_dataset._step_count >= max_steps:
                break


def _synthetic_dataloader(config: MASSIFConfig, max_steps: int):
    """
    Generates random token batches. Used when no dataset is available.
    Sufficient for pipeline testing and verifying the training loop runs.
    Loss will not decrease meaningfully on random data.
    """
    print("Using synthetic random-token data (pipeline test only).")
    for step in range(max_steps):
        input_ids = torch.randint(
            0, config.vocab_size,
            (config.batch_size, config.max_seq_len)
        )
        yield {'input_ids': input_ids}


# ===========================================================================
# Checkpoint
# ===========================================================================

def save_checkpoint(model, optimizer, scheduler, step: int, config: MASSIFConfig):
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    path = os.path.join(config.checkpoint_dir, f"step_{step:06d}.pt")
    torch.save({
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'config': config,
    }, path)
    print(f"Checkpoint saved: {path}")


# ===========================================================================
# MASSIF evaluation at checkpoint
# ===========================================================================

def run_checkpoint_massif_eval(model, config: MASSIFConfig, step: int,
                                tokenizer=None, device='cuda'):
    """
    Run a lightweight MASSIF sweep at a training checkpoint.
    Uses N=10 runs for speed during training; full N=50 sweep is in
    experiments/massif_sweep.py.
    """
    try:
        from experiments.massif_sweep import run_massif_sweep
        from massif.taxonomy import classify_model

        print(f"\n{'='*60}")
        print(f"MASSIF CHECKPOINT EVAL — Step {step}")
        print(f"{'='*60}")

        # Use tokenizer if available, else encode prompt as raw token ids
        prompt = config.massif_prompt
        if tokenizer is not None:
            prompt_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
        else:
            # Fallback: encode "I " as token id 40 in GPT-2 vocab (approximate)
            prompt_ids = torch.tensor([[40, 40, 40, 40]], device=device)

        metrics = run_massif_sweep(
            model=model,
            prompt_ids=prompt_ids,
            n_runs=10,                          # fast checkpoint sweep
            max_tokens=config.massif_max_tokens,
            temperature=config.massif_temperature,
            device=device,
        )

        dyn_class = classify_model(metrics)

        print(f"  Flip rate:    {metrics['flip_rate']*100:.1f}%")
        print(f"  Delta_t:      {metrics['delta_t']}")
        print(f"  tau_eff:      {metrics['tau_eff']:.2f} steps")
        print(f"  Mean kappa:   {metrics['mean_kappa']:.3f} rad")
        print(f"  CV:           {metrics['mean_cv']:.4f}")
        print(f"  Max norm:     {metrics['max_norm']:.1f}")
        print(f"  Class:        {dyn_class}")
        print(f"{'='*60}\n")

    except Exception as e:
        print(f"MASSIF eval skipped at step {step}: {e}")


# ===========================================================================
# Main training loop
# ===========================================================================

def train(config: MASSIFConfig, max_steps: int = None,
          tokenizer=None, device: str = 'cuda'):
    """
    Main training loop.

    Args:
        config:     MASSIFConfig instance (built from yaml)
        max_steps:  Override config.total_steps if provided (useful for quick tests)
        tokenizer:  HuggingFace tokenizer. If None, synthetic data is used.
        device:     'cuda' or 'cpu'
    """
    total_steps = max_steps if max_steps is not None else config.total_steps

    print(f"\n{'='*60}")
    print(f"MASSIF LLM Training — {config.run_name}")
    print(f"Device:      {device}")
    print(f"Model:       d_model={config.d_model}, n_layers={config.n_layers}, "
          f"n_heads={config.n_heads}")
    print(f"Vocab:       {config.vocab_size}")
    print(f"Steps:       {total_steps}")
    print(f"Batch size:  {config.batch_size}")
    print(f"MASSIF reg:  {'enabled' if config.massif_reg_enabled else 'disabled'}")
    print(f"{'='*60}\n")

    # --- Model ---
    model = MASSIFModel(config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params/1e6:.1f}M")

    # --- Regularizer ---
    regularizer = build_regularizer(config)

    # --- Optimizer ---
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
        weight_decay=config.weight_decay,
    )

    # --- Scheduler ---
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=total_steps,
        eta_min=config.learning_rate * config.min_lr_ratio,
    )

    # --- Dataloader ---
    dataloader = get_dataloader(config, tokenizer, total_steps)

    # --- Training loop ---
    model.train()
    step = 0

    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)

        # Shift: input is all but last token, label is all but first
        inputs = input_ids[:, :-1]
        labels = input_ids[:, 1:].contiguous()

        # Forward pass
        logits = model(inputs)                          # (B, T-1, vocab)

        # LM cross-entropy loss
        lm_loss = F.cross_entropy(
            logits.reshape(-1, config.vocab_size),
            labels.reshape(-1),
        )

        # MASSIF regularization loss
        hidden = model.get_hidden_states()              # (B, T-1, d_model) or None
        if hidden is not None:
            reg_losses = regularizer.compute(hidden, step=step)
            reg_loss = reg_losses['total']
        else:
            reg_loss = torch.tensor(0.0, device=device)

        total_loss = lm_loss + reg_loss

        # Backward
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()
        scheduler.step()

        # Logging
        if step % config.log_every == 0:
            reg_val = reg_loss.item() if isinstance(reg_loss, torch.Tensor) \
                      else reg_loss
            print(f"Step {step:>6} | "
                  f"LM: {lm_loss.item():.4f} | "
                  f"Reg: {reg_val:.4f} | "
                  f"LR: {scheduler.get_last_lr()[0]:.2e}")

        # MASSIF checkpoint eval
        if step % config.massif_eval_every == 0 and step > 0:
            model.eval()
            run_checkpoint_massif_eval(
                model, config, step,
                tokenizer=tokenizer, device=device
            )
            model.train()

        # Checkpoint save
        if step % config.checkpoint_every == 0 and step > 0:
            save_checkpoint(model, optimizer, scheduler, step, config)

        step += 1
        if step >= total_steps:
            break

    # --- Final MASSIF evaluation ---
    print(f"\n{'='*60}")
    print("FINAL MASSIF EVALUATION")
    print(f"{'='*60}")
    model.eval()
    run_checkpoint_massif_eval(
        model, config, step,
        tokenizer=tokenizer, device=device
    )

    return model


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a MASSIF-optimized language model."
    )
    parser.add_argument(
        '--config', type=str, required=True,
        help="Path to YAML config file (e.g. configs/skeleton_100m.yaml)"
    )
    parser.add_argument(
        '--max_steps', type=int, default=None,
        help="Override total_steps from config (useful for quick tests)"
    )
    parser.add_argument(
        '--device', type=str, default=None,
        help="Device: 'cuda' or 'cpu'. Auto-detected if not specified."
    )
    parser.add_argument(
        '--tokenizer', type=str, default='gpt2',
        help="HuggingFace tokenizer name or path. Default: gpt2"
    )
    parser.add_argument(
        '--no_tokenizer', action='store_true',
        help="Skip tokenizer loading and use synthetic data (pipeline test)"
    )
    args = parser.parse_args()

    # Device
    if args.device:
        device = args.device
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Config
    config_dict = load_config_dict(args.config)
    config = build_massif_config(config_dict)

    # Override max_steps if passed on command line
    if args.max_steps is not None:
        config.total_steps = args.max_steps

    # Tokenizer
    tokenizer = None
    if not args.no_tokenizer:
        try:
            from transformers import AutoTokenizer
            print(f"Loading tokenizer: {args.tokenizer}")
            tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            # Resize vocab if tokenizer vocab differs from config
            if tokenizer.vocab_size != config.vocab_size:
                print(f"Warning: tokenizer vocab size ({tokenizer.vocab_size}) "
                      f"differs from config vocab_size ({config.vocab_size}). "
                      f"Using tokenizer vocab size.")
                config.vocab_size = tokenizer.vocab_size
        except Exception as e:
            print(f"Tokenizer load failed ({e}). Using synthetic data.")
            tokenizer = None

    # Train
    model = train(
        config=config,
        max_steps=args.max_steps,
        tokenizer=tokenizer,
        device=device,
    )

    print("\nTraining complete.")
