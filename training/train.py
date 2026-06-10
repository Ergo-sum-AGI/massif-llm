# training/train.py
#
# Main training loop for the MASSIF-optimized LLM.
#
# Usage (Colab):
#   !python training/train.py --config configs/skeleton_100m.yaml --max_steps 1000 --mount_drive
#
# Usage (SageMaker):
#   !python training/train.py --config configs/skeleton_100m.yaml --max_steps 1000 --s3_bucket my-bucket
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
    """Build MASSIFConfig from a nested yaml dict structure."""
    # Extract nested blocks safely
    massif_reg = config_dict.get('massif_regularization', {})
    persist = massif_reg.get('persistence', {})
    norm_stab = massif_reg.get('norm_stability', {})
    curvature = massif_reg.get('curvature', {})

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
        
        # Fixed: Reading from nested structure
        massif_reg_enabled=massif_reg.get('enabled', True),
        massif_warmup_steps=massif_reg.get('warmup_steps', 1_000),
        
        persist_enabled=persist.get('enabled', True),
        persist_lambda=persist.get('lambda', 0.01),
        persist_threshold=persist.get('threshold', 0.3),
        
        norm_stab_enabled=norm_stab.get('enabled', True),
        norm_stab_lambda=norm_stab.get('lambda', 0.005),
        
        kappa_enabled=curvature.get('enabled', True),
        kappa_lambda=curvature.get('lambda', 0.01),
        kappa_target=curvature.get('target_kappa', 1.5),  # Matches 'target_kappa' in full_512m.yaml
        
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
# Cloud persistence: Google Drive & S3
# ===========================================================================

def mount_google_drive():
    """
    Mount Google Drive if running in Colab.
    Call this before training starts.
    """
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        print("Google Drive mounted at /content/drive")
        return '/content/drive/MyDrive'
    except ImportError:
        print("Not running in Colab — Google Drive mount skipped.")
        return None


def get_s3_client():
    """Initialize boto3 S3 client if available."""
    try:
        import boto3
        return boto3.client('s3')
    except ImportError:
        print("boto3 not installed — S3 sync disabled.")
        return None


def save_to_drive(local_path: str, drive_base: str, relative_path: str = None):
    """
    Copy a local file/folder to Google Drive.
    """
    if drive_base is None:
        return
    
    import shutil
    if relative_path is None:
        relative_path = os.path.basename(local_path)
    
    drive_path = os.path.join(drive_base, 'MASSIF_Checkpoints', relative_path)
    os.makedirs(os.path.dirname(drive_path), exist_ok=True)
    
    if os.path.isdir(local_path):
        if os.path.exists(drive_path):
            shutil.rmtree(drive_path)
        shutil.copytree(local_path, drive_path)
    else:
        shutil.copy2(local_path, drive_path)
    
    print(f"Synced to Drive: {drive_path}")


def save_to_s3(local_path: str, bucket: str, s3_prefix: str = 'massif-checkpoints'):
    """
    Upload a file or directory to S3.
    """
    s3 = get_s3_client()
    if s3 is None or bucket is None:
        return
    
    import shutil
    
    if os.path.isdir(local_path):
        # Upload directory recursively
        for root, dirs, files in os.walk(local_path):
            for file in files:
                local_file = os.path.join(root, file)
                rel_path = os.path.relpath(local_file, os.path.dirname(local_path))
                s3_key = f"{s3_prefix}/{rel_path}"
                s3.upload_file(local_file, bucket, s3_key)
                print(f"  S3 upload: s3://{bucket}/{s3_key}")
    else:
        s3_key = f"{s3_prefix}/{os.path.basename(local_path)}"
        s3.upload_file(local_path, bucket, s3_key)
        print(f"  S3 upload: s3://{bucket}/{s3_key}")


def sync_checkpoint(local_path: str, drive_base: str = None, s3_bucket: str = None):
    """
    Universal sync: local -> Drive (if mounted) AND S3 (if bucket provided).
    Always keeps local copy in /content/ as safety net.
    """
    # Local safety net already exists at local_path
    print(f"Local checkpoint: {local_path}")
    
    # Google Drive
    if drive_base:
        save_to_drive(local_path, drive_base)
    
    # S3
    if s3_bucket:
        save_to_s3(local_path, s3_bucket)


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
    """Wrap a HuggingFace streaming dataset into batches of input_ids efficiently."""
    seq_len = config.max_seq_len
    batch_size = config.batch_size
    buffer = []
    
    # Initialize state tracking cleanly
    if not hasattr(_wrap_dataset, '_batch'):
        _wrap_dataset._batch = []
    if not hasattr(_wrap_dataset, '_step_count'):
        _wrap_dataset._step_count = 0

    for example in dataset:
        text = example.get('text', '')
        if not text:
            continue
        
        # FIXED: Added truncation and max_length to prevent the warning
        ids = tokenizer.encode(text, add_special_tokens=True, truncation=True, max_length=seq_len)
        buffer.extend(ids)

        # Process chunks out of the buffer
        cursor = 0
        while len(buffer) - cursor >= seq_len + 1:
            chunk = buffer[cursor : cursor + seq_len + 1]
            cursor += seq_len + 1
            
            input_ids = torch.tensor(chunk[:seq_len], dtype=torch.long).unsqueeze(0)
            _wrap_dataset._batch.append(input_ids)

            if len(_wrap_dataset._batch) == batch_size:
                batch_tensor = torch.cat(_wrap_dataset._batch, dim=0)
                _wrap_dataset._batch = []
                yield {'input_ids': batch_tensor}
                
                _wrap_dataset._step_count += 1
                if _wrap_dataset._step_count >= max_steps:
                    return

        # Clear out the consumed portion of the buffer
        if cursor > 0:
            buffer = buffer[cursor:]


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

def save_checkpoint(model, optimizer, scheduler, step: int, config: MASSIFConfig, 
                    drive_base: str = None, s3_bucket: str = None):
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
    
    # Sync to cloud(s)
    sync_checkpoint(path, drive_base=drive_base, s3_bucket=s3_bucket)


def save_final_model(model, config: MASSIFConfig, drive_base: str = None, s3_bucket: str = None):
    """Save the final trained model weights separately for easy loading."""
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    final_path = os.path.join(config.checkpoint_dir, "final_model.pt")
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
    }, final_path)
    print(f"Final model saved: {final_path}")
    
    # Sync to cloud(s)
    sync_checkpoint(final_path, drive_base=drive_base, s3_bucket=s3_bucket)
    # Also sync the entire checkpoint directory
    sync_checkpoint(config.checkpoint_dir, drive_base=drive_base, s3_bucket=s3_bucket)


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

        # Clean string handoff
        prompt = config.massif_prompt

        metrics = run_massif_sweep(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
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
          tokenizer=None, device: str = 'cuda', 
          drive_base: str = None, s3_bucket: str = None):
    """
    Main training loop.

    Args:
        config:     MASSIFConfig instance (built from yaml)
        max_steps:  Override config.total_steps if provided (useful for quick tests)
        tokenizer:  HuggingFace tokenizer. If None, synthetic data is used.
        device:     'cuda' or 'cpu'
        drive_base: Google Drive base path (e.g., '/content/drive/MyDrive')
        s3_bucket:  S3 bucket name for checkpoint sync
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
    print(f"Drive sync:   {'enabled' if drive_base else 'disabled'}")
    print(f"S3 sync:      {'enabled (' + s3_bucket + ')' if s3_bucket else 'disabled'}")
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

        # Checkpoint save — every 500 steps as requested
        if step % 500 == 0 and step > 0:
            save_checkpoint(model, optimizer, scheduler, step, config, 
                          drive_base=drive_base, s3_bucket=s3_bucket)

        step += 1
        if step >= total_steps:
            break

    # --- Final save ---
    save_checkpoint(model, optimizer, scheduler, step, config,
                    drive_base=drive_base, s3_bucket=s3_bucket)
    save_final_model(model, config, drive_base=drive_base, s3_bucket=s3_bucket)

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
    parser.add_argument(
        '--mount_drive', action='store_true',
        help="Mount Google Drive and sync checkpoints there (Colab only)"
    )
    parser.add_argument(
        '--s3_bucket', type=str, default=None,
        help="S3 bucket name for checkpoint sync (SageMaker)"
    )
    args = parser.parse_args()

    # HF token should be set via environment variable, NOT hardcoded.
    # Set it in a notebook cell before running this script:
    #   import os; os.environ['HF_TOKEN'] = 'hf_...'
    hf_token = os.environ.get('HF_TOKEN')
    if hf_token:
        print("HF_TOKEN found in environment.")
    else:
        print("Warning: HF_TOKEN not set. Unauthenticated HF Hub requests may be rate-limited.")

    # Device
    if args.device:
        device = args.device
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Mount Google Drive if requested
    drive_base = None
    if args.mount_drive:
        drive_base = mount_google_drive()

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
        drive_base=drive_base,
        s3_bucket=args.s3_bucket,
    )

    print("\nTraining complete.")