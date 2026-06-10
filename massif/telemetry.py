# massif/telemetry.py
#
# Forward hook system for extracting hidden states from any model layer.
#
# Two use cases:
#
#   1. MASSIFModel (architecture/model.py):
#      Hidden states are stored directly via MASSIFBlock._hidden_state.
#      Use model.get_hidden_states() — no hooks needed.
#
#   2. External models (GPT-2, TinyLlama, DeepSeek, etc.):
#      Use TelemetryHook to register a forward hook on any named layer.
#      This is how the 13-architecture taxonomy experiments were run.
#      Use attach_telemetry() / detach_telemetry() for clean lifecycle.
#
# Usage (external model):
#   from massif.telemetry import attach_telemetry, detach_telemetry, get_hidden_state
#
#   hooks = attach_telemetry(model, layer_name='transformer.h.11')
#   # ... run model ...
#   h = get_hidden_state(hooks, 'transformer.h.11')
#   detach_telemetry(hooks)

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Callable


# ===========================================================================
# Core hook class
# ===========================================================================

class TelemetryHook:
    """
    A single forward hook attached to one nn.Module layer.

    Stores the output of the hooked layer after each forward pass.
    Supports both full-sequence storage (for MASSIF sweeps) and
    single-step storage (for real-time monitoring).

    Args:
        layer_name (str):     Human-readable name for this hook.
        capture_input (bool): If True, capture layer input instead of output.
                              Default False (capture output).
        sequence_mode (bool): If True, accumulate outputs across steps into
                              a list (for post-hoc MASSIF analysis).
                              If False, store only the most recent output.
    """

    def __init__(self, layer_name: str, capture_input: bool = False,
                 sequence_mode: bool = False):
        self.layer_name     = layer_name
        self.capture_input  = capture_input
        self.sequence_mode  = sequence_mode

        self._handle        = None      # the registered hook handle
        self._last_output   = None      # most recent captured tensor
        self._sequence      = []        # accumulated sequence (sequence_mode only)
        self._call_count    = 0

    def _hook_fn(self, module: nn.Module, input, output):
        """Called automatically by PyTorch on each forward pass."""
        if self.capture_input:
            tensor = input[0] if isinstance(input, tuple) else input
        else:
            tensor = output[0] if isinstance(output, tuple) else output

        # Detach immediately to avoid holding onto the computation graph
        captured = tensor.detach()
        self._last_output = captured
        self._call_count += 1

        if self.sequence_mode:
            # Store last-token hidden state: (B, d) for generation
            if captured.dim() == 3:
                self._sequence.append(captured[:, -1, :].cpu())
            else:
                self._sequence.append(captured.cpu())

    def attach(self, module: nn.Module) -> 'TelemetryHook':
        """Register the hook on a module. Returns self for chaining."""
        self._handle = module.register_forward_hook(self._hook_fn)
        return self

    def detach(self):
        """Remove the hook from the module."""
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def get_last(self) -> Optional[torch.Tensor]:
        """Return the most recent captured output."""
        return self._last_output

    def get_sequence(self) -> Optional[torch.Tensor]:
        """
        Return accumulated sequence as a tensor (T, d) or (B, T, d).
        Only populated in sequence_mode=True.
        Returns None if sequence is empty.
        """
        if not self._sequence:
            return None
        return torch.stack(self._sequence, dim=0)

    def clear_sequence(self):
        """Clear the accumulated sequence. Call between runs."""
        self._sequence = []
        self._call_count = 0

    def is_attached(self) -> bool:
        return self._handle is not None

    def __repr__(self):
        return (f"TelemetryHook(layer='{self.layer_name}', "
                f"attached={self.is_attached()}, "
                f"calls={self._call_count}, "
                f"seq_len={len(self._sequence)})")


# ===========================================================================
# Multi-layer telemetry manager
# ===========================================================================

class TelemetryManager:
    """
    Manages multiple TelemetryHooks across a model.

    Provides a clean interface for attaching hooks to named layers,
    collecting hidden states, and cleaning up after sweeps.

    Usage:
        mgr = TelemetryManager(model)
        mgr.attach('transformer.h.11')          # by layer name
        mgr.attach_final_layer()                 # auto-detect final layer
        mgr.set_sequence_mode(True)              # accumulate across steps

        for step in range(max_tokens):
            model(ids)
            h = mgr.get_last('transformer.h.11')  # (B, T, d)

        seq = mgr.get_sequence('transformer.h.11')  # (T, d)
        mgr.clear_all()
        mgr.detach_all()
    """

    def __init__(self, model: nn.Module):
        self.model  = model
        self.hooks: Dict[str, TelemetryHook] = {}

    def attach(self, layer_path: str, capture_input: bool = False,
               sequence_mode: bool = False) -> TelemetryHook:
        """
        Attach a hook to the layer at `layer_path`.

        layer_path uses dot notation to navigate the module tree:
            'transformer.h.11'           — GPT-2 final transformer block
            'model.layers.23'            — LLaMA final layer
            'blocks.11'                  — MASSIFModel final block
            'gpt_neox.layers.5'          — Pythia

        To find the right path for any model:
            print(list(model.named_modules()))
        """
        module = self._resolve_path(layer_path)
        hook   = TelemetryHook(layer_path, capture_input, sequence_mode)
        hook.attach(module)
        self.hooks[layer_path] = hook
        return hook

    def attach_final_layer(self, sequence_mode: bool = False) -> Optional[TelemetryHook]:
        """
        Auto-detect and attach to the final transformer layer.
        Supports GPT-2, LLaMA, DeepSeek, TinyLlama, Bloom, and MASSIFModel.
        Returns the hook, or None if detection fails.
        """
        candidates = [
            # MASSIFModel
            (f'blocks.{len(list(self.model.children()))-3}', None),
            # GPT-2
            (f'transformer.h.{self._count_layers("transformer.h")-1}', None),
            # LLaMA / TinyLlama / DeepSeek / Mistral
            (f'model.layers.{self._count_layers("model.layers")-1}', None),
            # Bloom
            (f'transformer.h.{self._count_layers("transformer.h")-1}', None),
            # RWKV
            (f'rwkv.blocks.{self._count_layers("rwkv.blocks")-1}', None),
        ]
        for path, _ in candidates:
            try:
                hook = self.attach(path, sequence_mode=sequence_mode)
                print(f"TelemetryManager: auto-attached to '{path}'")
                return hook
            except (AttributeError, KeyError):
                continue
        print("TelemetryManager: auto-detect failed. Use attach(layer_path) manually.")
        print("Available layers:")
        for name, _ in self.model.named_modules():
            if name:
                print(f"  {name}")
        return None

    def set_sequence_mode(self, enabled: bool):
        """Enable or disable sequence accumulation on all hooks."""
        for hook in self.hooks.values():
            hook.sequence_mode = enabled

    def get_last(self, layer_path: str) -> Optional[torch.Tensor]:
        """Get most recent output from a hooked layer."""
        if layer_path not in self.hooks:
            raise KeyError(f"No hook registered for '{layer_path}'")
        return self.hooks[layer_path].get_last()

    def get_sequence(self, layer_path: str) -> Optional[torch.Tensor]:
        """Get accumulated sequence from a hooked layer."""
        if layer_path not in self.hooks:
            raise KeyError(f"No hook registered for '{layer_path}'")
        return self.hooks[layer_path].get_sequence()

    def clear_all(self):
        """Clear all accumulated sequences. Call between ensemble runs."""
        for hook in self.hooks.values():
            hook.clear_sequence()

    def detach_all(self):
        """Remove all hooks. Call when done with telemetry."""
        for hook in self.hooks.values():
            hook.detach()
        self.hooks.clear()

    def status(self):
        """Print status of all registered hooks."""
        print(f"TelemetryManager: {len(self.hooks)} hooks registered")
        for name, hook in self.hooks.items():
            print(f"  {hook}")

    def _resolve_path(self, path: str) -> nn.Module:
        """Navigate dot-separated path through the module tree."""
        module = self.model
        for part in path.split('.'):
            if part.isdigit():
                module = list(module.children())[int(part)]
            else:
                module = getattr(module, part)
        return module

    def _count_layers(self, path: str) -> int:
        """Count the number of layers at a given path prefix."""
        try:
            module = self._resolve_path(path)
            return len(list(module.children()))
        except (AttributeError, KeyError, IndexError):
            return 0


# ===========================================================================
# Convenience functions for external model sweeps
# ===========================================================================

def attach_telemetry(model: nn.Module,
                     layer_name: str,
                     sequence_mode: bool = True) -> Dict[str, TelemetryHook]:
    """
    Attach a single telemetry hook to a named layer.
    Returns a dict of hooks for passing to detach_telemetry().

    Example (GPT-2):
        hooks = attach_telemetry(model, 'transformer.h.11', sequence_mode=True)
        # ... run generation ...
        h_seq = get_hidden_state(hooks, 'transformer.h.11')  # (T, d)
        detach_telemetry(hooks)
    """
    mgr = TelemetryManager(model)
    mgr.attach(layer_name, sequence_mode=sequence_mode)
    return mgr.hooks


def detach_telemetry(hooks: Dict[str, TelemetryHook]):
    """Detach all hooks in a hooks dict."""
    for hook in hooks.values():
        hook.detach()


def get_hidden_state(hooks: Dict[str, TelemetryHook],
                     layer_name: str) -> Optional[torch.Tensor]:
    """Get the accumulated sequence from a named hook."""
    if layer_name not in hooks:
        raise KeyError(f"No hook for '{layer_name}'")
    return hooks[layer_name].get_sequence()


def clear_hooks(hooks: Dict[str, TelemetryHook]):
    """Clear all sequences in a hooks dict. Call between ensemble runs."""
    for hook in hooks.values():
        hook.clear_sequence()


# ===========================================================================
# MASSIF sweep wrapper for external (non-MASSIFModel) models
# ===========================================================================

def run_external_massif_sweep(model: nn.Module, tokenizer,
                               layer_name: str, prompt: str,
                               n_runs: int = 50, max_tokens: int = 30,
                               temperature: float = 0.5,
                               device: str = 'cuda') -> dict:
    """
    Run a MASSIF sweep on any external HuggingFace model by attaching
    telemetry hooks to the specified layer.

    This is how the 13-architecture taxonomy experiments were run on
    GPT-2, TinyLlama, DeepSeek, etc.

    Args:
        model:      Any HuggingFace causal LM (must have .generate() or manual loop)
        tokenizer:  Corresponding tokenizer
        layer_name: Dot-path to the final transformer layer, e.g.:
                      GPT-2:      'transformer.h.11'
                      LLaMA:      'model.layers.31'
                      TinyLlama:  'model.layers.21'
                      DeepSeek:   'model.layers.27'
                      Bloom:      'transformer.h.23'
        prompt:     Input prompt string
        n_runs:     Ensemble size
        max_tokens: Generation steps
        temperature: Sampling temperature
        device:     'cuda' or 'cpu'

    Returns:
        Standard MASSIF metrics dict (same format as run_massif_sweep)
    """
    from massif.observables import (
        compute_persistence, detect_flip, compute_directional_alignment,
        compute_curvature, compute_radial_variance, compute_lead_lag,
        compute_tau_eff,
    )
    import numpy as np

    model.eval()
    input_ids = tokenizer.encode(
        prompt, return_tensors='pt', add_special_tokens=True
    ).to(device)

    # Attach hook to specified layer
    hook = TelemetryHook(layer_name, sequence_mode=True)
    module = TelemetryManager(model)._resolve_path(layer_name)
    hook.attach(module)

    all_hidden_states = []
    all_flip_times    = []
    all_curvatures    = []
    all_norms         = []

    try:
        with torch.no_grad():
            for run in range(n_runs):
                torch.manual_seed(run)
                hook.clear_sequence()
                ids = input_ids.clone()

                for _ in range(max_tokens):
                    out = model(ids)
                    logits = out.logits if hasattr(out, 'logits') else out[0]
                    last_logit = logits[0, -1, :] / (temperature + 1e-8)
                    probs = torch.softmax(last_logit, dim=-1)
                    next_token = torch.multinomial(probs, 1)
                    ids = torch.cat([ids, next_token.unsqueeze(0)], dim=-1)

                h_seq_raw = hook.get_sequence()  # (T, d) or (T, B, d)
                if h_seq_raw is None or h_seq_raw.shape[0] < 3:
                    all_flip_times.append(None)
                    continue

                # Ensure shape is (T, d)
                if h_seq_raw.dim() == 3:
                    h_seq = h_seq_raw[:, 0, :]
                else:
                    h_seq = h_seq_raw

                all_hidden_states.append(h_seq)
                persistence = compute_persistence(h_seq)
                all_flip_times.append(detect_flip(persistence))
                all_curvatures.append(compute_curvature(h_seq))
                sigma, cv = compute_radial_variance(h_seq)
                all_norms.append((sigma, cv))

    finally:
        hook.detach()

    if len(all_hidden_states) == 0:
        return {
            'flip_rate': 0.0, 'delta_t': None, 'tau_eff': 0.0,
            'mean_kappa': 0.0, 'mean_sigma': 0.0, 'mean_cv': 0.0,
            'max_norm': 0.0, 'V_peak': 0.0,
            'R_t': np.array([]), 'flip_times': all_flip_times,
            'n_valid_runs': 0,
        }

    min_T = min(h.shape[0] for h in all_hidden_states)
    h_ensemble = torch.stack([h[:min_T] for h in all_hidden_states])

    R_t     = compute_directional_alignment(h_ensemble)
    delta_t = compute_lead_lag(R_t, all_flip_times)
    V_t     = torch.diff(R_t) if len(R_t) > 1 else torch.zeros(1)
    V_peak  = float(V_t.max().item())

    flip_rate  = sum(1 for f in all_flip_times if f is not None) / n_runs
    valid_flips = [f for f in all_flip_times if f is not None]
    tau_collapse = float(np.mean(valid_flips)) if valid_flips else 0.0

    return {
        'flip_rate':    flip_rate,
        'delta_t':      delta_t,
        'tau_eff':      compute_tau_eff(tau_collapse, flip_rate),
        'mean_kappa':   float(np.mean(all_curvatures)) if all_curvatures else 0.0,
        'mean_sigma':   float(np.mean([s for s,_ in all_norms])) if all_norms else 0.0,
        'mean_cv':      float(np.mean([cv for _,cv in all_norms])) if all_norms else 0.0,
        'max_norm':     float(h_ensemble.norm(dim=-1).max().item()),
        'V_peak':       V_peak,
        'R_t':          R_t.numpy(),
        'flip_times':   all_flip_times,
        'n_valid_runs': len(all_hidden_states),
    }


# ===========================================================================
# Known layer paths for the 13-architecture taxonomy
# ===========================================================================

KNOWN_LAYER_PATHS = {
    'gpt2':              'transformer.h.11',
    'gpt2-medium':       'transformer.h.23',
    'gpt2-large':        'transformer.h.35',
    'gpt2-xl':           'transformer.h.47',
    'bloom-560m':        'transformer.h.23',
    'tinyllama-1.1b':    'model.layers.21',
    'deepseek-1.3b':     'model.layers.27',
    'gemma-2-2b':        'model.layers.17',
    'qwen2-0.5b':        'model.layers.23',
    'phi-2':             'model.layers.31',
    'mamba-130m':        'backbone.layers.23',
    'rwkv-169m':         'rwkv.blocks.11',
    'nemotron-mini-4b':  'model.layers.31',
}


def get_layer_path(model_name: str) -> Optional[str]:
    """
    Look up the known final-layer path for a model by name fragment.
    Case-insensitive partial match.

    Example:
        path = get_layer_path('gpt2-large')   # returns 'transformer.h.35'
        path = get_layer_path('tinyllama')     # returns 'model.layers.21'
    """
    model_name_lower = model_name.lower()
    for key, path in KNOWN_LAYER_PATHS.items():
        if key in model_name_lower or model_name_lower in key:
            return path
    return None
