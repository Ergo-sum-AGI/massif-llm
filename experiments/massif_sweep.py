# experiments/massif_sweep.py

import torch
from massif.observables import (compute_persistence, detect_flip,
                                  compute_directional_alignment,
                                  compute_curvature, compute_radial_variance,
                                  compute_lead_lag, compute_tau_eff)

def run_massif_sweep(model, tokenizer, prompt, n_runs=50, max_tokens=30,
                     temperature=0.5, device='cuda'):
    """
    Full MASSIF N=50 sweep for a single prompt.
    Returns complete taxonomy metrics.
    """
    model.eval()
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)

    all_hidden_states = []  # (N, T, d)
    all_flip_times = []
    all_curvatures = []
    all_norms = []

    with torch.no_grad():
        for run in range(n_runs):
            torch.manual_seed(run)  # independent seeds
            hidden_sequence = []
            ids = input_ids.clone()

            for step in range(max_tokens):
                logits = model(ids)
                last_logit = logits[0, -1, :] / temperature
                probs = torch.softmax(last_logit, dim=-1)
                next_token = torch.multinomial(probs, 1)
                ids = torch.cat([ids, next_token.unsqueeze(0)], dim=-1)

                h = model.get_hidden_states()
                if h is not None:
                    hidden_sequence.append(h[0, -1, :].cpu())

            if len(hidden_sequence) < 3:
                continue

            h_seq = torch.stack(hidden_sequence)  # (T, d)
            all_hidden_states.append(h_seq)

            # Per-run metrics
            persistence = compute_persistence(h_seq)
            flip_t = detect_flip(persistence)
            all_flip_times.append(flip_t)
            all_curvatures.append(compute_curvature(h_seq))
            sigma, cv = compute_radial_variance(h_seq)
            all_norms.append((sigma, cv))

    # Ensemble metrics
    h_ensemble = torch.stack(all_hidden_states)  # (N, T, d)
    R_t = compute_directional_alignment(h_ensemble)
    delta_t = compute_lead_lag(R_t, all_flip_times)

    flip_rate = sum(1 for f in all_flip_times if f is not None) / n_runs
    mean_kappa = sum(all_curvatures) / len(all_curvatures)
    mean_sigma = sum(s for s,_ in all_norms) / len(all_norms)
    mean_cv = sum(cv for _,cv in all_norms) / len(all_norms)
    max_norm = max(h_ensemble.norm(dim=-1).max().item() for _ in [1])

    # Effective warning
    valid_flips = [f for f in all_flip_times if f is not None]
    tau_collapse = sum(valid_flips) / len(valid_flips) if valid_flips else 0
    tau_eff = compute_tau_eff(tau_collapse, flip_rate)

    return {
        'flip_rate': flip_rate,
        'delta_t': delta_t,
        'tau_eff': tau_eff,
        'mean_kappa': mean_kappa,
        'mean_sigma': mean_sigma,
        'mean_cv': mean_cv,
        'max_norm': max_norm,
        'R_t': R_t.numpy(),
        'flip_times': all_flip_times,
    }
    
def run_checkpoint_massif_eval(model, config, step, tokenizer, device='cuda'):
    """
    Run full MASSIF evaluation at a checkpoint.
    """
    print(f"\n{'='*60}")
    print(f"📊 MASSIF CHECKPOINT EVALUATION - STEP {step}")
    print(f"{'='*60}")
    
    # Use smaller N for checkpoint speed (can adjust)
    n_runs = 30
    max_tokens = 20
    
    results = run_massif_sweep(
        model=model,
        tokenizer=tokenizer,
        prompt="I " * 4,
        n_runs=n_runs,
        max_tokens=max_tokens,
        temperature=0.5,
        device=device
    )
    
    print(f"   Flip rate: {results['flip_rate']:.1%} ({results['flip_rate']*n_runs:.0f}/{n_runs})")
    print(f"   Delta_t: {results['delta_t']:.1f}")
    print(f"   τ_eff: {results['tau_eff']:.1f} steps")
    print(f"   Mean curvature (κ̄): {results['mean_kappa']:.4f} rad")
    print(f"   Radial variance CV: {results['mean_cv']:.4f}")
    print(f"   Max norm: {results['max_norm']:.2f}")
    
    # Classify
    if results['max_norm'] > 200:
        if results['delta_t'] is not None and results['delta_t'] < -2:
            dyn_class = "Accelerator (Runaway-norm)"
        else:
            dyn_class = "Runaway"
    elif results['delta_t'] is None:
        dyn_class = "Unknown"
    elif results['delta_t'] < -2:
        dyn_class = "Accelerator"
    elif results['delta_t'] > 2:
        dyn_class = "Decelerator"
    else:
        dyn_class = "Neutral"
    
    print(f"   Dynamical class: {dyn_class}")
    print(f"{'='*60}\n")
    
    return results    