"""
Experiment runners for curvature bifurcation analysis.
Generates all figures and statistics from the paper.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import argparse

try:
    from .curvature_model import f, J, H_f, S, create_task_hessian
except ImportError:
    from curvature_model import f, J, H_f, S, create_task_hessian

# Create figures directory
os.makedirs('figures', exist_ok=True)


def run_transition(n=50, n_trials=20, alpha_max=5.0, n_alphas=300, 
                   task_hessian=None, seed=456, savefig=True):
    """
    Run curvature transition experiment.
    Plots λ_min(H_total) vs α for multiple random θ points.
    """
    np.random.seed(seed)
    alphas = np.linspace(0, alpha_max, n_alphas)
    
    if task_hessian is None:
        H_task = create_task_hessian(n, seed=seed)
    else:
        H_task = task_hessian
    
    alpha_c_values = []
    plt.figure(figsize=(12, 6))
    
    for trial in tqdm(range(n_trials), desc="Running trials"):
        theta = np.random.randn(n) * 0.5
        S_theta = S(theta)
        
        lambda_vals = []
        for alpha in alphas:
            H_total = H_task + alpha * S_theta
            eigenvals = np.linalg.eigvals(H_total)
            lambda_min = np.min(eigenvals)
            lambda_vals.append(lambda_min)
        
        lambda_vals = np.array(lambda_vals)
        plt.plot(alphas, lambda_vals, linewidth=1, alpha=0.5, color='blue')
        
        crossing = np.where(lambda_vals <= 0)[0]
        if len(crossing) > 0:
            alpha_c_values.append(alphas[crossing[0]])
    
    plt.axhline(0, color='red', linestyle='--', linewidth=2, label=r'$\lambda_{\min}=0$')
    plt.title(f'Curvature Transition: {n_trials} θ points, n={n}', fontsize=14, fontweight='bold')
    plt.xlabel(r'$\alpha$ (self-consistency weight)', fontsize=12)
    plt.ylabel(r'$\lambda_{\min}(H_{\text{total}})$', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    if savefig:
        plt.savefig(f'figures/curvature_transition_n{n}.png', dpi=300, bbox_inches='tight')
    plt.close()
    return alpha_c_values


def run_histogram(n=50, n_trials=20, alpha_max=5.0, n_alphas=300, seed=456, savefig=True):
    """Run histogram experiment to show distribution of α_c."""
    alpha_c_values = run_transition(n, n_trials, alpha_max, n_alphas, seed=seed, savefig=False)
    
    plt.figure(figsize=(10, 5))
    plt.hist(alpha_c_values, bins=8, color='skyblue', edgecolor='black', alpha=0.7)
    
    mean_alpha = np.mean(alpha_c_values)
    std_alpha = np.std(alpha_c_values)
    
    plt.axvline(mean_alpha, color='red', linestyle='--', linewidth=2, label=f'Mean $\\alpha_c$ = {mean_alpha:.2f}')
    plt.axvspan(mean_alpha - std_alpha, mean_alpha + std_alpha, alpha=0.2, color='red', label=f'Std = {std_alpha:.2f}')
    
    plt.title(f'Distribution of $\\alpha_c$ across {n_trials} θ points (n={n})', fontsize=14, fontweight='bold')
    plt.xlabel(r'$\alpha_c$', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.legend()
    plt.tight_layout()
    
    if savefig:
        plt.savefig(f'figures/alpha_c_histogram_n{n}.png', dpi=300, bbox_inches='tight')
    plt.close()
    return alpha_c_values


def run_scaling(n_list=[50, 75, 100, 150, 200], n_trials=50, seed=456, savefig=True):
    """Test dimensional scaling with heatmaps."""
    alphas = np.linspace(0, 5, 200)
    results = {}
    
    for n in n_list:
        H_task = create_task_hessian(n, seed=seed)
        lambda_matrix = np.zeros((n_trials, len(alphas)))
        alpha_c_values = []
        
        for trial in tqdm(range(n_trials), desc=f"n={n}"):
            theta = np.random.randn(n) * 0.5
            S_theta = S(theta)
            for a_idx, alpha in enumerate(alphas):
                H_total = H_task + alpha * S_theta
                lambda_matrix[trial, a_idx] = np.min(np.linalg.eigvals(H_total))
            
            crossing = np.where(lambda_matrix[trial, :] <= 0)[0]
            if len(crossing) > 0: alpha_c_values.append(alphas[crossing[0]])
        
        results[n] = {'mean_alpha': np.mean(alpha_c_values), 'std_alpha': np.std(alpha_c_values)}
        
        plt.figure(figsize=(12, 6))
        plt.imshow(lambda_matrix, aspect='auto', origin='lower', extent=[0, 5, 0, n_trials], cmap='coolwarm', vmin=-2, vmax=2)
        plt.axvline(results[n]['mean_alpha'], color='black', linestyle='--')
        plt.title(f'Curvature Heatmap: n={n}')
        plt.xlabel(r'$\alpha$')
        plt.ylabel('Trial')
        if savefig: plt.savefig(f'figures/heatmap_n{n}.png')
        plt.close()
    return results

def run_sensitivity(n=50, n_trials=20, n_hessians=10, seed=456, savefig=True):
    """Test sensitivity to different task Hessian realizations."""
    all_alpha_c = []
    for h_idx in tqdm(range(n_hessians), desc="Hessians"):
        H_task = create_task_hessian(n, seed=seed + h_idx)
        all_alpha_c.extend(run_transition(n, n_trials, savefig=False, task_hessian=H_task))
    
    plt.figure(figsize=(10, 5))
    plt.hist(all_alpha_c, bins=15, color='skyblue', edgecolor='black', alpha=0.7)
    plt.title('Sensitivity to Task Hessian')
    if savefig: plt.savefig('figures/sensitivity.png')
    plt.close()
    return all_alpha_c

def run_illusion(n=50, n_trials=20, seed=456, savefig=True):
    """Demonstrate the -102 illusion vs true bifurcation."""
    alphas = np.linspace(0, 5, 300)
    H_task = create_task_hessian(n, seed=seed)
    plt.figure(figsize=(12, 6))
    for trial in range(n_trials):
        theta = np.random.randn(n) * 0.5
        S_theta = S(theta)
        l_vals = [np.min(np.linalg.eigvals(H_task + a * S_theta)) for a in alphas]
        plt.plot(alphas, l_vals, alpha=0.3, color='blue')
        # Illusion scaling
        if any(v <= 0 for v in l_vals):
            ac = alphas[np.where(np.array(l_vals) <= 0)[0][0]]
            plt.plot(alphas, np.array(l_vals) * (-102/(ac*10)), alpha=0.2, color='gray', linestyle='--')
    plt.axhline(0, color='red', linestyle='--')
    plt.title('The -102 Illusion vs Reality')
    if savefig: plt.savefig('figures/illusion.png')
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=str, default='all')
    parser.add_argument('--n', type=int, default=50)
    parser.add_argument('--trials', type=int, default=20)
    args = parser.parse_args()
    
    if args.experiment in ['transition', 'all']: run_transition(n=args.n, n_trials=args.trials)
    if args.experiment in ['histogram', 'all']: run_histogram(n=args.n, n_trials=args.trials)
    if args.experiment in ['scaling', 'all']: run_scaling(n_trials=args.trials)
    if args.experiment in ['sensitivity', 'all']: run_sensitivity(n=args.n)
    if args.experiment in ['illusion', 'all']: run_illusion(n=args.n)

if __name__ == "__main__":
    main()
