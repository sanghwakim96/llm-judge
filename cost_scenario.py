import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
from joblib import Parallel, delayed
import matplotlib.ticker as ticker

# --- 0. Matplotlib Settings ---
mpl.style.use("classic")
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['mathtext.fontset'] = 'dejavusans'

def calculate_lp_error(shat, s_k, p):
    diff = np.abs(shat - s_k)
    if np.isinf(p): return np.max(diff)
    return np.power(np.sum(np.power(diff, p)), 1/p)

# --- 1. Simulation Worker Functions ---

def run_oracle_worker(B, p, env, costs, exponent, seed):
    rng = np.random.default_rng(seed)
    K, J = env['K'], env['J']
    s_k, true_vars, alphas, betas = env['s_k'], env['true_vars'], env['alphas'], env['betas']
    
    j_s = np.argmin(costs[np.newaxis, :] * true_vars, axis=1)
    v_star = costs[j_s] * true_vars[np.arange(K), j_s]
    w_s = (v_star)**exponent
    w_s /= np.sum(w_s)
    
    shat_o = np.zeros(K)
    for k in range(K):
        n_k = max(1, int(w_s[k] * B / costs[j_s[k]]))
        shat_o[k] = np.mean(rng.beta(alphas[k, j_s[k]], betas[k, j_s[k]], n_k))
    return calculate_lp_error(shat_o, s_k, p)

def run_est_ivwe_worker(B, p, env, costs, delta, R, log_factor, exponent, seed):
    rng = np.random.default_rng(seed)
    K, J = env['K'], env['J']
    s_k, alphas, betas = env['s_k'], env['alphas'], env['betas']
    
    if p >= 2:
        theoretical_N0 = (2 * B)**(1/3) * (log_factor)**(2/3)
    else:
        theoretical_N0 = 2**(p/(2*p+2)) * R**((3*p+2)/(2*p+2)) * log_factor**((3*p+2)/(4*p+4)) * B**((p+2)/(4*p+4))
    
    N_0 = max(2, min(int(np.ceil(theoretical_N0)), int(0.5 * B / (K * np.sum(costs)))))
    tau = R * np.sqrt(2 / (N_0 - 1) * np.log(4 * K * J / delta))

    hat_v, shat_p1 = np.zeros((K, J)), np.zeros((K, J))
    for k in range(K):
        for j in range(J):
            samples = rng.beta(alphas[k,j], betas[k,j], N_0)
            shat_p1[k, j], hat_v[k, j] = np.mean(samples), np.var(samples, ddof=1)
    
    B_p = B - N_0 * K * np.sum(costs) - K * np.max(costs)
    if B_p <= 0:
        raise ValueError(f"Budget {B} is too small.")
    bar_v = (tau + np.sqrt(np.maximum(hat_v, 0)))**2
    h_j_s = np.argmin(costs[np.newaxis, :] * bar_v, axis=1)
    h_w_s = (costs[h_j_s] * bar_v[np.arange(K), h_j_s])**exponent
    h_w_s /= np.sum(h_w_s)
    
    shat_est = np.zeros(K)
    for k in range(K):
        n_p2 = int(np.ceil(h_w_s[k] * B_p / costs[h_j_s[k]]))
        shat_p2 = np.mean(rng.beta(alphas[k, h_j_s[k]], betas[k, h_j_s[k]], n_p2))
        n_tot = np.array([N_0 if j != h_j_s[k] else N_0 + n_p2 for j in range(J)])
        m_tot = shat_p1[k, :].copy()
        m_tot[h_j_s[k]] = (N_0 * shat_p1[k, h_j_s[k]] + n_p2 * shat_p2) / n_tot[h_j_s[k]]
        ivw_w = n_tot / np.maximum(bar_v[k, :], 1e-7)
        shat_est[k] = np.sum(ivw_w * m_tot) / np.sum(ivw_w)
    return calculate_lp_error(shat_est, s_k, p)

# --- 2. Experiment Execution and Integrated Visualization Function ---

def run_cost_analysis(p, num_runs, budgets, delta=0.05, R=1.0, seed=0):
    K, J = 1000, 10
    log_factor = np.log(4 * K * J / delta)
    exponent = 1 if np.isinf(p) else p / (p + 2)
    p_fname = 'inf' if np.isinf(p) else str(p)

    rng = np.random.default_rng(seed)
    s_k = rng.uniform(0.1, 0.9, K)
    max_v = s_k * (1 - s_k)
    true_vars = np.zeros((K, J))
    for j in range(J):
        true_vars[:, j] = rng.uniform(max_v * (j + 1) / (2 * J), max_v * (J + j) / (2 * J))
    
    nu = (s_k[:, np.newaxis] * (1 - s_k[:, np.newaxis]) / true_vars) - 1
    env = {'K': K, 'J': J, 's_k': s_k, 'true_vars': true_vars,
           'alphas': s_k[:, np.newaxis] * nu, 'betas': (1 - s_k[:, np.newaxis]) * nu}

    mean_vars = np.mean(true_vars, axis=0)
    mean_inv_vars = np.mean(1.0 / true_vars, axis=0)
    scenarios = [
        ("Bad is Expensive", mean_vars / np.mean(mean_vars) * 0.1, '#e74c3c', 's'),
        ("Bad is Cheap", mean_inv_vars / np.mean(mean_inv_vars) * 0.1, '#2ecc71', 'd'),
        ("Uniform Costs", np.full(J, 0.1), '#3498db', 'o')
    ]

    algorithms = ["Oracle", "Est-IVWE"]
    
    for alg in algorithms:
        print(f"\n--- Testing Algorithm: {alg} ---")
        plt.figure(figsize=(8.5, 6))
        results_to_save = {}

        for scenario_idx, (title, costs, color, marker) in enumerate(scenarios):
            print(f"Scenario: {title}")
            history = np.zeros((num_runs, len(budgets)))
            
            for b_idx, B in enumerate(budgets):
                worker_seeds = [seed + scenario_idx * 100_000 + b_idx * 10_000 + i for i in range(num_runs)]
                if alg == "Oracle":
                    res = Parallel(n_jobs=-1)(delayed(run_oracle_worker)(B, p, env, costs, exponent, worker_seed) for worker_seed in worker_seeds)
                else:
                    res = Parallel(n_jobs=-1)(delayed(run_est_ivwe_worker)(B, p, env, costs, delta, R, log_factor, exponent, worker_seed) for worker_seed in worker_seeds)
                history[:, b_idx] = res

            results_to_save[title] = history
            plt.fill_between(budgets, np.percentile(history, 10, axis=0), np.percentile(history, 90, axis=0), color=color, alpha=0.15, edgecolor='none')
            plt.plot(budgets, np.mean(history, axis=0), marker=marker, label=f"{alg} ({title})", color=color, linewidth=2)

        max_b = np.max(budgets)
        scale = 10**int(np.floor(np.log10(max_b)))
        plt.xticks(budgets, [f'{b/scale:.1f}' for b in budgets])
        
        plt.xlim(min(budgets) - max_b * 0.05, max_b * 1.05)
        plt.xlabel(f'Budget (B) [$\\times 10^{{{int(np.log10(scale))}}}$]', fontsize=12)
        
        ax = plt.gca()
        y_max = ax.get_ylim()[1]
        y_exp = int(np.floor(np.log10(y_max)))
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{x/10**y_exp:.1f}'))
        ax.text(0, 1.02, fr'$\times 10^{{{y_exp}}}$', transform=ax.transAxes, fontsize=11, va='bottom', ha='left')
        ax.set_ylim(0, y_max * 1.05)

        p_label = r'\infty' if np.isinf(p) else str(p)
        plt.title(fr"{alg} $L_{{{p_label}}}$ Error under 3 Cost Scenarios" + f"\n(K={K}, J={J}, Runs={num_runs})", fontsize=14)
        plt.legend(frameon=True, loc='upper right', fontsize=10)
        plt.subplots_adjust(left=0.05, right=0.99, top=0.92, bottom=0.08)
        save_dir = "./fig/synthetic"
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(f"{save_dir}/cost_scenarios_{alg.replace('-', '_')}_l{p_fname}.pdf",  dpi=300)
        np.savez(f"{save_dir}/cost_scenarios_{alg.replace('-', '_')}_l{p_fname}.npz", budgets=budgets, **results_to_save)
        plt.close()

# --- 3. Execution ---
if __name__ == "__main__":
    
    P_VALS = [1, 2, np.inf]
    NUM_RUNS = 25
    
    BUDGETS = [1e6, 3e6, 5e6, 7e6, 9e6] 

    for P_VAL in P_VALS:
        run_cost_analysis(p=P_VAL, num_runs=NUM_RUNS, budgets=BUDGETS, seed=0)
