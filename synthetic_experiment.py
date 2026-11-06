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
    if np.isinf(p): 
        return np.max(diff)
    return np.power(np.sum(np.power(diff, p)), 1/p)

# --- 1. Environment Creation Function ---
def make_environment(K, J, seed=0):
    """Generates experimental environment variables given K (number of queries) and J (number of judges)"""
    rng = np.random.default_rng(seed)
    costs = rng.uniform(0.5, 1.5, J)
    s_k = rng.uniform(0.1, 0.9, K)
    true_vars = np.zeros((K, J))
    for k in range(K):
        max_v = s_k[k] * (1 - s_k[k]) * 0.9
        true_vars[k, :] = rng.uniform(1e-4, max_v, J)
    
    nu = (s_k[:, np.newaxis] * (1 - s_k[:, np.newaxis]) / true_vars) - 1
    alphas = s_k[:, np.newaxis] * nu
    betas = (1 - s_k[:, np.newaxis]) * nu
    
    return {
        "K": K, "J": J, 
        "costs": costs, "s_k": s_k, 
        "true_vars": true_vars, 
        "alphas": alphas, "betas": betas
    }

# --- 2. Single Execution Worker Function for Parallel Processing ---
def run_single_iteration(seed, p, budgets, env, delta, R, log_factor, exponent):
    rng = np.random.default_rng(seed)
    K, J = env["K"], env["J"]
    costs, s_k = env["costs"], env["s_k"]
    true_vars, alphas, betas = env["true_vars"], env["alphas"], env["betas"]
    
    run_results = {alg: np.zeros(len(budgets)) 
                   for alg in ["Uniform", "Oracle", "Est-IVWE (Bounded)", "Est-IVWE (Gaussian)"]}

    for b_idx, B in enumerate(budgets):
        # --- A. Parameter Settings (Bounded) ---
        if p >= 2:
            theoretical_N0 = (2 * B)**(1/3) * (log_factor)**(2/3)
        else:
            term1 = 2**(p / (2*p + 2))
            term2 = R**((3*p + 2) / (2*p + 2))
            term3 = log_factor**((3*p + 2) / (4*p + 4))
            term4 = B**((p + 2) / (4*p + 4))
            theoretical_N0 = term1 * term2 * term3 * term4
        
        N_0_ivwe = max(2, min(int(np.ceil(theoretical_N0)), int(0.5 * B / (K * np.sum(costs)))))
        tau_ivwe = R * np.sqrt(2 / (N_0_ivwe - 1) * np.log(4 * K * J / delta))

        # --- B. Parameter Settings (Gaussian) ---
        N_0_g = max(2, min(1 + int(np.ceil(16 * np.log(4 * K * J / delta))), int(0.5 * B / (K * np.sum(costs)))))

        # --- 1. Uniform Allocation ---
        n_u = max(1, int(B / (K * np.sum(costs))))
        shat_u = np.array([np.mean([np.mean(rng.beta(alphas[k,j], betas[k,j], n_u)) for j in range(J)]) for k in range(K)])
        run_results["Uniform"][b_idx] = calculate_lp_error(shat_u, s_k, p)

        # --- 2. Oracle Allocation ---
        j_s = np.argmin(costs[np.newaxis, :] * true_vars, axis=1)
        w_s = (costs[j_s] * true_vars[np.arange(K), j_s])**exponent
        w_s /= np.sum(w_s)
        shat_o = np.array([np.mean(rng.beta(alphas[k, j_s[k]], betas[k, j_s[k]], 
                                                 max(1, int(w_s[k]*B/costs[j_s[k]])))) for k in range(K)])
        run_results["Oracle"][b_idx] = calculate_lp_error(shat_o, s_k, p)

        # --- 3. Est-IVWE (Bounded) ---
        hat_v, shat_p1 = np.zeros((K, J)), np.zeros((K, J))
        for k in range(K):
            for j in range(J):
                samples = rng.beta(alphas[k,j], betas[k,j], N_0_ivwe)
                shat_p1[k, j], hat_v[k, j] = np.mean(samples), np.var(samples, ddof=1)
        
        B_p_s = B - N_0_ivwe * K * np.sum(costs) - K * np.max(costs)
        if B_p_s <= 0:
            raise ValueError(f"Budget {B} is too small.")
        bar_v_s = (tau_ivwe + np.sqrt(np.maximum(hat_v, 0)))**2
        h_j_s = np.argmin(costs[np.newaxis, :] * bar_v_s, axis=1)
        h_w_s = (costs[h_j_s] * bar_v_s[np.arange(K), h_j_s])**exponent
        h_w_s /= np.sum(h_w_s)
        
        shat_est = np.zeros(K)
        for k in range(K):
            n_p2 = int(np.ceil(h_w_s[k] * B_p_s / costs[h_j_s[k]]))
            shat_p2 = np.mean(rng.beta(alphas[k, h_j_s[k]], betas[k, h_j_s[k]], n_p2))
            n_tot = np.array([N_0_ivwe if j != h_j_s[k] else N_0_ivwe + n_p2 for j in range(J)])
            m_tot = shat_p1[k, :].copy()
            m_tot[h_j_s[k]] = (N_0_ivwe * shat_p1[k, h_j_s[k]] + n_p2 * shat_p2) / n_tot[h_j_s[k]]
            ivw_w = n_tot / np.maximum(bar_v_s[k, :], 1e-7)
            shat_est[k] = np.sum(ivw_w * m_tot) / np.sum(ivw_w)
        run_results["Est-IVWE (Bounded)"][b_idx] = calculate_lp_error(shat_est, s_k, p)

        # --- 4. Est-IVWE (Gaussian) ---
        hat_v_g, shat_p1_g = np.zeros((K, J)), np.zeros((K, J))
        for k in range(K):
            for j in range(J):
                samples = rng.beta(alphas[k,j], betas[k,j], N_0_g)
                shat_p1_g[k, j], hat_v_g[k, j] = np.mean(samples), np.var(samples, ddof=1)
        
        B_p_g = B - N_0_g * K * np.sum(costs) - K * np.max(costs)
        if B_p_g <= 0:
            raise ValueError(f"Budget {B} is too small.")
        bar_V_g = costs[np.newaxis, :] * np.maximum(hat_v_g, 1e-7)
        h_j_g = np.argmin(bar_V_g, axis=1)
        h_w_g = (bar_V_g[np.arange(K), h_j_g])**exponent
        h_w_g /= np.sum(h_w_g)
        
        shat_est_g = np.zeros(K)
        for k in range(K):
            n_p2_g = int(np.ceil(h_w_g[k] * B_p_g / costs[h_j_g[k]]))
            shat_p2_g = np.mean(rng.beta(alphas[k, h_j_g[k]], betas[k, h_j_g[k]], n_p2_g))
            n_tot_g = np.array([N_0_g if j != h_j_g[k] else N_0_g + n_p2_g for j in range(J)])
            m_tot_g = shat_p1_g[k, :].copy()
            m_tot_g[h_j_g[k]] = (N_0_g * shat_p1_g[k, h_j_g[k]] + n_p2_g * shat_p2_g) / n_tot_g[h_j_g[k]]
            ivw_w_g = n_tot_g / np.maximum(hat_v_g[k, :], 1e-7)
            shat_est_g[k] = np.sum(ivw_w_g * m_tot_g) / np.sum(ivw_w_g)
        run_results["Est-IVWE (Gaussian)"][b_idx] = calculate_lp_error(shat_est_g, s_k, p)
        
    return run_results

# --- 3. Experiment Execution Function (Parallel Processing Structure) ---
def run_experiment(p, budgets, num_runs, env, delta=0.05, R=1.0, seed=0):
    K, J = env["K"], env["J"]
    log_factor = np.log(4 * K * J / delta)
    exponent = 1 if np.isinf(p) else p / (p + 2)
    
    print(f"Starting Experiment for p={p} (Parallelized with {num_runs} runs)...")
    
    results = Parallel(n_jobs=-1)(
        delayed(run_single_iteration)(seed + i, p, budgets, env, delta, R, log_factor, exponent) 
        for i in range(num_runs)
    )
    
    raw_history = {alg: np.zeros((num_runs, len(budgets))) 
                   for alg in ["Uniform", "Oracle", "Est-IVWE (Bounded)", "Est-IVWE (Gaussian)"]}
    
    for run_idx, res in enumerate(results):
        for alg in raw_history.keys():
            raw_history[alg][run_idx] = res[alg]

    p_fname = 'inf' if np.isinf(p) else str(p)
    save_dir = "./fig/synthetic"
    os.makedirs(save_dir, exist_ok=True)
    data_save_path = f"{save_dir}/synthetic_K{K}_J{J}_runs{num_runs}_l{p_fname}.npz"
    
    np.savez(data_save_path, budgets=budgets, **raw_history)
    
    print(f"Data successfully saved to: {data_save_path}")

    visualize_and_save(p, budgets, raw_history, K, J, num_runs)

# --- 4. Visualization and Saving Function ---
def visualize_and_save(p, budgets, raw_history, K, J, num_runs):
    
    plt.figure(figsize=(8.5, 6))
    colors = {'Uniform': '#bdc3c7', 'Oracle': '#3498db', 
              'Est-IVWE (Bounded)': '#e74c3c', 'Est-IVWE (Gaussian)': '#2ecc71'}
    markers = {'Uniform': 'o', 'Oracle': 's', 
               'Est-IVWE (Bounded)': '^', 'Est-IVWE (Gaussian)': 'd'}

    for alg in raw_history:
        data = raw_history[alg]
        plt.fill_between(budgets, np.percentile(data, 10, axis=0), np.percentile(data, 90, axis=0), 
                         color=colors[alg], alpha=0.15, edgecolor='none')
        plt.plot(budgets, np.mean(data, axis=0), marker=markers[alg], label=alg, 
                 color=colors[alg], linewidth=2, linestyle='--' if alg == 'Uniform' else '-')

    max_b = np.max(budgets)
    common_exponent = int(np.floor(np.log10(max_b)))
    scale_factor = 10**common_exponent

    plt.xticks(budgets, [f'{b/scale_factor:.1f}' for b in budgets])
    plt.xlim(min(budgets) - max_b * 0.05, max_b * 1.05)
    plt.xlabel(f'Budget (B) [$\\times 10^{{{common_exponent}}}$]', fontsize=12)

    
    y_max = plt.gca().get_ylim()[1]
    y_exp = int(np.floor(np.log10(y_max)))
    y_formatter = ticker.FuncFormatter(lambda x, pos: f'{x/10**y_exp:.1f}')

    ax = plt.gca()
    ax.yaxis.set_major_formatter(y_formatter)

    ax.text(0, 1.02, fr'$\times 10^{{{y_exp}}}$', 
            transform=ax.transAxes, 
            fontsize=11,
            va='bottom', ha='left',
            fontfamily='sans-serif')
    
    ax.set_ylim(0, y_max * 1.15)

    p_label = r'\infty' if np.isinf(p) else str(p)
    plt.title(fr'$L_{{{p_label}}}$ Error by Budget (Synthetic Data)' + f'\n(K={K}, J={J}, runs={num_runs})', fontsize=14)
    plt.legend(frameon=True, loc='upper right')
    plt.tight_layout()

    p_fname = 'inf' if np.isinf(p) else str(p)
    save_dir = "./fig/synthetic"
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f"{save_dir}/synthetic_K{K}_J{J}_runs{num_runs}_l{p_fname}.pdf", bbox_inches='tight', dpi=300)
    plt.close()

# --- 5. Main Execution Block ---
if __name__ == "__main__":
    K, J = 1000, 10
    num_runs = 50
    budgets = [1e7, 3e7, 5e7, 7e7, 9e7]
    
    print("Generating environment...")
    environment = make_environment(K, J, seed=0)
    
    p_values = [1, 2, np.inf]
    
    for p_val in p_values:
        run_experiment(
            p=p_val, 
            budgets=budgets, 
            num_runs=num_runs, 
            env=environment,
            seed=0
        )
