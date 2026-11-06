import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
from joblib import Parallel, delayed
import matplotlib.ticker as ticker

import scipy.stats as stats
import json

# --- 0. Matplotlib Settings ---
mpl.style.use("classic")
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['mathtext.fontset'] = 'dejavusans'

# --- 1. Data Preprocessing Functions (Majority Voting / Mode-based) ---
def load_judge_data(file_path, files, K_total=1017):
    if len(files) != 3:
        raise ValueError(f"Expected exactly 3 judge files, found {len(files)}: {files}")

    all_judge_data = []
    for actual_filename in files:
        filename = os.path.join(file_path, actual_filename) if os.path.exists(file_path) else actual_filename
        if not filename.endswith(".jsonl"): filename += ".jsonl"
        with open(filename, "r") as f:
            rewards = np.fromiter((float(json.loads(line)["predicted_score"]) for line in f), dtype=float)
        if len(rewards) % K_total != 0:
            raise ValueError(f"{filename} contains {len(rewards)} records, which is not divisible by {K_total} questions.")
        all_judge_data.append(rewards.reshape(K_total, -1))
    return all_judge_data

def valid_scores(scores):
    return scores[(scores >= 0) & (scores <= 4)]

def load_and_preprocess_majority_voting(file_path, files):
    K_total = 1017
    all_judge_data = load_judge_data(file_path, files, K_total)
    J = len(files)

    valid_indices, s_k_list = [], []
    for k in range(K_total):
        if any(len(valid_scores(all_judge_data[j][k])) < 25 for j in range(J)):
            continue
        
        current_modes = [stats.mode(valid_scores(all_judge_data[j][k]), keepdims=True).mode[0] for j in range(J)]
        
        if all(m == current_modes[0] for m in current_modes):
            valid_indices.append(k)
            s_k_list.append(current_modes[0])
            
    new_K = len(valid_indices)
    s_k = np.array(s_k_list)
    rewards_pool = [[valid_scores(all_judge_data[j][k]) for j in range(J)] for k in valid_indices]

    true_var = np.zeros((new_K, J))
    for k_idx in range(new_K):
        for j in range(J):
            samples = rewards_pool[k_idx][j]
            true_var[k_idx, j] = np.mean((samples - s_k[k_idx])**2)
    true_var = np.maximum(true_var, 1e-6)

    print(f"Preprocessing complete: {new_K} questions survived. (Applied Majority Voting s_k)")
    return new_K, J, rewards_pool, s_k, true_var, valid_indices

def load_and_preprocess_graybill_deal(file_path, files):
    K_total = 1017
    all_judge_data = load_judge_data(file_path, files, K_total)
    J = len(files)

    valid_indices, s_k_list = [], []
    for k in range(K_total):
        if any(len(valid_scores(all_judge_data[j][k])) < 25 for j in range(J)):
            continue
        
        current_modes = [stats.mode(valid_scores(all_judge_data[j][k]), keepdims=True).mode[0] for j in range(J)]
        
        if all(m == current_modes[0] for m in current_modes):
            means, vars_emp, counts = [], [], []
            for j in range(J):
                samples = valid_scores(all_judge_data[j][k])
                means.append(np.mean(samples))
                vars_emp.append(np.maximum(np.var(samples, ddof=1), 1e-6))
                counts.append(len(samples))
            
            weights = np.array(counts) / np.array(vars_emp)
            graybill_s_k = np.sum(weights * np.array(means)) / np.sum(weights)
            
            valid_indices.append(k)
            s_k_list.append(graybill_s_k)
            
    new_K = len(valid_indices)
    s_k = np.array(s_k_list)
    rewards_pool = [[valid_scores(all_judge_data[j][k]) for j in range(J)] for k in valid_indices]

    true_var = np.zeros((new_K, J))
    for k_idx in range(new_K):
        for j in range(J):
            samples = rewards_pool[k_idx][j]
            true_var[k_idx, j] = np.mean((samples - s_k[k_idx])**2)
    true_var = np.maximum(true_var, 1e-6)

    print(f"Preprocessing complete: {new_K} questions survived. (Based on Graybill-Deal Estimator)")
    return new_K, J, rewards_pool, s_k, true_var, valid_indices

# --- 2. Helper Functions ---
def calculate_lp_norm(vector, p_val):
    abs_vec = np.abs(vector)
    if p_val == np.inf: return np.max(abs_vec)
    return np.power(np.sum(np.power(abs_vec, p_val)), 1/p_val)

# --- 3. Single Execution Simulation ---
def run_single_simulation(B, p, delta, K, J, rewards_pool, s_k, true_var, costs, seed):
    rng = np.random.default_rng(seed)
    exponent = 1.0 if np.isinf(p) else p / (p + 2)
    R = 4.0
    log_factor = np.log(4 * K * J / delta)
    
    # 1) Calculate N_0
    if p >= 2:
        theoretical_N0 = (2 * B)**(1/3) * (R**2 * log_factor)**(2/3) / R**2
    else:
        term1 = 2**(p / (2*p + 2))
        term2 = R**((3*p + 2) / (2*p + 2))
        term3 = log_factor**((3*p + 2) / (4*p + 4))
        term4 = B**((p + 2) / (4*p + 4))
        theoretical_N0 = term1 * term2 * term3 * term4
    
    N_0 = int(np.ceil(theoretical_N0))
    N_0 = max(2, min(N_0, int(0.5 * B / (K * np.sum(costs)))))
    
    tau = R * np.sqrt((2 / (N_0 - 1)) * log_factor)
    N_0_g = 1 + int(np.ceil(16 * np.log(4 * K * J / delta)))
    N_0_g = max( 2, min(N_0_g, int(0.5 * B / (K * np.sum(costs)))))

    errors = {}

    # A. Uniform
    n_kj_u = max(1, int(B / (K * np.sum(costs))))
    shat_u = np.zeros(K)
    for k in range(K):
        j_means = [np.mean(rng.choice(rewards_pool[k][j], n_kj_u, replace=True)) for j in range(J)]
        shat_u[k] = np.mean(j_means)
    errors["Uniform"] = calculate_lp_norm(shat_u - s_k, p)

    # B. Oracle
    j_star = np.argmin(costs.reshape(1, J) * true_var, axis=1)
    v_k_star = costs[j_star] * true_var[np.arange(K), j_star]
    w_star = (v_k_star**exponent) / np.sum(v_k_star**exponent)
    shat_o = np.zeros(K)
    for k in range(K):
        n_k = max(1, int((w_star[k] * B) / costs[j_star[k]]))
        shat_o[k] = np.mean(rng.choice(rewards_pool[k][j_star[k]], n_k, replace=True))
    errors["Oracle"] = calculate_lp_norm(shat_o - s_k, p)

    # C. Est-IVWE
    shat_p1, hat_var = np.zeros((K, J)), np.zeros((K, J))
    for k in range(K):
        for j in range(J):
            samples = rng.choice(rewards_pool[k][j], N_0, replace=True)
            shat_p1[k, j], hat_var[k, j] = np.mean(samples), np.var(samples, ddof=1)
    
    B_prime =  B - N_0 * K * np.sum(costs) - K * np.max(costs)
    if B_prime <= 0:
        raise ValueError(f"Budget {B} is too small.")
    bar_sigma = tau + np.sqrt(np.maximum(hat_var, 0))
    hat_j_star = np.argmin(costs.reshape(1, J) * (bar_sigma**2), axis=1)
    hat_v_k = costs[hat_j_star] * (bar_sigma[np.arange(K), hat_j_star]**2)
    hat_w_star = (hat_v_k**exponent) / np.sum(hat_v_k**exponent)
    
    shat_est = np.zeros(K)
    for k in range(K):
        n_k_p2 = int(np.ceil(hat_w_star[k] * B_prime / costs[hat_j_star[k]]))
        shat_p2 = np.mean(rng.choice(rewards_pool[k][hat_j_star[k]], n_k_p2, replace=True))
        
        f_means = shat_p1[k, :].copy()
        n_counts = np.full(J, N_0)
        f_means[hat_j_star[k]] = (N_0 * shat_p1[k, hat_j_star[k]] + n_k_p2 * shat_p2) / (N_0 + n_k_p2)
        n_counts[hat_j_star[k]] = N_0 + n_k_p2
        
        inv_v = n_counts / np.maximum(bar_sigma[k, :]**2, 1e-6)
        shat_est[k] = np.sum(inv_v * f_means) / np.sum(inv_v)
    errors["Est-IVWE (Bounded)"] = calculate_lp_norm(shat_est - s_k, p)
    # --- D. Est-IVWE (Gaussian) ---
    hat_v_g, shat_p1_g = np.zeros((K, J)), np.zeros((K, J))
    for k in range(K):
        for j in range(J):
            samples_g = rng.choice(rewards_pool[k][j], N_0_g, replace=True)
            shat_p1_g[k, j] = np.mean(samples_g)
            hat_v_g[k, j] = np.var(samples_g, ddof=1)
    
    B_p_g = (B- N_0_g * K * np.sum(costs)- K * np.max(costs))
    if B_p_g <= 0:
        raise ValueError(f"Budget {B} is too small.")
    bar_V_g_select = ( costs[np.newaxis, :] * np.maximum(hat_v_g, 1e-7) )
    h_j_g = np.argmin(bar_V_g_select, axis=1)
    
    h_w_g = (bar_V_g_select[np.arange(K), h_j_g])**exponent
    h_w_g /= (np.sum(h_w_g) + 1e-9)
    
    shat_est_g = np.zeros(K)
    for k in range(K):
        n_p2_g = int(np.ceil(h_w_g[k] * B_p_g / costs[h_j_g[k]]))
        
        shat_p2_val = np.mean(rng.choice(rewards_pool[k][h_j_g[k]], n_p2_g, replace=True))
        n_tot_best = N_0_g + n_p2_g
        mean_best = (N_0_g * shat_p1_g[k, h_j_g[k]] + n_p2_g * shat_p2_val) / n_tot_best

        n_tot_k = np.full(J, N_0_g)
        n_tot_k[h_j_g[k]] = n_tot_best
        
        means_k = shat_p1_g[k, :].copy()
        means_k[h_j_g[k]] = mean_best
        
        weights_k = n_tot_k / np.maximum(hat_v_g[k, :], 1e-7)
        shat_est_g[k] = np.sum(weights_k * means_k) / np.sum(weights_k)
        
    errors["Est-IVWE (Gaussian)"] = calculate_lp_norm(shat_est_g - s_k, p)
    return errors

# --- 4. Main Experiment Loop ---
def main_budget_sweep(budgets, num_runs, data_type='correctness', p=2, delta=0.05, costs=None, graybill=False, seed=0):
    file_path = './helpsteer2_neurips2026/'
    files = sorted(f for f in os.listdir(file_path) if f.endswith('.jsonl') and data_type in f)
    if len(files) != 3:
        raise ValueError(f"Expected 3 files for {data_type}, found {len(files)}: {files}")
    print(f"Judge order: {', '.join(files)}")
    
    if graybill:
        K, J, rewards_pool, s_k, true_var, _ = load_and_preprocess_graybill_deal(file_path, files)
    else:
        K, J, rewards_pool, s_k, true_var, _ = load_and_preprocess_majority_voting(file_path, files)
    
    if costs is None:
        costs = np.full(J, 0.1, dtype=float)
    else:
        costs = np.asarray(costs, dtype=float)
        if len(costs) != J:
            raise ValueError(f"The length of the input cost vector ({len(costs)}) does not match the number of judges ({J}).")
        if np.any(costs <= 0):
            raise ValueError("All costs must be positive.")

    raw_history = {"Uniform": [], "Oracle": [], "Est-IVWE (Bounded)": [], "Est-IVWE (Gaussian)": []}
    history = {"Uniform": [], "Oracle": [], "Est-IVWE (Bounded)": [], "Est-IVWE (Gaussian)": []}
    
    for b_idx, B in enumerate(budgets):
        print(f"Testing Budget: {B:,} with costs: {costs}")
        
        results = Parallel(n_jobs=-1)(
            delayed(run_single_simulation)(B, p, delta, K, J, rewards_pool, s_k, true_var, costs, seed + b_idx * num_runs + run_idx)
            for run_idx in range(num_runs)
        )
        
        for alg in raw_history.keys():
            raw_history[alg].append([res[alg] for res in results])
            history[alg].append(np.mean([res[alg] for res in results]))


    p_fname = 'inf' if np.isinf(p) else str(p)
    save_dir = "./fig/real"
    os.makedirs(save_dir, exist_ok=True)
    data_save_path = f"{save_dir}/{data_type}_K{K}_J{J}_runs{num_runs}_l{p_fname}.npz"
    np.savez(data_save_path, budgets=budgets, **raw_history)

    plt.figure(figsize=(8.5, 6))
    colors = {'Uniform': '#bdc3c7', 'Oracle': '#3498db', 
              'Est-IVWE (Bounded)': '#e74c3c', 'Est-IVWE (Gaussian)': '#2ecc71'}
    markers = {'Uniform': 'o', 'Oracle': 's', 
               'Est-IVWE (Bounded)': '^', 'Est-IVWE (Gaussian)': 'd'}
    
    

    for alg in ["Uniform", 'Oracle', 'Est-IVWE (Bounded)', 'Est-IVWE (Gaussian)']:
        low_q = np.percentile(raw_history[alg], 10, axis=1)
        high_q = np.percentile(raw_history[alg], 90, axis=1)
        
        plt.fill_between(budgets, low_q, high_q, color=colors[alg], alpha=0.15, edgecolor='none')
        
        plt.plot(budgets, history[alg], marker=markers[alg], label=alg, color=colors[alg], 
                linewidth=2, linestyle='--' if alg == 'Uniform' else '-')
    
    max_b = np.max(budgets)
    common_exponent = int(np.floor(np.log10(max_b)))
    scale_factor = 10**common_exponent

    plt.xticks(budgets, [f'{b/scale_factor:.1f}' for b in budgets])
    plt.xlim(min(budgets) - max_b * 0.05, max_b * 1.05)
    plt.xlabel(f'Budget (B) [$\\times 10^{{{common_exponent}}}$]', fontsize=12)
    minmax = [[np.min(history[alg]), np.max(history[alg])] for alg in ["Uniform", 'Oracle', 'Est-IVWE (Bounded)', 'Est-IVWE (Gaussian)']]
    y_min, y_max = np.min(minmax), np.max(minmax)

    y_exp = int(np.floor(np.log10(y_max)))
    y_formatter = ticker.FuncFormatter(lambda x, pos: f'{x/10**y_exp:.1f}')

    ax = plt.gca()
    ax.yaxis.set_major_formatter(y_formatter)

    ax.text(0, 1.02, fr'$\times 10^{{{y_exp}}}$', 
            transform=ax.transAxes, 
            fontsize=11,
            va='bottom', ha='left',
            fontfamily='sans-serif')
    
    ax.set_ylim(max(y_min - y_max * 0.1,0), y_max * 1.1)


    p_str = r'\infty' if np.isinf(p) else str(p)
    plt.title(fr'$L_{{{p_str}}}$ Error by Budget ({data_type.capitalize()} Dataset)' + f'\n(K={K}, J={J}, runs={num_runs})', fontsize=14)   
    plt.legend(frameon=True, loc='upper right')
    plt.tight_layout()

    save_path = f"./fig/real/{data_type}_K{K}_J{J}_runs{num_runs}_l{p_fname}.pdf"
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

# --- Execution Block ---
if __name__ == "__main__":
    my_budgets = [int(i*10) for i in range(20, 501, 40)]
    
    graybill = False
    p_values = [1, 2, np.inf]
    
    for data_type in ['complexity', 'correctness', 'helpfulness', 'verbosity']:
        for p_val in p_values:
            main_budget_sweep(
                budgets=my_budgets,
                num_runs=50,
                data_type=data_type,
                p=p_val,
                delta=0.05,
                graybill=graybill,
                seed=0
            )
