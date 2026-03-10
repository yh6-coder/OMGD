import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import random
import os
import pandas as pd
from scipy.linalg import lstsq
from scipy.stats import norm, qmc
from tqdm import tqdm

# =====================================================
# Hyperparameters
# =====================================================
n = 1000 #10000
d = 10
RUNS = 1
EPOCHS = 10000 #100000
T = n * EPOCHS
r = 0.5 # keep ratio
k = int(r * d)  
M = int(1 / r)  

# =====================================================
# Synthetic least squares problem
# =====================================================
wgen = np.random.rand(d)  # parameter for generating data
w0 = np.zeros(d) # initial point

def gen_xy(q: np.ndarray):
    xi = norm.ppf(q[:d])  # x ~ N(0, I_d)
    yi = norm.ppf(q[-1], loc=xi @ wgen, scale=1.0)  # y ~ N(x'wgen, 1)
    return xi, yi

def gen_syn_data(num_examples: int, dim: int, seed: int):
    np.random.seed(seed)
    qmc_samples = np.random.rand(num_examples, dim)  # IID uniform
    syn_data = []
    for i in range(num_examples):
        q = np.mod(qmc_samples[i], 1.0)  # wrap to [0,1)
        syn_data.append(gen_xy(q))
    return syn_data


# =====================================================
# SGD update
# =====================================================

def sgd(T: int, w0: np.ndarray, wopt: np.ndarray,
        samples: list, ordering: np.ndarray, A: np.matrix, b: np.ndarray,
        masks: np.ndarray = None, method: str = "SGD", proj=False):
    subopt = np.zeros(T)
    decay_term = np.zeros(T)
    data_reshuffle = np.zeros(T)
    compression_error = np.zeros(T)
    wk = w0.copy()
    alpha = 1.0 / (d + 2 + d / np.linalg.norm(wgen) ** 2)

    error = wk - wopt
    error2 = np.zeros(d)
    error3 = np.zeros(d)
    mask = None
    for t in tqdm(range(T), desc=f"SGD ({method})", leave=False):
        xi, yi = samples[ordering[t] - 1]  # Julia 是 1-based
        alpha = 1.0 / (1.0 / alpha + (1 - alpha * (d + 2)) / (1 - alpha))
        # alpha = 1 / (t + 1)
        grad = (xi @ wk - yi) * xi
        grad_m = grad
        tempt = np.identity(d) - alpha * A
        error = tempt @ error
        decay_term[t] = np.linalg.norm(error)**2
        error2 = tempt @ error2 + alpha * (A @ wk - b - grad)
        data_reshuffle[t] = np.linalg.norm(error2)**2
        if masks is not None and t > 10**2:
            if method == "RR_mask_iid":
                mask = generate_masks_iid_oneshot(d, k)
            elif method == "RR_mask_without_replacement":
                mask = masks[t//n]
            else:
                raise ValueError("Invalid method!")
            grad_m = grad * mask * (1.0 / r)  # masked + rescaled
        if proj and t > 10**2:
            Z = np.random.randn(d, 5)
            St = Z @ np.linalg.inv(np.linalg.cholesky(Z.T @ Z).T)
            grad_m = 2*St@(St.T@grad)
        if (masks is not None or proj) and t > 10**2:
            error3 = tempt @ error3 + alpha * (grad-grad_m)
        compression_error[t] = np.linalg.norm(error3)**2
        wk -= alpha * grad_m
        subopt[t] = np.linalg.norm(wk - wopt) ** 2

    return subopt, decay_term, data_reshuffle, compression_error

# =====================================================
# Mask generator
# =====================================================

def generate_masks_iid_oneshot(d:int, k:int):
    idx = np.random.choice(d, size=k, replace=False)
    mask = np.zeros(d)
    mask[idx] = 1
    return mask


def generate_masks_without_replacement_oneshot(T: int, n: int, d: int, k: int, M: int):

    steps_per_block = M * n
    num_blocks = T // steps_per_block
    block_masks = []

    for b in range(num_blocks):
        coords = np.random.permutation(d)
        groups = np.array_split(coords, M)
        for g in groups:
            mask = np.zeros(d)
            mask[g] = 1
            block_masks.append(mask)

    return block_masks

# =====================================================
# Plot
# =====================================================
def compute_A_list(syn_data):
    X = np.array([x_i for x_i, y_i in syn_data])
    A = X.T @ X /len(syn_data)
    return A

def compute_xy_mean_list(syn_data):
    X = np.array([x_i for x_i, y_i in syn_data])
    y = np.array([y_i for x_i, y_i in syn_data])
    result = X.T @ y / len(syn_data)
    return result

def main(SEED):
    np.random.seed(SEED)
    random.seed(SEED)
    print("generating data...")
    samples_offline = gen_syn_data(n, d + 1, SEED)
    A = compute_A_list(samples_offline)
    b = compute_xy_mean_list(samples_offline)
    X = np.vstack([pair[0] for pair in samples_offline])
    y = np.array([pair[1] for pair in samples_offline])
    wopt, _, _, _ = lstsq(X, y)

    results = {}
    errors = {}
    tempts = {}
    decay_term = {}
    data_reshuffle = {}
    compression_error = {}
    for method in ["RR_proj", "RR", "RR_mask_iid", "RR_mask_without_replacement"]:
        print(f"begin training ({method})...")
        subopt_list = np.zeros((RUNS, T))
        grad_error2 = np.zeros((RUNS, T))
        grad_error3 = np.zeros((RUNS, T))
        grad_error4 = np.zeros((RUNS, T))
        for run in range(RUNS):
            sample_indices = np.zeros(T, dtype=int)
            for epoch in range(EPOCHS):
                np.random.seed(SEED + run + epoch)
                perm = np.random.permutation(np.arange(1, n + 1))
                sample_indices[epoch * n:(epoch + 1) * n] = perm
            proj = False
            if method == "RR":
                masks = None
            elif method == "RR_mask_iid":
                masks = generate_masks_iid_oneshot(d, k)
            elif method == "RR_mask_without_replacement":
                masks = generate_masks_without_replacement_oneshot(T, n, d, k, M)
            elif method == "RR_proj":
                masks = None
                proj = True
            else:
                raise ValueError("Unknown method")

            subopt_list[run, :], grad_error2[run, :], grad_error3[run, :], grad_error4[run, :]= sgd(T, w0, wopt, samples_offline, sample_indices, A=A, b=b,
                                         masks=masks, method=method, proj=proj)

        results[method] = subopt_list
        decay_term[method] = grad_error2
        data_reshuffle[method] = grad_error3
        compression_error[method] = grad_error4

    print("training completed")

    # save results
    os.makedirs("./results/total_error", exist_ok=True)
    os.makedirs("./results/decay_term", exist_ok=True)
    os.makedirs("./results/data_reshuffle", exist_ok=True)
    os.makedirs("./results/compression_error", exist_ok=True)
    try:
        for method in results.keys():
            np.save(f"./results/total_error/{method}_SEED_{SEED}.npy", results[method])
        for method in decay_term.keys():
            np.save(f"./results/decay_term/{method}_SEED_{SEED}.npy", decay_term[method])
        for method in data_reshuffle.keys():
            np.save(f"./results/data_reshuffle/{method}_SEED_{SEED}.npy", data_reshuffle[method])
        for method in compression_error.keys():
            np.save(f"./results/compression_error/{method}_SEED_{SEED}.npy", compression_error[method])
        print("Results saved to ./results/")
    except Exception as e:
        print(f"Error saving results: {e}")

def plot_main(SEED, type):

    methods = ["RR_proj", "RR_mask_iid", "RR_mask_without_replacement", "RR"]
    labels = {
        "RR_proj": r"\textbf{RR\_proj}",
        "RR_mask_iid": r"\textbf{RR\_mask\_iid}",
        "RR_mask_without_replacement": r"\textbf{RR\_mask\_wor}",
        "RR": r"\textbf{RR}"
    }
    results = {}

    for method in methods:
        filename = f"./results/{type}/{method}_SEED_{SEED}.npy"
        if os.path.exists(filename):
            results[method] = np.load(filename)
            # print(f"Loaded {method}: {results[method].shape}")
        else:
            print(f"File not found: {filename}")

    plt.figure(figsize=(4, 3.8))
    colors = {"RR": "blue", "RR_mask_iid": "orange",
              "RR_mask_without_replacement": "green",
              "RR_proj": "red"}

    T = results[methods[0]].shape[1]  
    iters = np.arange(1000, T + 1, 100)

    for method in methods:
        if method in results:
            subopt_list = results[method]
            subopt_mean = np.mean(subopt_list, axis=0)
            ws = 2000
            subopt_mean = pd.Series(subopt_mean).rolling(ws, min_periods=1).mean().values
            plt.plot(iters, subopt_mean[999::100], label=labels[method], color=colors[method], linewidth=1.25)

    t_values = iters.astype(float)
    plt.plot(iters, 10*1 / t_values, label=r'$1/t$', color='purple', linestyle='--', linewidth=2)
    if type == 'decay_term':
        plt.plot(iters, 100*1 / (t_values ** 2), label=r'$1/t^2$', color='deeppink', linestyle=':', linewidth=2)
    else:
        plt.plot(iters, 5000 * 1 / (t_values ** 2), label=r'$1/t^2$', color='deeppink', linestyle=':', linewidth=2)
    
    plt.xlabel(r"\textbf{Iteration} $t$", fontsize=20)
    if type == 'total_error': plt.legend(loc="lower left", fontsize=12)
    if type == 'total_error':
        plt.ylabel(r"$\Vert \theta_t - \theta^*\Vert^2$", fontsize=14)
    elif type == 'decay_term':
        plt.ylabel(r"$L^2$ norm of decay term", fontsize=14)
    elif type == 'data_reshuffle':
        plt.ylabel(r"$L^2$ norm of data_reshuffle term", fontsize=14)
    elif type == 'compression_error':
        plt.ylabel(r"$L^2$ norm of compression_error term", fontsize=14)
    plt.tick_params(axis='y', which='major', colors='black', labelsize=10.9)
    plt.tick_params(axis='x', which='major', colors='black', labelsize=18)
    plt.tick_params(which='minor')

    ax = plt.gca()

    plt.yscale('log')
    plt.xscale('log')
    plt.grid(which="major", color="0.9", linestyle='-', )
    plt.grid(which="minor", color="0.9", linestyle='-',)

    plt.tight_layout()
    plot_fn = os.path.join("figs", f"{type}_SEED_{SEED}.pdf")
    out_dir = os.path.dirname(plot_fn)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    plt.savefig(plot_fn)
    print(f"Plot saved at {plot_fn}")

    # plt.show()

if __name__ == '__main__':
    for SEED in {666}:
        print(SEED)
        matplotlib.rcParams['text.usetex'] = True
        matplotlib.rcParams['font.family'] = 'serif'
        # main(SEED)
        for type in ['total_error', 'decay_term', 'data_reshuffle', 'compression_error']:
            plot_main(SEED, type)
