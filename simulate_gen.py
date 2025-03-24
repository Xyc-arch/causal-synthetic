import os
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Clip propensity scores to avoid 0 or 1
def clip_scores(ps, lower=1e-6, upper=1-1e-6):
    return np.clip(ps, lower, upper)

# IPTW estimator using RF for PS
def compute_iptw(df):
    df["Y"] = df["Y"].round()
    df["A"] = df["A"].round()
    cov = ['W1', 'W2', 'W3', 'W4', 'W5', 'W6']
    rf = RandomForestClassifier(random_state=42)
    rf.fit(df[cov], df['A'])
    ps = clip_scores(rf.predict_proba(df[cov])[:, 1])
    w = np.where(df['A'] == 1, 1 / ps, 1 / (1 - ps))
    treated = df[df['A'] == 1]
    control = df[df['A'] == 0]
    y1 = np.sum(treated['Y'] * w[df['A'] == 1]) / np.sum(w[df['A'] == 1])
    y0 = np.sum(control['Y'] * w[df['A'] == 0]) / np.sum(w[df['A'] == 0])
    return y1 - y0

# AIPW estimator using RF for PS and outcome
def compute_aipw(df):
    df["Y"] = df["Y"].round()
    df["A"] = df["A"].round()
    cov = ['W1', 'W2', 'W3', 'W4', 'W5', 'W6']
    rf_ps = RandomForestClassifier(random_state=42)
    rf_ps.fit(df[cov], df['A'])
    ps = clip_scores(rf_ps.predict_proba(df[cov])[:, 1])
    rf_out = RandomForestClassifier(random_state=42)
    X_out = df[cov + ['A']]
    rf_out.fit(X_out, df['Y'])
    X1 = df[cov].copy(); X1['A'] = 1
    X0 = df[cov].copy(); X0['A'] = 0
    m1 = rf_out.predict_proba(X1)[:, 1]
    m0 = rf_out.predict_proba(X0)[:, 1]
    return np.mean(m1 - m0 + df['A'] * (df['Y'] - m1) / ps - (1 - df['A']) * (df['Y'] - m0) / (1 - ps))

# Simple substitution estimator using RF for outcome model
def compute_simple_substitution(df):
    cov = ['W1', 'W2', 'W3', 'W4', 'W5', 'W6']
    rf_out = RandomForestClassifier(random_state=42)
    X_out = df[cov + ['A']]
    rf_out.fit(X_out, df['Y'])
    X1 = df[cov].copy(); X1['A'] = 1
    X0 = df[cov].copy(); X0['A'] = 0
    m1 = rf_out.predict_proba(X1)[:, 1]
    m0 = rf_out.predict_proba(X0)[:, 1]
    return np.mean(m1 - m0)

# Simulation function: subsample, compute estimator, then bias/var/MSE vs. full-data truth
def simulate_estimator(baseline_file, estimator_func, sample_size=1000, iterations=100, data_path=""):
    print(f"Starting simulation for baseline: {data_path + baseline_file}")
    full_df = pd.read_csv(data_path + baseline_file)
    truth = estimator_func(full_df)
    estimates = []
    for i in range(iterations):
        if i % 10 == 0:
            print(f"  Iteration: {i}")
        subsample = full_df.sample(n=sample_size, random_state=i)
        est = estimator_func(subsample)
        estimates.append(est)
    print(f"Finished simulation for baseline: {data_path + baseline_file}\n")
    estimates = np.array(estimates)
    result = {
        "ground_truth": truth,
        "bias": np.mean(estimates - truth),
        "variance": np.var(estimates),
        "mse": np.mean((estimates - truth) ** 2),
        "estimates": estimates.tolist()
    }
    return result

if __name__ == '__main__':
    # Paths for baselines (50k samples each)
    parent_path_llm = "./llm_data/"
    parent_path_gan = "./gan_data/"
    truth_file = "data_truth.csv"
    llm_file = "syn_hybrid.csv"
    gan_file = "syn_hybrid.csv"
    
    sample_size = 1000
    iterations = 100
    
    # Run simulations for each estimator
    iptw_results = {
        "LLM": simulate_estimator(llm_file, compute_iptw, sample_size, iterations, data_path=parent_path_llm),
        "GAN": simulate_estimator(gan_file, compute_iptw, sample_size, iterations, data_path=parent_path_gan),
        "Truth": simulate_estimator(truth_file, compute_iptw, sample_size, iterations, data_path="")
    }
    
    aipw_results = {
        "LLM": simulate_estimator(llm_file, compute_aipw, sample_size, iterations, data_path=parent_path_llm),
        "GAN": simulate_estimator(gan_file, compute_aipw, sample_size, iterations, data_path=parent_path_gan),
        "Truth": simulate_estimator(truth_file, compute_aipw, sample_size, iterations, data_path="")
    }
    
    ss_results = {
        "LLM": simulate_estimator(llm_file, compute_simple_substitution, sample_size, iterations, data_path=parent_path_llm),
        "GAN": simulate_estimator(gan_file, compute_simple_substitution, sample_size, iterations, data_path=parent_path_gan),
        "Truth": simulate_estimator(truth_file, compute_simple_substitution, sample_size, iterations, data_path="")
    }
    
    # Save results to JSON
    results = {
        "IPTW": iptw_results,
        "AIPW": aipw_results,
        "Simple_Substitution": ss_results
    }
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    output_path = os.path.join(results_dir, "simulation_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)
    
    # Print summary: group IPTW and AIPW together, then SS
    print("=== IPTW Results ===")
    for baseline, res in iptw_results.items():
        print(f"{baseline}: GT: {res['ground_truth']:.4f} | Bias: {res['bias']:.4f} | Var: {res['variance']:.6f} | MSE: {res['mse']:.6f}")
    print("\n=== AIPW Results ===")
    for baseline, res in aipw_results.items():
        print(f"{baseline}: GT: {res['ground_truth']:.4f} | Bias: {res['bias']:.4f} | Var: {res['variance']:.6f} | MSE: {res['mse']:.6f}")
    print("\n=== Simple Substitution Results ===")
    for baseline, res in ss_results.items():
        print(f"{baseline}: GT: {res['ground_truth']:.4f} | Bias: {res['bias']:.4f} | Var: {res['variance']:.6f} | MSE: {res['mse']:.6f}")
    
    print(f"\nSimulation results saved to {output_path}")
