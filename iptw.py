import os
import json
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Set random seed for reproducibility
np.random.seed(42)

def iptw(file_path, data_path=""):
    """
    Standard IPTW ATE estimator using Random Forest for PS estimation.
    Expects CSV with columns: W1, W2, W3, W4, W5, W6, A, Y.
    PS are clipped to avoid division by zero.
    """
    data = pd.read_csv(data_path + file_path)
    data["Y"] = data["Y"].round()
    data["A"] = data["A"].round()
    covariates = ['W1', 'W2', 'W3', 'W4', 'W5', 'W6']
    
    rf = RandomForestClassifier(random_state=42)
    rf.fit(data[covariates], data['A'])
    # Clip PS to avoid exactly 0 or 1
    data['ps'] = np.clip(rf.predict_proba(data[covariates])[:, 1], 1e-6, 1 - 1e-6)
    
    # Compute IPTW weights
    data['w'] = np.where(data['A'] == 1, 1 / data['ps'], 1 / (1 - data['ps']))
    
    # Compute weighted outcomes for treated and control groups
    treated = data[data['A'] == 1]
    control = data[data['A'] == 0]
    y1 = np.sum(treated['Y'] * treated['w']) / np.sum(treated['w'])
    y0 = np.sum(control['Y'] * control['w']) / np.sum(control['w'])
    
    ate = y1 - y0
    print(f"{data_path + file_path}: Estimated ATE = {ate}")
    return ate

def iptw_truncated(file_path, data_path=""):
    """
    Baseline estimator mitigating positivity issues by truncating IPTW weights.
    Applied only to ('data.csv', "").
    Truncates weights to a maximum of 10.
    """
    data = pd.read_csv(data_path + file_path)
    covariates = ['W1', 'W2', 'W3', 'W4', 'W5', 'W6']
    
    rf = RandomForestClassifier(random_state=42)
    rf.fit(data[covariates], data['A'])
    # data['ps'] = np.clip(rf.predict_proba(data[covariates])[:, 1], 1e-6, 1 - 1e-6)
    data['ps'] = rf.predict_proba(data[covariates])[:, 1]
    
    data['w'] = np.where(data['A'] == 1, 1 / data['ps'], 1 / (1 - data['ps']))
    data['w'] = np.where(data['w'] > 10, 10, data['w'])
    
    treated = data[data['A'] == 1]
    control = data[data['A'] == 0]
    y1 = np.sum(treated['Y'] * treated['w']) / np.sum(treated['w'])
    y0 = np.sum(control['Y'] * control['w']) / np.sum(control['w'])
    
    ate = y1 - y0
    print(f"{data_path + file_path} (truncated): Estimated ATE = {ate}")
    return ate

if __name__ == '__main__':
    gens = {0: "llm", 1: "gan"}
    gen_llm = gens[0]
    gen_gan = gens[1]
    
    parent_path_llm = f"./{gen_llm}_data/"
    parent_path_gan = f"./{gen_gan}_data/"
    
    # Define file list; files without an explicit path are in the current directory.
    file_list = [
        ('data_seed.csv', ""),
        ('data.csv', ""),
        ('syn_full.csv', parent_path_llm),
        ('syn_hybrid.csv', parent_path_llm),
        ('pair.csv', parent_path_llm),
        ('syn_full.csv', parent_path_gan),
        ('syn_hybrid.csv', parent_path_gan),
        ('pair.csv', parent_path_gan)
    ]
    
    results = {}
    # Compute standard IPTW ATE for each file
    for file_name, path in file_list:
        full_key = path + file_name
        results[full_key] = iptw(file_name, data_path=path)
    
    # Add baseline using truncated weights for 'data.csv' as a separate baseline
    truncated_key = "truncated.csv"
    results[truncated_key] = iptw_truncated('data.csv', data_path="")
    
    # Save all baseline results together in one JSON file
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    output_path = os.path.join(results_dir, "iptw.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)
    
    print(f"\nIPTW ATE results (including truncated baseline) saved to {output_path}")
