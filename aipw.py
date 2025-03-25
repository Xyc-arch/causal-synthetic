import os
import json
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

np.random.seed(42)

def aipw(file_path, data_path=""):
    """
    Standard AIPW ATE estimator using Random Forest for both propensity score and outcome estimation.
    Expects CSV with columns: W1, W2, W3, W4, W5, W6, A, Y.
    Propensity scores are clipped to [1e-6, 1-1e-6] to avoid division by zero.
    """
    # Read and preprocess data
    data = pd.read_csv(data_path + file_path)
    data["Y"] = data["Y"].round()
    data["A"] = data["A"].round()
    covariates = ['W1', 'W2', 'W3', 'W4', 'W5', 'W6']
    
    # Estimate propensity scores with clipping
    rf_ps = RandomForestClassifier(random_state=42)
    rf_ps.fit(data[covariates], data['A'])
    data['ps'] = np.clip(rf_ps.predict_proba(data[covariates])[:, 1], 1e-6, 1 - 1e-6)
    
    # Fit outcome model using Random Forest
    rf_out = RandomForestClassifier(random_state=42)
    X_outcome = data[covariates + ['A']]
    rf_out.fit(X_outcome, data['Y'])
    
    # Predict potential outcomes for treatment and control
    X1 = data[covariates].copy()
    X1['A'] = 1
    m1 = rf_out.predict_proba(X1)[:, 1]
    
    X0 = data[covariates].copy()
    X0['A'] = 0
    m0 = rf_out.predict_proba(X0)[:, 1]
    
    A = data['A']
    Y = data['Y']
    p = data['ps']
    
    # Compute AIPW estimator (no additional truncation on augmentation terms)
    aipw_est = np.mean(m1 - m0 + A * (Y - m1) / p - (1 - A) * (Y - m0) / (1 - p))
    print(f"{data_path + file_path}: Estimated ATE (AIPW) = {aipw_est}")
    return aipw_est

def aipw_truncated(file_path, data_path=""):
    """
    AIPW ATE estimator baseline for 'data.csv' with augmentation terms truncated.
    Same as aipw() except that if A*(Y-m1)/p or (1-A)*(Y-m0)/(1-p) exceed 10, they are set to 10.
    Propensity scores are clipped to [1e-6, 1-1e-6].
    """
    # Read and preprocess data
    data = pd.read_csv(data_path + file_path)
    data["Y"] = data["Y"].round()
    data["A"] = data["A"].round()
    covariates = ['W1', 'W2', 'W3', 'W4', 'W5', 'W6']
    
    # Estimate propensity scores with clipping
    rf_ps = RandomForestClassifier(random_state=42)
    rf_ps.fit(data[covariates], data['A'])
    data['ps'] = np.clip(rf_ps.predict_proba(data[covariates])[:, 1], 1e-6, 1 - 1e-6)
    
    # Fit outcome model using Random Forest
    rf_out = RandomForestClassifier(random_state=42)
    X_outcome = data[covariates + ['A']]
    rf_out.fit(X_outcome, data['Y'])
    
    # Predict potential outcomes
    X1 = data[covariates].copy()
    X1['A'] = 1
    m1 = rf_out.predict_proba(X1)[:, 1]
    
    X0 = data[covariates].copy()
    X0['A'] = 0
    m0 = rf_out.predict_proba(X0)[:, 1]
    
    A = data['A']
    Y = data['Y']
    p = data['ps']
    
    # Compute augmentation terms
    term_treated = A * (Y - m1) / p
    term_control = (1 - A) * (Y - m0) / (1 - p)
    
    # Truncate augmentation terms at 10 if they exceed 10
    term_treated = np.where(term_treated > 10, 10, term_treated)
    term_control = np.where(term_control > 10, 10, term_control)
    
    # Compute AIPW estimator with truncated augmentation terms
    aipw_est = np.mean(m1 - m0 + term_treated - term_control)
    print(f"{data_path + file_path} (truncated): Estimated ATE = {aipw_est}")
    return aipw_est

if __name__ == '__main__':
    gens = {0: "llm", 1: "gan"}
    gen_llm = gens[0]
    gen_gan = gens[1]
    
    parent_path_llm = f"./{gen_llm}_data/"
    parent_path_gan = f"./{gen_gan}_data/"
    
    # Baseline file list; files without an explicit path are in the current directory.
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
    # Compute AIPW ATE using the standard estimator for each file
    for file_name, path in file_list:
        full_key = path + file_name
        results[full_key] = aipw(file_name, data_path=path)
    
    # Add a baseline for 'data.csv' with truncated augmentation terms
    truncated_key = "data.csv (truncated)"
    results[truncated_key] = aipw_truncated('data.csv', data_path="")
    
    # Save all results to a JSON file
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    output_path = os.path.join(results_dir, "aipw.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)
    
    print(f"\nAIPW ATE results (including baseline with truncation) saved to {output_path}")
