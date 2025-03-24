import os
import json
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

def aipw(file_path, data_path=""):
    """
    AIPW ATE estimator using Random Forest models for both the propensity score and outcome.
    Expects CSV with columns: W1, W2, W3, W4, W5, W6, A, Y.
    """
    # Read data from the combined path
    data = pd.read_csv(data_path + file_path)
    # Ensure Y and A are appropriately rounded (if needed)
    data["Y"] = data["Y"].round()
    data["A"] = data["A"].round()
    covariates = ['W1', 'W2', 'W3', 'W4', 'W5', 'W6']
    
    # Estimate propensity scores using Random Forest
    rf_ps = RandomForestClassifier(random_state=42)
    rf_ps.fit(data[covariates], data['A'])
    data['ps'] = rf_ps.predict_proba(data[covariates])[:, 1]
    
    # Fit outcome model using Random Forest with covariates and treatment A
    rf_out = RandomForestClassifier(random_state=42)
    X_outcome = data[covariates + ['A']]
    rf_out.fit(X_outcome, data['Y'])
    
    # Predict potential outcomes under treatment (A=1) and control (A=0)
    X1 = data[covariates].copy()
    X1['A'] = 1
    m1 = rf_out.predict_proba(X1)[:, 1]
    
    X0 = data[covariates].copy()
    X0['A'] = 0
    m0 = rf_out.predict_proba(X0)[:, 1]
    
    A = data['A']
    Y = data['Y']
    p = data['ps']
    
    # Compute the AIPW estimator
    aipw_est = np.mean(m1 - m0 + A * (Y - m1) / p - (1 - A) * (Y - m0) / (1 - p))
    
    print(f"{data_path + file_path}:")
    print("  Estimated ATE (AIPW):", aipw_est)
    return aipw_est

if __name__ == '__main__':
    gens = {0: "llm", 1: "gan"}
    gen_llm = gens[0]
    gen_gan = gens[1]
    
    parent_path_llm = f"./{gen_llm}_data/"
    parent_path_gan = f"./{gen_gan}_data/"
    
    # Baseline file list using the updated structure:
    file_list = [
        ('data_seed.csv', ""),                   # file in current directory
        ('data.csv', ""),                        # file in current directory
        ('syn_full.csv', parent_path_llm),         # LLM files
        ('syn_hybrid.csv', parent_path_llm),
        ('pair.csv', parent_path_llm),
        ('syn_full.csv', parent_path_gan),         # GAN files
        ('syn_hybrid.csv', parent_path_gan),
        ('pair.csv', parent_path_gan)
    ]
    
    aipw_results = {}
    for file_name, path in file_list:
        full_path = path + file_name
        aipw_results[full_path] = aipw(file_name, data_path=path)
    
    # Create output directory if it doesn't exist
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    
    output_path = os.path.join(results_dir, "aipw.json")
    with open(output_path, "w") as f:
        json.dump(aipw_results, f, indent=4)
    
    print(f"\nAIPW ATE results saved to {output_path}")
