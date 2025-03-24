import os
import json
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

def iptw(file_path, data_path=""):
    """
    IPTW ATE estimator using random forest for propensity score estimation.
    Expects CSV with columns: W1, W2, W3, W4, W5, W6, A, Y.
    """
    data = pd.read_csv(data_path + file_path)
    # Ensure that Y and A are binary (or rounded) values
    data["Y"] = data["Y"].round()
    data["A"] = data["A"].round()
    covariates = ['W1', 'W2', 'W3', 'W4', 'W5', 'W6']
    
    # Estimate propensity scores using Random Forest
    rf = RandomForestClassifier(random_state=42)
    rf.fit(data[covariates], data['A'])
    data['ps'] = rf.predict_proba(data[covariates])[:, 1]
    
    # Compute IPTW weights (if ps is 0 or 1 this could be unstable)
    data['w'] = np.where(data['A'] == 1, 1 / data['ps'], 1 / (1 - data['ps']))
    
    # Compute weighted outcomes for treated (y1) and control (y0)
    treated = data[data['A'] == 1]
    control = data[data['A'] == 0]
    
    y1 = np.sum(treated['Y'] * treated['w']) / np.sum(treated['w'])
    y0 = np.sum(control['Y'] * control['w']) / np.sum(control['w'])
    
    ate = y1 - y0
    print(f"{data_path + file_path}:")
    print("  Weighted Y for treated:", y1)
    print("  Weighted Y for control:", y0)
    print("  Estimated ATE:", ate)
    return ate

if __name__ == '__main__':
    gens = {0: "llm", 1: "gan"}
    # Choose paths for each generator
    gen_llm = gens[0]
    gen_gan = gens[1]
    
    parent_path_llm = f"./{gen_llm}_data/"
    parent_path_gan = f"./{gen_gan}_data/"
    
    # Define the list of training files with appropriate paths.
    # Files without an explicit path will be assumed to be in the current directory.
    file_list = [
        ('data_seed.csv', ""),                   # file in current directory
        ('data.csv', ""),                        # file in current directory
        ('syn_full.csv', parent_path_llm),         # LLM files
        ('syn_hybrid.csv', parent_path_llm),
        ('pair.csv', parent_path_llm),
        ('syn_full.csv', parent_path_gan),         # LLM files
        ('syn_hybrid.csv', parent_path_gan),
        ('pair.csv', parent_path_gan)
    ]
    
    # Evaluate IPTW ATE for each file and store the results in a dictionary
    results = {}
    for file_name, path in file_list:
        results[path + file_name] = iptw(file_name, data_path=path)
    
    # Create output directory if it doesn't exist
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    
    output_path = os.path.join(results_dir, "iptw.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)
    
    print(f"\nIPTW ATE results saved to {output_path}")
