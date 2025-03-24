import os
import json

# Load the simulation results
results_path = os.path.join("results", "simulation_results.json")
with open(results_path, "r") as f:
    results = json.load(f)

# Print results grouped by estimator
for estimator in ["IPTW", "AIPW", "Simple_Substitution"]:
    print(f"=== {estimator} Results ===")
    for baseline, res in results[estimator].items():
        print(f"{baseline}: GT: {res['ground_truth']:.4f} | Bias: {res['bias']:.4f} | Var: {res['variance']:.6f} | MSE: {res['mse']:.6f}")
    print()
