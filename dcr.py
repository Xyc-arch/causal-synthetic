import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def compute_dcr(seed_path, syn_path, test_path):
    """
    Compute the Distance to Closest Record (DCR) for a synthetic dataset (syn_path)
    relative to a seed dataset (seed_path). Both datasets are standardized using the seed stats.
    The synthetic dataset is randomly sub-sampled to match the number of rows in the test dataset.
    """
    # Load data
    seed = pd.read_csv(seed_path)
    syn = pd.read_csv(syn_path)
    test = pd.read_csv(test_path)
    
    w_cols = ['W1', 'W2', 'W3', 'W4', 'W5', 'W6']
    
    # Standardize using seed means and stds
    means = seed[w_cols].mean()
    stds = seed[w_cols].std()
    seed[w_cols] = (seed[w_cols] - means) / stds
    
    # Randomly sample synthetic rows to match test data size
    n_test = test.shape[0]
    if syn.shape[0] > n_test:
        syn = syn.sample(n=n_test, random_state=42)
    # Standardize synthetic and test data using seed stats
    syn[w_cols] = (syn[w_cols] - means) / stds
    test[w_cols] = (test[w_cols] - means) / stds
    
    def min_distance(sample, seed_data):
        # Compute Euclidean distance from one sample to each row in seed_data and return the minimum.
        return np.sqrt(((seed_data - sample[w_cols]) ** 2).sum(axis=1)).min()
    
    # Apply min_distance function to each row of synthetic and test data
    syn_dcr = syn.apply(lambda row: min_distance(row, seed[w_cols]), axis=1)
    test_dcr = test.apply(lambda row: min_distance(row, seed[w_cols]), axis=1)
    
    return syn_dcr.tolist(), test_dcr.tolist()

def main():
    # Define file paths
    seed_path = 'data_seed.csv'
    test_path = 'data_test.csv'
    
    # Parent directories for synthetic data from LLM and GAN
    parent_path_llm = "./llm_data/"
    parent_path_gan = "./gan_data/"
    
    syn_file_llm = "syn_full.csv"
    syn_file_gan = "syn_full.csv"
    
    # Compute DCR for both synthetic datasets relative to seed, with test sample size.
    syn_dcr_llm, test_dcr = compute_dcr(seed_path, parent_path_llm + syn_file_llm, test_path)
    syn_dcr_gan, _ = compute_dcr(seed_path, parent_path_gan + syn_file_gan, test_path)
    
    # Print basic summary statistics
    print("Test Data: Mean DCR = {:.4f}, Std DCR = {:.4f}".format(np.mean(test_dcr), np.std(test_dcr)))
    print("LLM Syn Full: Mean DCR = {:.4f}, Std DCR = {:.4f}".format(np.mean(syn_dcr_llm), np.std(syn_dcr_llm)))
    print("GAN Syn Full: Mean DCR = {:.4f}, Std DCR = {:.4f}".format(np.mean(syn_dcr_gan), np.std(syn_dcr_gan)))
    
    # Save the DCR results to JSON
    results = {
        "test": test_dcr,
        "syn_full_llm": syn_dcr_llm,
        "syn_full_gan": syn_dcr_gan
    }
    
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    output_json = os.path.join(results_dir, "dcr.json")
    with open(output_json, "w") as f:
        json.dump(results, f, indent=4)
    print("DCR results saved to:", output_json)
    
    # (Optional) Create a boxplot comparing test and both synthetic datasets
    plt.boxplot([test_dcr, syn_dcr_llm, syn_dcr_gan],
                labels=['Data Test', 'Syn Full LLM', 'Syn Full GAN'])
    plt.ylabel('Distance to Closest Record')
    plt.title('DCR Boxplot')
    boxplot_path = os.path.join(results_dir, "dcr_boxplot.png")
    plt.savefig(boxplot_path)
    print("Boxplot saved to:", boxplot_path)

if __name__ == '__main__':
    main()
