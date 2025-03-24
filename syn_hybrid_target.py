import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import statsmodels.api as sm
from scipy.special import expit, logit

def hybrid(syn_full_path, data_seed_path, data_path):
    # helper function to clip probabilities away from 0 and 1
    def clip(p, low=1e-6, high=1-1e-6):
        return np.clip(p, low, high)
    
    # Read the seed and synthetic data
    seed = pd.read_csv(data_seed_path)
    syn = pd.read_csv(data_path + syn_full_path).drop(columns=['A', 'Y'], errors='ignore')
    
    # --- Propensity Score (PS) Step ---
    X_ps = seed[['W1', 'W2', 'W3', 'W4', 'W5', 'W6']]
    ps_model = RandomForestClassifier(random_state=42)
    ps_model.fit(X_ps, seed['A'])
    
    # Predict PS on synthetic data, clip to avoid 0/1, then assign treatment
    syn['ps'] = clip(ps_model.predict_proba(syn[['W1', 'W2', 'W3', 'W4', 'W5', 'W6']])[:, 1])
    syn['A'] = syn['ps'].apply(lambda p: np.random.binomial(1, p))
    
    # --- Outcome Regression Initial Fit ---
    X_outcome = seed[['W1', 'W2', 'W3', 'W4', 'W5', 'W6', 'A']]
    outcome_model = RandomForestClassifier(random_state=42)
    outcome_model.fit(X_outcome, seed['Y'])
    
    # Get the initial predictions Q(W,A) on seed data and clip to avoid 0/1 issues
    Q_seed = clip(outcome_model.predict_proba(seed[['W1', 'W2', 'W3', 'W4', 'W5', 'W6', 'A']])[:, 1])
    
    # --- Targeting Step (Logistic Link Update) on Seed ---
    # Get PS for seed and clip it as well
    ps_seed = clip(ps_model.predict_proba(seed[['W1', 'W2', 'W3', 'W4', 'W5', 'W6']])[:, 1])
    
    # Define the clever covariate H: for A=1, H=1/ps; for A=0, H=-1/(1-ps)
    H_seed = np.where(seed['A'] == 1, 1/ps_seed, -1/(1-ps_seed))
    
    # Use the logistic fluctuation model:
    # logit(Q*(W,A)) = logit(Q(W,A)) + epsilon * H
    # Offset is logit(Q_seed)
    offset = logit(Q_seed)
    
    # Fit logistic regression with no intercept; H_seed is the only covariate
    model = sm.GLM(seed['Y'], H_seed, family=sm.families.Binomial(), offset=offset)
    result = model.fit()
    epsilon = result.params[0]
    
    # Update the initial predictions using the logistic fluctuation update
    Q_star_seed = expit(logit(Q_seed) + epsilon * H_seed)
    
    # --- Apply the Targeting Update to Synthetic Data ---
    # Get initial predictions Q(W,A) on synthetic data and clip
    Q_syn = clip(outcome_model.predict_proba(syn[['W1', 'W2', 'W3', 'W4', 'W5', 'W6', 'A']])[:, 1])
    # Compute the clever covariate for synthetic data using the synthetic PS (which is clipped)
    ps_syn = clip(syn['ps'])
    H_syn = np.where(syn['A'] == 1, 1/ps_syn, -1/(1-ps_syn))
    
    # Update predictions using the same fluctuation parameter epsilon
    Q_star_syn = expit(logit(Q_syn) + epsilon * H_syn)
    
    # Simulate outcome Y for synthetic data using the targeted predictions
    syn['y_prob_target'] = Q_star_syn
    syn['Y'] = syn['y_prob_target'].apply(lambda p: np.random.binomial(1, p))

    
    # Save the final synthetic data with targeted outcome regression
    syn_hybrid = syn[['W1', 'W2', 'W3', 'W4', 'W5', 'W6', 'A', 'Y']]
    syn_hybrid.to_csv(data_path + "syn_hybrid_target.csv", index=False)
    return syn_hybrid

if __name__ == '__main__':
    gens = {0: "llm", 1: "gan"}
    gen = gens[0]
    parent_path = "./{}_data/".format(gen)
    hybrid("syn_full.csv", "data_seed.csv", parent_path)
