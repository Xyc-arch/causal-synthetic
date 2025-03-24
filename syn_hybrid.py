import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

def hybrid(syn_full_path, data_seed_path, data_path):
    seed = pd.read_csv(data_seed_path)
    syn = pd.read_csv(data_path + syn_full_path).drop(columns=['A', 'Y'], errors='ignore')
    
    # Train PS model using RF
    X_ps = seed[['W1', 'W2', 'W3', 'W4', 'W5', 'W6']]
    ps_model = RandomForestClassifier(random_state=42)
    ps_model.fit(X_ps, seed['A'])
    
    # Predict propensity scores and assign A
    syn['ps'] = ps_model.predict_proba(syn[['W1', 'W2', 'W3', 'W4', 'W5', 'W6']])[:, 1]
    syn['A'] = syn['ps'].apply(lambda p: np.random.binomial(1, p))
    
    # Train outcome model using RF
    X_outcome = seed[['W1', 'W2', 'W3', 'W4', 'W5', 'W6', 'A']]
    outcome_model = RandomForestClassifier(random_state=42)
    outcome_model.fit(X_outcome, seed['Y'])
    
    # Predict outcome probabilities and assign Y
    X_syn_outcome = syn[['W1', 'W2', 'W3', 'W4', 'W5', 'W6', 'A']]
    syn['y_prob'] = outcome_model.predict_proba(X_syn_outcome)[:, 1]
    syn['Y'] = syn['y_prob'].apply(lambda p: np.random.binomial(1, p))
    
    syn_hybrid = syn[['W1', 'W2', 'W3', 'W4', 'W5', 'W6', 'A', 'Y']]
    syn_hybrid.to_csv(data_path + "syn_hybrid.csv", index=False)
    print("Synthetic hybrid data saved as syn_hybrid.csv")
    return syn_hybrid

if __name__ == '__main__':
    gens = {0: "llm", 1: "gan"}
    gen = gens[0]
    parent_path = "./{}_data/".format(gen)
    hybrid("syn_full.csv", "data_seed.csv", parent_path)
