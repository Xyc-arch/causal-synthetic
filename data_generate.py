import numpy as np
import pandas as pd
from scipy.special import expit

def generate_dataset(n, seed=42, rct=False, truth=False):
    np.random.seed(seed)
    W1 = np.random.binomial(1, 0.5, n)
    W2 = np.random.binomial(1, 0.5, n)
    # W3 depends on W1 and W2: probability increases with W1+W2
    pW3 = 0.3 + 0.35 * ((W1 + W2) / 2)
    W3 = np.random.binomial(1, pW3, n)
    W4 = np.random.normal(0, 1, n)
    W5 = np.random.normal(0, 1, n)
    # W6 depends on W4 and W5
    W6 = 0.5 * W4 + 0.5 * W5 + np.random.normal(0, 1, n)
    
    logits = -30 + 16*W1 - 24*W2 + 12*W3 + 6*W4 - 10*W5 + 16*W6
    pA = expit(logits)
    
    A = np.random.binomial(1, 0.5, n) if rct else np.random.binomial(1, pA, n)
    
    # W3, W5 neg
    tau = (2.0 +
           0.5 * np.sin(W1) +
           0.3 * np.log(np.abs(W2) + 1) -
           0.2 * (W3 ** 2) +
           0.1 * np.exp(W4) -
           0.3 * np.tanh(W5) +
           0.2 * np.cos(W6))
    
    outcome_logits = -0.5 + tau*A + 0.5*W1 + 1.0*W2 - 1.0*W3 + 0.2*W4 - 0.3*W5 + 0.1*W6
    pY = expit(outcome_logits)
    Y = np.random.binomial(1, pY, n)
    
    outcome_logits_treated = -0.5 + tau + 0.5*W1 + 1.0*W2 - 1.0*W3 + 0.2*W4 - 0.3*W5 + 0.1*W6
    outcome_logits_control = -0.5 + 0.5*W1 + 1.0*W2 - 1.0*W3 + 0.2*W4 - 0.3*W5 + 0.1*W6
    pY1 = expit(outcome_logits_treated)
    pY0 = expit(outcome_logits_control)
    ate = np.mean(pY1 - pY0)
    
    data = pd.DataFrame({
        'W1': W1, 'W2': W2, 'W3': W3,
        'W4': W4, 'W5': W5, 'W6': W6,
        'A': A, 'Y': Y, 'pA': pA, 'pY': pY
    })
    
    print("Count of A=1:", (data['A'] == 1).sum())
    print("Count of A=0:", (data['A'] == 0).sum())
    print("Count of Y=1:", (data['Y'] == 1).sum())
    print("Count of Y=0:", (data['Y'] == 0).sum())
    
    if truth:
        return data, ate, np.mean(pY1), np.mean(pY0)
    else:
        return data


if __name__ == '__main__':
    
    data_truth, ate_true, y1_truth, y0_truth = generate_dataset(50000, rct=True, truth=True)
    data_truth.drop(['pA', 'pY'], axis=1).to_csv("data_truth.csv", index=False)
    
    print("Theoretical ATE:", ate_true)
    print("Theoretical E[Y(1)]:", y1_truth)
    print("Theoretical E[Y(0)]:", y0_truth)
    
    data = generate_dataset(200, seed=1)
    # print(data.head())
    data.drop(['pA', 'pY'], axis=1).to_csv("data.csv", index=False)
    
    
    data_seed = generate_dataset(1000, rct=True, seed=2)
    data_seed.drop(['pA', 'pY'], axis=1).to_csv("data_seed.csv", index=False)
    
    
    data_test = generate_dataset(1000, rct=True, seed=3)
    data_test.drop(['pA', 'pY'], axis=1).to_csv("data_test.csv", index=False)
        

