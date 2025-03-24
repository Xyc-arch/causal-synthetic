import os
import json
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

def evaluate_model(training_file, test_file='data_test.csv'):
    # Load training data
    train = pd.read_csv(training_file)
    X_train = train[['W1', 'W2', 'W3', 'W4', 'W5', 'W6', 'A']]
    y_train = train['Y']
    
    # Train Random Forest model
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train, y_train)
    
    # Load test data
    test = pd.read_csv(test_file)
    X_test = test[['W1', 'W2', 'W3', 'W4', 'W5', 'W6', 'A']]
    y_test = test['Y']
    
    # Predict probabilities and compute AUC
    auc = roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1])
    print(f"AUC on test set (trained on {training_file}): {auc}")
    return auc

if __name__ == '__main__':
    gens = {0: "llm", 1: "gan"}
    gen_gan = gens[1]
    gen_llm = gens[0]
    
    parent_path_gan = f"./{gen_gan}_data/"
    parent_path_llm = f"./{gen_llm}_data/"
    
    # List of training files to evaluate
    training_files = [
        'data_seed.csv',
        parent_path_gan + 'syn_hybrid.csv',
        parent_path_gan + 'syn_full.csv',
        parent_path_llm + 'syn_hybrid.csv',
        parent_path_llm + 'syn_full.csv'
    ]
    
    auc_results = {}
    for file in training_files:
        auc_results[file] = evaluate_model(file)
    
    # Create output directory if it doesn't exist
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Save the AUC results as a JSON file named tstr.json
    output_path = os.path.join(results_dir, "tstr.json")
    with open(output_path, "w") as f:
        json.dump(auc_results, f, indent=4)
    
    print(f"AUC results saved to {output_path}")
