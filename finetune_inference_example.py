import time
import os
from be_great import GReaT
from sklearn.datasets import fetch_california_housing
import pandas as pd
from utils import *


data_names = {0: "craft", 1: "gender", 2: "diabetes", 3: "adult"}
data_name = data_names[0]

for idx_data in [2, 3]:
    
    data_name = data_names[idx_data]

    info = load_data_info('data_info.json')
    data_info = info.get(data_name, {})
    batch_size = data_info.get('batch_size')
    epochs = data_info.get('epochs')
    syn_samples = data_info.get('syn_samples')


    # Define paths for data, model save directory, and synthetic data
    seed_data_path = f"../data/{data_name}/use.csv"
    model_save_path = f"../model/{data_name}"
    synthetic_data_save_path = f"../data/{data_name}/synthetic_allseed.csv"
    os.makedirs(os.path.dirname(synthetic_data_save_path), exist_ok=True)

    # Timer to measure execution time
    start_time = time.time()

    data = pd.read_csv(seed_data_path)

    print(data)

    # Step 2: Initialize and train the model
    print("Training the model...")
    model = GReaT(llm='gpt2', batch_size=batch_size, epochs=epochs, save_steps=400000)
    model.fit(data)

    # Step 3: Save the trained model
    model.save(path=model_save_path)
    print(f"Model saved to {model_save_path}")

    # Step 4: Reinitialize a new model instance and load from the saved model
    print("Reinitializing the model and loading from saved model...")
    new_model = GReaT.load_from_dir(path=model_save_path)
    print(f"Model loaded from {model_save_path}")

    # Step 5: Sample synthetic data from the loaded model
    synthetic_data = new_model.sample(n_samples=syn_samples, max_length=2000)

    # Step 6: Save the synthetic data to a CSV file
    synthetic_data.to_csv(synthetic_data_save_path, index=False)
    print(f"Synthetic data saved to {synthetic_data_save_path}")

    # Measure and print the elapsed time
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time taken {data_name}: {elapsed_time} seconds")
