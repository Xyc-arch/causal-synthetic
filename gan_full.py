import pandas as pd
from ctgan import CTGAN

# Load seed data with columns: W1, W2, W3, W4, W5, W6, A, Y
data = pd.read_csv('data_seed.csv')

columns = ['W1', 'W2', 'W3', 'W4', 'W5', 'W6', 'A', 'Y']

# Initialize and train CTGAN on the full dataset
epochs = 50
ctgan = CTGAN(epochs=epochs)
ctgan.fit(data, columns)

# Generate synthetic data (with all columns) and save as "syn_cova.csv"
sample_size = 50000
synthetic_data = ctgan.sample(sample_size)
synthetic_data.to_csv("./gan_data/syn_full.csv", index=False)

print("finsih")
