import os
import json
import matplotlib.pyplot as plt
import numpy as np

# Load AIPW JSON data
json_path = "./results/aipw.json"
with open(json_path, "r") as f:
    data = json.load(f)

# Build a dictionary for LLM and GAN data, ignoring "data_seed.csv" and "data.csv"
ate_values = {}
for key, value in data.items():
    if key in ["data_seed.csv", "data.csv"]:
        continue
    if key.startswith("./llm_data/"):
        name = key.replace("./llm_data/", "").replace("_", " ").replace(".csv", "").title()
        ate_values["LLM " + name] = value
    elif key.startswith("./gan_data/"):
        name = key.replace("./gan_data/", "").replace("_", " ").replace(".csv", "").title()
        ate_values["GAN " + name] = value

# Add Theoretical ATE
theoretical_label = "Theoretical"
theoretical_value = 0.41825623603939066
ate_values[theoretical_label] = theoretical_value

# Remove theoretical entry, then sort remaining keys in descending order by absolute difference,
# so that baselines with smaller differences (closer to theory) are near the right.
theo_val = ate_values.pop(theoretical_label)
sorted_keys = sorted(ate_values.keys(), key=lambda k: abs(ate_values[k] - theoretical_value), reverse=True)
sorted_keys.append(theoretical_label)

# Prepare lists for plotting
names = sorted_keys
# For non-theoretical keys, get their values in sorted order, then append theoretical value.
values = [ate_values[k] for k in sorted_keys if k != theoretical_label] + [theoretical_value]

# Generate distinct colors
cmap = plt.cm.get_cmap('tab10', len(names))
colors = [cmap(i) for i in range(len(names))]

plt.figure(figsize=(10, 6))
bars = plt.bar(names, values, color=colors, edgecolor='black')
plt.xlabel("Baseline")
plt.ylabel("Estimated ATE")
plt.title("AIPW ATE Estimates by Baseline")
plt.xticks(rotation=45, ha='right')

# Annotate bars and add dashed horizontal lines
for i, bar in enumerate(bars):
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, f"{yval:.4f}",
             ha="center", va="bottom")
    plt.axhline(y=yval, color=colors[i], linestyle='--', linewidth=1, alpha=0.7)

# Save the plot
output_path = "./results/aipw_ate.png"
plt.tight_layout()
plt.savefig(output_path)
plt.close()

print(f"Bar plot saved to {output_path}")
