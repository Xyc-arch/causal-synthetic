import os
import json
import matplotlib.pyplot as plt
import numpy as np

# Load IPTW JSON data
json_path = "./results/iptw.json"
with open(json_path, "r") as f:
    data = json.load(f)

# Rename keys for LLM and GAN data
ate_values = {}
for key, value in data.items():
    if key.startswith("./llm_data/"):
        name = key.replace("./llm_data/", "").replace("_", " ").replace(".csv", "").title()
        ate_values["LLM " + name] = value
    elif key.startswith("./gan_data/"):
        name = key.replace("./gan_data/", "").replace("_", " ").replace(".csv", "").title()
        ate_values["GAN " + name] = value

# Add theoretical ATE
theoretical_label = "Theoretical"
theoretical_value = 0.41825623603939066
ate_values[theoretical_label] = theoretical_value

# Remove theoretical entry and sort others in descending order by absolute difference
# so that those with smaller differences (better) come closer to theoretical on the right.
theo_val = ate_values.pop(theoretical_label)
sorted_keys = sorted(ate_values.keys(), key=lambda k: abs(ate_values[k] - theoretical_value), reverse=True)
sorted_keys.append(theoretical_label)

# Prepare ordered lists for plotting
names = sorted_keys
values = [ate_values[k] for k in sorted_keys if k != theoretical_label] + [theoretical_value]

# Generate distinct colors for each bar
cmap = plt.cm.get_cmap('tab10', len(names))
colors = [cmap(i) for i in range(len(names))]

plt.figure(figsize=(10, 6))
bars = plt.bar(names, values, color=colors, edgecolor='black')
plt.xlabel("Baseline")
plt.ylabel("Estimated ATE")
plt.title("IPTW ATE Estimates by Baseline")
plt.xticks(rotation=45, ha='right')

# Annotate bars and draw dashed horizontal lines at each bar's height
for i, bar in enumerate(bars):
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, f"{yval:.4f}", ha="center", va="bottom")
    plt.axhline(y=yval, color=colors[i], linestyle='--', linewidth=1, alpha=0.7)

# Save the figure
output_path = "./results/iptw_ate.png"
plt.tight_layout()
plt.savefig(output_path)
plt.close()

print(f"Bar plot saved to {output_path}")
