import os

import matplotlib.pyplot as plt
import pandas as pd


CSV_PATH = "results/training_log.csv"
OUTPUT_PATH = "results/training_curve.png"


# Load training data
df = pd.read_csv(CSV_PATH)

# Plot
plt.figure(figsize=(10, 6))

# Raw episode returns
plt.plot(
    df["global_step"],
    df["episode_reward"],
    alpha=0.25,
    label="Episode Return",
)

# Smoothed episode returns
plt.plot(
    df["global_step"],
    df["smooth_reward"],
    linewidth=2,
    label="Smoothed Return",
)

plt.xlabel("Training Steps")
plt.ylabel("Episode Return")
plt.title("G1 PPO Training Curve")

plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()

# Save
os.makedirs("results", exist_ok=True)

plt.savefig(
    OUTPUT_PATH,
    dpi=200,
)

print(
    f"Plot saved to: {OUTPUT_PATH}"
)

plt.show()