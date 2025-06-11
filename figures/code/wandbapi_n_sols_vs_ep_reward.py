import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
import wandb
import matplotlib as mpl
import numpy as np

# --- Matplotlib and LaTeX Configuration ---
# Sets up the plot to use LaTeX for rendering text, ensuring consistency with academic documents.
mpl.use("pgf")
plt.rcParams.update({
    "pgf.texsystem": "pdflatex",
    "font.family": "serif",
    "text.usetex": True,
    "pgf.rcfonts": False,
    "font.size": 10, # Standard font size for documents
})

# --- Configuration ---
# Define the size of the rolling window for smoothing the plot lines.
window_size = 100
# Define the target group to analyze.
target_group = "unknown_c_unknown_value_at_fathomed_naive_new"

# --- Main Script ---
# Initialize the wandb API to fetch run data.
api = wandb.Api()

# Define the output directory for saving the plot.
output_dir = "figures/plots"
os.makedirs(output_dir, exist_ok=True)

# --- Plotting Setup ---
# Create a 1x2 subplot grid for side-by-side comparison.
# Adjust figsize for a wide, compact layout suitable for an A4 page.
fig, axes = plt.subplots(1, 2, figsize=(6, 2.5))

try:
    # --- Data Fetching ---
    # Fetch runs from the specified group.
    print(f"Fetching runs for group: '{target_group}'...")
    runs = api.runs(
        path="eliasaar_org/OptDMaking",
        filters={"group": target_group}
    )

    if not runs:
        raise ValueError(f"No runs found for group '{target_group}'.")

    # Select the first run as a representative for the group.
    run = runs[0]
    print(f"Processing run '{run.name}'...")

    # --- Fetch each metric separately to handle asynchronous logging ---
    # Fetch n_sols history
    history_sols = run.scan_history(keys=["n_sols", "_step"],page_size = 10000, max_step=5e4)
    df_sols = pd.DataFrame(history_sols)

    # Fetch ep_reward history
    history_reward = run.scan_history(keys=["ep_reward", "_step"],page_size = 10000, max_step=5e4)
    df_reward = pd.DataFrame(history_reward)


    # --- Plot 1: Number of Solutions (n_sols) ---
    ax1 = axes[0]
    df_sols.dropna(inplace=True) # Ensure no missing values
    if not df_sols.empty:
        # Apply the rolling window to smooth the data
        smoothed_n_sols = df_sols["n_sols"].rolling(window=window_size, min_periods=1).mean()
        ax1.plot(df_sols["_step"].iloc[101:], smoothed_n_sols.iloc[101:], color='C5')
        ax1.set_ylabel(f'Solutions')
        ax1.set_title('Solutions') # Use LaTeX for n_sols
    else:
        ax1.text(0.5, 0.5, 'No "n_sols" data', ha='center', va='center')
        ax1.set_title('$n\\_sols$')

    # --- Plot 2: Episodic Reward ---
    ax2 = axes[1]
    df_reward.dropna(inplace=True) # Ensure no missing values
    if not df_reward.empty:
        # Apply the rolling window to smooth the data
        smoothed_ep_reward = df_reward["ep_reward"].rolling(window=window_size, min_periods=1).mean()
        # Normalize the smoothed data between 0 and 1
        min_reward = smoothed_ep_reward.min()
        max_reward = smoothed_ep_reward.max()

            # Avoid division by zero if all values are the same
        if max_reward == min_reward:
            normalized_ep_reward = pd.Series([0.5] * len(smoothed_ep_reward)) # Assign a mid-point or 0 if constant
        else:
            normalized_ep_reward = (smoothed_ep_reward - min_reward) / (max_reward - min_reward)
        ax2.plot(df_reward["_step"].iloc[101:], normalized_ep_reward.iloc[101:], color='C6')
        ax2.set_ylabel(f'Episodic Cost')
        ax2.set_title('Episodic Cost')
    else:
        ax2.text(0.5, 0.5, 'No "ep_reward" data', ha='center', va='center')
        ax2.set_title('Episodic Reward')

    # --- Final Plot Configuration for Both Subplots ---
    # fig.suptitle(f'Run Analysis: {run.name}', y=1.0) # Add a main title
    for ax in axes:
        ax.set_xlabel('Step')
        # Remove x-axis ticks and labels
        ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        # Remove grid by not calling ax.grid()

except Exception as e:
    print(f"An error occurred: {e}")
    # Add error text to plots if something fails
    axes[0].text(0.5, 0.5, 'Plotting failed', ha='center', va='center')
    axes[1].text(0.5, 0.5, 'Plotting failed', ha='center', va='center')


plt.tight_layout(rect=[0, 0, 1, 0.95]) # Adjust layout to make room for suptitle

# --- Save and Show Plot ---
filename = os.path.join(output_dir, 'fathomed_naive_sols_vs_reward.pgf')
plt.savefig(filename)
plt.show()

print(f"\nSide-by-side plot saved successfully as '{filename}'")
