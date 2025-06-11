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

# --- Main Script ---
# Initialize the wandb API to fetch run data.
api = wandb.Api()

# Define the output directory for saving the plot.
output_dir = "figures/plots"
os.makedirs(output_dir, exist_ok=True)

# Define the groups to fetch runs from.
groups = [
    "unknown_c_unknown_value_at_fathomed_nn",
    "unknown_c_unknown_value_at_sample_nn",
    "unknown_c_unknown_value_at_sample_naive_new",
    "unknown_c_unknown_value_at_fathomed_naive_new"
]

# Create a mapping for clear plot labels.
label_map = {
    "unknown_c_unknown_value_at_fathomed_nn": "pruned nn",
    "unknown_c_unknown_value_at_sample_nn": "sample nn",
    "unknown_c_unknown_value_at_sample_naive_new": "sample uniform",
    "unknown_c_unknown_value_at_fathomed_naive_new": "pruned uniform"
}

# --- Plotting Setup ---
# Adjust figsize to be more compact for an A4 page (width, height in inches).
# A width of ~5.5 inches fits well within a standard LaTeX article text block.
fig, ax = plt.subplots(figsize=(5.5, 3))
# Use the 'tab10' color map for distinct, high-contrast line colors.
colors = plt.cm.tab10(np.linspace(0, 1, len(groups)))

# --- Data Fetching and Plotting ---
# Iterate through each group to fetch and plot data for one run.
for i, group_name in enumerate(groups):
    try:
        # Fetch runs from the specified group.
        runs = api.runs(
            path="eliasaar_org/OptDMaking",
            filters={"group": group_name}
        )

        if not runs:
            print(f"Warning: No runs found for group '{group_name}'. Skipping.")
            continue

        # Select the first run as a representative for the group.
        run = runs[0]
        print(f"Processing run '{run.name}' from group '{group_name}'...")

        # Scan the run's history to get 'n_sols' and '_step'.
        history = run.scan_history(keys=["n_sols", "_step"], page_size=10000,max_step=5e4)

        # Create a pandas DataFrame from the history data.
        df = pd.DataFrame(history)

        # Drop any rows where 'n_sols' is missing to ensure data integrity.
        df.dropna(subset=['n_sols'], inplace=True)

        if df.empty:
            print(f"Warning: No 'n_sols' data found for run '{run.name}'. Skipping.")
            continue

        # Apply the rolling window to smooth the data
        smoothed_n_sols = df["n_sols"].rolling(window=window_size, min_periods=1).mean()
        smoothed_n_sols = smoothed_n_sols.iloc[101:]
        df = df.iloc[101:]
        # Plot the smoothed 'n_sols' against '_step'.
        ax.plot(df["_step"], smoothed_n_sols, label=label_map.get(group_name, group_name), color=colors[i])

    except Exception as e:
        print(f"An error occurred while processing group '{group_name}': {e}")

# --- Final Plot Configuration ---
ax.set_xlabel('Step')
ax.set_ylabel(f'Solutions') # Shortened label for a smaller plot
# ax.set_title('SiSolutions')

# Remove x-axis ticks and labels
ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

# Remove grid
# The ax.grid(...) line has been removed.

ax.legend(loc='best', ncol=2)
fig = plt.gcf()
fig.set_size_inches(6,2)
plt.tight_layout()

# --- Save and Show Plot ---
filename = os.path.join(output_dir, 'n_sols_comparison_compact.pgf')
plt.savefig(filename)
plt.show()

print(f"\nCompact plot saved successfully as '{filename}'")