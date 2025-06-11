import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
from itertools import groupby
import numpy as np # Added for np.concatenate and nanmin/nanmax
import operator
import wandb
import matplotlib as mpl

mpl.use("pgf")
plt.rcParams.update({
    "pgf.texsystem": "pdflatex",  # or "lualatex" / "xelatex" to match your document
    "font.family": "serif",       # use same font family as LaTeX (e.g., serif, sans-serif)
    "text.usetex": True,          # let matplotlib use LaTeX to render all text
    "pgf.rcfonts": False,         # disable matplotlib's default font setup
    "font.size": 10,              # match your document font size (e.g., 10pt, 11pt, 12pt)
})
def format_thousands(x, pos):
    return f'{x/1000:.0f}k'

api = wandb.Api()

output_dir = "figures/plots"
os.makedirs(output_dir, exist_ok=True) # Ensure the output directory exists

runs_api = api.runs(
    path="eliasaar_org/OptDMaking",
    filters={"$or": [{"group": "unknown_c_unknown_value_at_fathomed_nn"},
                     {"group": "unknown_c_unknown_value_at_sample_nn"}]},
)

window_size = 100
plot_index = 1
# Sort runs by the last character of their name (assumed to be the seed)
runs_sorted = sorted(list(runs_api), key=lambda x: x.name.strip()[-1])

color_map = {
    "sample" : "C0",
    "pruned" : "C1",
    "nn" : "C2",
    "uniform" : "C3"
}

# Group runs by the seed
# plt.figure(figsize= (7.5,2.5))
fig = plt.figure()
for seed, group in groupby(runs_sorted, lambda x: x.name.strip()[-1]):
    plt.subplot(3, 2, plot_index)
    plot_index += 1

    all_data_for_subplot = []
    run_data_list = [] # To store dataframes for later normalization and plotting

    # --- First pass: Collect all data for the current subplot ---
    for run in list(group): # Iterate through a copy of the group
        history = run.scan_history(
            keys=["ep_reward", "_step"], page_size=100000, min_step=None, max_step=5e4
        )
        # Store raw data and steps for this run
        raw_rewards = [row["ep_reward"] for row in history if row["ep_reward"] is not None] # Filter out None values
        steps = [row["_step"] for row in history if row["ep_reward"] is not None] # Ensure steps align with filtered rewards

        if not raw_rewards: # Skip if no valid reward data
            print(f"Skipping run {run.name} for seed {seed} due to no reward data.")
            continue

        df_run = pd.DataFrame({'ep_reward': raw_rewards, "_step": steps, 'run_name': run.name})
        all_data_for_subplot.extend(raw_rewards) # Add to list for overall min/max
        run_data_list.append(df_run)

    if not all_data_for_subplot: # If no data was collected for this subplot
        plt.title(f'No data for seed = {seed}')
        plt.xlabel('Step')
        plt.ylabel('Episodic Cost')
        plt.grid(False)
        plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(format_thousands))
        print(f"No data to plot for seed {seed}.")
        continue # Skip to the next subplot

    # --- Calculate min and max for the current subplot ---
    # Convert to numpy array for robust min/max calculation, ignoring NaNs
    all_data_np = np.array(all_data_for_subplot, dtype=float)
    subplot_min = np.nanmin(all_data_np)
    subplot_max = np.nanmax(all_data_np)

    # --- Second pass: Normalize and plot each run in the subplot ---
    for df_run_to_plot in run_data_list:
        col_data = df_run_to_plot["ep_reward"]

        # Normalize using subplot's min and max
        if subplot_max == subplot_min: # Avoid division by zero if all values are the same
            normalized_data = pd.Series(np.zeros_like(col_data.values), index=col_data.index)
        else:
            normalized_data = (col_data - subplot_min) / (subplot_max - subplot_min)  # Normalize to [0, 1]

        smoothed_normalized_data = normalized_data.rolling(window=window_size, min_periods=1).mean() # min_periods=1 to handle edges

        label = "pruned" if "fathomed" in df_run_to_plot['run_name'].iloc[0] else "sample"
        color = color_map[label]
        plt.plot(df_run_to_plot["_step"].iloc[101:], smoothed_normalized_data.iloc[101:], label=label,color = color)
        # loc_fig = plt.gcf()
        # loc_fig.set_size_inches(2,1)
    plt.xlabel('Step', fontsize=9)
    plt.ylabel('Episodic Cost', fontsize=9)
    plt.title(f'Seed = {seed}')
    plt.grid(False)
    plt.tick_params(
        axis="both",   # apply to x and y axes
        which="both",  # major and minor ticks
        bottom=False, top=False,          # x-axis ticks off
        left=False, right=False,          # y-axis ticks off
        labelbottom=False, labelleft=False  # hide tick labels too
    )
handles, labels = plt.gca().get_legend_handles_labels()

hl = sorted(zip(handles, labels),
        key=operator.itemgetter(1))
handles2, labels2 = zip(*hl)

fig.legend(handles2, labels2,loc='upper center', ncol=2)
    # plt.legend()
    # plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(format_thousands))

plt.tight_layout()
# plt.suptitle("Gradient sampling comparison, nearest neighbour")

# fig = plt.gcf()
fig.set_size_inches(6,4)
fig.subplots_adjust(
    left=0.1,    # space from left edge
    right=0.9,   # space from right edge
    top=0.8,     # space from top
    bottom=0.1,  # space from bottom
    hspace=1.1,  # vertical spacing between subplots
    wspace=0.3   # horizontal spacing between subplots
)
# fig.patch.set_linewidth(1)
# fig.patch.set_edgecolor('grey')

filename = os.path.join(output_dir, 'pruned_vs_sample_nn.pgf')
plt.savefig(filename)
plt.show()
# plt.close()
print(f"Combined per-subplot normalized plot saved as '{filename}'")