
# # import pandas as pd
# # import matplotlib.pyplot as plt
# # import matplotlib.ticker as ticker
# # import os

# # import wandb
# # def format_thousands(x, pos):
# #     return f'{x/1000:.0f}k'

# # api = wandb.Api()

# # output_dir = "figures/plots"

# # runs = api.runs(
# #     path = "eliasaar_org/OptDMaking",
# #      filters = {"$and": [{"group" : "unknown_c_unknown_value_at_fathomed_naive" }]},
# # )

# # window_size  = 10
# # i = 0
# # for run in runs:
    
# #     df = run.scan_history(
# #     keys=["smooth_ep_reward"], page_size=1000, min_step=None, max_step=5e4
# # )
# #     # df = run.history(    samples=1e6, keys=["smooth_ep_reward"], x_axis="_step", pandas=(True), stream="default"
# #     # )
# #     # print(df)
# #     # col_data = df["smooth_ep_reward"][:100]
# #     col_data  = [row["smooth_ep_reward"] for row in df]
# #     col_data = pd.DataFrame({'smooth_ep_reward':col_data})
# #     print(col_data)
# #     normalized_data = (col_data - col_data.min()) / (col_data.max() - col_data.min())  # Normalize to [0, 1]
# #     normalized_data = normalized_data.rolling(window = window_size).mean()
# #     # normalized_data = np.array(normalized_data)
# #     # normalized_data.plot()
# #     print(normalized_data)
# #     plt.plot(normalized_data.index*1000, normalized_data, label=f"seed = {run.name.strip()[-1]}")
# #     i+=1  # Plot on the same figure

# # plt.xlabel('Step')
# # plt.ylabel('Normalized Episodic Cost')
# # plt.title('Comparison of training with different seeds')
# # plt.grid(False)
# # handles, labels = plt.gca().get_legend_handles_labels()
# # order = [1,0,2,3,4]
# # plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order])
# # # plt.legend()
# # plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(format_thousands))
# # plt.show()

# # filename = os.path.join(output_dir, 'all_columns_normalized_k_xaxis.png')
# # plt.savefig(filename)
# # plt.close()
# # print(f"Combined normalized plot saved as '{filename}'")




# import pandas as pd
# import matplotlib.pyplot as plt
# import matplotlib.ticker as ticker
# import os
# from itertools import groupby

# import wandb
# def format_thousands(x, pos):
#     return f'{x/1000:.0f}k'

# api = wandb.Api()

# output_dir = "figures/plots"

# runs = api.runs(
#     path = "eliasaar_org/OptDMaking",
#      filters = {"$or": [{"group" : "unknown_c_unknown_value_at_fathomed_naive" },{"group" : "unknown_c_unknown_value_at_fathomed_nn" }]},
# )

# window_size  = 10
# i = 1
# runs_sorted = sorted(list(runs), key=lambda x: x.name.strip()[-1])

# for key,group in groupby(runs_sorted,lambda x : x.name.strip()[-1]):
#     plt.subplot(3,2,i)
#     i+=1 
#     # print(len(list(group))) 

#     for run in list(group):
#         df = run.scan_history(
#         keys=["smooth_ep_reward","_step"], page_size=100000, min_step=None, max_step=5e4
#     )
#     #     df = run.scan_history(
#     #     keys=["n_sols"], page_size=1000, min_step=None, max_step=5e4
#     # )
#         # df = run.history(    samples=1e6, keys=["smooth_ep_reward"], x_axis="_step", pandas=(True), stream="default"
#         # )
#         # print(df)
#         # col_data = df["smooth_ep_reward"][:100]
#         col_data  = [row["smooth_ep_reward"] for row in df]
#         steps = [row["_step"] for row in df]
#         df = pd.DataFrame({'smooth_ep_reward':col_data,"_step":steps})
#         col_data = df["smooth_ep_reward"]
#         # col_data  = [row["n_sols"] for row in df]
#         # col_data = pd.DataFrame({'n_sols':col_data})
#         normalized_data = (col_data - col_data.min()) / (col_data.max() - col_data.min())  # Normalize to [0, 1]
#         normalized_data = normalized_data.rolling(window = window_size).mean()
#         # normalized_data = np.array(normalized_data)
#         # normalized_data.plot()
#         label = "uniform" if "naive" in run.name else "nn"
#         print(normalized_data)
#         plt.plot(df["_step"], normalized_data, label= label)


#         plt.xlabel('Step')
#         plt.ylabel('Episodic Cost')
#         plt.title(f'Sampling comparison, seed = {key}')
#         plt.grid(False)
# # handles, labels = plt.gca().get_legend_handles_labels()
# # order = [1,0,2,3,4]
# # plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order])
#         plt.legend()
#         plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(format_thousands))
# plt.tight_layout()
# plt.show()

# filename = os.path.join(output_dir, 'all_columns_normalized_k_xaxis.png')
# plt.savefig(filename)
# plt.close()
# print(f"Combined normalized plot saved as '{filename}'")



import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
from itertools import groupby
import numpy as np # Added for np.concatenate and nanmin/nanmax

import wandb

def format_thousands(x, pos):
    return f'{x/1000:.0f}k'

api = wandb.Api()

output_dir = "figures/plots"
os.makedirs(output_dir, exist_ok=True) # Ensure the output directory exists

runs_api = api.runs(
    path="eliasaar_org/OptDMaking",
    filters={"$or": [{"group": "unknown_c_unknown_value_at_fathomed_naive_new"},
                     {"group": "unknown_c_unknown_value_at_fathomed_nn"}]},
)

window_size = 10
plot_index = 1
# Sort runs by the last character of their name (assumed to be the seed)
runs_sorted = sorted(list(runs_api), key=lambda x: x.name.strip()[-1])

# Group runs by the seed
for seed, group in groupby(runs_sorted, lambda x: x.name.strip()[-1]):
    plt.subplot(3, 2, plot_index)
    plot_index += 1

    all_data_for_subplot = []
    run_data_list = [] # To store dataframes for later normalization and plotting

    # --- First pass: Collect all data for the current subplot ---
    for run in list(group): # Iterate through a copy of the group
        history = run.scan_history(
            keys=["smooth_ep_reward", "_step"], page_size=100000, min_step=None, max_step=5e4
        )
        # Store raw data and steps for this run
        raw_rewards = [row["smooth_ep_reward"] for row in history if row["smooth_ep_reward"] is not None] # Filter out None values
        steps = [row["_step"] for row in history if row["smooth_ep_reward"] is not None] # Ensure steps align

        if not raw_rewards: # Skip if no valid reward data
            print(f"Skipping run {run.name} for seed {seed} due to no reward data.")
            continue

        df_run = pd.DataFrame({'smooth_ep_reward': raw_rewards, "_step": steps, 'run_name': run.name})
        all_data_for_subplot.extend(raw_rewards) # Add to list for overall min/max
        run_data_list.append(df_run)

    if not all_data_for_subplot: # If no data was collected for this subplot
        plt.title(f'No data for seed = {seed}')
        plt.xlabel('Step')
        plt.ylabel('Normalized Episodic Cost') # Changed ylabel
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
        col_data = df_run_to_plot["smooth_ep_reward"]

        # Normalize using subplot's min and max
        if subplot_max == subplot_min: # Avoid division by zero if all values are the same
            normalized_data = pd.Series(np.zeros_like(col_data.values), index=col_data.index)
        else:
            normalized_data = (col_data - subplot_min) / (subplot_max - subplot_min)  # Normalize to [0, 1]

        smoothed_normalized_data = normalized_data.rolling(window=window_size, min_periods=1).mean() # min_periods=1 to handle edges

        label = "uniform" if "naive" in df_run_to_plot['run_name'].iloc[0] else "nn"
        plt.plot(df_run_to_plot["_step"], smoothed_normalized_data, label=label)

    plt.xlabel('Step')
    plt.ylabel('Episodic Cost') # Changed ylabel
    plt.title(f'Sample method comparison, seed = {seed}') # Changed title for clarity
    plt.grid(False)
    plt.legend()
    plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(format_thousands))

plt.tight_layout()
plt.show()

# Changed filename for better description
filename = os.path.join(output_dir, 'per_subplot_normalized_fathomed_comparison.png')
plt.savefig(filename)
plt.close()
print(f"Combined per-subplot normalized plot saved as '{filename}'")