
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os

import wandb
def format_thousands(x, pos):
    return f'{x/1000:.0f}k'

api = wandb.Api()

output_dir = "figures/plots"

runs = api.runs(
    path = "eliasaar_org/OptDMaking",
     filters = {"$and": [{"group" : "unknown_c_unknown_value_at_fathomed_naive" }]},
)

window_size  = 10
i = 0
for run in runs:
    
    df = run.scan_history(
    keys=["smooth_ep_reward","_step"], page_size=1000, min_step=None, max_step=5e4
)
    # df = run.history(    samples=1e6, keys=["smooth_ep_reward"], x_axis="_step", pandas=(True), stream="default"
    # )
    # print(df)
    # col_data = df["smooth_ep_reward"][:100]
    col_data  = [row["smooth_ep_reward"] for row in df]
    steps = [row["_step"] for row in df]
    
    df = pd.DataFrame({'smooth_ep_reward':col_data,"_step":steps})
    print(col_data)
    col_data = df["smooth_ep_reward"]
    
    normalized_data = (col_data - col_data.min()) / (col_data.max() - col_data.min())  # Normalize to [0, 1]
    normalized_data = normalized_data.rolling(window = window_size).mean()
    # normalized_data = np.array(normalized_data)
    # normalized_data.plot()
    print(normalized_data)
    plt.plot(df["_step"], normalized_data, label=f"seed = {run.name.strip()[-1]}")
    i+=1  # Plot on the same figure

plt.xlabel('Step')
plt.ylabel('Normalized Episodic Cost')
plt.title('Comparison of training with different seeds')
plt.grid(False)
handles, labels = plt.gca().get_legend_handles_labels()
order = [1,0,2,3,4]
plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order])
# plt.legend()
plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(format_thousands))
plt.show()

filename = os.path.join(output_dir, 'all_columns_normalized_k_xaxis.png')
plt.savefig(filename)
plt.close()
print(f"Combined normalized plot saved as '{filename}'")
