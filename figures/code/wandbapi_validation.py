
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
import matplotlib as mpl

import wandb
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

runs = api.runs(
    path = "eliasaar_org/OptDMaking",
     filters = {"$and": [{"group" : "unknown_c_unknown_value_at_fathomed_naive_new" }]},
)

window_size  = 100
i = 0
for run in runs:
    
    df = run.scan_history(
    keys=["ep_reward","_step"], page_size=100000, min_step=None, max_step=5e4
)
    # df = run.history(    samples=1e6, keys=["smooth_ep_reward"], x_axis="_step", pandas=(True), stream="default"
    # )
    # print(df)
    # col_data = df["smooth_ep_reward"][:100]
    col_data  = [row["ep_reward"] for row in df]
    steps = [row["_step"] for row in df]
    
    df = pd.DataFrame({'smooth_ep_reward':col_data,"_step":steps})
    print(col_data)
    col_data = df["smooth_ep_reward"]
    col_data = col_data.rolling(window = window_size).mean()
    normalized_data = (col_data - col_data.min()) / (col_data.max() - col_data.min())  # Normalize to [0, 1]
    # normalized_data = normalized_data.rolling(window = window_size).mean()
    # normalized_data = np.array(normalized_data)
    # normalized_data.plot()
    print(normalized_data)
    plt.plot(df["_step"].iloc[101:], normalized_data.iloc[101:], label=f"seed = {run.name.strip()[-1]}")
    i+=1  # Plot on the same figure

plt.xlabel('Step')
plt.ylabel('Episodic Cost')
# plt.title('Comparison of training with different seeds')
# plt.tick_params(
#     axis="both",   # apply to x and y axes
#     which="both",  # major and minor ticks
#     bottom=False, top=False,          # x-axis ticks off
#     left=False, right=False,          # y-axis ticks off
#     labelbottom=False, labelleft=False  # hide tick labels too
# )
# plt.tick_params(
#     axis='y',          # changes apply to the x-axis
#     which='both',      # both major and minor ticks are affected
#     bottom=False,      # ticks along the bottom edge are off
#     top=False,         # ticks along the top edge are off
#     labelbottom=False) 
plt.grid(False)
fig = plt.gcf()
fig.set_size_inches(6,2)
# handles, labels = plt.gca().get_legend_handles_labels()
# order = [1,0,2,3,4]
# plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order])
# plt.legend()
plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(format_thousands))
filename = os.path.join(output_dir, 'validation.pgf')
plt.savefig(filename)
plt.show()

plt.close()
print(f"Combined normalized plot saved as '{filename}'")
