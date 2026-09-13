"""
Training loss curve from W&B export.
Loads wandb_training_loss.csv from results/ and saves plot to output/.
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'results')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

CSV_PATH = os.path.join(RESULTS_DIR, 'wandb_training_loss.csv')
STEPS_PER_EPOCH = 8982
METRIC_SUBSTR = "loss_3d_pos"
SAVE_PATH = os.path.join(OUTPUT_DIR, 'plot_loss.png')


def load_epoch_averages():
    """The points the curve is drawn from: one mean loss per epoch, plus epoch 0.

    Returns (epoch_avg, loss_col). epoch_avg carries `epoch`, the mean loss in
    metres, `n_steps` (how many logged steps went into that mean) and `loss_log`,
    the value actually plotted -- the y axis is log-transformed and then
    relabelled in the original units.
    """
    df = pd.read_csv(CSV_PATH)

    step_col = next(c for c in df.columns if c.strip().lower() == "step")
    loss_candidates = [c for c in df.columns
                       if METRIC_SUBSTR in c and not c.endswith("__MIN") and not c.endswith("__MAX")]
    if not loss_candidates:
        loss_candidates = [c for c in df.columns if METRIC_SUBSTR in c]
    loss_col = loss_candidates[0]

    df = df[[step_col, loss_col]].copy()
    df[step_col] = pd.to_numeric(df[step_col], errors="coerce")
    df[loss_col] = pd.to_numeric(df[loss_col], errors="coerce")
    df = df.dropna(subset=[step_col, loss_col])

    df["epoch"] = ((df[step_col] - 1) // STEPS_PER_EPOCH) + 1
    epoch_avg = df.groupby("epoch", as_index=False)[loss_col].mean()
    epoch_avg["n_steps"] = df.groupby("epoch", as_index=False)[loss_col].size()["size"].values

    first_point = pd.DataFrame({"epoch": [0], loss_col: [df.iloc[0][loss_col]], "n_steps": [1]})
    epoch_avg = pd.concat([first_point, epoch_avg], ignore_index=True)
    epoch_avg = epoch_avg[epoch_avg["epoch"] <= 10].reset_index(drop=True)
    epoch_avg["loss_log"] = np.log(epoch_avg[loss_col])
    return epoch_avg, loss_col


def main():
    epoch_avg, loss_col = load_epoch_averages()

    plt.figure(figsize=(9, 5))
    plt.plot(epoch_avg["epoch"], epoch_avg["loss_log"],
             linestyle="-", color="green", linewidth=1.5, zorder=3)
    plt.scatter(epoch_avg["epoch"], epoch_avg["loss_log"],
                marker="s", s=80, facecolor="yellow", edgecolor="darkgreen",
                linewidths=1.8, zorder=5, label="Avg Loss / Epoch")

    yticks = np.linspace(epoch_avg["loss_log"].min(), epoch_avg["loss_log"].max(), 6)
    yticklabels = [f"{np.exp(y):.3f}" for y in yticks]
    plt.yticks(yticks, yticklabels, fontsize=15)
    plt.xticks(fontsize=15)
    plt.xlabel("Epoch", fontsize=16)
    plt.ylabel("Distance (Meter) Loss", fontsize=16)
    plt.title("Training Loss", fontsize=18)
    plt.legend(fontsize=15)
    plt.grid(False)
    plt.tight_layout()

    plt.savefig(SAVE_PATH, dpi=150)
    plt.savefig(SAVE_PATH.replace('.png', '.pdf'))
    plt.close()
    print(f"Saved: {SAVE_PATH}")


if __name__ == '__main__':
    main()
