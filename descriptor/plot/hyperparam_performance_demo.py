from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score

from plot.draw_settings import get_log_dir

file_list = [
    "hyperparam_eval_for_ZKNN-Z=0.5,k=4.csv",
    "hyperparam_eval_for_ZKNN-Z=1.0,k=3.csv",
    "hyperparam_eval_for_ZKNN-Z=1.5,k=1.csv",
    "hyperparam_eval_for_ZKNN-Z=3.0,k=1.csv",
]
hp_list = [{"Z": 0.5, "k": 4}, {"Z": 1.0, "k": 3}, {"Z": 1.5, "k": 1}, {"Z": 3.0, "k": 1}]
colors = ["#c6262f", "#5b97c8", "#464b84", "#7a3090"]


def hp_performance_plot(dataset_type):
    data_dir = get_log_dir(dataset_type=dataset_type, sub_dir="SynAD")

    fig, axes = plt.subplots(1, 4, figsize=(24, 5), dpi=300)
    plt.subplots_adjust(wspace=0.2, hspace=0.1)
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.5)

    for i, (f, color) in enumerate(zip(file_list, colors)):
        df = pd.read_csv(data_dir / f)
        coverage = f"{len(df[df['AD_type'] == 'IAD'])/len(df):.3f}"
        df = df[df["AD_type"] == "IAD"]

        sns.scatterplot(data=df, x="y_pred", y="y_true", ax=axes[i], color=color, s=50, alpha=0.2, edgecolor="w", linewidth=0.3)
        axes[i].axline((0, 0), slope=1, color="gray", linestyle="--", alpha=0.5)
        title = f.split(".")[0].replace("hyperparam_eval_for_", "")
        axes[i].set_xlabel("Predict Yield/%", fontsize=14)
        axes[i].set_ylabel("Experimental Yield/%", fontsize=14) if i == 0 else axes[i].set_ylabel("")

        min_val = min(df["y_pred"].min(), df["y_true"].min())
        max_val = max(df["y_pred"].max(), df["y_true"].max())
        axes[i].set_xlim(min_val - 5, max_val + 5)
        axes[i].set_ylim(min_val - 5, max_val + 5)

        axes[i].grid(True, linestyle="--", alpha=0.3)
        r2 = r2_score(df["y_true"], df["y_pred"])
        axes[i].text(
            0.05,
            0.95,
            f"Z = {hp_list[i]['Z']}, k = {hp_list[i]['k']}\nR² = {r2:.3f}\nCoverage = {coverage}",
            transform=axes[i].transAxes,
            fontsize=12,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

    output_path = Path(__file__).parent / "hyperparam_performance.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.2)
    plt.close()


if __name__ == "__main__":
    hp_performance_plot(dataset_type="ULD")
