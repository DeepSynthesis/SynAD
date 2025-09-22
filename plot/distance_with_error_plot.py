from pathlib import Path
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from plot.draw_settings import get_log_dir

text_size = 14

plt.rcParams.update(
    {
        "font.size": text_size,
        "axes.labelsize": text_size,
        "axes.titlesize": text_size,
        "xtick.labelsize": text_size,
        "ytick.labelsize": text_size,
        "font.family": "Arial",
    }
)


def load_logfile(dataset_type, method_type):
    log_dir = get_log_dir(dataset_type)
    df = pd.read_csv(log_dir / Path(f"SynAD/synad_results_for_{method_type}.csv"))
    return df


def distance_with_error_plot(dataset_type, method_type, error_line=[5, 10, 15, 20, 30, 40], palette=None):
    df = load_logfile(dataset_type, method_type)
    df["error"] = abs(df["y_pred"] - df["y_true"])

    Q1 = df["metrics"].quantile(0.25)
    Q3 = df["metrics"].quantile(0.75)
    IQR = Q3 - Q1
    upper_bound = Q3 + 1.5 * IQR
    lower_bound = Q1 - 1.5 * IQR
    df = df[df["metrics"] <= upper_bound]
    df = df[df["metrics"] >= lower_bound]

    bins = [0] + error_line + [np.inf]
    labels = [f"[0, {error_line[0]})"]
    labels += [f"[{error_line[i]}, {error_line[i+1]})" for i in range(len(error_line) - 1)]
    labels.append(f"[{error_line[-1]}, 100]")
    df["error_interval"] = pd.cut(df["error"], bins=bins, labels=labels, right=False)

    plt.figure(figsize=(8, 6))
    data = [df[df["error_interval"] == interval]["metrics"].dropna().values for interval in labels]

    ax = sns.boxplot(x="error_interval", y="metrics", data=df, palette=palette, order=labels, hue="error_interval", legend=False)
    ax.set_xlabel("Absolute error of prediction/%")
    ax.set_ylabel("Metrics value")
    ax.set_title(f"Metrics Distribution by Error Interval for {method_type}")
    plt.xticks(rotation=45)
    plt.grid(True, axis="y", alpha=0.4)

    plt.tight_layout()
    plt.savefig(Path(__file__).parent / Path(f"distance_with_error_plot_{method_type}.png"), dpi=300)


if __name__ == "__main__":
    custom_colors = ["#5b97c8", "#464b84", "#c6262f", "#c6262f", "#c6262f", "#c6262f", "#c6262f"]
    for method in ["ZKNN", "KDE", "ensemble", "BNN_probe"]:
        distance_with_error_plot(dataset_type="ULD", method_type=method, palette=custom_colors)
