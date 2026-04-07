import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error

from draw_settings import get_log_dir

text_size = 14
plt.rcParams.update(
    {
        "font.size": text_size,
        "axes.labelsize": text_size,
        "axes.titlesize": text_size,
        "xtick.labelsize": text_size,
        "ytick.labelsize": text_size,
    }
)


def plot_prediction_performace(train_data, test_data, dataset_type, model_name, return_figure=False):
    output_dir = get_log_dir(dataset_type, "predict_performace")
    output_dir.mkdir(parents=True, exist_ok=True)
    from matplotlib.colors import LinearSegmentedColormap

    crest_white = LinearSegmentedColormap.from_list("crest_white", ["white", "#5b97c8", "#02335c"])
    flare_white = LinearSegmentedColormap.from_list("flare_white", ["white", "#c6262f", "#3e0c0f"])

    fig = plt.figure(figsize=(18, 8), dpi=300)
    grid = plt.GridSpec(4, 8, hspace=0.5, wspace=0.5)

    test_main_ax = fig.add_subplot(grid[1:4, 0:3])
    test_x_ax = fig.add_subplot(grid[0, 0:3])
    test_y_ax = fig.add_subplot(grid[1:4, 3])

    train_main_ax = fig.add_subplot(grid[1:4, 4:7])
    train_x_ax = fig.add_subplot(grid[0, 4:7])
    train_y_ax = fig.add_subplot(grid[1:4, 7])

    sns.kdeplot(data=train_data, x="predict", y="real", cmap=crest_white, fill=True, levels=20, thresh=0.05, ax=test_main_ax)
    test_main_ax.set(xlim=(0, 100), ylim=(0, 100))
    test_main_ax.set_xlabel("Train Predicted Yield")
    test_main_ax.set_ylabel("Experimental Yield")

    test_x_ax.plot(train_data["predict"], np.zeros_like(train_data["predict"]), "|", color="gray", alpha=0.3)
    sns.kdeplot(data=train_data, x="predict", ax=test_x_ax, color="#5b97c8", fill=True)
    test_x_ax.set_ylim(0, None)
    test_x_ax.set_xticks([])
    test_x_ax.set_yticks([])
    test_x_ax.spines["right"].set_visible(False)
    test_x_ax.spines["top"].set_visible(False)
    test_x_ax.spines["left"].set_visible(False)
    test_x_ax.spines["bottom"].set_visible(False)

    test_y_ax.plot(np.zeros_like(train_data["real"]), train_data["real"], "|", color="gray", alpha=0.3)
    sns.kdeplot(data=train_data, y="real", ax=test_y_ax, color="#5b97c8", fill=True)
    test_y_ax.set_xlim(0, None)
    test_y_ax.set_xticks([])
    test_y_ax.set_yticks([])
    test_y_ax.spines["right"].set_visible(False)
    test_y_ax.spines["top"].set_visible(False)
    test_y_ax.spines["left"].set_visible(False)
    test_y_ax.spines["bottom"].set_visible(False)

    sns.kdeplot(data=test_data, x="predict", y="real", cmap=flare_white, fill=True, levels=20, thresh=0.05, ax=train_main_ax)
    train_main_ax.set(xlim=(0, 100), ylim=(0, 100))
    train_main_ax.set_xlabel("Test Predicted Yield")
    train_main_ax.set_ylabel("")

    train_x_ax.plot(test_data["predict"], np.zeros_like(test_data["predict"]), "|", color="gray", alpha=0.3)
    sns.kdeplot(data=test_data, x="predict", ax=train_x_ax, color="#c6262f", fill=True)
    train_x_ax.set_ylim(0, None)
    train_x_ax.set_xticks([])
    train_x_ax.set_yticks([])
    train_x_ax.spines["right"].set_visible(False)
    train_x_ax.spines["top"].set_visible(False)
    train_x_ax.spines["left"].set_visible(False)
    train_x_ax.spines["bottom"].set_visible(False)

    train_y_ax.plot(np.zeros_like(test_data["real"]), test_data["real"], "|", color="gray", alpha=0.3)
    sns.kdeplot(data=test_data, y="real", ax=train_y_ax, color="#c6262f", fill=True)
    train_y_ax.set_xlim(0, None)
    train_y_ax.set_xticks([])
    train_y_ax.set_yticks([])
    train_y_ax.spines["right"].set_visible(False)
    train_y_ax.spines["top"].set_visible(False)
    train_y_ax.spines["left"].set_visible(False)
    train_y_ax.spines["bottom"].set_visible(False)

    plt.savefig(output_dir / f"{model_name}.png", dpi=300, bbox_inches="tight")

    if return_figure:
        return plt.show()


# Global constants
CHEMPROP_DATA = {
    "R2": [0.512, 0.468, 0.491, 0.548, 0.553, 0.402, 0.420, 0.493, 0.496, 0.449],
    "MAE": [14.870, 14.329, 14.166, 13.261, 13.267, 14.912, 15.606, 14.430, 14.430, 15.281],
}

YIELDBERT_DATA = {
    "R2": [0.158, 0.345, 0.344, 0.273, 0.421, 0.187, 0.348, 0.218, 0.324, 0.089],
    "MAE": [14.449, 13.180, 12.636, 13.343, 12.058, 14.574, 12.169, 13.368, 13.504, 14.028],
}

METHOD_MAPPING = {
    "ADA": "AdaBoost",
    "BAG": "Bagging",
    "BNN": "Bayesian NN",
    "CAT": "CatBoost",
    "LGB": "LightGBM",
    "NN": "Neural Network",
    "RF": "Random Forest",
    "SVM": "SVM",
    "XGB": "XGBoost",
    "ChemProp": "ChemProp",
    "YieldBERT": "YieldBERT",
}

METRICS = ["R2", "MAE", "RMSE"]
PALETTE = sns.color_palette("tab10") + sns.color_palette("Set2")[:2]

# Method grouping and coloring
METHOD_GROUPS = {
    "Boost/RF": ["AdaBoost", "CatBoost", "LightGBM", "Random Forest", "XGBoost"],
    "NN": ["Bayesian NN", "Neural Network", "ChemProp", "YieldBERT"],
    "Others": ["Bagging", "SVM"],
}

COLOR_PALETTE = {"Boost/RF": "#5b97c8", "NN": "#464b84", "Others": "#c6262f"}  # Blue  # Orange  # Green


def plot_all_method_performance(dataset_type, return_figure=False):
    input_dir = get_log_dir(dataset_type)
    test_csv_files = [{"method_type": f.stem.split("_")[0], "filepath": f} for f in input_dir.glob("*_yield_test.csv")]

    performance_data = []
    for df, file_info in zip([pd.read_csv(f["filepath"]) for f in test_csv_files], test_csv_files):
        for fold, fold_df in df.groupby("fold_id"):
            y_true, y_pred = fold_df["real"], fold_df["predict"]
            performance_data.extend(
                [
                    {"Method": file_info["method_type"], "Metric": "R2", "Value": r2_score(y_true, y_pred)},
                    {"Method": file_info["method_type"], "Metric": "MAE", "Value": mean_absolute_error(y_true, y_pred)},
                    {"Method": file_info["method_type"], "Metric": "RMSE", "Value": root_mean_squared_error(y_true, y_pred)},
                ]
            )

    for method, data in [("ChemProp", CHEMPROP_DATA), ("YieldBERT", YIELDBERT_DATA)]:
        performance_data.extend([{"Method": method, "Metric": metric, "Value": val} for metric in ["R2", "MAE"] for val in data[metric]])

    performance_df = pd.DataFrame(performance_data)
    performance_df["Method"] = performance_df["Method"].map(METHOD_MAPPING)

    # Add group information and custom sorting
    performance_df["Group"] = performance_df["Method"].apply(lambda x: next((k for k, v in METHOD_GROUPS.items() if x in v), "Others"))

    # Create custom sort order
    group_order = ["Boost/RF", "NN", "Others"]
    method_order = [m for group in group_order for m in METHOD_GROUPS.get(group, [])]
    performance_df["Method"] = pd.Categorical(performance_df["Method"], categories=method_order, ordered=True)
    performance_df = performance_df.sort_values("Method")

    figures = []
    for metric in METRICS:
        plt.figure(figsize=(14, 7))
        sns.set_style("whitegrid")
        plt.rcParams.update({"font.size": 14})

        metric_df = performance_df[performance_df["Metric"] == metric]
        stats_df = metric_df.groupby("Method")["Value"].agg(["mean", "std"]).reset_index()

        # Assign colors based on method groups
        stats_df["Color"] = stats_df["Method"].apply(
            lambda x: next((v for k, v in COLOR_PALETTE.items() if x in METHOD_GROUPS.get(k, [])), COLOR_PALETTE["Others"])
        )

        color_dict = dict(zip(stats_df["Method"], stats_df["Color"]))

        ax = sns.barplot(
            x="Method",
            y="mean",
            data=stats_df,
            order=method_order,
            palette=color_dict,
            edgecolor="black",
            linewidth=1.5,
        )

        ax.errorbar(x=range(len(stats_df)), y=stats_df["mean"], yerr=stats_df["std"], fmt="none", ecolor="black", capsize=8, elinewidth=1)

        for i, (mean_val, std_val) in enumerate(zip(stats_df["mean"], stats_df["std"])):
            ax.text(i, mean_val + 0.08 * mean_val, f"{mean_val:.2f}±{std_val:.2f}", ha="center", va="bottom", fontsize=14)

        plt.xlabel("Method", fontsize=15, fontweight="bold")
        plt.ylabel(f"{metric} (mean ± std)", fontsize=15, fontweight="bold")
        plt.xticks(rotation=45, ha="right", fontsize=14)
        plt.yticks(fontsize=14)

        if metric == "R2":
            plt.ylim(max(0, stats_df["mean"].min() - 0.1), min(1, stats_df["mean"].max() + 0.1))
        elif metric == "MAE":
            plt.ylim(0, stats_df["mean"].max() * 1.2)

        plt.tight_layout()
        plt.savefig(Path(__file__).parent / f"Model_performance_{dataset_type}_{metric}.png", dpi=300, bbox_inches="tight")
        if return_figure:
            figures.append(ax.get_figure())
        else:
            plt.show()
            plt.close()

    return figures if return_figure else None


def plot_all_desc_performance(dataset_type, return_figure=False, colors=None, rotation=45):
    """
    Plot boxplot comparing performance of different descriptors with customizable styling.

    Parameters:
    - dataset_type: str, name/type of dataset (used in title)
    - return_figure: bool, whether to return the figure object
    - title_size: int, font size for the title
    - label_size: int, font size for axis labels
    - colors: dict or list, colors for each descriptor category.
              If dict: {"OneHot": "color1", "baseline": "color2", ...}
              If list: ["color1", "color2", ...] in order of descriptors
              If None, uses default palette
    - rotation: int, rotation angle for x-axis labels

    Returns:
    - If return_figure=True, returns matplotlib figure object
    - Otherwise shows the plot and returns None
    """
    # Prepare the data in long format for seaborn
    data = {
        "SPOC": [0.644, 0.635, 0.594, 0.624, 0.642, 0.586, 0.641, 0.646, 0.652, 0.562],
        "fingerprint": [0.568, 0.515, 0.602, 0.572, 0.599, 0.618, 0.584, 0.579, 0.582, 0.598],
        "QM desc": [0.559, 0.607, 0.523, 0.57, 0.609, 0.534, 0.535, 0.613, 0.583, 0.524],
        "MFP": [0.598, 0.608, 0.566, 0.577, 0.579, 0.561, 0.574, 0.601, 0.608, 0.508],
        "OneHot": [0.499, 0.483, 0.507, 0.499, 0.465, 0.502, 0.499, 0.519, 0.456, 0.477],
    }
    colors = {"SPOC": "#464b84", "fingerprint": "#5b97c8", "QM desc": "#c6262f", "MFP": "#844669", "OneHot": "#535353"}

    # Convert to long format DataFrame
    df = pd.DataFrame(data).melt(var_name="Descriptor", value_name="Performance")

    # Create the plot
    plt.figure(figsize=(8, 6))
    sns.set_style("whitegrid")

    # Handle custom colors
    if colors is not None:
        if isinstance(colors, dict):
            # Map descriptor names to colors
            palette = [colors[desc] for desc in df["Descriptor"].unique()]
        else:
            # Assume list/array of colors in order
            palette = colors
    else:
        palette = "Set2"

    ax = sns.boxplot(data=df, x="Descriptor", y="Performance", palette=palette, width=0.5)

    # Add title and labels with customizable sizes
    ax.set_xlabel("Descriptor Type", fontsize=15, fontweight="bold")
    ax.set_ylabel("Prediction R2", fontsize=15, fontweight="bold")

    # Customize tick labels
    ax.tick_params(axis="both", which="major", labelsize=14)

    # Rotate x-axis labels if needed
    plt.xticks(rotation=rotation, ha="right")

    # Adjust layout
    plt.tight_layout()
    plt.savefig(Path(__file__).parent / f"Descriptor_performance_{dataset_type}.png", dpi=300, bbox_inches="tight")

    if return_figure:
        return ax.get_figure()


if __name__ == "__main__":
    # plot_all_method_performance("ULD")
    plot_all_desc_performance("ULD")
