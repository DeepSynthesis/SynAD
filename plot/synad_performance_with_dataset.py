from pathlib import Path
from cycler import V
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns


def synad_performance_with_dataset_hyperparam(dataset_type, method_type, target, split_mode, color="#873A86"):
    log_path = Path(__file__).parent / Path(f"../src/logs/{dataset_type}/SynAD")
    data = log_path / Path(f"{method_type}_for_{target}_synad_hyperparam_search_with_{split_mode}_partition.csv")
    df = pd.read_csv(data)

    # Create figure with higher DPI and custom size
    plt.figure(figsize=(8, 7), dpi=300)

    # Create scatter plot with custom styling
    scatter = sns.scatterplot(
        data=df,
        x="coverage",
        y="r2_iad",
        color=color,
        s=200,  # Increase point size
        edgecolor="white",  # Add black border to points
        linewidth=0.5,  # Border thickness
    )

    # Set plot title and labels with custom font sizes
    plt.title(f"SynAD Performance: {method_type} on {dataset_type} ({target})", fontsize=14, pad=20)
    plt.xlabel("Coverage", fontsize=12)
    plt.ylabel("R2 IAD", fontsize=12)

    # Ensure y-axis doesn't go below 0
    plt.ylim(bottom=0, top=1)
    plt.xlim(left=0, right=0.5)

    # Add grid for better readability
    plt.grid(True, linestyle="--", alpha=0.7)

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    # Save figure
    output_path = Path(__file__).parent / Path(f"SynAD_performance/{dataset_type}_{method_type}_for_{target}_hp_search.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def synad_heatmap_with_hyper_param(dataset_type, method_type, target, split_mode, label1, label2):
    log_path = Path(__file__).parent / Path(f"../src/logs/{dataset_type}/SynAD")
    data = log_path / Path(f"{method_type}_for_{target}_synad_hyperparam_search_with_{split_mode}_partition.csv")
    df = pd.read_csv(data)
    hyper_param_df = pd.DataFrame.from_records(df.iloc[:, 0].apply(lambda x: [n.split("=")[-1] for n in x.split(",")]))

    hyper_param_df.columns = [label1, label2]
    df = pd.concat([df, hyper_param_df], axis=1)
    df[label1] = pd.to_numeric(df[label1])
    df[label2] = pd.to_numeric(df[label2])
    df = df.sort_values([label1, label2])
    df = df.fillna(1.0)
    plt.rcParams.update({"axes.labelsize": 20, "xtick.labelsize": 20, "ytick.labelsize": 20, "axes.titlesize": 20, "figure.titlesize": 18})

    plt.figure(figsize=(13, 10))
    # from IPython import embed; embed(); exit()
    df = df.drop_duplicates(subset=[label1, label2])
    Z_r2_df = df.pivot(index=label2, columns=label1, values="r2_iad")

    ax = sns.heatmap(Z_r2_df, cmap="flare", cbar=True)
    cbar = ax.collections[0].colorbar
    cbar.set_label("Inner-SynAD R2", size=16)
    cbar.ax.tick_params(labelsize=15)
    plt.xlabel(label1, fontsize=20)
    plt.ylabel(label2, fontsize=20)
    plt.title(f"{dataset_type}_{method_type}_for_{target}_hyperparm_with_r2", pad=20)
    plt.tight_layout()
    output_path = Path(__file__).parent / Path(f"SynAD_performance/{dataset_type}_{method_type}_for_{target}_hyperparm_with_r2_heatmap.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(13, 10))
    Z_cov_df = df.pivot(index=label2, columns=label1, values="coverage")
    ax = sns.heatmap(Z_cov_df, cmap="flare", cbar=True)
    cbar = ax.collections[0].colorbar
    cbar.set_label("Coverage", size=16)
    cbar.ax.tick_params(labelsize=15)
    plt.xlabel(label1, fontsize=20)
    plt.ylabel(label2, fontsize=20)
    plt.title(f"{dataset_type}_{method_type}_for_{target}_hyperparm_with_coverage", pad=20)
    plt.tight_layout()
    output_path = Path(__file__).parent / Path(
        f"SynAD_performance/{dataset_type}_{method_type}_for_{target}_hyperparm_with_coverage_heatmap.png"
    )
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def synad_single_variance_with_hyper_param(dataset_type, method_type, target, split_mode, col_name):
    log_path = Path(__file__).parent / Path(f"../src/logs/{dataset_type}/SynAD")
    data = log_path / Path(f"{method_type}_for_{target}_synad_hyperparam_search_with_{split_mode}_partition.csv")
    df = pd.read_csv(data)
    df = df[df["r2_iad"] > -50]

    if method_type == "KDE":
        df = df.iloc[100:350, :]
        df.reset_index(inplace=True, drop=True)
    hyper_param_df = pd.DataFrame(df.iloc[:, 0].apply(lambda x: x.split("=")[-1]))
    hyper_param_df.columns = [col_name]
    df = pd.concat([df, hyper_param_df], axis=1)
    df[col_name] = pd.to_numeric(df[col_name])
    df = df.sort_values([col_name])
    plt.rcParams.update({"axes.labelsize": 20, "xtick.labelsize": 20, "ytick.labelsize": 20, "axes.titlesize": 20, "figure.titlesize": 18})
    plt.figure(figsize=(13, 10))
    ax1 = sns.lineplot(data=df, x=col_name, y="r2_iad", color="#1f77b4", linewidth=3, label="in-SynAD R²")
    ax1.set_xlabel(col_name, fontsize=20)
    ax1.set_ylabel("in-SynAD R²", fontsize=20)
    ax1.tick_params(axis="x", labelsize=20)
    ax1.tick_params(axis="y", labelsize=20)
    ax2 = ax1.twinx()
    sns.lineplot(data=df, x=col_name, y="coverage", color="#ff7f0e", linewidth=3, label="Coverage", ax=ax2)
    ax2.set_ylabel("Coverage", fontsize=20)
    ax2.tick_params(axis="y", labelsize=20)
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="best", fontsize=18)
    plt.title(f"{dataset_type}_{method_type}_for_{target}_hyperparam_with_R2_and_Coverage", pad=20, fontsize=20)
    plt.tight_layout()
    output_path = Path(__file__).parent / Path(
        f"SynAD_performance/{dataset_type}_{method_type}_for_{target}_hyperparam_with_R2_and_Coverage_plot.png"
    )
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    synad_heatmap_with_hyper_param("ULD", "KDE", "yield", "by_paper", "threshold", "expand_idx")
    synad_single_variance_with_hyper_param("ULD", "leverage", "yield", "by_paper", "threshold")
    synad_single_variance_with_hyper_param("ULD", "ensemble", "yield", "by_paper", "threshold")
    synad_single_variance_with_hyper_param("ULD", "BNN_probe", "yield", "by_paper", "threshold")
