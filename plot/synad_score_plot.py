from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_style("whitegrid")

dataset_type = "ULD"
log_path = Path(__file__).parent / f"../src/logs/{dataset_type}/"
output_dir = Path(__file__).parent / "synad_score_plots"
output_dir.mkdir(parents=True, exist_ok=True)
colors = ["#5b97c8", "#c6262f"]

category_order = [
    "acetone_imide",
    "acetylacetone",
    "alcohol_amine",
    "amino_acid",
    "bipyridine",
    "carbine",
    "diimide",
    "normal_bialcohol",
    "normal_biamine",
    "others",
    "oxalamide",
    "oxalamidic_acid",
    "phenanthroline",
    "phosphorus",
]


category_colors = {
    "acetone_imide": "#535353",
    "acetylacetone": "#535353",
    "alcohol_amine": "#535353",
    "amino_acid": "#535353",
    "bipyridine": "#464b84",
    "carbine": "#5b97c8",
    "diimide": "#c6262f",
    "normal_bialcohol": "#464b84",
    "normal_biamine": "#464b84",
    "others": "#464b84",
    "oxalamide": "#5b97c8",
    "oxalamidic_acid": "#5b97c8",
    "phenanthroline": "#464b84",
    "phosphorus": "#c6262f",
}


def synad_score_plot_with_ligand(filename):
    data = pd.read_csv(log_path / filename)

    for target_type in ["predict", "synad_score"]:
        plt.figure(figsize=(8, 5), dpi=300)

        sns.boxplot(
            x="category",
            y=target_type,
            data=data,
            order=category_order,
            palette=[category_colors[cat] for cat in category_order],
            width=0.6,
            linewidth=1,
            flierprops={"marker": "o", "markersize": 4, "markerfacecolor": "0.4", "markeredgewidth": 1},
        )

        plt.ylabel("SynAD score", fontsize=12)
        plt.xlabel("Ligand category", fontsize=12)
        plt.xticks(rotation=35, ha="right", fontsize=10)
        plt.yticks(fontsize=10)

        plt.grid(axis="y", alpha=0.4)
        plt.tight_layout()
        plt.savefig(output_dir / f"{target_type}_plot.png", bbox_inches="tight", dpi=300)
        plt.close()


def synad_score_plot_with_year(filename):
    data = pd.read_csv(log_path / filename)
    data["time"] = data["year"] + (data["month"] - 1) / 12
    data["synad_score_label"] = data["synad_score"].apply(lambda x: "High SynAD score" if x > 0.5 else "Low SynAD score")

    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    sns.set_style("white")
    ax.axhspan(0, 0.5, color="#D0D0D0", alpha=0.3)
    sns.scatterplot(
        data=data,
        x="time",
        y="synad_score",
        hue="synad_score_label",
        s=100,
        edgecolor="black",
        linewidth=1,
        palette=colors,
        legend=False,
    )

    min_year = int(data["year"].min())
    max_year = int(data["year"].max())
    plt.xticks(
        ticks=np.arange(min_year, max_year + 1, 2),
        labels=np.arange(min_year, max_year + 1, 2),
        fontsize=11,
        rotation=45 if (max_year - min_year) > 10 else 0,
    )

    plt.ylabel("SynAD score", fontsize=12)
    plt.xlabel("Publication year", fontsize=12)
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)

    plt.grid(alpha=0.0)
    plt.axhline(y=0.5, color="gray", linestyle="--", linewidth=1)
    plt.tight_layout()
    plt.savefig(output_dir / "synad_score_with_year.png", bbox_inches="tight", dpi=300)
    plt.close()


def synad_score_boxplot_by_year(filename):
    data = pd.read_csv(log_path / filename)
    data["year"] = data["year"].astype(int)
    plt.rcParams.update({"axes.edgecolor": "0.3", "axes.linewidth": 0.8, "grid.color": "0.9", "font.size": 11})
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    sns.set_style("white")
    sns.boxplot(
        data=data,
        x="year",
        y="synad_score",
        linewidth=1.2,
        width=0.6,
        flierprops={"marker": "o", "markerfacecolor": "gray", "markersize": 4, "markeredgecolor": "none"},
    )

    ax.axhspan(0, 0.5, color="#D0D0D0", alpha=0.3)
    plt.axhline(y=0.5, color="gray", linestyle="--", linewidth=1)

    plt.ylabel("SynAD score Score", fontsize=12)
    plt.xlabel("Publication Year", fontsize=12)
    plt.yticks(fontsize=11)

    plt.tight_layout()
    plt.savefig(output_dir / "synad_score_boxplot_by_year.png", bbox_inches="tight", dpi=300)
    plt.close()


def synad_score_with_year_and_r2(filename, x_thresh, y_thresh):
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    data = pd.read_csv(log_path / filename)
    data["r2_test"] = data["r2_test"].apply(lambda x: max(x, 0))
    data["synad_score_label"] = data["synad_score"].apply(lambda x: "High SynAD score" if x > y_thresh else "Low SynAD score")

    plt.rcParams.update({"axes.edgecolor": "0.3", "axes.linewidth": 0.8, "grid.color": "0.9", "font.size": 11})
    colors = {"High SynAD score": "#5b97c8", "Low SynAD score": "#c6262f"}
    fig, ax = plt.subplots(figsize=(5.5, 5), dpi=300)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.axvspan(0, x_thresh, color="#D0D0D0", alpha=0.3)
    sns.set_style("white")
    sns.scatterplot(
        data=data, x="synad_score", y="r2_test", hue="synad_score_label", palette=colors, s=100, edgecolor="black", linewidth=1, ax=ax
    )

    ax.set_xlabel("SynAD score", fontsize=12)
    ax.set_ylabel("R² score", fontsize=12)
    ax.grid(alpha=0.4)
    ax.legend().remove()
    ax.tick_params(axis="both", labelsize=11)
    plt.xlim(-0.1, 1.1)
    plt.ylim(-0.1, 1.1)
    plt.tight_layout()
    plt.savefig(output_dir / "synad_score_combined.png", bbox_inches="tight", dpi=300)
    plt.close()


if __name__ == "__main__":
    synad_score_plot_with_ligand("expanded_data/ligand1_synad_score_results.csv")
    synad_score_plot_with_year("synad_score_with_year.csv")
    synad_score_boxplot_by_year("synad_score_with_year.csv")
    synad_score_with_year_and_r2("synad_score_with_year.csv", 0.5, 0.5)
