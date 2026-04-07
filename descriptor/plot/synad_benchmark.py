from pathlib import Path
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import CubicSpline, BPoly

log_path = Path(__file__).parent / "logs/ullmann_with_ligand"

if __name__ == "__main__":
    dataset_type = "ULD"
    target_type = "yield"
    method = "XGB"
    split_mode = "by_year"
    synad_log_path = Path(__file__).parent / Path(f"../src/logs/{dataset_type}/SynAD")

    train_path = synad_log_path / Path(f"{method}_{target_type}_train_with_{split_mode}_partition.csv")
    test_path = synad_log_path / Path(f"{method}_{target_type}_test_with_{split_mode}_partition.csv")

    all_results_train = pd.read_csv(train_path)
    all_results_test = pd.read_csv(test_path)

    all_results_test.loc[:, "error"] = (all_results_test.loc[:, "predict"] - all_results_test.loc[:, "real"]).abs()
    all_results_test.sort_values(by="error", inplace=True, ascending=True)
    all_results_test.reset_index(drop=True, inplace=True)
    for i in range(all_results_test.shape[0]):
        all_results_test.loc[i, "R2_max"] = r2_score(all_results_test.loc[:i, "real"], all_results_test.loc[:i, "predict"])

    x = np.arange(all_results_test.shape[0]) / all_results_test.shape[0]

    performance_list = []
    for logfile in synad_log_path.glob(f"*_for_*_synad_hyperparam_search_with_{split_mode}_partition.csv"):
        df = pd.read_csv(logfile)
        df = df.loc[:, ["r2_iad", "coverage"]]
        df["type"] = logfile.stem.split("_")[0]
        performance_list.append(df.values)

    performance_array = np.concatenate(performance_list, axis=0).T
    performance_array = performance_array[:, performance_array[1].astype(float) >= 0.001]
    sns.set_theme(style="whitegrid", palette="Dark2", font_scale=1.2)
    plt.figure(figsize=(12, 6))
    sns.lineplot(x=x, y=all_results_test["R2_max"], color="#929292", label="Max value", linewidth=3)
    plt.fill_between(x, all_results_test["R2_max"], y2=-2, color="#929292", alpha=0.2)
    df_perf = pd.DataFrame(
        {"r2_iad": performance_array[0].astype(float), "coverage": performance_array[1].astype(float), "type": performance_array[2]}
    )

    model_colors = {
        "BNN": "#8198da",
        "KDE": "#c6262f",
        "SVM": "#75bb99",
        "ZKNN-chebyshev": "#bfcb45",
        "ZKNN-cityblock": "#5b97c8",
        "ZKNN-cosine": "#ebdd85",
        "ZKNN-euclidean": "#d68768",
        "ensemble": "#464b84",
        "leverage": "#e8a4b5",
    }

    model_names = {
        "BNN": "Bayesian NN",
        "KDE": "KDE",
        "SVM": "One-class SVM",
        "ZKNN-chebyshev": "ZKNN with Chebyshev distance",
        "ZKNN-cityblock": "ZKNN with Manhattan distance",
        "ZKNN-cosine": "ZKNN with cosine distance",
        "ZKNN-euclidean": "ZKNN with Euclidean distance",
        "ensemble": "Ensemble",
        "leverage": "Leverage",
    }

    for model_type in sorted(df_perf["type"].unique()):
        type_data = df_perf[df_perf["type"] == model_type]
        type_data = type_data.sort_values("coverage")
        agg_data = type_data.groupby("coverage").agg({"r2_iad": ["mean", "std", "count"]}).reset_index()
        agg_data.columns = ["coverage", "r2_mean", "r2_std", "count"]
        agg_data["r2_std"] = agg_data["r2_std"].fillna(0)
        x_new = np.linspace(agg_data["coverage"].min(), agg_data["coverage"].max(), 300)
        if len(agg_data) > 1:
            x_interp = np.linspace(agg_data["coverage"].min(), agg_data["coverage"].max(), len(agg_data) * 2)
            cs = CubicSpline(agg_data["coverage"], agg_data["r2_mean"])
            y_interp = cs(x_interp)
            control_points = np.column_stack([x_interp, y_interp])
            t = np.linspace(0, 1, len(x_interp))
            n = len(control_points) - 1
            coeffs = np.array([control_points[:, 1]]).T
            bezier_curve = BPoly(coeffs, [0, 1])
            y_smooth = bezier_curve(t)
            lw = 3 if model_type == "KDE" or model_type == "ZKNN-cityblock" else 2
            plt.plot(
                x_interp, y_smooth, color=model_colors.get(model_type, "black"), label=model_names.get(model_type, model_type), linewidth=lw
            )
        else:
            plt.plot(
                agg_data["coverage"],
                agg_data["r2_mean"],
                color=model_colors.get(model_type, "black"),
                label=model_names.get(model_type, model_type),
                linewidth=2,
            )

    plt.ylim(-0.3, 1.1) if split_mode != "by_year" else plt.ylim(-1.5, 1.1)
    plt.xlim(0, 1)
    plt.xlabel("IAD ratio")
    plt.ylabel("R2")
    plt.legend(loc="center left", bbox_to_anchor=(1.05, 0.5))
    plt.tight_layout()
    plt.savefig(Path(__file__).parent / Path(f"synad_benchmark_for_{split_mode}_split.png"), dpi=300)
    plt.close()
