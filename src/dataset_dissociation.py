from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.discriminant_analysis import StandardScaler
from modules.data_load import load_reaction_data
from modules.desc_process import ReactionDesc
from scipy.spatial.distance import cdist

from synad.utils_func import decomponent_reactions, generate_split_info


def matrix_kl_divergence(A, B, epsilon=1e-10):
    A_probs = A / A.sum(axis=1, keepdims=True)
    B_probs = B / B.sum(axis=1, keepdims=True)

    P_mean = np.mean(A_probs, axis=0)
    Q_mean = np.mean(B_probs, axis=0)
    P_mean = np.clip(P_mean, epsilon, None)
    Q_mean = np.clip(Q_mean, epsilon, None)

    P_mean /= P_mean.sum()
    Q_mean /= Q_mean.sum()
    kl_div = np.sum(P_mean * np.log(P_mean / Q_mean))

    return kl_div


def average_mean_distance(A, B):
    min_dist = np.mean(cdist(A, B, "euclidean"))
    return min_dist


def sigular_value_difference(A, B):
    s1, s2 = np.linalg.svd(A, full_matrices=False)[1], np.linalg.svd(B, full_matrices=False)[1]
    return abs(np.mean(s1) - np.mean(s2))


def relavance_dist(A, B):
    from scipy.spatial.distance import jensenshannon

    mean_dist_A, mean_dist_B = np.mean(A, axis=0), np.mean(B, axis=0)
    prob_dist_A = mean_dist_A / np.sum(mean_dist_A)
    prob_dist_B = mean_dist_B / np.sum(mean_dist_B)
    js_distance = jensenshannon(prob_dist_A, prob_dist_B)
    cov_A = np.cov(A, rowvar=False)
    cov_B = np.cov(B, rowvar=False)
    cov_distance = np.linalg.norm(cov_A - cov_B)
    return js_distance, cov_distance


def dist_eval(A, B):
    kl_divergence = matrix_kl_divergence(A, B)
    ave_mean_dist = average_mean_distance(A, B)
    sigular_value_diff = sigular_value_difference(A, B)
    js_dist, cov_dist = relavance_dist(A, B)

    print(f"KL divergence: {kl_divergence}")
    print(f"Average mean distance: {ave_mean_dist}")
    print(f"Singular value difference: {sigular_value_diff}")
    print(f"JS distance: {js_dist}")
    print(f"Covariance distance: {cov_dist}")

    return [kl_divergence, ave_mean_dist, sigular_value_diff, js_dist, cov_dist]


def calculate_dataset_dissociation(dataset_type, split_mode, n_splits, year_thresh=2015, do_prediction=False):
    """Calculate the dissociation of Training set and test set."""
    data_df = load_reaction_data(dataset_type)  # load reaction data, default droping duplicated reactions

    reagent_columns = ["ligand1", "ligand2", "metal", "reactant1", "reactant2", "product", "add1", "add2", "solv"]
    data_columns = ["cat_amount", "temperature", "time"]
    reaction_desc = ReactionDesc(data_df, reagent_columns=reagent_columns, data_columns=data_columns, dataset_type="ULD")
    reaction_desc.load_reaction_desc()
    X_data, y_data = reaction_desc.generate_descriptor_matrix("yield")
    data_df = reaction_desc.data_df

    split_info = generate_split_info(data_df, X_data, split_mode, n_splits=n_splits, year_thresh=2015)
    if do_prediction:
        from synad.core.train_model import MLMethod

        method = "XGB"
        ml_method = MLMethod(
            X_data=X_data.values, y_data=y_data.values, dataset_type=dataset_type, model_name=method, split_info=split_info
        )
        _, all_results_test = ml_method.model_train(verbose=True)
        r2_full_list = [r2_score(group["real"], group["predict"]) for _, group in all_results_test.groupby("fold_id")]
        ml_method_spoc = MLMethod(
            X_data[[x for x in X_data.columns if "SPOC" in x]].values,
            y_data=y_data.values,
            dataset_type=dataset_type,
            model_name=method,
            split_info=split_info,
        )
        _, all_results_test_spoc = ml_method_spoc.model_train(verbose=True)
        r2_spoc_list = [r2_score(group["real"], group["predict"]) for _, group in all_results_test_spoc.groupby("fold_id")]

        ml_method_qpoc = MLMethod(
            X_data[[x for x in X_data.columns if "QM" in x or "solv_param" in x]].values,
            y_data=y_data.values,
            dataset_type=dataset_type,
            model_name=method,
            split_info=split_info,
        )
        _, all_results_test_qpoc = ml_method_qpoc.model_train(verbose=True)
        r2_qpoc_list = [r2_score(group["real"], group["predict"]) for _, group in all_results_test_qpoc.groupby("fold_id")]

        logpath = Path(__file__).parent / Path(f"logs/{dataset_type}/divide_by_{split_mode}.log")
        with open(logpath, "w") as f:
            f.write(f"R2 for full descriptor: {r2_full_list}\n")
            f.write(f"R2 for SPOC descriptor: {r2_spoc_list}\n")
            f.write(f"R2 for QPOC descriptor: {r2_qpoc_list}\n")

    X_train, X_test = X_data.values[split_info[0][0]], X_data.values[split_info[0][1]]
    X_scaler = StandardScaler()
    X_train = X_scaler.fit_transform(X_train)
    X_test = X_scaler.transform(X_test)

    X_decomped = decomponent_reactions(X_train, X_test)
    metrics = dist_eval(
        X_decomped[X_decomped["type"] == "training"].loc[:, ["x", "y"]].values,
        X_decomped[X_decomped["type"] == "test"].loc[:, ["x", "y"]].values,
    )

    return metrics


if __name__ == "__main__":
    metrics_name_list = ["kl_divergence", "average_mean_distance", "sigular_value_difference", "js_distance", "cov_distance"]
    split_mode_list = ["random", "by_ligand1", "by_paper", "by_year", "in"]
    metrics_list = []
    for split_mode in split_mode_list:
        metrics_list.append(
            calculate_dataset_dissociation(dataset_type="ULD", split_mode=split_mode, n_splits=5, year_thresh=2015, do_prediction=False)
        )

    df = pd.DataFrame(metrics_list, columns=metrics_name_list, index=split_mode_list)
    df.to_csv(Path(__file__).parent / Path("logs/ULD/dataset_dissociation.csv"))
