from pathlib import Path
import pandas as pd
import numpy as np
import synad
from modules.data_load import load_reaction_data
from modules.desc_process import ReactionDesc
from predict_settings import data_info


def load_data(dataset_type, target_type, synad_log_path, split_mode, retrain=True):
    """Load and prepare data for SynAD evaluation using data processing modules."""
    data_df = load_reaction_data(dataset_type, target_type)  # load reaction data

    reagent_columns = data_info[dataset_type]["reagent_columns"]
    data_columns = data_info[dataset_type]["data_columns"]
    reaction_desc = ReactionDesc(data_df, reagent_columns=reagent_columns, data_columns=data_columns, dataset_type=dataset_type)
    reaction_desc.generate_spoc_descriptor()
    reaction_desc.load_reaction_desc()
    X_data, y_data = reaction_desc.generate_descriptor_matrix(target_type)  # yield or ee
    data_df = reaction_desc.data_df

    # Use synad package function for split generation
    n_splits = 5
    split_info = synad.generate_split_info(data_df, X_data, split_mode, n_splits=n_splits, year_thresh=2015)

    if not synad_log_path.exists():
        synad_log_path.mkdir(parents=True, exist_ok=True)

    method = "XGB"
    train_path = synad_log_path / Path(f"{method}_{target_type}_train_with_{split_mode}_partition.csv")
    test_path = synad_log_path / Path(f"{method}_{target_type}_test_with_{split_mode}_partition.csv")

    if not (train_path.exists() and test_path.exists()) or retrain:
        # Use synad package function for model training
        all_results_train, all_results_test = synad.train_and_evaluate_model(
            X_data.values, y_data.values, split_info, model_name=method, dataset_type=dataset_type, verbose=True
        )
        all_results_train.to_csv(train_path, index=False)
        all_results_test.to_csv(test_path, index=False)
    all_results_train = pd.read_csv(train_path)
    all_results_test = pd.read_csv(test_path)

    # Use synad package function for model performance evaluation
    train_metrics = synad.evaluate_model_performance(all_results_train["real"], all_results_train["predict"])
    test_metrics = synad.evaluate_model_performance(all_results_test["real"], all_results_test["predict"])

    print(f"for {method} model,")
    print(f"    r2_train: {train_metrics['r2']}, r2_test: {test_metrics['r2']}")
    print(f"    mae_train: {train_metrics['mae']}, mae_test: {test_metrics['mae']}")
    print(f"    rmse_train: {train_metrics['rmse']}, rmse_test: {test_metrics['rmse']}")

    # Use synad package function for feature importance calculation
    importance_lists_filepath = synad_log_path / f"{method}_{target_type}_importance_with_{split_mode}_partition.csv"
    if not Path(importance_lists_filepath).exists():
        importance_lists = synad.calculate_feature_importance(
            X_data.values, y_data.values, split_info, model_name=method, dataset_type=dataset_type
        )
        np.savetxt(importance_lists_filepath, importance_lists, delimiter=",")
    else:
        importance_lists = np.loadtxt(importance_lists_filepath, delimiter=",")

    return X_data, y_data, all_results_test, split_info, importance_lists


def synad_hyperparam_opt(hyperparams_range, split_mode, dataset_type="ULD", target_type="yield", method_type="ZKNN", max_num=100):
    """Perform SynAD hyperparameter optimization using synad package functions."""
    synad_log_path = Path(__file__).parent / Path(f"../logs/{dataset_type}/SynAD")

    X_data, y_data, all_results_test, split_info, importance_lists = load_data(dataset_type, target_type, synad_log_path, split_mode)
    all_results_test = all_results_test.set_index("idx").sort_index() if "idx" in all_results_test.columns else all_results_test

    # Use synad package function for hyperparameter optimization
    results_df = synad.synad_hyperparameter_optimization(
        X_data.values,
        y_data,
        all_results_test["predict"].values,
        split_info,
        opt_range=hyperparams_range,
        importance_lists=importance_lists,
        method_type=method_type,
        opt_method="hyperopt",
        max_num=max_num,
    )
    return results_df


def single_synad_evaluation(hyperparams, split_mode, dataset_type="ULD", target_type="yield", method_type="ZKNN"):
    """Perform single SynAD evaluation using synad package functions."""
    synad_log_path = Path(__file__).parent / Path(f"../logs/{dataset_type}/SynAD")

    X_data, y_data, all_results_test, split_info, importance_lists = load_data(dataset_type, target_type, synad_log_path, split_mode)
    all_results_test = all_results_test.set_index("idx").sort_index() if "idx" in all_results_test.columns else all_results_test
    # Use synad package function for SynAD evaluation
    r2_iad, r2_oad, coverage = synad.single_synad_evaluation(
        X_data.values,
        y_data,
        all_results_test["predict"].values,
        split_info,
        split_mode,
        importance_lists,
        hyperparams=hyperparams,
        method_type=method_type,
        dataset_type=dataset_type,
        save_results_df=True,
    )
    print(f"SynAD Results: r2_iad={r2_iad}, r2_oad={r2_oad}, coverage={coverage}")
    return r2_iad, r2_oad, coverage


def synad_eval(opt_range, split_mode, dataset_type="ULD", target_type="yield", method_type="ZKNN", max_num=100):
    """Perform SynAD hyperparameter optimization using synad package functions."""
    synad_log_path = Path(__file__).parent / Path(f"../logs/{dataset_type}/SynAD")

    X_data, y_data, all_results_test, split_info, importance_lists = load_data(dataset_type, target_type, synad_log_path, split_mode)
    all_results_test = all_results_test.set_index("idx").sort_index() if "idx" in all_results_test.columns else all_results_test

    # Use synad package function for hyperparameter optimization
    results_df = synad.synad_hyperparameter_optimization(
        X_data.values,
        y_data,
        all_results_test["predict"].values,
        split_info,
        opt_range,
        importance_lists,
        method_type=method_type,
        max_num=max_num,
    )

    # Save results
    results_df.to_csv(synad_log_path / f"{method_type}_for_{target_type}_synad_hyperparam_search_with_{split_mode}_partition.csv")
    return results_df


if __name__ == "__main__":
    ZKNN_opt_range = {"Z": np.arange(0.5, 5.1, 0.5), "k": np.arange(1, 10, 1), "metric": ["cityblock"]}
    synad_eval(ZKNN_opt_range, split_mode="by_paper", dataset_type="ULD", target_type="yield", method_type="ZKNN")
