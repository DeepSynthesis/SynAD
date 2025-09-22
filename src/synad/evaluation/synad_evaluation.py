"""
SynAD evaluation functions that operate on provided data matrices.

This module provides functions for evaluating SynAD performance without
dependencies on specific data loading or description generation modules.
"""

import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error

from ..core.train_model import MLMethod
from ..core.synad import SynADJudgementor
from ..utils_func import SHAP_value_calculation


def single_synad_evaluation(
    X_data,
    y_data,
    y_pred_data,
    split_info,
    split_mode,
    importance_lists=None,
    hyperparams=None,
    method_type="ZKNN",
    dataset_type="custom",
    save_results_df=False,
):
    """
    Perform single SynAD evaluation on provided data matrices.

    Args:
        X_data (np.ndarray): Feature matrix
        y_data (np.ndarray): Target values
        y_pred_data (np.ndarray): Predicted values
        split_info (list): List of (train_idx, test_idx) tuples for cross-validation
        importance_lists (np.ndarray, optional): Feature importance lists for each fold
        hyperparams (dict): Hyperparameters for SynAD method
        method_type (str): SynAD method type (default: "ZKNN")
        dataset_type (str): Dataset identifier for logging
        save_results_df (bool): Whether to save results to CSV

    Returns:
        tuple: (r2_iad, r2_oad, coverage) metrics
    """
    if hyperparams is None:
        hyperparams = {}

    synad = SynADJudgementor(method_type=method_type, n_jobs=24)
    synad.load_data_kfold(X_data, y_data, y_pred_data, split_info, importance_lists)

    r2_iad, r2_oad, coverage = synad.kfold_get_synad(
        hyperparams, save_results_df=save_results_df, dataset_type=dataset_type, split_mode=split_mode
    )

    return r2_iad, r2_oad, coverage


def synad_hyperparameter_optimization(
    X_data, y_data, y_pred_data, split_info, opt_range, importance_lists=None, method_type="ZKNN", opt_method="grid_search", max_num=100
):
    """
    Perform hyperparameter optimization for SynAD methods.

    Args:
        X_data (np.ndarray): Feature matrix
        y_data (np.ndarray): Target values
        y_pred_data (np.ndarray): Predicted values
        split_info (list): List of (train_idx, test_idx) tuples for cross-validation
        importance_lists (np.ndarray, optional): Feature importance lists for each fold
        opt_range (dict): Hyperparameter optimization ranges
        method_type (str): SynAD method type (default: "ZKNN")
        dataset_type (str): Dataset identifier for logging
        max_num (int): Maximum number of optimization iterations

    Returns:
        pd.DataFrame: Hyperparameter optimization results
    """

    synad = SynADJudgementor(method_type=method_type, n_jobs=24)
    synad.load_data_kfold(X_data, y_data, y_pred_data, split_info, importance_lists)

    results_df = synad.hyper_param_search(opt_range, opt_method=opt_method, max_num=max_num, verbose=True)

    return results_df


def train_and_evaluate_model(X_data, y_data, split_info, model_name="XGB", dataset_type="custom", verbose=True):
    """
    Train a model and return predictions for SynAD evaluation.

    Args:
        X_data (np.ndarray): Feature matrix
        y_data (np.ndarray): Target values
        split_info (list): List of (train_idx, test_idx) tuples for cross-validation
        model_name (str): Name of the model to train (default: "XGB")
        dataset_type (str): Dataset identifier
        verbose (bool): Whether to print training progress

    Returns:
        tuple: (all_results_train, all_results_test) DataFrames
    """
    ml_method = MLMethod(X_data=X_data, y_data=y_data, dataset_type=dataset_type, model_name=model_name, split_info=split_info)
    all_results_train, all_results_test = ml_method.model_train(verbose=verbose)

    return all_results_train, all_results_test


def calculate_feature_importance(X_data, y_data, split_info, model_name="XGB", dataset_type="custom"):
    """
    Calculate feature importance using SHAP for each fold in cross-validation.

    Args:
        X_data (np.ndarray): Feature matrix
        y_data (np.ndarray): Target values
        split_info (list): List of (train_idx, test_idx) tuples for cross-validation
        model_name (str): Name of the model to train (default: "XGB")
        dataset_type (str): Dataset identifier

    Returns:
        np.ndarray: Feature importance lists for each fold
    """
    ml_method = MLMethod(X_data=X_data, y_data=y_data, dataset_type=dataset_type, model_name=model_name, split_info=split_info)
    ml_method.model_train(verbose=False)
    models, scalers = ml_method.get_model()

    importance_lists = []
    for (train_idx, _), model, scaler in zip(split_info, models, scalers):
        sub_X_train = X_data[train_idx]
        importance_lists.append(SHAP_value_calculation(model, sub_X_train, scaler=scaler))

    return np.array(importance_lists)


def evaluate_model_performance(y_true, y_pred):
    """
    Calculate standard regression metrics.

    Args:
        y_true (array-like): True values
        y_pred (array-like): Predicted values

    Returns:
        dict: Dictionary containing r2, mae, and rmse metrics
    """
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)

    return {"r2": round(r2, 3), "mae": round(mae, 3), "rmse": round(rmse, 3)}
