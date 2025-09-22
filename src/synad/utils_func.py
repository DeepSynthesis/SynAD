"""
Utility functions for SynAD package.
Contains general-purpose utilities that don't depend on specific data formats.
"""

from loguru import logger
from rdkit import Chem
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold
import shap


def results_generate(res: dict, prefix: str, key_selector=None, res_dev: dict = {}) -> str:
    """Generate results string from a dict"""
    res_str = ""
    key_list = list(res.keys())[0:key_selector]
    if res_dev == {}:
        for i, k in enumerate(key_list):
            res_str += f"{prefix}_{k}: {res[k]}, " if i + 1 < len(key_list) else f"{prefix}_{k}: {res[k]}"
    else:
        for i, k in enumerate(key_list):
            res_str += f"{prefix}_{k}: {res[k]}±{res_dev[k]}, " if i + 1 < len(key_list) else f"{prefix}_{k}: {res[k]}±{res_dev[k]}"
    return res_str


def SMILES_canonicalization(smiles: str):
    """Canonicalize SMILES string using RDKit"""
    try:
        x = Chem.MolFromSmiles(smiles)
        return Chem.MolToSmiles(x, isomericSmiles=True)
    except:
        return smiles


def split_by_type(data_df, n_splits, tp):
    tp = tp.split("_", 1)[-1]

    if n_splits > len(data_df[tp].unique()):
        n_splits = len(data_df[tp].unique())
        logger.warning(f"n_splits is too large, set to {n_splits}")

    data_df = data_df.reset_index(drop=True)
    if "H-" in tp or "Z-" in tp:
        new_paper_tag = [tp]
        matches = data_df["paper"].isin(new_paper_tag)
        test_index = data_df[matches].index
        train_index = data_df.drop(test_index).index
        assert test_index.shape[0] > 0, "No data found for the specified paper tag."
        return [(train_index, test_index)]

    paper_list = data_df[tp].unique()
    np.random.seed(42)
    np.random.shuffle(paper_list)
    split_indices = np.linspace(0, len(paper_list), num=n_splits + 1, dtype=int)
    paper_lists = [paper_list[start:end] for start, end in zip(split_indices[:-1], split_indices[1:])]
    index_lists = []
    for i, paper in enumerate(paper_lists):
        test_index = data_df[data_df[tp].isin(paper)].index
        train_index = data_df.drop(test_index).index
        index_lists.append((train_index, test_index))
    return index_lists


def generate_split_info(data_df, X_data, split_mode, n_splits=5, rd_state=42, year_thresh=0):
    """
    Generate split information based on the specified splitting mode.

    Parameters:
        data_df (DataFrame): The dataset containing metadata or index for splitting.
        X_data (array-like): The feature matrix used for splitting.
        split_mode (str): The mode of splitting, options include:
                          - "random": Random split using KFold.
                          - "no_split": No splitting, use the full dataset as one fold.
        n_splits (int): Number of splits for the KFold (default=5).
        rd_state (int): Random state for reproducibility (default=42).

    Returns:
        list: A list of tuples containing train and test indices for each split.

    Raises:
        Exception: If the provided split_mode is not recognized.
    """
    if split_mode == "random":
        kf = KFold(n_splits=n_splits, random_state=rd_state, shuffle=True)
        split_info = list(kf.split(X_data))
    elif split_mode == "no_split":
        split_info = [(data_df.index, [])]
    elif split_mode == "by_year":
        split_info = [(data_df[data_df["year"] < year_thresh].index, data_df[data_df["year"] >= year_thresh].index)]
    elif "by_" in split_mode:
        split_info = split_by_type(data_df=data_df, n_splits=n_splits, tp=split_mode)
    else:
        raise Exception(f"ERROR: no split mode called: {split_mode}.")

    return split_info


def metric_cal(y_true: list, y_pred: list, precise: int = 3) -> dict:
    """Calculate RMSE, MAE and R^2

    Args:
        y_true (list): The real number list
        y_pred (list): The predict number list
        precise (int, optional): The number of digits after the decimal point. Defaults to 3.

    Returns:
        dict: RMSE, MAE, R^2 values
    """
    rmse = round(float(np.sqrt(mean_squared_error(y_true, y_pred))), precise)
    mae = round(float(mean_absolute_error(y_true, y_pred)), precise)
    r2 = round(float(r2_score(y_true, y_pred)), precise)
    return {"r2": r2, "mae": mae, "rmse": rmse}


def SHAP_value_calculation(model, X_train, scaler=None):
    """
    Calculate SHAP values for the given model and training data.

    Parameters:
    model: The trained machine learning model.
    X_train: The training data used to fit the model.
    scaler: Optional scaler to transform the data.

    Returns:
    shap_values: SHAP values for each feature in the training data.
    """
    if scaler != None:
        X_train = scaler.transform(X_train)
    else:
        logger.warning("No scaler is provided, using raw data")

    # Initialize the SHAP explainer
    explainer = shap.TreeExplainer(model, X_train)

    # Calculate SHAP values
    shap_values = explainer(X_train).values
    importance_list = np.mean(np.abs(shap_values), axis=0)
    importance_list = importance_list.reshape(-1, 1)
    importance_list = MinMaxScaler().fit_transform(importance_list)
    importance_list = importance_list.flatten()

    return importance_list


def decomponent_reactions(X_train, X_test=None):
    """Decompose reactions using UMAP"""
    from umap import UMAP

    logger.info(f"Start UMAP decomposition...")
    decomp_model = UMAP(n_components=2, n_jobs=1, verbose=0, n_neighbors=15, min_dist=0.9, random_state=42)
    X_train_decomp = decomp_model.fit_transform(X_train)
    X_test_decomp = decomp_model.transform(X_test) if X_test is not None else None
    X_train_decomp_df = pd.DataFrame(X_train_decomp, columns=["x", "y"])
    X_test_decomp_df = pd.DataFrame(X_test_decomp, columns=["x", "y"])
    X_train_decomp_df["type"] = "training"
    X_test_decomp_df["type"] = "test"
    X_decomp = pd.concat([X_train_decomp_df, X_test_decomp_df], axis=0)
    return X_decomp
