from pathlib import Path
import pandas as pd
from loguru import logger

global_data_path = Path(__file__).parent / Path("../../data")
ULD_subset = ["ligand1", "ligand2", "reactant1", "reactant2", "product", "add1", "add2", "solv", "temperature", "time", "cat_amount"]


def cl(s):
    return "blank_cell" if s == "blank_cell" else s + "."


def excel_loader(filepath: str, target_prop: str) -> pd.DataFrame:
    """Used to read excel's file contents and delete select_Tag's False content\n
    meanwhile generate 'l123' and 'r12' data.
    """
    data = pd.read_excel(filepath, dtype=object, index_col=False)

    logger.info("reading data...")
    logger.info("--------> length of original data: " + str(len(data)))

    if "select_Tag" in data.columns:
        data["select_Tag"] = data["select_Tag"].astype(bool)
        data = data[data["select_Tag"] == True]
    data = data.dropna(subset=[target_prop])
    data.reset_index(drop=True, inplace=True)
    logger.info("--------> length of cleaned data: " + str(len(data)))

    data["r12"] = ""
    if "reactant1" in data.columns and "reactant2" in data.columns:
        for i in data.index:
            data.loc[i, "r12"] = cl(data.loc[i, "reactant1"]) + cl(data.loc[i, "reactant2"])
            data.loc[i, "r12"] = data.loc[i, "r12"][0:-1]

    return data


def process_duplicates(group, target_prop, keep_threshold=15):
    if len(group) == 1:
        return group
    else:
        prod_diff = group[target_prop].max() - group[target_prop].min()
        if prod_diff < keep_threshold:
            first_row = group.iloc[0].copy()
            first_row[target_prop] = group[target_prop].mean()
            return pd.DataFrame([first_row])
        else:
            return pd.DataFrame()


def load_reaction_data(dataset_type="ULD", target_prop="yield", drop_duplicate=False, keep_threshold=15):
    """Load and process reaction data from excel file.

    Args:
        data_file (str): Name of excel file (without extension).
        dataset_type (str, optional): Dataset type identifier. Defaults to "ULD".
        drop_duplicate (bool, optional): Whether to drop duplicates. Defaults to False.
        target_prop (str, optional): Target property column name. Defaults to "yield".
        keep_threshold (int, optional): Threshold for keeping duplicates. Defaults to 15.

    Returns:
        pd.DataFrame: Processed reaction data.
    """
    data_df = excel_loader(global_data_path / Path(f"{dataset_type}.xlsx"), target_prop)
    if drop_duplicate:
        duplicates = data_df.duplicated(subset=ULD_subset, keep=False)
        processed_groups = data_df[duplicates].groupby(ULD_subset).apply(lambda x: process_duplicates(x, target_prop, keep_threshold))
        unique_rows = data_df[~duplicates]
        processed_rows = processed_groups.reset_index(drop=True)
        data_df = pd.concat([unique_rows, processed_rows], ignore_index=True)

    logger.info(f"shape of DataFrame: {data_df.shape}")
    return data_df
