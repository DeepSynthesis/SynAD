from pathlib import Path
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error

from modules.data_load import load_reaction_data
from modules.desc_process import ReactionDesc
from synad.core.train_model import MLMethod
from synad.utils_func import generate_split_info
from predict_settings import data_info


def train_models(X_data, y_data, dataset_type, method, split_info, target):
    ml_method = MLMethod(X_data=X_data.values, y_data=y_data.values, dataset_type=dataset_type, model_name=method, split_info=split_info)
    all_results_train, all_results_test = ml_method.model_train(verbose=True, supervise_eval=True)
    all_results_train.to_csv(Path(__file__).parent / Path(f"logs/{dataset_type}/{method}_{target}_train.csv"), index=False)
    all_results_test.to_csv(Path(__file__).parent / Path(f"logs/{dataset_type}/{method}_{target}_test.csv"), index=False)
    return ml_method


def dataset_prediction(dataset_type="ULD", target_type="yield", split_mode="random", method="XGB", do_hyperopt=False):
    # ULD, B-H_HTE, xxx
    data_df = load_reaction_data(dataset_type, target_type)  # load reaction data
    reagent_columns = data_info[dataset_type]["reagent_columns"]
    data_columns = data_info[dataset_type]["data_columns"]
    reaction_desc = ReactionDesc(data_df, reagent_columns=reagent_columns, data_columns=data_columns, dataset_type=dataset_type)
    # reaction_desc.generate_spoc_descriptor()
    reaction_desc.load_reaction_desc()
    X_data, y_data = reaction_desc.generate_descriptor_matrix(target_type)  # yield or ee
    data_df = reaction_desc.data_df

    # split
    n_splits = 10
    split_info = generate_split_info(data_df, X_data, split_mode, n_splits=n_splits)

    train_path = Path(__file__).parent / Path(f"logs/{dataset_type}/{method}_{target_type}_train.csv")
    test_path = Path(__file__).parent / Path(f"logs/{dataset_type}/{method}_{target_type}_test.csv")

    ml_method = train_models(X_data, y_data, dataset_type, method, split_info, target_type)

    all_results_train = pd.read_csv(train_path)
    all_results_test = pd.read_csv(test_path)
    r2_train = r2_score(all_results_train["real"], all_results_train["predict"])
    r2_test = r2_score(all_results_test["real"], all_results_test["predict"])
    mae_train = mean_absolute_error(all_results_train["real"], all_results_train["predict"])
    mae_test = mean_absolute_error(all_results_test["real"], all_results_test["predict"])
    rmse_train = root_mean_squared_error(all_results_train["real"], all_results_train["predict"])
    rmse_test = root_mean_squared_error(all_results_test["real"], all_results_test["predict"])

    print(f"for {method} model,")
    print(f"    r2_train: {r2_train:.2f}, r2_test: {r2_test:.2f}")
    print(f"    mae_train: {mae_train:.2f}, mae_test: {mae_test:.2f}")
    print(f"    rmse_train: {rmse_train:.2f}, rmse_test: {rmse_test:.2f}")


if __name__ == "__main__":
    dataset_prediction(dataset_type="ULD", target_type="yield", split_mode="random", method="XGB", do_hyperopt=False)
