from pathlib import Path
import pandas as pd
from sklearn.metrics import r2_score

from modules.data_load import load_reaction_data
from modules.desc_process import ReactionDesc
from synad.core.synad import SynADJudgementor
from synad.core.train_model import MLMethod
from predict_settings import data_info

dataset_type = "ULD"
model_type = "XGB"
reagent_type = "ligand1"
target_type = "yield"

synad_log_path = Path(__file__).parent / Path(f"logs/{dataset_type}/SynAD")
split_mode = "no_split"

SYNAD_PARAMS = {"hard": {"Z": 0.5, "k": 4, "metric": "cityblock"}, "easy": {"Z": 1.5, "k": 3, "metric": "cityblock"}}


data_df = load_reaction_data(dataset_type, target_type)  # load reaction data

reagent_columns = data_info[dataset_type]["reagent_columns"]
data_columns = data_info[dataset_type]["data_columns"]

data_df["date_seq"] = data_df["year"] * 12 + data_df["month"]
data_df = data_df.sort_values(by="date_seq").reset_index(drop=True)
data_df["paper_id"] = data_df.groupby("paper", sort=False).ngroup()

reaction_desc = ReactionDesc(data_df, reagent_columns=reagent_columns, data_columns=data_columns, dataset_type=dataset_type)
reaction_desc.generate_spoc_descriptor()
reaction_desc.load_reaction_desc()
X_data, y_data = reaction_desc.generate_descriptor_matrix(target_type)  # yield or ee
data_df = reaction_desc.data_df
data_df.reset_index(drop=True, inplace=True)

paper_info_columns = ["paper_id", "paper", "year", "month"]

synad_score_with_paper = pd.DataFrame({}, columns=["paper_id", "paper", "year", "month", "synad_score"])
for paper_id in data_df["paper_id"].unique():
    if paper_id == 0:
        continue
    X_train, y_train = X_data.loc[data_df["paper_id"] < paper_id], y_data.loc[data_df["paper_id"] < paper_id]
    X_test, y_test = X_data.loc[data_df["paper_id"] == paper_id], y_data.loc[data_df["paper_id"] == paper_id]
    ml_method = MLMethod(
        X_data.values,
        y_data.values,
        split_info=[(list(X_train.index), list(X_test.index))],
        model_name=model_type,
        dataset_type=dataset_type,
    )
    _, all_test_results = ml_method.model_train()
    r2_test = r2_score(all_test_results["real"], all_test_results["predict"])
    if not len(X_test) > 0 and len(X_train) > 0:
        continue
    synad_evaluator = SynADJudgementor(method_type="ZKNN")
    synad_evaluator.load_data(X_train.values, X_test.values, None)
    hard_synad_results = synad_evaluator.get_synad(SYNAD_PARAMS["hard"])
    easy_synad_results = synad_evaluator.get_synad(SYNAD_PARAMS["easy"])

    synad_score = (sum(easy_synad_results["AD_type"] == "IAD") - sum(hard_synad_results["AD_type"] == "IAD")) * 0.5
    synad_score += sum(easy_synad_results["AD_type"] == "OAD")
    synad_score = synad_score / len(X_test)
    synad_score = f"{synad_score:.3f}"

    paper_info = data_df.loc[data_df["paper_id"] == paper_id, paper_info_columns].head(1).squeeze()
    synad_score_with_paper.loc[paper_id, paper_info_columns] = paper_info
    synad_score_with_paper.loc[paper_id, "synad_score"] = synad_score
    synad_score_with_paper.loc[paper_id, "r2_test"] = r2_test

    print(
        f"for {synad_score_with_paper.loc[paper_id, 'paper']}, the average synad_score is {synad_score} ({paper_id}/{len(data_df['paper_id'].unique())})"
    )

synad_score_with_paper.to_csv(Path(__file__).parent / Path(f"logs/{dataset_type}/synad_score_with_year.csv"), index=False)
