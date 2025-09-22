from pathlib import Path
from synad.core.synad import SynADJudgementor
from plot.decomponent import decomponent_reactions, plot_decomponent
from modules.synad_eval import load_data

dataset_type = "ULD"
target_type = "yield"
synad_log_path = synad_log_path = Path(__file__).parent / Path(f"logs/{dataset_type}/SynAD")
split_mode = "by_paper"

X_data, y_data, all_results_test, split_info, importance_lists = load_data(dataset_type, target_type, synad_log_path, split_mode)
synad = SynADJudgementor(method_type="ZKNN", n_jobs=24)

synad.load_data(X_data.loc[split_info[0][0]], X_data.loc[split_info[0][1]], importance_lists[0])
output_df = synad.get_synad({"Z": 0.5, "k": 4, "metric": "cityblock"})

output_df["index"] = split_info[0][1]
X_decomped = decomponent_reactions(X_data)
X_decomped.loc[:, "type"] = "blank_cell"
X_decomped.loc[split_info[0][0], "type"] = "training"
X_decomped.loc[output_df[output_df["AD_type"] == "IAD"]["index"], "type"] = "inner synad"
X_decomped.loc[output_df[output_df["AD_type"] == "OAD"]["index"], "type"] = "out of synad"
plot_decomponent(X_decomped, "ULD", palette=["#5b97c892", "#c6262e8e", "#464b8425"], remarks="synad-visual")
