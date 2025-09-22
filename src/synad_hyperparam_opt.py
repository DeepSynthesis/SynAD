import numpy as np
from modules.synad_eval import synad_hyperparam_opt

opt_range = {
    "Z": np.arange(-1, 5, 0.02),
    "k": np.arange(1, 10, 1),
    "metric": ["cityblock", "cosine", "euclidean"],
}

output_df = synad_hyperparam_opt(opt_range, split_mode="by_ligand_smiles", dataset_type="B-H_HTE", target_type="yield", method_type="ZKNN")
output_df.to_csv("output.csv")
