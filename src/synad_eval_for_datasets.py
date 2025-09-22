from pathlib import Path
import pandas as pd
from modules.synad_eval import single_synad_evaluation

eval_hyperparam = {
    "CPA_lit": [
        ("by_ligand_smiles", "ddG", {"Z": 5.092, "k": 33, "metric": "cosine"}),
    ],
    "NiCOlit": [
        ("by_doi", "yield", {"Z": 1.414, "k": 1, "metric": "cosine"}),
    ],
    "Pd-catalyzed_lit": [
        ("by_literature_id", "yield", {"Z": 4.157, "k": 37, "metric": "cosine"}),
    ],
    "B-H_HTE": [
        ("by_ligand_smiles", "yield", {"Z": 2.494, "k": 5, "metric": "cityblock"}),
    ],
    "CPA_HTE": [
        ("by_catalyst", "ddG", {"Z": 5.879, "k": 7, "metric": "euclidean"}),
    ],
    "Suzuki_HTE": [
        ("by_ligand_smiles", "yield", {"Z": 13.571, "k": 23, "metric": "cityblock"}),
    ],
}

result_df = pd.DataFrame({})
for dataset, test_examples in eval_hyperparam.items():
    for example in test_examples:
        print(f"Now computing {example[0]} for {dataset}")
        result = single_synad_evaluation(
            hyperparams=example[2], split_mode=example[0], dataset_type=dataset, target_type=example[1], method_type="ZKNN"
        )
        result_df.loc[dataset, f"in-SynAD-R2"] = result[0]
        result_df.loc[dataset, f"coverage"] = result[2]

result_df.to_csv(Path(__file__).parent / "logs/ULD/synad_eval_other_datasets.csv")
