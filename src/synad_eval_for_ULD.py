from pathlib import Path
import pandas as pd
from modules.synad_eval import single_synad_evaluation

eval_hyperparam = {
    "ZKNN": [
        ("random", {"Z": 0.862, "k": 35, "metric": "cityblock"}),
        ("by_ligand1", {"Z": 1.239, "k": 25, "metric": "euclidean"}),
        ("by_paper", {"Z": 0.500, "k": 4, "metric": "cityblock"}),
        ("by_year", {"Z": 1.193, "k": 7, "metric": "cityblock"}),
    ],
    "KDE": [
        ("random", {"threshold": 0.0524, "expand_idx": 1.027}),
        ("by_ligand1", {"threshold": 0.0443, "expand_idx": 1.025}),
        ("by_paper", {"threshold": 0.0335, "expand_idx": 1.000}),
        ("by_year", {"threshold": 0.0432, "expand_idx": 1.318}),
    ],
    "ensemble": [
        ("random", {"threshold": 1.9}),
        ("by_ligand1", {"threshold": 2.0}),
        ("by_paper", {"threshold": 0.9}),
        ("by_year", {"threshold": 1.3}),
    ],
    "leverage": [
        ("random", {"n": 6.142}),
        ("by_ligand1", {"n": 1.175}),
        ("by_paper", {"n": 0.322}),
        ("by_year", {"n": 6.411}),
    ],
        "BNN_probe": [
        ("random", {"threshold": 2.3}),
        ("by_ligand1", {"threshold": 2.7}),
        ("by_paper", {"threshold": 1.4}),
        ("by_year", {"threshold": 3.3}),
    ],
}

result_df = pd.DataFrame({})
for method, test_examples in eval_hyperparam.items():
    for example in test_examples:
        print(f"Now computing {example[0]} for {method}")
        result = single_synad_evaluation(
            hyperparams=example[1], split_mode=example[0], dataset_type="ULD", target_type="yield", method_type=method
        )
        result_df.loc[f"{example[0]}_in-SynAD-R2", method] = result[0]
        result_df.loc[f"{example[0]}_coverage", method] = result[2]

result_df.to_csv(Path(__file__).parent / "logs/ULD/synad_eval_ULD.csv")
