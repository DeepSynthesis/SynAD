"""
SynAD: Synthesis Applicability Domain for chemical reaction prediction.

This package provides tools for:
- Training machine learning models for chemical reaction prediction
- Evaluating synthesis applicability domain (SynAD) of predictions
- Computing SynAD scores for molecular compounds
"""

__version__ = "0.1.0"
__author__ = "Zhenzhi Tan"
__email__ = "zhenzhi-tan@outlook.com"

# Core imports
from .core.synad import SynADJudgementor
from .core.train_model import MLMethod

# Evaluation functions
from .evaluation.synad_evaluation import (
    single_synad_evaluation,
    synad_hyperparameter_optimization,
    train_and_evaluate_model,
    calculate_feature_importance,
    evaluate_model_performance
)

from .evaluation.synad_score_evaluate import (
    SynADScoreEvaluator,
    SynADScoreVisualizer
)

# Utility functions
from .utils_func import (
    results_generate,
    SMILES_canonicalization,
    generate_split_info,
    metric_cal,
    SHAP_value_calculation,
    decomponent_reactions
)

# Public API
__all__ = [
    # Core classes
    "SynADJudgementor",
    "MLMethod",

    # Evaluation functions
    "single_synad_evaluation",
    "synad_hyperparameter_optimization",
    "train_and_evaluate_model",
    "calculate_feature_importance",
    "evaluate_model_performance",
    "SynADScoreEvaluator",
    "SynADScoreVisualizer",

    # Utility functions
    "results_generate",
    "SMILES_canonicalization",
    "generate_split_info",
    "metric_cal",
    "SHAP_value_calculation",
    "decomponent_reactions"
]