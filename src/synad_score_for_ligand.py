"""
SynAD Score Evaluation using synad package functions.

This module provides simplified functions for SynAD score evaluation
by leveraging the synad package functionality.
"""

from pathlib import Path
from loguru import logger
import pandas as pd
import synad
from modules.data_load import load_reaction_data
from modules.desc_process import ReactionDesc
from predict_settings import data_info
from tqdm import tqdm


def expand_reaction_data(data_df, reagent_type):
    """
    Expand reaction data by generating virtual combinations.

    Args:
        data_df (pd.DataFrame): Original reaction data
        reagent_type (str): Type of reagent to expand (e.g., 'ligand1')

    Returns:
        pd.DataFrame: Expanded data with virtual reactions
    """
    REACTION_PARTS = [
        "metal",
        "ligand1",
        "ligand2",
        "reactant1",
        "reactant2",
        "r12",
        "product",
        "add1",
        "add2",
        "solv",
        "time",
        "temperature",
        "cat_amount",
    ]

    assert reagent_type in REACTION_PARTS, f"{reagent_type} not found in reaction parts"

    reaction_parts = REACTION_PARTS.copy()
    reaction_parts.remove(reagent_type)

    all_expand_reagents = data_df[reagent_type].unique()
    all_unique_reactions = data_df[reaction_parts].drop_duplicates()
    expanded_data_list = []

    logger.info("Expanding data now...")
    for reagent in tqdm(all_expand_reagents, desc="Expanding Reagents"):
        expanded_df = all_unique_reactions.copy()
        expanded_df["is_virtual"] = True

        real_data_mask = data_df[reagent_type] == reagent
        expanded_df.loc[real_data_mask.reset_index(drop=True), "is_virtual"] = False
        expanded_df[reagent_type] = reagent
        expanded_data_list.append(expanded_df)

    expanded_data = pd.concat(expanded_data_list, ignore_index=True)
    logger.info(f"Expanded data shape: {expanded_data.shape}, expanded {expanded_data.shape[0] / data_df.shape[0]:.1f}x")

    return expanded_data


def load_and_prepare_descriptors(data_df, dataset_type, target_type):
    """
    Load reaction descriptors for the given data.

    Args:
        data_df (pd.DataFrame): Reaction data
        dataset_type (str): Dataset type (e.g., 'ULD')
        target_type (str): Target property (e.g., 'yield')

    Returns:
        tuple: (X_data, y_data, processed_data_df)
    """
    reagent_columns = data_info[dataset_type]["reagent_columns"]
    data_columns = data_info[dataset_type]["data_columns"]

    reaction_desc = ReactionDesc(data_df, reagent_columns=reagent_columns, data_columns=data_columns, dataset_type=dataset_type)
    reaction_desc.load_reaction_desc()
    X_data, y_data = reaction_desc.generate_descriptor_matrix(target_type)
    processed_data_df = reaction_desc.data_df

    return X_data, y_data, processed_data_df


def calculate_synad_scores_for_expanded_data(training_X, expanded_X):
    """
    Calculate SynAD scores for expanded data using synad package.

    Args:
        training_X (np.ndarray): Training descriptor matrix
        expanded_X (np.ndarray): Expanded descriptor matrix

    Returns:
        tuple: (synad_scores, synad_types, predictions)
    """
    logger.info("Calculating SynAD scores using synad package...")

    # Use synad package for SynAD score calculation
    score_evaluator = synad.SynADScoreEvaluator()
    synad_scores, synad_types, predictions = score_evaluator.calculate_synad_scores(training_X, expanded_X)

    logger.info("SynAD score calculation completed.")
    return synad_scores, synad_types, predictions


def export_synad_results(data_df_expanded, reagent_type, dataset_type, base_path):
    """
    Export SynAD evaluation results to CSV.

    Args:
        data_df_expanded (pd.DataFrame): Expanded data with SynAD results
        reagent_type (str): Type of reagent
        dataset_type (str): Dataset type
        base_path (Path): Base path for saving results
    """
    category_path = base_path / f"../data/{dataset_type}_append/{reagent_type}_data_category_with_skeleton.csv"

    if not category_path.exists():
        logger.warning(f"Category file not found: {category_path}")
        # Export without category information
        result_data = data_df_expanded.groupby(reagent_type)[["predict", "synad_score"]].mean().reset_index()
    else:
        category_df = pd.read_csv(category_path, index_col=0)

        predict_data = data_df_expanded.groupby(reagent_type)["predict"].mean()
        synad_score_data = data_df_expanded.groupby(reagent_type)["synad_score"].mean()

        result_data = pd.concat([predict_data, synad_score_data], axis=1).reset_index()

        # Add category information
        for idx in result_data.index:
            reagent_value = result_data.loc[idx, reagent_type]
            category_match = category_df[category_df[reagent_type] == reagent_value]
            if not category_match.empty:
                result_data.loc[idx, "category"] = category_match.iloc[0, 2]
            else:
                result_data.loc[idx, "category"] = "unknown"

    # Save results
    output_dir = base_path / f"logs/{dataset_type}/expanded_data"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{reagent_type}_synad_score_results.csv"

    result_data.to_csv(output_path, index=False)
    logger.info(f"Results exported to {output_path}")


def train_model_if_needed(data_df, dataset_type, model_type, target_type, model_path):
    """
    Train prediction model if it doesn't exist.

    Args:
        data_df (pd.DataFrame): Training data
        dataset_type (str): Dataset type
        model_type (str): Model type (e.g., 'XGB')
        target_type (str): Target property
        model_path (Path): Path where model should be saved

    Returns:
        bool: True if model was trained, False if loaded existing
    """
    if model_path.exists():
        logger.info(f"Model {model_type} already exists at {model_path}")
        return False

    logger.info(f"Training {model_type} model...")

    # Load descriptors
    X_data, y_data, _ = load_and_prepare_descriptors(data_df, dataset_type, target_type)

    # Use synad package for training
    split_info = synad.generate_split_info(data_df, X_data, "no_split")
    synad.train_and_evaluate_model(X_data.values, y_data.values, split_info, model_name=model_type, dataset_type=dataset_type, verbose=True)

    logger.info(f"Model training completed and saved to {model_path}")
    return True


def evaluate_synad_scores_for_reagent(dataset_type, model_type, reagent_type, target_type="yield"):
    """
    Main function to evaluate SynAD scores for a specific reagent type.

    Args:
        dataset_type (str): Dataset type (e.g., 'ULD')
        model_type (str): Model type (e.g., 'XGB')
        reagent_type (str): Reagent type (e.g., 'ligand1')
        target_type (str): Target property (default: 'yield')

    Returns:
        pd.DataFrame: Expanded data with SynAD scores
    """
    base_path = Path(__file__).parent
    model_path = base_path / f"logs/{dataset_type}/models/{model_type}.pkl"

    # Load original data
    logger.info(f"Loading {dataset_type} data for {target_type}...")
    original_data = load_reaction_data(dataset_type, target_type)

    # Train model if needed
    train_model_if_needed(original_data, dataset_type, model_type, target_type, model_path)

    # Load original descriptors
    logger.info("Loading original descriptors...")
    original_X, _, _ = load_and_prepare_descriptors(original_data, dataset_type, target_type)

    # Expand data
    logger.info(f"Expanding data for reagent type: {reagent_type}")
    expanded_data = expand_reaction_data(original_data, reagent_type)

    # Load expanded descriptors
    logger.info("Loading expanded descriptors...")
    expanded_desc = ReactionDesc(
        expanded_data,
        reagent_columns=data_info[dataset_type]["reagent_columns"],
        data_columns=data_info[dataset_type]["data_columns"],
        dataset_type=dataset_type,
    )
    expanded_desc.load_reaction_desc()
    expanded_data = expanded_desc.drop_no_desc_data()

    expanded_X, _ = expanded_desc.generate_descriptor_matrix(verbose=True)
    expanded_X = expanded_X.loc[:, original_X.columns]  # Ensure same features

    # Validate data size
    if expanded_data.shape[0] == 0:
        raise ValueError("No data found for the specified reagent class.")
    elif expanded_data.shape[0] < 100:
        logger.warning("Less than 100 data points may lead to poor prediction results.")

    logger.info(f"Using {expanded_data.shape[0]} expanded data points for evaluation.")

    # Calculate SynAD scores
    synad_scores, synad_types, predictions = calculate_synad_scores_for_expanded_data(original_X.values, expanded_X.values)

    # Add results to expanded data
    expanded_data = expanded_data.copy()
    expanded_data["synad_score"] = synad_scores
    expanded_data["synad_type"] = synad_types
    if predictions is not None:
        expanded_data["predict"] = predictions
    else:
        logger.warning("No predictions generated - model might not be available")

    # Export results
    if "predict" in expanded_data.columns:
        export_synad_results(expanded_data, reagent_type, dataset_type, base_path)

    return expanded_data


def visualize_synad_results(expanded_data, original_X, expanded_X, save_path=None):
    """
    Visualize SynAD results using synad package visualizer.

    Args:
        expanded_data (pd.DataFrame): Expanded data with SynAD results
        original_X (np.ndarray): Original descriptor matrix
        expanded_X (np.ndarray): Expanded descriptor matrix
        save_path (str, optional): Path to save the plot

    Returns:
        matplotlib.figure.Figure: The generated plot
    """
    logger.info("Generating UMAP visualization...")

    # Use synad package for decomposition and visualization
    score_evaluator = synad.SynADScoreEvaluator()
    score_evaluator.X_data = original_X
    score_evaluator.X_data_expanded = expanded_X

    # Generate mask for original points
    origin_points_mask = ~expanded_data["is_virtual"].values

    decomposed_coords, data_types = score_evaluator.decompose_data_for_visualization(origin_points_mask=origin_points_mask)

    # Create visualizer and plot
    visualizer = synad.SynADScoreVisualizer(save_path=save_path)
    figure = visualizer.plot_umap_results(decomposed_coords, data_types, save_path=save_path)

    return figure


# Configuration
dataset_type = "ULD"
model_type = "XGB"
reagent_type = "ligand1"
target_type = "yield"

result_data = evaluate_synad_scores_for_reagent(dataset_type, model_type, reagent_type, target_type)
logger.info("SynAD score evaluation completed successfully!")
