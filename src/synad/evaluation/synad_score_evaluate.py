"""
SynAD score evaluation module.

This module provides classes for calculating SynAD scores on molecular descriptors
without dependencies on specific data loading or description generation modules.
"""

from pathlib import Path
from loguru import logger
import numpy as np
import umap
from scipy.spatial.distance import cdist
from matplotlib import pyplot as plt
import seaborn as sns

from ..core.synad import SynADJudgementor


class SynADScoreEvaluator:
    """SynAD score evaluator that works with provided descriptor matrices."""

    # Default SynAD parameters for confidence thresholds
    DEFAULT_SYNAD_PARAMS = {"hard": {"Z": 0.5, "k": 4, "metric": "cityblock"}, "easy": {"Z": 1.5, "k": 3, "metric": "cityblock"}}

    DEFAULT_DECOMP_PARAMS = {"n_origin_points": 50, "distance_threshold": 10, "distance_range": (20, 20.1), "random_sample_size": 1000}

    DEFAULT_UMAP_PARAMS = {"n_neighbors": 10, "n_components": 2, "min_dist": 0.9, "n_jobs": 16}

    def __init__(self, model=None, model_scaler=None):
        """
        Initialize the SynAD score evaluator.

        Args:
            model: Pre-trained model for predictions (optional)
            model_scaler: Pre-fitted scaler for features (optional)
        """
        self.model = model
        self.model_scaler = model_scaler
        self.X_data = None
        self.X_data_expanded = None
        self.predict_results = None

    def calculate_synad_scores(self, training_data, expanded_data):
        """
        Calculate SynAD scores for expanded data based on training data.

        Args:
            training_data (np.ndarray): Training descriptor matrix
            expanded_data (np.ndarray): Expanded descriptor matrix

        Returns:
            tuple: (synad_scores_hard, synad_scores_easy, predictions)
        """
        self.X_data = training_data
        self.X_data_expanded = expanded_data

        # Normalize data if scaler is provided
        if self.model_scaler is not None:
            training_data_scaled = self.model_scaler.transform(training_data)
            expanded_data_scaled = self.model_scaler.transform(expanded_data)
        else:
            training_data_scaled = training_data
            expanded_data_scaled = expanded_data

        # Calculate SynAD results
        synad = SynADJudgementor(method_type="ZKNN", n_jobs=24)
        synad.load_data(training_data_scaled, expanded_data_scaled, None)

        synad_eval_results_hard = synad.get_synad(hyper_param=self.DEFAULT_SYNAD_PARAMS["hard"])
        synad_eval_results_easy = synad.get_synad(hyper_param=self.DEFAULT_SYNAD_PARAMS["easy"])

        # Assign SynAD types
        synad_types = self._assign_synad_types(synad_eval_results_hard, synad_eval_results_easy)

        # Calculate SynAD scores
        synad_scores = self._calculate_synad_scores(synad_types)

        # Make predictions if model is available
        predictions = None
        if self.model is not None:
            predictions = self.model.predict(expanded_data_scaled)

        return synad_scores, synad_types, predictions

    def _assign_synad_types(self, synad_eval_results_hard, synad_eval_results_easy):
        """Assign SynAD types based on evaluation results."""
        mask_easy = synad_eval_results_easy["AD_type"] == "IAD"
        mask_hard = synad_eval_results_hard["AD_type"] == "IAD"

        synad_types = np.array(["bad"] * len(synad_eval_results_hard))
        synad_types[mask_easy] = "medium"
        synad_types[mask_hard] = "good"

        return synad_types

    def _calculate_synad_scores(self, synad_types):
        """Calculate numerical SynAD scores from types."""

        def get_score(synad_type):
            if synad_type == "good":
                return 0
            elif synad_type == "medium":
                return 0.5
            else:  # bad
                return 1

        return np.array([get_score(t) for t in synad_types])

    def decompose_data_for_visualization(self, origin_points_mask=None):
        """
        Decompose data using UMAP for visualization.

        Args:
            origin_points_mask (np.ndarray): Boolean mask indicating original data points

        Returns:
            tuple: (decomposed_coordinates, data_types)
        """
        if self.X_data is None or self.X_data_expanded is None:
            raise ValueError("Data not loaded. Call calculate_synad_scores first.")

        # Load or compute distance matrix for sampling
        sampled_origin_points, X_data_sampled, dist_matrix = self._load_or_compute_distance_matrix()

        # Sample points based on distance criteria
        threshold = self.DEFAULT_DECOMP_PARAMS["distance_threshold"]
        range_min, range_max = self.DEFAULT_DECOMP_PARAMS["distance_range"]

        mask = (dist_matrix < threshold) | ((dist_matrix > range_min) & (dist_matrix < range_max))
        rows, cols = np.where(mask)

        X_data_expanded_sampled = self.X_data_expanded[cols]
        data_df_location = cols.tolist()

        # Remove original points if mask is provided
        if origin_points_mask is not None:
            mask_non_origin = ~origin_points_mask[data_df_location]
            X_data_expanded_sampled = X_data_expanded_sampled[mask_non_origin]
            data_df_location = np.array(data_df_location)[mask_non_origin]

        # Sample random bad points
        sample_size = min(self.DEFAULT_DECOMP_PARAMS["random_sample_size"], len(self.X_data_expanded))
        random_sample_indices = np.random.choice(len(self.X_data_expanded), size=sample_size, replace=False)
        X_data_random_sampled = self.X_data_expanded[random_sample_indices]

        # Combine all data
        total_X_data = np.concatenate([X_data_sampled, X_data_expanded_sampled, X_data_random_sampled], axis=0)

        n_origin = len(sampled_origin_points)
        total_X_data_types = np.concatenate(
            [
                np.array(["origin"] * n_origin),
                np.array(["sampled"] * len(X_data_expanded_sampled)),
                np.array(["random"] * len(X_data_random_sampled)),
            ],
            axis=0,
        )

        # UMAP decomposition
        umap_model = umap.UMAP(**self.DEFAULT_UMAP_PARAMS)
        total_X_data_decomp = umap_model.fit_transform(total_X_data)

        return total_X_data_decomp, total_X_data_types

    def _load_or_compute_distance_matrix(self):
        """Compute distance matrix for sampling."""
        n_points = self.DEFAULT_DECOMP_PARAMS["n_origin_points"]
        sampled_origin_points = np.random.choice(len(self.X_data), size=n_points, replace=False)
        X_data_sampled = self.X_data[sampled_origin_points]
        dist_matrix = cdist(X_data_sampled, self.X_data_expanded)

        return sampled_origin_points, X_data_sampled, dist_matrix


class SynADScoreVisualizer:
    """Visualizer for SynAD score results."""

    PLOT_CONFIG = {
        "figsize": (10, 9),
        "dpi": 300,
        "categories": ["bad", "medium", "good", "origin"],
        "colors": {"origin": "#c6262f", "good": "#464b84", "medium": "#5b97c8", "bad": "#dfceec"},
        "sizes": {"origin": 80, "good": 30, "medium": 30, "bad": 30},
        "alphas": {"origin": 1, "good": 0.5, "medium": 0.5, "bad": 0.5},
    }

    def __init__(self, save_path=None):
        """
        Initialize visualizer.

        Args:
            save_path: Path to save plots (optional)
        """
        self.save_path = save_path

    def plot_umap_results(self, decomposed_coords, data_types, save_path=None):
        """
        Plot UMAP decomposition results.

        Args:
            decomposed_coords (np.ndarray): 2D coordinates from UMAP
            data_types (np.ndarray): Types of data points
            save_path (str): Path to save the plot (optional)
        """
        plt.figure(figsize=self.PLOT_CONFIG["figsize"], dpi=self.PLOT_CONFIG["dpi"])

        for category in self.PLOT_CONFIG["categories"]:
            if category in data_types:
                mask = data_types == category
                x = decomposed_coords[mask, 0]
                y = decomposed_coords[mask, 1]

                sns.scatterplot(
                    x=x,
                    y=y,
                    color=self.PLOT_CONFIG["colors"][category],
                    size=[self.PLOT_CONFIG["sizes"][category]] * len(x),
                    sizes=(self.PLOT_CONFIG["sizes"][category], self.PLOT_CONFIG["sizes"][category]),
                    alpha=self.PLOT_CONFIG["alphas"][category],
                    label=category,
                )

        plt.xlabel("UMAP1", fontsize=16)
        plt.ylabel("UMAP2", fontsize=16)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.legend()
        plt.tight_layout()

        # Save plot if path is provided
        if save_path or self.save_path:
            save_path = save_path or self.save_path
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, bbox_inches="tight")
            logger.info(f"Plot saved to {save_path}")

        return plt.gcf()

    def plot_score_distribution(self, synad_scores, save_path=None):
        """
        Plot distribution of SynAD scores.

        Args:
            synad_scores (np.ndarray): Array of SynAD scores
            save_path (str): Path to save the plot (optional)
        """
        plt.figure(figsize=(8, 6))
        plt.hist(synad_scores, bins=30, alpha=0.7, edgecolor="black")
        plt.xlabel("SynAD Score", fontsize=14)
        plt.ylabel("Frequency", fontsize=14)
        plt.title("Distribution of SynAD Scores", fontsize=16)
        plt.grid(True, alpha=0.3)

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, bbox_inches="tight")
            logger.info(f"Score distribution plot saved to {save_path}")

        return plt.gcf()
