import itertools
from pathlib import Path
from loguru import logger
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from joblib import Parallel, delayed
from sklearn.discriminant_analysis import StandardScaler
from sklearn.metrics import r2_score
from sklearn.decomposition import PCA
from sklearn.svm import OneClassSVM
from scipy.stats import gaussian_kde
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK

from ..utils_func import generate_split_info


def compute_distances(target_mat, background_mat, metric, n_jobs):
    """
    Computes the distance matrix between target_mat and background_mat using the specified distance metric.

    Args:
        target_mat (np.ndarray): Target matrix.
        background_mat (np.ndarray): Background matrix.

    Returns:
        np.ndarray: Distance matrix.
    """
    target_mat = target_mat.astype(float)
    background_mat = background_mat.astype(float)
    split_target = np.array_split(target_mat, n_jobs)

    def compute_dist(sub_target):
        if metric == "mahalanobis":
            VI = np.linalg.pinv(np.cov(background_mat, rowvar=False))
            return cdist(sub_target, background_mat, metric="mahalanobis", VI=VI)
        elif metric in ["euclidean", "cityblock", "cosine", "chebyshev"]:
            return cdist(sub_target, background_mat, metric=metric)
        else:
            raise ValueError(f"Unsupported metric: {metric}")

    results = Parallel(n_jobs=n_jobs)(delayed(compute_dist)(sub) for sub in split_target)
    distance_matrix = np.vstack(results)
    return distance_matrix


def get_nearest_distance(target_mat, background_mat, k, metric, n_jobs):
    """
    Computes the average of the k nearest distances for each row in the target matrix.

    Args:
        target_mat (np.ndarray): Target matrix.
        background_mat (np.ndarray): Background matrix.
        k (int): Number of nearest neighbors to consider.

    Returns:
        np.ndarray: Mean of the k nearest distances for each row in the target matrix.
    """

    distance_matrix = compute_distances(target_mat, background_mat, metric, n_jobs)
    distance_matrix[distance_matrix == 0] = np.inf  # Avoid selecting self-distances
    sorted_distances = np.sort(distance_matrix, axis=1)
    dist_info = np.mean(sorted_distances[:, :k], axis=1)

    return dist_info


def get_d_and_sigma_of_dataset(self, data_X):
    """
    Calculates the mean distance (d) and standard deviation (sigma) of distances within the dataset.

    Args:
        data_X (np.ndarray): Input dataset.

    Returns:
        tuple: Mean distance (d) and standard deviation (sigma).
    """
    nearest_distance_list = self.get_nearest_distance(data_X, data_X, k=1)
    return np.mean(nearest_distance_list), np.std(nearest_distance_list)


class SynADJudgementor:
    def __init__(self, method_type="ZKNN", n_jobs=24):
        """
        Initializes the SynADJudgementor with the specified method type and distance metric.

        Args:
            method_type (str): Method for calculating distances, e.g., "parallel".
            distance_metric (str): Distance metric to use, e.g., "euclidean", "mahalanobis", "manhattan".
            n_jobs (int): Number of parallel jobs (used for "parallel" method_type).
        """
        self.method_type = method_type
        self.n_jobs = n_jobs

        if method_type == "ZKNN":
            self.method = self.ZKNN_judgement
        elif method_type == "leverage":
            self.method = self.leverage_judgement
        elif method_type == "ensemble":
            self.method = self.ensemble_judgement
        elif method_type == "SVM":
            self.method = self.SVM_judgement
        elif method_type == "BB":
            self.method = self.BoundingBox_judgement
        elif method_type == "KDE":
            self.method = self.KernelDensityEstimation_judgement
        elif method_type == "RF_probe":
            self.method = self.RandomForestProbe_judgement
        elif method_type == "GR_probe":
            self.method = self.GaussianRandomProbe_judgement
        elif method_type == "BNN_probe":
            self.method = self.BayesianNeuralNetworkProbe_judgement
        else:
            raise Exception(f"Unsupported method type: {method_type}")

    def load_data_kfold(self, X_data, y_data, y_pred_data, split_info, importance_lists=None):
        self.X_data = X_data
        self.y_data = y_data
        self.y_pred_data = y_pred_data
        self.split_info = split_info
        self.importance_lists = importance_lists

    def load_data(self, training_data, validation_data, importance_list=None, drop_duplicate=False, normalize=True):
        if importance_list is not None:
            self.training_data = training_data * importance_list
            self.validation_data = validation_data * importance_list
        else:
            self.training_data = training_data
            self.validation_data = validation_data

        if drop_duplicate:
            self.training_data = np.unique(self.training_data, axis=0)
            self.validation_data = np.unique(self.validation_data, axis=0)

        if normalize:
            self.data_scaler = StandardScaler()
            self.training_data = self.data_scaler.fit_transform(self.training_data)
            self.validation_data = self.data_scaler.transform(self.validation_data)

    def get_synad(self, hyper_param={}, training_y_data=None):
        """
        Calculates the Inner- and Out-of-Distribution (IAD and OAD) matrices for the validation data based on training data.

        Returns:
            pd.DataFrame: IAD and OAD matrices.
        """
        if self.method_type == "BNN_probe" or self.method_type == "ensemble":
            return self.method(training_y_data=training_y_data.values, **hyper_param)
        else:
            return self.method(**hyper_param)

    def kfold_get_synad(self, params={}, save_results_df=False, dataset_type=None, split_mode=None):
        results_list = []
        for id, (train_idx, test_idx) in enumerate(self.split_info):
            X_data_evaluate = self.X_data * self.importance_lists[id] if self.importance_lists is not None else self.X_data
            training_data, validation_data = X_data_evaluate[train_idx], X_data_evaluate[test_idx]
            self.load_data(training_data, validation_data)
            training_y_data = self.y_data[train_idx]
            res_df = self.get_synad(params, training_y_data)
            res_df["idx"] = test_idx
            results_list.append(res_df)
        results_df = pd.concat(results_list)
        results_df = results_df.set_index("idx").sort_index()
        results_df["y_pred"] = self.y_pred_data
        results_df["y_true"] = self.y_data
        # evaluation of SynAD performace
        r2_iad, r2_oad, coverage = self.evaluate_synad(results_df)

        if save_results_df:

            log_path = Path(__file__).parent / Path(f"../../logs/{dataset_type}/SynAD")
            results_df.to_csv(log_path / Path(f"synad_results_for_{self.method_type}_with_{split_mode}_split.csv"))
        return r2_iad, r2_oad, coverage

    def hyper_param_search(self, hyper_param_range={}, opt_method="hyperopt", max_num=100, verbose=False):

        def object_func(params):
            r2_iad, r2_oad, coverage = self.kfold_get_synad(params)
            if not verbose:
                print(",".join([f"{k}: {v}" for k, v in params.items()]), end=" ")
                print(f"r2_iad: {r2_iad}, r2_oad: {r2_oad}, coverage: {coverage}")
            # We want to maximize (r2_iad + r2_oad + coverage), so we minimize the negative
            loss = -(r2_iad)

            return {"loss": loss, "status": STATUS_OK, "params": params, "r2_iad": r2_iad, "r2_oad": r2_oad, "coverage": coverage}

        if not hasattr(self, "split_info"):
            raise Exception("Please define K-Fold evalutaion before use hyper parameter optimization.")

        if hyper_param_range == {}:
            match self.method_type:
                case _:
                    raise Exception(f"Unsupported method type: {self.method_type}")

        hyper_param_search_result = pd.DataFrame(columns=["r2_iad", "r2_oad", "coverage"])

        if opt_method == "hyperopt":
            # Define the search space for hyperopt
            space = {}
            for param_name, param_range in hyper_param_range.items():
                if isinstance(param_range[0], int):
                    space[param_name] = hp.choice(param_name, param_range)
                elif isinstance(param_range[0], float):
                    space[param_name] = hp.uniform(param_name, min(param_range), max(param_range))
                else:
                    space[param_name] = hp.choice(param_name, param_range)

            # Define the objective function for hyperopt

            # Run hyperopt optimization
            trials = Trials()
            best = fmin(object_func, space, algo=tpe.suggest, max_evals=max_num, trials=trials)

            # Collect results from all trials
            hyper_param_search_result = pd.DataFrame(columns=["r2_iad", "r2_oad", "coverage"])
            for trial in trials:
                params = trial["result"]["params"]
                comb_str = ",".join([f"{k}={v}" for k, v in params.items()])
                hyper_param_search_result.loc[comb_str] = [
                    trial["result"]["r2_iad"],
                    trial["result"]["r2_oad"],
                    trial["result"]["coverage"],
                ]

        elif opt_method == "grid_search":  # Default to grid search
            hyper_param_search_result = pd.DataFrame(columns=["r2_iad", "r2_oad", "coverage"])
            for comb in itertools.product(*hyper_param_range.values()):
                hyper_param = dict(zip(hyper_param_range.keys(), comb))
                res = object_func(hyper_param)
                comb_str = ",".join([f"{k}={v}" for k, v in zip(hyper_param_range.keys(), comb)])
                logger.info(f"{comb_str} | r2_iad: {res['r2_iad']} | r2_oad: {res['r2_oad']} | coverage: {res['coverage']}")
                hyper_param_search_result.loc[comb_str] = [res["r2_iad"], res["r2_oad"], res["coverage"]]
        else:
            raise Exception(f"Unsupported optimization method: {opt_method}")

        return hyper_param_search_result

    def evaluate_synad(self, results_df):
        # Calculate the r2_iad and r2_oad
        iad_data = results_df[results_df["AD_type"] == "IAD"]
        oad_data = results_df[results_df["AD_type"] == "OAD"]
        r2_iad = r2_score(iad_data["y_true"], iad_data["y_pred"]) if len(iad_data) > 0 else -100
        r2_oad = r2_score(oad_data["y_true"], oad_data["y_pred"]) if len(oad_data) > 0 else -100
        coverage = len(iad_data) / len(results_df)

        r2_iad, r2_oad, coverage = round(r2_iad, 3), round(r2_oad, 3), round(coverage, 3)
        return r2_iad, r2_oad, coverage

    def ZKNN_judgement(self, k=-100, Z=-100, trim_percent=5, metric="cityblock"):
        # "euclidean", "cityblock", "cosine", "chebyshev"
        # Calculate the mean distance (d) and standard deviation (sigma) for the training data
        nearest_distance_list_train = get_nearest_distance(self.training_data, self.training_data, k=1, metric=metric, n_jobs=self.n_jobs)
        assert not any(np.isnan(nearest_distance_list_train)), "There are nan values in the distance information"

        d = np.mean(nearest_distance_list_train)
        sigma = np.std(nearest_distance_list_train)
        D_T_value = d + Z * sigma  # Calculate the D_T threshold
        nearest_distance_list_test = get_nearest_distance(self.validation_data, self.training_data, k=k, metric=metric, n_jobs=self.n_jobs)
        # Create a DataFrame and label the AD types
        return pd.DataFrame(
            {
                "metrics": nearest_distance_list_test,
                "AD_type": ["OAD" if dist > D_T_value else "IAD" for dist in nearest_distance_list_test],
                "judge_value": [D_T_value] * len(nearest_distance_list_test),
            }
        )

    def leverage_judgement(self, n=3):
        M, N = self.training_data.shape[1], self.training_data.shape[0]  # M = number of descriptors, N = number of training examples
        h_star = n * (M + 1) / N  # Threshold h*

        leverage_values_test = np.sum(
            self.validation_data @ np.linalg.pinv(self.training_data.T @ self.training_data) * self.validation_data, axis=1
        )  # h = x_i^T (X^T X)^-1 x_i

        return pd.DataFrame(
            {
                "leverage": leverage_values_test,
                "AD_type": ["OAD" if lev > h_star else "IAD" for lev in leverage_values_test],
            }
        )

    def ensemble_judgement(self, training_y_data, threshold=10, ensemble_variance=True):

        from synad.core.train_model import MLMethod

        split_info = generate_split_info(None, self.training_data, "random", n_splits=5, rd_state=42)
        ml_model = MLMethod(
            self.training_data,
            training_y_data,
            dataset_type="ULD",
            model_name="XGB",
            split_info=split_info,
        )
        ml_model.model_train()
        models_list, scalers_list = ml_model.get_model()
        test_predict_list = []
        for model, scaler in zip(models_list, scalers_list):
            validation_data_scaled = scaler.transform(self.validation_data)
            test_predict_list.append(model.predict(validation_data_scaled))

        test_predict_dev = np.std(test_predict_list, axis=0)

        return pd.DataFrame(
            {
                "metrics": test_predict_dev,
                "AD_type": ["OAD" if dev > threshold else "IAD" for dev in test_predict_dev],
            }
        )

    def SVM_judgement(self, kernel="rbf", nu=0.7, gamma="scale"):
        model = OneClassSVM(kernel=kernel, nu=nu, gamma=gamma)
        model.fit(self.training_data)

        decision_scores = model.decision_function(self.validation_data)  # Decision function values
        predictions = model.predict(self.validation_data)  # +1 for inliers, -1 for outliers

        return pd.DataFrame(
            {
                "metrics": decision_scores,
                "AD_type": ["IAD" if pred == 1 else "OAD" for pred in predictions],
            }
        )

    def BoundingBox_judgement(self):
        min_bounds = np.min(self.training_data, axis=0)
        max_bounds = np.max(self.training_data, axis=0)

        within_bounds_matrix = (self.validation_data >= min_bounds) & (self.validation_data <= max_bounds)
        decision_scores = np.sum(within_bounds_matrix, axis=1)  # Count number of descriptors within bounds

        AD_types = ["IAD" if score == len(min_bounds) else "OAD" for score in decision_scores]

        return pd.DataFrame(
            {
                "metrics": decision_scores,
                "AD_type": AD_types,
            }
        )

    def KernelDensityEstimation_judgement(self, threshold=0.05, expand_idx=1.0):

        pca = PCA(n_components=0.95)
        self.training_data = pca.fit_transform(self.training_data)
        self.validation_data = pca.transform(self.validation_data)

        kde = gaussian_kde(self.training_data.T)  # Transpose to match KDE input format

        train_log_density = kde.logpdf(self.training_data.T)
        test_log_density = kde.logpdf(self.validation_data.T)

        log_threshold = np.percentile(train_log_density, threshold * 100) * expand_idx
        AD_types = ["IAD" if log_dens >= log_threshold else "OAD" for log_dens in test_log_density]

        return pd.DataFrame(
            {
                "density": np.exp(test_log_density),
                "log_density": test_log_density,
                "AD_type": AD_types,
            }
        )

    def GaussianRandomProbe_judgement(self, training_y_data, threshold=0.01):
        from sklearn.gaussian_process import GaussianProcessRegressor as GPR

        gpr = GPR(alpha=0.27761504296810335, n_restarts_optimizer=8, normalize_y=False)
        print("doing GPR")
        gpr.fit(self.training_data, training_y_data)

        y_pred, sigma = gpr.predict(self.validation_data, return_std=True)
        return pd.DataFrame(
            {
                "metrics": sigma * 100,
                "y_pred_test": y_pred,
                "AD_type": ["IAD" if score >= threshold else "OAD" for score in 1 - sigma],  # IAD: Inlier, OAD: Outlier
            }
        )

    def BayesianNeuralNetworkProbe_judgement(self, training_y_data, threshold=0.01):
        from ..models.methods_NN import NN_MODEL_HYPER_PARAM, BayesianNeuralNetworkRegressor

        hyperparam = NN_MODEL_HYPER_PARAM["BNN_hyper_parameters"]

        training_y_data = training_y_data.reshape(-1, 1)
        bnnr = BayesianNeuralNetworkRegressor(**hyperparam)
        bnnr.fit(self.training_data, training_y_data)

        test_data_scaled = bnnr.X_scaler.transform(self.validation_data)
        y_pred, bias_list = bnnr.predict_with_uncertainty(test_data_scaled)

        return pd.DataFrame(
            {
                "metrics": bias_list.flatten(),
                "predict": y_pred.flatten(),
                "AD_type": ["IAD" if bias >= threshold else "OAD" for bias in bias_list],  # IAD: Inlier, OAD: Outlier
            }
        )
