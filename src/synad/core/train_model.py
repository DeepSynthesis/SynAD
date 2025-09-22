from sklearn.discriminant_analysis import StandardScaler
from sklearn.metrics import r2_score
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger
from hyperopt import fmin, tpe, Trials, space_eval
import ast, re
import time

from ..utils_func import results_generate
from ..models.methods_traditional import TREE_MODELS, TREE_MODEL_HYPER_PARAM, TREE_MODEL_HSPACE
from ..models.methods_NN import NN_MODELS, NN_MODEL_HYPER_PARAM, NN_MODEL_HSPACE

from ..utils_func import metric_cal

model_path = Path(__file__).parent / Path("models")
log_path = Path(__file__).parent / Path("../logs")

method_list = ["LGB", "XGB", "CAT", "ADA", "RF", "GAU", "SVM", "BAG", "NN", "GCN", "BNN"]
graph_models = ["GCN"]


class MLMethod:
    def __init__(self, X_data, y_data, split_info=None, model_name: str = "XGB", dataset_type: str = "ULD"):
        """Initialize TreeMethod class with model name

        Args:
            model_name (str): aviliable for `lightgbm`, `xgboost`, `catboost`, `adaboost`, `random_forest`, `gaussian`, `svr`, `bagging`
        """
        if split_info == None:
            split_info = [(X_data.index, [])]

        self.model_name = model_name
        self.X_data = X_data
        self.y_data = y_data
        self.split_info = split_info
        self.dataset_type = dataset_type
        if self.model_name in ["LGB", "XGB", "CAT", "ADA", "RF", "GAU", "SVM", "BAG"]:
            self.model = TREE_MODELS[model_name]
            self.hyper_param = TREE_MODEL_HYPER_PARAM[f"{model_name}_hyper_parameters"]
            self.hyper_param.update(self.__read_parameter(model_name, self.dataset_type))
            self.hspace = TREE_MODEL_HSPACE[f"{model_name}_hspace"]
        elif self.model_name in ["NN", "GCN", "BNN"]:
            self.model = NN_MODELS[model_name]
            self.hyper_param = NN_MODEL_HYPER_PARAM[f"{model_name}_hyper_parameters"]
            self.hyper_param.update(self.__read_parameter(model_name, self.dataset_type))
        else:
            raise Exception(f"No model name called {self.model_name}!")
        self.model_save_path = log_path / Path(f"{self.dataset_type}/models/{self.model_name}.pkl")
        self.scaler_save_path = log_path / Path(f"{self.dataset_type}/models/{self.model_name}-scaler.pkl")

    def model_train(self, supervise_eval=False, verbose=False, add_hp={}):
        self.models, self.standard_scalers = [], []
        self.hyper_param.update(add_hp)
        for k, (train_index, test_index) in enumerate(self.split_info):
            logger.info(f"in the {k+1}th split.") if verbose else None
            X_train, y_train = self.X_data[train_index], self.y_data[train_index]
            X_test, y_test = self.X_data[test_index], self.y_data[test_index]

            # Normalize
            X_scaler = StandardScaler()
            X_train = X_scaler.fit_transform(X_train)
            if X_test.shape[0] == 0:
                logger.warning("No test data...")
            else:
                X_test = X_scaler.transform(X_test)

            if self.model_name in graph_models:
                X_train.append(self.X_atom_data[train_index], self.X_bond_data[train_index])
                X_test.append(self.X_atom_data[test_index], self.X_bond_data[test_index])
            my_model = self.model(**self.hyper_param)

            if supervise_eval and "NN" in self.model_name:
                my_model = my_model.fit(X_train, y_train, X_test, y_test)
            else:
                my_model = my_model.fit(X_train, y_train)

            y_pred_train = my_model.predict(X_train)

            train_infos = {"y_true": y_train, "y_pred": y_pred_train}
            ave_train = metric_cal(**train_infos, precise=3)
            if list(self.split_info[0][1]) != []:
                y_pred_test = my_model.predict(X_test)
                train_infos = {"y_true": y_train, "y_pred": y_pred_train}
                ave_test = metric_cal(y_test, y_pred_test, precise=3)
            else:
                y_pred_test, ave_test = "no_inner_predict", {}

            logger.info(f"{results_generate(ave_train, 'train', 2)}. {results_generate(ave_test, 'test', 2)}") if verbose else None

            # result save
            y_pred_train = np.array([i[0] for i in y_pred_train]) if isinstance(y_pred_train[0], np.ndarray) else y_pred_train
            y_pred_test = np.array([i[0] for i in y_pred_test]) if isinstance(y_pred_test[0], np.ndarray) else y_pred_test
            results_train = pd.DataFrame(
                {
                    "fold_id": [k] * len(train_index),
                    "idx": train_index,
                    "predict": y_pred_train,
                    "real": y_train,
                }
            )
            results_test = pd.DataFrame(
                {
                    "fold_id": [k] * len(test_index),
                    "idx": test_index,
                    "predict": y_pred_test,
                    "real": y_test,
                }
            )
            all_results_train = results_train.copy() if k == 0 else pd.concat([all_results_train, results_train])
            all_results_test = results_test.copy() if k == 0 else pd.concat([all_results_test, results_test])

            self.models.append(my_model)
            self.standard_scalers.append(X_scaler)

        return all_results_train, all_results_test

    def hyper_opt(self):

        def loss(config):
            self.hyper_param.update(config)
            _, _, all_results_test, _, _ = self.model_train()
            return -r2_score(y_true=all_results_test["real"], y_pred=all_results_test["predict"])
            # https://zhuanlan.zhihu.com/p/147663370

        opt_save_file = log_path / Path(f"{self.dataset_type}/parameter_opt_ditails-{self.model_name}.pkl")
        #  find a better way to replace pickle
        trials = pickle.load(opt_save_file.open("rb")) if opt_save_file.exists() else Trials()
        best = fmin(fn=loss, space=self.hspace, algo=tpe.suggest, max_evals=len(trials) + 50, trials=trials)
        best_params = space_eval(self.hspace, best)
        hyper_opt = logger.add(log_path / Path(f"{self.dataset_type}/parameters-{self.model_name}.log"), level="INFO")
        logger.info(f"-----generate at {time.ctime()}-----")
        logger.info(best_params)
        logger.info("-" * 46 + "\n\n")
        logger.remove(hyper_opt)

        # save opt results

        pickle.dump(trials, opt_save_file.open("wb"))

    def save_model(self):
        assert len(self.models) == 1, "Multiple models cannot be saved. Use `no_split` to train models."
        output_model, output_scaler = self.models[0], self.standard_scalers[0]

        self.model_save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(self.model_save_path, "wb") as f:
            pickle.dump(output_model, f)
        with open(self.scaler_save_path, "wb") as f:
            pickle.dump(output_scaler, f)

        logger.info(f"{self.model_name} model has been saved.")

    def load_model(self, return_model=False):
        with open(self.model_save_path, "rb") as f:
            loaded_model = pickle.load(f)
        with open(self.scaler_save_path, "rb") as f:
            loaded_scaler = pickle.load(f)

        self.models, self.standard_scalers = [loaded_model], [loaded_scaler]
        if return_model:
            return loaded_model, loaded_scaler
        else:
            return None

    def get_model(self):
        return self.models, self.standard_scalers

    def predict(self, X_test_data=None, y_test_data=None):
        assert hasattr(self, "models"), "Model not trained or loaded. Please train or load model first."
        # assert len(self.models) == 1, "Multiple models cannot be loaded. Use `no_split` to train models."
        if X_test_data == None:
            logger.warning("X_test_data is None, use X_data instead.")
            X_test = self.X_data
        else:
            X_test = X_test_data

        y_pred_list = []
        for k, (model, X_scaler) in enumerate(zip(self.models, self.standard_scalers)):
            X_test = X_scaler.transform(X_test)
            y_pred_test = model.predict(X_test)
            results_test = pd.DataFrame(
                {
                    "fold_id": [k] * len(X_test),
                    "predict": y_pred_test,
                    "real": y_test_data,
                }
            )
            y_pred_list.append(results_test)

        y_pred = pd.concat(y_pred_list)
        return y_pred

    def __read_parameter(self, method_name, dataset_type):
        if (log_path / Path(f"{dataset_type}/parameters-{method_name}.log")).exists():
            with (log_path / Path(f"{dataset_type}/parameters-{method_name}.log")).open("r") as parm_file:
                parms = parm_file.read()
            parms = ast.literal_eval(re.findall(r"\{.*\}", parms)[-1])
            return parms
        else:
            return {}
