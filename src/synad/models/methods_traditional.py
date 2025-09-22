import lightgbm, xgboost, catboost
from sklearn import ensemble, gaussian_process, svm
from hyperopt import hp

TREE_MODELS = {
    "LGB": lightgbm.LGBMRegressor,
    "XGB": xgboost.XGBRegressor,
    "CAT": catboost.CatBoostRegressor,
    "ADA": ensemble.AdaBoostRegressor,
    "RF": ensemble.RandomForestRegressor,
    "GAU": gaussian_process.GaussianProcessRegressor,
    "SVM": svm.SVR,
    "BAG": ensemble.BaggingRegressor,
}


TREE_MODEL_HYPER_PARAM = {
    "LGB_hyper_parameters": {
        "task": "train",
        "boosting_type": "gbdt",
        "objective": "regression",
        "learning_rate": 0.03,
        "max_depth": 20,
        "num_leaves": 2528,
        "subsample": 1,
        "min_child_weight": 0.04235,
        "min_split_gain": 0.006577,
        "colsample_bytree": 0.8043,
        "reg_alpha": 1,
        "reg_lambda": 1,
        "verbosity": -1,
    },
    "XGB_hyper_parameters": {
        "booster": "gbtree",
        "nthread": 28,
        "n_estimators": 500,  # early stop steps, do not need to adjust.
        "eta": 0.04652048820461861,
        "max_depth": 11,
        "subsample": 0.8702021783627013,
        "min_child_weight": 9.662743821684952,
        "gamma": 8.004764599050205,
        "lambda": 0.2988771193011762,
        "alpha": 0.8567172944601594,
    },
    "CAT_hyper_parameters": {
        "iterations": 5000,
        "task_type": "CPU",
        "logging_level": "Verbose",
        "metric_period": int(5000 / 10),
        "learning_rate": 0.03,
        "l2_leaf_reg": 3,
        "bagging_temperature": 1,
        "random_seed": 42,
        "depth": 6,
        "leaf_estimation_method": "Gradient",
        "fold_len_multiplier": 2,
        "border_count": 128,
    },
    "RF_hyper_parameters": {
        "n_estimators": 4000,
        # "criterion": "squared_error",
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
        "n_jobs": 30,
        "random_state": 42,
    },
    "ADA_hyper_parameters": {
        "n_estimators": 100,
        "learning_rate": 0.03,
        "random_state": 42,
        # "loss": "linear",
    },
    "GAU_hyper_parameters": {
        "alpha": 0.27761504296810335,
        # "kernel": Matern(length_scale=1, nu=1.5),
        "n_restarts_optimizer": 8,
    },
    "SVM_hyper_parameters": {
        "C": 1.0,
        "kernel": "rbf",
        "degree": 3,
        "gamma": "auto",
        "coef0": 0.0,
        "tol": 0.001,
        "cache_size": 200,
        "verbose": False,
        "max_iter": -1,
    },
    "BAG_hyper_parameters": {
        "n_estimators": 10,
        "max_samples": 1.0,
        "max_features": 1.0,
        "n_jobs": -1,
        "random_state": 42,
    },
}


TREE_MODEL_HSPACE = {
    "LGB_hspace": {
        "learning_rate": hp.uniform("learning_rate", 1e-5, 0.1),
        "max_depth": hp.uniformint("max_depth", 3, 30),
        "num_leaves": hp.uniformint("num_leaves", 2**3 - 1, 10000),
        "subsample": hp.choice("subsample", [0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]),
        "colsample_bytree": hp.choice("colsample_bytree", [0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]),
        "min_child_weight": hp.uniform("min_child_weight", 0, 0.1),
        "min_split_gain": hp.uniform("min_split_gain", 0, 0.1),
        "reg_alpha": hp.uniform("reg_alpha", 0, 100),
        "reg_lambda": hp.uniform("reg_lambda", 0, 100),
    },
    "XGB_hspace": {
        "eta": hp.uniform("eta", 1e-5, 0.3),
        "max_depth": hp.choice("max_depth", range(7, 15)),
        "subsample": hp.uniform("subsample", 0.5, 1.0),
        "min_child_weight": hp.uniform("min_child_weight", 0.0, 10.0),
        "gamma": hp.uniform("gamma", 0.0, 10.0),
        "lambda": hp.uniform("lambda", 0.0, 10.0),
        "alpha": hp.uniform("alpha", 0.0, 10),
    },
    "CAT_hspace": {
        "learning_rate": hp.uniform("learning_rate", 1e-4, 1e-1),
        "l2_leaf_reg": hp.choice("l2_leaf_reg", range(3, 10)),
        "depth": hp.uniformint("depth", 1, 15),
        "fold_len_multiplier": hp.uniform("min_child_weight", 1, 10),
    },
    "ADA_hspace": {
        "learning_rate": hp.uniform("learning_rate", 1e-6, 1e-2),
        "n_estimators": hp.uniformint("n_estimators", 10, 1000),
        "loss": hp.choice("loss", ["linear", "square", "exponential"]),
    },
    "RF_hspace": {
        "n_estimators": hp.uniformint("n_estimators", 100, 1000),
        "criterion": hp.choice("criterion", ["squared_error", "absolute_error"]),
        "min_samples_split": hp.uniformint("min_samples_split", 2, 20),
        "min_samples_leaf": hp.uniformint("min_samples_leaf", 1, 10),
        "max_features": hp.choice("max_features", ["sqrt", "log2"]),
    },
    "GAU_hspace": {
        "alpha": hp.uniform("alpha", 0.01, 1),
        # "optimizer": hp.choice("optimizer", ["fmin_l_bfgs_b", "CG", "Newton-CG", "TR"]),
        "n_restarts_optimizer": hp.uniformint("n_restarts_optimizer", 1, 10),
    },
    "SVM_hspace": {
        "C": hp.uniform("regularization_parametere", 0.0, 1000.0),
        # "coef0": hp.choice("subsample", [0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]),
        "tol": hp.uniform("tolerance", 1e-5, 1e-0),
    },
    "BAG_hspace": {
        "n_estimators": hp.uniformint("n_estimators", 10, 10000),
        "max_samples": hp.uniform("max_samples", 0.0, 1.0),
        "max_features": hp.uniform("max_features", 0.0, 1.0),
    },
}
