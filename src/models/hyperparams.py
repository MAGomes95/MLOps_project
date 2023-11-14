from typing import Union, Callable, Dict, Tuple

import optuna
import pandas as pd
import numpy as np
from comet_ml import Experiment
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from logger import get_console_logger
from data.preprocessing import get_preprocessing_pipeline

logger = get_console_logger()

def sample_hyperparams(
        model_fn: Callable,
        trial: optuna.trial.Trial,
) -> Dict[str, Union[str, int, float]]:
    
    if model_fn == SVC:
        return {
            "C": trial.suggest_float("C", 0.001, 1000),
            "kernel": trial.suggest_categorical("kernel", ["linear", "poly", "rbf", "sigmoid"]),
            "degree": trial.suggest_int("degree", 2, 5),
            "gamma": trial.suggest_float("gamma", 0.001, 100)
        }
    elif model_fn == KNeighborsClassifier:
        return {
            "n_neighbors": trial.suggest_int("n_neighbors", 1, 5),
            "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
        }
    elif model_fn == RandomForestClassifier:
        return {
            "n_estimators": trial.suggest_int("n_estimators", 10, 1000),
            "criterion": trial.suggest_categorical("criterion", ["gini", "entropy", "log_loss"]),
            "max_depth": trial.suggest_int("max_depth", 5, 30),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 8),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
        }
    else:
        raise NotImplementedError
    

def find_best_hyperparams(
        model_fn: Callable,
        nr_trials: int,
        X: pd.DataFrame,
        y: pd.Series,
        experiment: Experiment,
) -> Tuple[Dict, Dict]:
    """
    The function aims to determine the bet set of hyperparameters for a given model,
    given nr_trials experiments.
    """
    assert model_fn in {SVC, RandomForestClassifier, KNeighborsClassifier}

    def objective(trial: optuna.trial.Trial) -> float:
        """
        Error function we weant to minimize using hyperparameter tuning.
        """
        hyperparameters = sample_hyperparams(model_fn, trial)

        # Evaluate model using cross-validation
        kfold = KFold(shuffle=True)
        performance = []
        logger.info(f"{trial.number=}")
        for split_number, (train_idx, validation_idx) in enumerate(kfold.split(X)):

            # Split data for training and validation
            X_train, X_val = X.iloc[train_idx], X.iloc[validation_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[validation_idx]

            logger.info(f'{split_number=}')
            logger.info(f'{len(X_train)=}')
            logger.info(f'{len(X_val)=}')

            pipeline = make_pipeline(
                get_preprocessing_pipeline(),
                model_fn(**hyperparameters)
            )
            pipeline.fit(X_train, y_train)

            # Model Evaluation
            y_pred = pipeline.predict(X_val)
            accuracy = accuracy_score(y_val, y_pred)
            performance.append(accuracy)

            logger.info(f"{accuracy=}")

        performance = np.array(performance).mean()

        return performance
    
    logger.info('Starting hyper-parameter search...')
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=nr_trials)

    # Get the best hyperparameters and their values
    best_hyperparams = study.best_params
    best_acc = study.best_value

    logger.info("Best Parameters:")

    for key, value in best_hyperparams.items():
        logger.info(f"{key}: {value}")
    logger.info(f"Best Acc: {best_acc}")

    experiment.log_metric('Cross_validation_Acc', best_acc)

    return best_hyperparams
