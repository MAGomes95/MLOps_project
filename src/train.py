import os
import pickle
from typing import Dict, Union, Optional, Callable, Literal


import pandas as pd
from comet_ml import (
    API,
    Experiment
)
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from src.logger import get_console_logger
from src.data.preprocessing import (
    get_features_and_target,
    get_preprocessing_pipeline
)
from src.models.hyperparams import find_best_hyperparams

logger = get_console_logger()

def get_model_fn_from_name(model_name: str = Literal["svm", "random_forest", "knn"]) -> Callable:
    """
    Returns the model object given model name.
    """
    if model_name == "svm":
        return SVC
    elif model_name == "knn":
        return KNeighborsClassifier
    else:
        return RandomForestClassifier
    

def train(
    X: pd.DataFrame,
    y: pd.Series,
    model: str,
    tune_hyperparams: Optional[bool] = True,
    hyperparams_trials: Optional[int] = 10
) -> None:
    """
    Train a machine learning model using X and y, while possibly
    performing hyperparameters tuning.
    """
    model_fn = get_model_fn_from_name(model)

    experiment = Experiment(
        api_key=os.environ["COMET_ML_API_KEY"],
        workspace=os.environ["COMET_ML_WORKSPACE"],
        project_name="MLOps_LungCancer_Project"
    )
    experiment.add_tag(model)

    # Split the data intro train and validation
    train_sample_size = int(0.8 * X.shape[0])
    X_train, X_validation = X.iloc[:train_sample_size], X.iloc[train_sample_size:]
    y_train, y_validation = y.iloc[:train_sample_size], y.iloc[train_sample_size:]

    logger.info(f"Train size: {X_train.shape[0]}")
    logger.info(f"Validation size: {X_validation.shape[0]}")

    if not tune_hyperparams:
        logger.info("Using default hyperparameters")
        pipeline = make_pipeline(
            get_preprocessing_pipeline(),
            model_fn()
        )
    else:
        logger.info("Finding best hyperparameters")
        best_hyperparams = find_best_hyperparams(
            model_fn=model_fn,
            nr_trials=hyperparams_trials,
            X=X,
            y=y,
            experiment=experiment
        )

        logger.info(f"Best Hyperparameters: {best_hyperparams}")

        pipeline = make_pipeline(
            get_preprocessing_pipeline(),
            model_fn.set_params(**best_hyperparams)
        )

        experiment.add_tag("Hyperparameters-tunning")

    # Performing final train

    logger.info("Fitting model")
    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_validation)
    validation_acc = accuracy_score(y_validation, predictions)
    logger.info(f"Validation accuracy: {validation_acc}")
    experiment.log_metrics({"Validation_accuracy": validation_acc})

    logger.info("Saving the Model to the Disk")
    with open("models/model.pkl", "wb") as handler:
        pickle.dump(pipeline, handler)

    experiment.log_model(str(model_fn), "models/model.pkl")
    experiment.register_model(str(model_fn))



if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--model", type=str, choices=["svm", "rf", "knn"])
    parser.add_argument("--tune-hypers", action="store_true")
    parser.add_argument("--hyperparms-trials", type=int, default=10)
    args = parser.parse_args()

    logger.info("Getting features and target")
    features, target = get_features_and_target(
        path_to_input="data/raw_data.csv"
    )

    logger.info("Performing Training")
    train(
        X=features,
        y=target,
        model=args.model,
        tune_hyperparams=args.tune_hypers,
        hyperparams_trials=args.hyperparms_trials
    )