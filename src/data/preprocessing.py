from typing import Optional, List, Tuple
from pathlib import Path

import fire
import pandas as pd
import numpy as np
from sklearn.preprocessing import (
    LabelEncoder,
    MinMaxScaler,
    PolynomialFeatures
)
from sklearn.feature_selection import (
    SelectKBest, 
    mutual_info_classif
)
from sklearn.pipeline import Pipeline
from logger import get_console_logger

logger = get_console_logger()

def get_features_and_target(path_to_input: Optional[Path]) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Loads the data from provided path and returns separatly features and target
    """
    dataset = pd.read_csv(path_to_input)
    features = dataset[np.setdiff1d(dataset.columns, "target")]
    target = dataset["target"]

    return features, target

    
def get_preprocessing_pipeline() -> Pipeline:
    """Returns the preprocessing pipeline."""
    pipeline_steps = [
        ("interaction features", PolynomialFeatures(interaction_only=True, include_bias=False)),
        ("feature selection", SelectKBest(mutual_info_classif, k=5)),
        ("scaling", MinMaxScaler()),
        ("label encoding", LabelEncoder())
    ]

    return Pipeline(steps=pipeline_steps)


if __name__ == "__main__":

    features, target = fire.Fire(get_features_and_target)
    
    preprocessing_pipeline = get_preprocessing_pipeline()

    preprocessing_pipeline.fit(features, target)
    X, y = preprocessing_pipeline.transform(features, target)

    print(X.head())