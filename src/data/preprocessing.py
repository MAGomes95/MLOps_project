from typing import Optional, List, Tuple
from pathlib import Path

import fire
import pandas as pd
import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from src.logger import get_console_logger

logger = get_console_logger()

def get_features_and_target(path_to_input: Optional[Path]) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Loads the data from provided path and returns separatly features and target
    """
    dataset = pd.read_csv(path_to_input)
    features = dataset[np.setdiff1d(dataset.columns, "target")]
    target = dataset["target"]

    return features, target


class FeatureCrossing(BaseEstimator, TransformerMixin):
    """
    Expands the input dataframe with feature crosses

    The new columns would be named like {feature_1}_x_{feature_2}
    """
    def __init__(self, features: Tuple[str]):
        self.features = features

    def fit(self, X: pd.DataFrame) -> "FeatureCrossing":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Computes the feature crosses given operation and feature names from X"""
        if len(self.features) <= 1:
            logger.error("Feature crosses require at least two features to be computed")
            raise TypeError
        if any([feature not in X.columns for feature in self.features]):
            logger.error("At least, one of the provided features is not in the dataset")
            raise ValueError
        
        feature_cross = np.prod(X[self.features].values, axis=1)
        X["_x_".join(self.features)] = feature_cross
        return X
    
    def inverse_transform(self, X: pd.DataFrame, feature_cross: str) -> pd.DataFrame:
        """Inverse the feature cross transformation"""
        X.drop(columns=[feature_cross], inplace=True)
        return X
    
def get_preprocessing_pipeline(feature_crosses: List[Tuple[str]] = None) -> Pipeline:
    """Returns the preprocessing pipeline."""
    pipeline_steps = [
        # Expanding the dataset with Feature Crosses
        (f"feature_cross_{i}", FeatureCrossing(features=this_set) for i, this_set in enumerate(feature_crosses)),
        # Dataset Normalization
        ("Normalization", StandardScaler())
    ]

    return Pipeline(steps=pipeline_steps)

if __name__ == "__main__":

    features, target = fire.Fire(get_features_and_target)
    
    preprocessing_pipeline = get_preprocessing_pipeline(
        feature_crosses=[
            ("Coughing of Blood", "Genetic Risk"), 
            ("Dust Allergy", "Weight Loss"), 
            ("Alcohol use", "Chest Pain")
        ]
    )

    preprocessing_pipeline.fit(features)
    X = preprocessing_pipeline.transform(features)

    print(X.head())