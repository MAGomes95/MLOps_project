from typing import Optional, List, Tuple
from pathlib import Path

import fire
import pandas as pd
import numpy as np
from sklearn.preprocessing import (
    LabelEncoder,
    MinMaxScaler,
    PolynomialFeatures,
    FunctionTransformer
)
from sklearn.feature_selection import (
    SelectKBest, 
    mutual_info_classif
)
from sklearn.pipeline import Pipeline
from sklearn.base import (
    TransformerMixin,
    BaseEstimator
)
#from logger import get_console_logger

#logger = get_console_logger()


class LabelEncoderTransformer(BaseEstimator, TransformerMixin):
    """
    This class implements a classic LabelEncoder class,
    with the turn-around of taking X,y as arguments instead
    of only y.
    """
    def __init__(self):
        self.encoder = LabelEncoder()

    def fit(self, X, y):
        self.encoder = self.encoder.fit(y)
        return self
    
    def transform(self, X, y):
        return self.encoder.transform(y)

    def fit_transform(self, X, y):
        self = self.fit(X, y)
        output = self.transform(X, y)
        return X, output.reshape(-1, 1)


def get_features_and_target(
        path_to_input: Optional[Path],
        target: str = "Level"
    ) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Loads the data from provided path and returns separatly features and target
    """
    dataset = pd.read_csv(path_to_input)
    features = dataset[np.setdiff1d(dataset.columns, [target])]
    target = dataset[target].values

    return features, target

def feature_dropper(X: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    """
    Drops feature from the received dataset X
    """
    return X[np.setdiff1d(X.columns, features)]

    
def get_preprocessing_pipeline() -> Pipeline:
    """Returns the preprocessing pipeline."""
    pipeline_steps = [
        ("drop ID", FunctionTransformer(feature_dropper, kw_args={"features": ["Patient Id"]})),
        ("interaction features", PolynomialFeatures(interaction_only=True, include_bias=False)),
        ("feature selection", SelectKBest(mutual_info_classif, k=5)),
        ("scaling", MinMaxScaler()),
        ("label encoding", LabelEncoderTransformer()),
    ]

    return Pipeline(steps=pipeline_steps)


if __name__ == "__main__":

    features, target = fire.Fire(get_features_and_target)
    
    preprocessing_pipeline = get_preprocessing_pipeline()

    X, y = preprocessing_pipeline.fit_transform(features, target)

    print(X[:5], y[:5], sep="\n")