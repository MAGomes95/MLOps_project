import os
from typing import Dict

from pydantic import BaseModel
import pandas as pd

from src.models.model_registry_api import load_model_from_registry
from src.logger import get_console_logger

logger = get_console_logger("deployer")

try:
    COMET_ML_WORKSPACE = os.environ["COMET_ML_WORKSPACE"]
    COMET_ML_API_KEY = os.environ["COMET_ML_API_KEY"]
    COMET_ML_MODEL_NAME = os.environ["COMET_ML_MODEL_NAME"]

except:
    raise KeyError("Could not load environmental variables")

model = load_model_from_registry(
    workspace=COMET_ML_WORKSPACE,
    api_key=COMET_ML_API_KEY,
    model_name=COMET_ML_MODEL_NAME
)

class Record(BaseModel):
    Age: float
    Gender: float
    Air_Pollution: float
    Alcohol_Use: float
    Dust_Allergy: float
    Occupational_Hazards: float
    Genetic_Risk: float
    Chronic_Lung_Disease: float
    Balanced_Diet: float
    Obesity: float
    Smoking: float
    Passive_Smoking: float
    Chest_Pain: float
    Wheezing: float
    Swallowing: float
    Clubbing: float
    Cold: float
    Cough: float
    Fatigue: float
    Weight_Loss: float
    Shortness_Breath: float
    Snoring: float
    Cough_x_Genetic: float
    Alergy_x_Weight: float
    Alcohol_x_Pain: float


def predict(record: Record):
    """
    Prediction Generation
    """
    item = Record(**record)
    row = pd.DataFrame([item.dict()])
    prediction = model.predict(row)[0]

    return {"prediction": prediction}


if __name__ == "__main__":
    item = {
        "Age": 23.0,
        "Gender": 0,
        "Air_Pollution": 5.0,
        "Alcohol_Use": 3.0,
        "Dust_Allergy": 7.0,
        "Occupational_Hazards": 2.0,
        "Genetic_Risk": 6.0,
        "Chronic_Lung_Disease": 1.0,
        "Balanced_Diet": 7.0,
        "Obesity": 1.0,
        "Smoking": 1.0,
        "Passive_Smoking": 3.0,
        "Chest_Pain": 1.0,
        "Wheezing": 4.0,
        "Swallowing": 1.0,
        "Clubbing": 1.0,
        "Cold": 2.0,
        "Cough": 1.0,
        "Fatigue": 5.0,
        "Weight_Loss": 1.0,
        "Shortness_Breath": 1.0,
        "Snoring": 6.0,
        "Cough_x_Snoring": 6.0,
        "Alergy_x_Weight": 7.0,
        "Alcohol_x_Pain": 3.0
    }

    prediction = predict(item)
    print(f"{prediction=}")