import os
from typing import Dict
from logging import Logger

from pydantic import (
    BaseModel,
    Field
)
import pandas as pd
from cerebrium import get_secret

from models.model_registry_api import load_model_from_registry
from logger import get_console_logger

logger = get_console_logger("deployer")

try:
    # Running inference on Cerebrium platfotm
    COMET_WORKSPACE = get_secret("COMET_WORKSPACE")
    COMET_API_KEY = get_secret("COMET_API_KEY")
    COMET_MODEL = get_secret("COMET_MODEL")
except:
    # Running inference locally
    from dotenv import load_dotenv
    load_dotenv()
    COMET_WORKSPACE = os.environ["COMET_WORKSPACE"]
    COMET_API_KEY = os.environ["COMET_API_KEY"]
    COMET_MODEL = os.environ["COMET_MODEL"]

model = load_model_from_registry(
    workspace=COMET_WORKSPACE,
    api_key=COMET_API_KEY,
    model_name=COMET_MODEL
)

class Record(BaseModel):
    Age: float
    Gender: float
    Air_Pollution: float = Field(..., alias="Air Pollution")
    Alcohol_Use: float = Field(..., alias="Alcohol use")
    Dust_Allergy: float = Field(..., alias="Dust Allergy")
    Occupational_Hazards: float = Field(..., alias="OccuPational Hazards")
    Genetic_Risk: float = Field(..., alias="Genetic Risk")
    Chronic_Lung_Disease: float = Field(..., alias="chronic Lung Disease")
    Balanced_Diet: float = Field(..., alias="Balanced Diet")
    Obesity: float
    Smoking: float
    Passive_Smoking: float = Field(..., alias="Passive Smoker")
    Chest_Pain: float = Field(..., alias="Chest Pain")
    Wheezing: float
    Swallowing: float = Field(..., alias="Swallowing Difficulty")
    Clubbing: float = Field(..., alias="Clubbing of Finger Nails")
    Cold: float = Field(..., alias="Frequent Cold")
    Cough: float = Field(..., alias="Coughing of Blood")
    DCough: float = Field(..., alias="Dry Cough")
    Fatigue: float
    Weight_Loss: float = Field(..., alias="Weight Loss")
    Shortness_Breath: float = Field(..., alias="Shortness of Breath")
    Snoring: float


def predict(record: dict, run_id: str, logger: Logger) -> Dict[str, int]:
    """
    Prediction Generation
    """
    item = Record(**record)
    row = pd.DataFrame([item.model_dump(by_alias=True)])
    prediction = model.predict(row)[0]

    return {"prediction": prediction}


if __name__ == "__main__":
    item = {
        "Age": 23.0,
        "Gender": 0,
        "Air Pollution": 5.0,
        "Alcohol use": 3.0,
        "Dust Allergy": 7.0,
        "OccuPational Hazards": 2.0,
        "Genetic Risk": 6.0,
        "chronic Lung Disease": 1.0,
        "Balanced Diet": 7.0,
        "Obesity": 1.0,
        "Smoking": 1.0,
        "Passive Smoker": 3.0,
        "Chest Pain": 1.0,
        "Coughing of Blood": 3.0,
        "Wheezing": 4.0,
        "Swallowing Difficulty": 1.0,
        "Clubbing of Finger Nails": 1.0,
        "Frequent Cold": 2.0,
        "Dry Cough": 1.0,
        "Fatigue": 5.0,
        "Weight Loss": 1.0,
        "Shortness of Breath": 1.0,
        "Snoring": 6.0,
    }

    prediction = predict(item)
    print(f"{prediction=}")