import pickle

from comet_ml import API
from sklearn.pipeline import Pipeline

from src.logger import get_console_logger

logger = get_console_logger()

def load_model_from_registry(
        workspace: str,
        api_key: str,
        model_name: str,
        status: str = "Production",
) -> Pipeline:
    """
    Loads model from remote model registry
    """
    api = API(api_key)
    model_details = api.get_registry_model_details(workspace, model_name)["versions"]
    model_versions = [md["version"] for md in model_details if md["status"] == status]
    if len(model_versions) == 0:
        logger.error("No production model found")
        raise ValueError
    else:
        logger.info(f"Found {status} model versions: {model_versions}")
        model_version = model_versions[0]

    api.download_registry_model(
        workspace,
        registry_name=model_name,
        version=model_version,
        output_path="../models/",
        expand=True
    )

    with open("../models/model.pkl", "rb") as handler:
        model = pickle.load(handler)

    return model

