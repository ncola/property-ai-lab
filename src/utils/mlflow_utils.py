import os
import mlflow
from dotenv import load_dotenv

load_dotenv()


def set_tracking():
    """Configure MLflow tracking URI from env.

    MLflow reads MLFLOW_TRACKING_USERNAME / MLFLOW_TRACKING_PASSWORD
    from env automatically for HTTP basic auth (DagsHub).
    """
    uri = os.getenv("MLFLOW_TRACKING_URI")
    if not uri:
        raise RuntimeError(
            "MLFLOW_TRACKING_URI is not set. "
            "Set it in .env (local) or st.secrets (Streamlit Cloud)."
        )
    mlflow.set_tracking_uri(uri)


def start_run(experiment_name: str, run_name: str):
    mlflow.set_experiment(experiment_name)
    return mlflow.start_run(run_name=run_name)
