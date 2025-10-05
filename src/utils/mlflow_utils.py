import os, mlflow

def set_tracking():
    uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5002")
    mlflow.set_tracking_uri(uri)

def start_run(experiment_name: str, run_name: str):
    mlflow.set_experiment(experiment_name)
    return mlflow.start_run(run_name=run_name)

