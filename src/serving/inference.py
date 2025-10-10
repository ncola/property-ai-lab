import src.data.preprocess as pre
import numpy as np, pandas as pd
import mlflow
import yaml
import xgboost as xgb

def features_inference_pipeline(df_raw):
    cfg = yaml.safe_load(open("configs/config.yaml"))
    f_cfg = cfg.get("feature_selection")
    include = f_cfg.get("include") # columns from  model signature, in case
    df = df_raw.copy()

    df = pre.basic_transformations_depending_on_database(df)
    df = pre.create_building_age(df)
    
    X = df[include]

    X, _ = pre.to_categorical(X)  
    X, _ = pre.to_float64(X)

    return X


def predict(df):
    X = xgb.DMatrix(df, enable_categorical=True)
    try: 
        cfg = yaml.safe_load(open("configs/config.yaml"))
        model_uri = cfg["inference"]["model_uri"] 

        model = mlflow.xgboost.load_model(model_uri)
    except Exception as e:
        raise RuntimeError(f"Couldn't load model from MLflow ({model_uri}): {e}")

    y_pred = model.predict(X).astype(int)
    return y_pred

