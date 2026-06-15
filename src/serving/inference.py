from functools import lru_cache

import src.data.preprocess as pre
import yaml
import pandas as pd
import mlflow.xgboost as mlfxgb


@lru_cache(maxsize=1)
def _load_model(model_uri: str):
    return mlfxgb.load_model(model_uri)

_BIN_COLS = [
    "balcony", "separate_kitchen", "air_conditioning", "roller_shutters", "dishwasher",
    "garage", "anti_burglary_door", "basement", "entryphone", "garden", "internet",
    "monitoring", "terrace", "alarm", "lift", "closed_area",
]
_CAT_STR_COLS = ["market", "building_material", "construction_status", "district", "floor_num"]
_INT_COLS = ["building_age"]
_FLOAT_COLS = ["area", "rooms_num", "building_floors_num"]


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
    X = features_inference_pipeline(df)
    try:
        cfg = yaml.safe_load(open("configs/config.yaml"))
        model_uri = cfg["inference"]["model_uri"]
        model = _load_model(model_uri)
    except Exception as e:
        raise RuntimeError(f"Couldn't load model from MLflow ({model_uri}): {e}")

    y_pred =  model.predict(X).astype(int)

    return y_pred


def predict_calculator(form_df):
    """Predict for a single-row DataFrame coming from the manual input form."""
    cfg = yaml.safe_load(open("configs/config.yaml"))
    features = cfg["feature_selection"]["include"]
    model_uri = cfg["inference"]["model_uri"]

    df = form_df.copy()
    df[_BIN_COLS] = (
        df[_BIN_COLS]
        .replace({"tak": 1, "nie": 0, True: 1, False: 0})
        .astype("int64")
    )
    for c in _INT_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype("int64")
    for c in _FLOAT_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
    for c in _CAT_STR_COLS:
        df[c] = df[c].astype("category")
    df = df[features]

    model = _load_model(model_uri)
    return model.predict(df)

