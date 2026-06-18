from functools import lru_cache
from pathlib import Path

import mlflow.xgboost as mlfxgb
import numpy as np
import pandas as pd
import yaml

import src.data.preprocess as pre
import src.data.price_per_m2_features as price_features


CONFIG_PATH = Path("configs/config.yaml")


@lru_cache(maxsize=1)
def _load_config():
    with CONFIG_PATH.open() as f:
        return yaml.safe_load(f)


@lru_cache(maxsize=4)
def _load_model(model_uri: str):
    return mlfxgb.load_model(model_uri)


def _prepare_inference_frame(df_raw):
    df = pre.clean_raw_data(df_raw.copy())
    return price_features.add_price_per_m2_features(df)


def _build_model_input(df, feature_columns):
    X = df.copy()
    for col in feature_columns:
        if col not in X.columns:
            X[col] = pd.NA
    X = X[feature_columns].copy()
    X, _ = pre.to_categorical(X)
    X, _ = pre.to_float64(X)
    return X


def predict(df_raw):
    inference_cfg = _load_config()["inference"]
    df = _prepare_inference_frame(df_raw)
    if "market" not in df.columns:
        raise ValueError("Missing required column: market")
    if "area" not in df.columns:
        raise ValueError("Missing required column: area")

    areas = pd.to_numeric(df["area"], errors="coerce")
    if areas.isna().any() or (areas <= 0).any():
        raise ValueError("Area must be present and greater than 0.")

    model_uris = inference_cfg["model_uris"]
    feature_columns = inference_cfg["feature_columns"]
    markets = df["market"]
    predicted_prices = pd.Series(index=df.index, dtype="float64")

    for market in markets.dropna().unique():
        if market not in model_uris:
            raise ValueError(f"No model URI configured for market: {market}")
        idx = markets == market
        X = _build_model_input(df.loc[idx], feature_columns)
        model = _load_model(model_uris[market])
        log_price_per_m2 = model.predict(X)
        predicted_prices.loc[idx] = np.exp(log_price_per_m2) * areas.loc[idx].to_numpy()

    if predicted_prices.isna().any():
        missing = markets[predicted_prices.isna()].unique().tolist()
        raise ValueError(f"Could not predict rows for markets: {missing}")

    return predicted_prices.round().astype("int64").to_numpy()
