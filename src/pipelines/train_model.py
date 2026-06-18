import argparse
from datetime import datetime

import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
import xgboost as xgb
import yaml
from mlflow.models.signature import infer_signature
from scipy.stats import randint
from sklearn.model_selection import KFold, RandomizedSearchCV, train_test_split

import src.data.preprocess as pre
import src.data.price_per_m2_features as price_features
import src.utils.metrics as metrics
import src.utils.mlflow_utils as ml_utils
from src.data.database.db_setup import Database
from src.data.database.service import DataService


TARGET_COL = "log_price_per_m"
MODEL_NAME_PREFIX = "property_price_per_m2"


def _parse_cli():
    p = argparse.ArgumentParser()
    p.add_argument("--n_estimators", type=int)
    p.add_argument("--learning_rate", type=float)
    p.add_argument("--max_depth", type=int)
    p.add_argument("--subsample", type=float)
    p.add_argument("--colsample_bytree", type=float)
    p.add_argument("--reg_alpha", type=float)
    p.add_argument("--reg_lambda", type=float)
    p.add_argument("--min_child_weight", type=float)
    p.add_argument("--tree_method", type=str)
    p.add_argument("--n_jobs", type=int)
    p.add_argument("--market", choices=["primary", "secondary"], action="append")
    p.add_argument("--tune", action="store_true", help="run randomized search before final training")
    return p.parse_args()


def _override_with_cli(hp, args):
    hp = hp.copy()
    for k, v in vars(args).items():
        if v is not None and k not in {"tune", "market"}:
            hp[k] = v
    return hp


def prepare_price_per_m2_training_frame(df_raw):
    df = pre.clean_raw_data(df_raw.copy())
    df = pre.ensure_price_per_m2(df)
    df = price_features.add_price_per_m2_features(df)

    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["price_per_m"] = pd.to_numeric(df["price_per_m"], errors="coerce")
    df["area"] = pd.to_numeric(df["area"], errors="coerce")
    df = df.dropna(subset=["market", "price", "price_per_m", "area"]).copy()
    df = df[(df["price"] > 0) & (df["price_per_m"] > 0) & (df["area"] > 0)].copy()

    keep = pd.Series(False, index=df.index)
    for market in df["market"].dropna().unique():
        market_mask = df["market"] == market
        lo, hi = df.loc[market_mask, "price_per_m"].quantile([0.01, 0.99])
        keep = keep | (market_mask & df["price_per_m"].between(lo, hi))
    df = df[keep].copy()
    df[TARGET_COL] = np.log(df["price_per_m"])
    return df


def build_price_per_m2_xy(df_market, feature_columns):
    df = df_market.dropna(subset=[TARGET_COL, "price", "area"]).copy()
    for col in feature_columns:
        if col not in df.columns:
            df[col] = pd.NA

    X = df[feature_columns].copy()
    y = df[TARGET_COL].astype(float).copy()
    aux = df[["price", "area"]].astype(float).copy()

    X, cat_cols = pre.to_categorical(X)
    X, _ = pre.to_float64(X)
    return X, y, aux, cat_cols


def tune_hyperparams(X, y, base_params, n_iter=30, cv_splits=5, random_state=42):
    model = xgb.XGBRegressor(
        enable_categorical=True,
        tree_method=base_params.get("tree_method", "hist"),
        random_state=random_state,
        n_jobs=base_params.get("n_jobs", -1),
        eval_metric="rmse",
    )

    search_space = {
        "max_depth": randint(3, 11),
        "learning_rate": np.linspace(0.01, 0.20, 15),
        "subsample": np.linspace(0.8, 1.0, 4),
        "colsample_bytree": np.linspace(0.8, 1.0, 4),
        "min_child_weight": randint(1, 10),
        "reg_alpha": np.linspace(0.0, 0.5, 6),
        "reg_lambda": np.linspace(0.5, 2.0, 7),
        "n_estimators": randint(400, 2001),
    }

    cv = KFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=search_space,
        n_iter=n_iter,
        scoring="neg_root_mean_squared_error",
        cv=cv,
        verbose=1,
        n_jobs=-1,
        random_state=random_state,
        refit=True,
    )
    search.fit(X, y)
    return search


def _predict_total_price(model, X, area):
    return np.exp(model.predict(X)) * area.to_numpy(dtype=float)


def _log_run_artifacts(model, X, cat_cols, cfg):
    mlflow.log_dict({"features": list(X.columns)}, artifact_file="feature_names.json")
    mlflow.log_dict({"categorical_columns": cat_cols}, artifact_file="categorical_columns.json")
    mlflow.log_dict(cfg, artifact_file="config_used.json")

    booster = model.get_booster()
    mlflow.log_dict(
        {
            "gain": booster.get_score(importance_type="gain"),
            "weight": booster.get_score(importance_type="weight"),
        },
        artifact_file="feature_importance.json",
    )


def train_market(df, market, cfg, hp, tune_flag=False):
    training_cfg = cfg.get("training", {})
    inference_cfg = cfg["inference"]
    feature_columns = inference_cfg["feature_columns"]
    test_size = training_cfg.get("test_size", 0.2)
    early_stopping = training_cfg.get("early_stopping_rounds", 50)
    eval_metric = training_cfg.get("eval_metric", "rmse")
    random_state = training_cfg.get("random_state", hp.get("random_state", 42))

    df_market = df[df["market"] == market].copy()
    if df_market.empty:
        raise ValueError(f"No rows available for market: {market}")

    X, y, aux, cat_cols = build_price_per_m2_xy(df_market, feature_columns)
    X_train, X_valid, y_train, y_valid, aux_train, aux_valid = train_test_split(
        X,
        y,
        aux,
        test_size=test_size,
        random_state=random_state,
    )

    tuned_params = {}
    cv_best_score = None
    if tune_flag:
        search = tune_hyperparams(
            X_train,
            y_train,
            base_params=hp,
            n_iter=training_cfg.get("tune_n_iter", 40),
            cv_splits=training_cfg.get("cv_splits", 5),
            random_state=random_state,
        )
        tuned_params = search.best_params_
        cv_best_score = float(search.best_score_)
        hp = {**hp, **tuned_params}

    model = xgb.XGBRegressor(
        enable_categorical=True,
        tree_method=hp.get("tree_method", "hist"),
        n_estimators=hp.get("n_estimators", 800),
        max_depth=hp.get("max_depth", 6),
        learning_rate=hp.get("learning_rate", 0.05),
        subsample=hp.get("subsample", 0.8),
        colsample_bytree=hp.get("colsample_bytree", 0.8),
        reg_alpha=hp.get("reg_alpha", 0.0),
        reg_lambda=hp.get("reg_lambda", 1.0),
        min_child_weight=hp.get("min_child_weight", 1.0),
        n_jobs=hp.get("n_jobs", -1),
        random_state=random_state,
        eval_metric=eval_metric,
        early_stopping_rounds=early_stopping,
    )

    run_name = f"price_per_m2_model__{market}"
    logged_model_name = f"{MODEL_NAME_PREFIX}_{market}"

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tag("model_contract", "price_per_m2_model")
        mlflow.set_tag("market", market)
        mlflow.set_tag("target_used", TARGET_COL)
        mlflow.set_tag("post_transform", "exp(pred)*area")
        mlflow.log_params({f"hp_{k}": v for k, v in hp.items()})
        mlflow.log_param("market", market)
        mlflow.log_param("target_col", TARGET_COL)
        mlflow.log_param("test_size", test_size)
        mlflow.log_param("early_stopping_rounds", early_stopping)
        mlflow.log_param("eval_metric", eval_metric)
        mlflow.log_param("n_features", X.shape[1])
        mlflow.log_param("n_train", len(X_train))
        mlflow.log_param("n_valid", len(X_valid))

        if tuned_params:
            mlflow.log_params({f"tuned_{k}": v for k, v in tuned_params.items()})
            mlflow.log_metric("cv_best_neg_rmse", cv_best_score)

        print(f"[{market}] fitting XGBoost...", flush=True)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_train, y_train), (X_valid, y_valid)],
            verbose=True,
        )
        print(f"[{market}] fit done; computing metrics...", flush=True)

        y_pred_valid = _predict_total_price(model, X_valid, aux_valid["area"])
        y_pred_train = _predict_total_price(model, X_train, aux_train["area"])
        y_valid_price = aux_valid["price"].to_numpy(dtype=float)
        y_train_price = aux_train["price"].to_numpy(dtype=float)

        valid_metrics = metrics.log_regression_metrics(y_valid_price, y_pred_valid)
        train_metrics = metrics.compute_regression_metrics(y_train_price, y_pred_train)
        mlflow.log_metrics({f"train_{k}": v for k, v in train_metrics.items()})

        log_rmse = float(np.sqrt(np.mean((y_valid.to_numpy() - model.predict(X_valid)) ** 2)))
        mlflow.log_metric("log_price_per_m_rmse", log_rmse)

        sig_X = X_train.head(min(500, len(X_train)))
        signature = infer_signature(sig_X, model.predict(sig_X))
        print(f"[{market}] logging model to MLflow...", flush=True)
        mlflow.xgboost.log_model(
            model,
            name=logged_model_name,
            signature=signature,
        )
        print(f"[{market}] logging artifacts...", flush=True)
        _log_run_artifacts(model, X, cat_cols, cfg)
        metrics.log_training_curves(model, metric_name=eval_metric, tag="validation_1")
        metrics.log_feature_importance(model, importance_type="gain", top_k=40)
        metrics.log_pred_vs_actual(y_valid_price, y_pred_valid, fname=f"pred_vs_actual_{market}.png")
        metrics.log_residuals_hist(y_valid_price, y_pred_valid, fname=f"residuals_hist_{market}.png", bins=40)
        print(f"[{market}] MLflow logging done.", flush=True)

    return {
        "run_id": run.info.run_id,
        "market": market,
        "model_name": logged_model_name,
        "metrics": valid_metrics,
        "features": list(X.columns),
    }


def main():
    args = _parse_cli()
    start = datetime.now()
    print(f"Starting training at: {start}")

    ml_utils.set_tracking()
    cfg = yaml.safe_load(open("configs/config.yaml"))
    mlflow.set_experiment(cfg["experiment"]["name"])

    hp = _override_with_cli(cfg["xgboost"], args)

    db = Database()
    df_raw = DataService(db).load_dataframe_all()
    if df_raw.empty:
        raise SystemExit("There is no data.")

    df = prepare_price_per_m2_training_frame(df_raw)
    markets = args.market or sorted(df["market"].dropna().unique().tolist())
    print(f"Training markets: {markets}")

    for market in markets:
        result = train_market(df, market, cfg, hp.copy(), tune_flag=args.tune)
        m = result["metrics"]
        print(
            f"{market}: run_id={result['run_id']} model={result['model_name']} "
            f"MAE={m['mae']:,.0f} MAPE={m['mape']:.2f}% R2={m['r2']:.4f}"
        )

    end = datetime.now()
    print(f"Training time: {end - start}")


if __name__ == "__main__":
    main()
