import yaml, mlflow, pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV, KFold
from scipy.stats import randint
from mlflow.models.signature import infer_signature

import xgboost as xgb
import argparse

import src.utils.mlflow_utils as ml_utils
import src.utils.metrics as metrics
from src.data.database.service import DataService
from src.data.prepare_features import features_train_pipeline
from src.data.database.db_setup import Database
from datetime import datetime


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
    p.add_argument("--tune", action="store_true", help="run randomized search before final training")
    return p.parse_args()

def _override_with_cli(hp):
    args = _parse_cli()
    tune_flag = args.tune

    for k, v in vars(args).items():
        if v is not None and k != "tune":
            hp[k] = v

    return hp, tune_flag


def tune_hyperparams(X, y, base_params, n_iter = 30, cv_splits = 5, random_state = 1):
    model = xgb.XGBRegressor(
        enable_categorical=True,
        tree_method=base_params.get("tree_method", "hist"),
        random_state=random_state,
        n_jobs=base_params.get("n_jobs", -1),
    )

    search_space = {
        "max_depth": randint(5, 13),                 
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


def main():
    start = datetime.now()
    print(f"⏱️ Starting time: {start}")

    ml_utils.set_tracking()
    cfg = yaml.safe_load(open("configs/config.yaml"))
    exp_name = cfg["experiment"]["name"]

    mlflow.set_experiment(exp_name)

    target_col = cfg["data"]["target_col"]
    include = cfg["feature_selection"]["include"]
    hp= cfg["xgboost"]
    trening_cfg = cfg.get("training", {})
    test_size = trening_cfg.get("test_size", 0.2)
    early_stopping = trening_cfg.get("early_stopping_rounds", 50)
    eval_metric = trening_cfg.get("eval_metric", "rmse")  
    random_state = hp.get("random_state", 1)

    hp, tune_flag = _override_with_cli(hp)

    # getting and prepare data
    db = Database()
    df_raw = DataService(db).load_dataframe_all()
    if df_raw.empty:
        raise SystemExit("There is no data.")

    X, y, cat_cols = features_train_pipeline(df_raw, target_col, include)

    # split the data
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # RandomizedSearchCV if flag --tune 
    best_params = {}
    if tune_flag:
        with mlflow.start_run(run_name="xgboost - RandomizedSearchCV"):
            mlflow.log_params({"tune": True, "cv_splits": trening_cfg.get("cv_splits", 5)})
            search = tune_hyperparams(
                X_train, y_train,
                base_params=hp,
                n_iter=trening_cfg.get("tune_n_iter", 40),
                cv_splits=trening_cfg.get("cv_splits", 5),
                random_state=random_state
            )
            best_params = search.best_params_
            mlflow.log_params({f"tuned_{k}": v for k, v in best_params.items()})
            mlflow.log_metric("cv_best_neg_rmse", float(search.best_score_))
            print("[tuning] best params:", best_params)
            print("[tuning] best score (neg RMSE):", search.best_score_)

            #overwrite parameters
            hp.update(best_params)

    model = xgb.XGBRegressor(
        enable_categorical=True,
        tree_method=hp.get("tree_method", "hist"),
        n_estimators=hp.get("n_estimators", 1200),
        max_depth=hp.get("max_depth", 8),
        learning_rate=hp.get("learning_rate", 0.05),
        subsample=hp.get("subsample", 0.8),
        colsample_bytree=hp.get("colsample_bytree", 0.8),
        reg_alpha=hp.get("reg_alpha", 0.0),
        reg_lambda=hp.get("reg_lambda", 1.0),
        min_child_weight=hp.get("min_child_weight", 1.0),
        n_jobs=hp.get("n_jobs", -1),
        random_state=random_state,
        eval_metric=eval_metric,
        early_stopping_rounds=20
    )


    # proper trening run 
    with mlflow.start_run(run_name="xgboost - price"):
        # log 
        mlflow.log_params({f"cfg_{k}": v for k, v in cfg.get("experiment", {}).items() if k != "name"})
        mlflow.log_params(hp)
        mlflow.log_param("target_col", target_col)
        mlflow.log_param("test_size", test_size)
        mlflow.log_param("early_stopping_rounds", early_stopping)
        mlflow.log_param("eval_metric", eval_metric)
        mlflow.log_param("n_features", X.shape[1])

        #training
        model.fit(
            X_train, y_train,
            eval_set=[(X_valid, y_valid)],
            verbose=True,
        )

        # evaluation
        y_pred_val = model.predict(X_valid)
        metrics.log_regression_metrics(y_valid.to_numpy(), y_pred_val)

        sig_X = X_train.head(500)
        signature = infer_signature(X_train, model.predict(sig_X))

        mlflow.xgboost.log_model(model, 
                                 name="model",
                                 signature=signature
                                 )
        mlflow.log_dict({"features": list(X.columns)}, artifact_file="feature_names.json")
        mlflow.log_dict({
            "categorical_columns": cat_cols,
            "categories": {
                c: X[c].cat.categories.astype(str).tolist()
                for c in cat_cols if pd.api.types.is_categorical_dtype(X[c])
            },
        }, artifact_file="categorical_columns.json")
        mlflow.log_dict(cfg, artifact_file="config_used.json")

        booster = model.get_booster()
        mlflow.log_dict(
            {
                "gain": booster.get_score(importance_type="gain"),
                "weight": booster.get_score(importance_type="weight"),
            },
            artifact_file="feature_importance.json",
        )

        metrics.log_training_curves(model, metric_name=eval_metric, tag="validation_0")

        metrics.log_feature_importance(model, importance_type="gain", top_k=40)
        metrics.log_pred_vs_actual(y_valid.to_numpy(), y_pred_val, fname="pred_vs_actual.png")
        metrics.log_residuals_hist(y_valid.to_numpy(), y_pred_val, fname="residuals_hist.png", bins=40)

        end = datetime.now()
        print(f"⏱️ Training time: {end-start}")

if __name__ == "__main__":
    main()