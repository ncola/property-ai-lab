import yaml, mlflow, pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import xgboost as xgb
from mlflow.models.signature import infer_signature

from src.utils.mlflow_utils import set_tracking, start_run
from src.data.database.service import TrainingDataService
from src.data.prepare_features import prepare_features
from src.data.database.db_setup import Database
def main():
    set_tracking()
    cfg = yaml.safe_load(open("configs/config.yaml"))
    exp_name   = cfg["experiment"]["name"]
    target_col = cfg["data"]["target_col"]
    include    = cfg["feature_selection"]["include"]
    hp         = cfg["xgboost"]

    db=Database()
    df_raw = TrainingDataService(db).load_dataframe_all()
    if df_raw.empty:
        raise SystemExit("Brak danych.")
    df, feature_cols, cat_cols = prepare_features(df_raw, target_col, include)

    X = df[feature_cols].copy()
    y = df[target_col].values

    for c in cat_cols:
        if c in X.columns:
            X[c] = X[c].astype("category")

    Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.2, random_state=42)

    dtr = xgb.DMatrix(Xtr, label=ytr, enable_categorical=True)
    dva = xgb.DMatrix(Xva, label=yva, enable_categorical=True)

    num_rounds = hp.get("n_estimators", 800)
    params = dict(
        objective="reg:squarederror",
        tree_method="hist",
        enable_categorical=True,   
        seed=42,                   
        **{k: v for k, v in hp.items() if k != "n_estimators"}
    )


    with start_run(exp_name, "xgboost_random_split"):
        mlflow.log_params(params | {"_features": ",".join(feature_cols)})

        model = xgb.train(
            params,
            dtr,
            num_boost_round=num_rounds,
            evals=[(dva, "val")],
            verbose_eval=False
        )

        pred = model.predict(dva)
        rmse = np.sqrt(mean_squared_error(yva, pred))
        mape = mean_absolute_percentage_error(yva, pred)

        mlflow.log_metric("val_rmse", float(rmse))
        mlflow.log_metric("val_mape", float(mape))

        score = model.get_score(importance_type="gain")
        imp_df = pd.DataFrame(sorted(score.items(), key=lambda x: x[1], reverse=True),
                              columns=["feature", "importance"])
        imp_df.to_csv("feature_importance_xgb.csv", index=False)
        mlflow.log_artifact("feature_importance_xgb.csv")

        ax = imp_df.head(20).plot(kind="barh", x="feature", y="importance", figsize=(6, 4))
        ax.invert_yaxis()
        ax.figure.tight_layout()
        ax.figure.savefig("feature_importance_xgb.png", dpi=150, bbox_inches="tight")
        mlflow.log_artifact("feature_importance_xgb.png")

        sig = infer_signature(Xtr, model.predict(xgb.DMatrix(Xtr, enable_categorical=True)))

        mlflow.xgboost.log_model(
            xgb_model=model,
            name="model",
            signature=sig
        )

        print(f"[XGBoost] RMSE={rmse:.3f} MAPE={mape:.3f} n_feats={len(feature_cols)}")


if __name__ == "__main__":
    main()