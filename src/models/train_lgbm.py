import yaml, mlflow, pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import lightgbm as lgb
from mlflow.models.signature import infer_signature

from src.utils.mlflow_utils import set_tracking, start_run
from src.data.database.service import TrainingDataService
from src.data.prepare_features import prepare_features
from src.data.database.db_setup import Database

def main():
    set_tracking()
    cfg = yaml.safe_load(open("configs/config.yaml"))
    target = cfg["data"]["target_col"]
    include = cfg["feature_selection"]["include"]
    hp = cfg["lightgbm"]

    db=Database()
    df_raw = TrainingDataService(db).load_dataframe_all()
    if df_raw.empty:
        raise SystemExit("Brak danych.")
    df, feats, cat_feats = prepare_features(df_raw, target, include)

    X, y = df[feats].copy(), df[target].values
    for c in cat_feats:
        if c in X.columns: X[c] = X[c].astype("category")

    Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.2, random_state=42)
    dtr = lgb.Dataset(Xtr, label=ytr, categorical_feature=cat_feats or None)
    dva = lgb.Dataset(Xva, label=yva, categorical_feature=cat_feats or None)

    with start_run(cfg["experiment"]["name"], "lgbm_random_split"):
        mlflow.log_params(hp | {"_features": ",".join(feats)})
        model = lgb.train(
            params=dict(objective="rmse", metric="rmse", **hp),
            train_set=dtr, valid_sets=[dva],
            num_boost_round=1500,
            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]
        )
        pred = model.predict(Xva, num_iteration=model.best_iteration)
        rmse = np.sqrt(mean_squared_error(yva, pred))
        mape = mean_absolute_percentage_error(yva, pred)
        mlflow.log_metric("val_rmse", float(rmse))
        mlflow.log_metric("val_mape", float(mape))

        imp_df = pd.DataFrame({
            "feature": Xtr.columns,
            "importance": model.feature_importance(importance_type="gain")
        }).sort_values("importance", ascending=False)

        imp_df.to_csv("feature_importance_lgb.csv", index=False)
        mlflow.log_artifact("feature_importance_lgb.csv")

        ax = imp_df.plot(kind="barh", x="feature", y="importance", figsize=(6,4))
        ax.invert_yaxis()
        ax.figure.tight_layout()
        ax.figure.savefig("feature_importance_lgb.png", dpi=150, bbox_inches="tight")
        mlflow.log_artifact("feature_importance_lgb.png")

        signature = infer_signature(Xtr, model.predict(Xtr, num_iteration=model.best_iteration))

        mlflow.lightgbm.log_model(
            lgb_model= model,
            name="model",
            signature=signature
        )
        print(f"[LightGBM] RMSE={rmse:.3f} MAPE={mape:.3f} n_feats={len(feats)}")

if __name__ == "__main__":
    main()
