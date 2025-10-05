import yaml, mlflow, pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from catboost import CatBoostRegressor, Pool
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
    hp = cfg["catboost"]

    db=Database()
    df_raw = TrainingDataService(db).load_dataframe_all()
    if df_raw.empty:
        raise SystemExit("Brak danych.")
    df, feats, cat_feats = prepare_features(df_raw, target, include)

    X, y = df[feats], df[target].values
    Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.2, random_state=42)

    cat_idx = [Xtr.columns.get_loc(c) for c in cat_feats] if cat_feats else None
    tr, va = Pool(Xtr, ytr, cat_features=cat_idx), Pool(Xva, yva, cat_features=cat_idx)

    with start_run(cfg["experiment"]["name"], "catboost_random_split"):
        mlflow.log_params(hp | {"_features": ",".join(feats)})
        model = CatBoostRegressor(loss_function="RMSE", eval_metric="RMSE",
                                  random_seed=42, verbose=False, **hp)
        model.fit(tr, eval_set=va, use_best_model=True)

        pred = model.predict(Xva)
        rmse = np.sqrt(mean_squared_error(yva, pred))
        mape = mean_absolute_percentage_error(yva, pred)
        mlflow.log_metric("val_rmse", float(rmse))
        mlflow.log_metric("val_mape", float(mape))

        imp_df = pd.DataFrame({
            "feature": Xtr.columns,
            "importance": model.get_feature_importance()
        }).sort_values("importance", ascending=False)

        imp_df.to_csv("feature_importance_cb.csv", index=False)  
        mlflow.log_artifact("feature_importance_cb.csv")

        ax = imp_df.plot(kind="barh", x="feature", y="importance", figsize=(6,4))
        ax.invert_yaxis()
        ax.figure.tight_layout()
        ax.figure.savefig("feature_importance_cb.png", dpi=150, bbox_inches="tight")
        mlflow.log_artifact("feature_importance_cb.png")

        signature = infer_signature(Xtr, model.predict(Xtr))

        mlflow.catboost.log_model(
            cb_model=model,
            name="model",
            signature=signature
        )

        print(f"[CatBoost] RMSE={rmse:.3f} MAPE={mape:.3f} n_feats={len(feats)}")

if __name__ == "__main__":
    main()
