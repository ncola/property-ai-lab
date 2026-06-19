from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, median_absolute_error
import numpy as np
import matplotlib.pyplot as plt 
import mlflow
import pandas as pd

def compute_regression_metrics(y_true, y_pred):
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    mape = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)
    rmsle = float(np.sqrt(np.mean(np.square(np.log1p(y_pred) - np.log1p(y_true)))))
    medae = float(median_absolute_error(y_true, y_pred))

    return {"rmse": rmse, "mae": mae, "mape": mape, "r2": r2, "rmsle": rmsle, "medae": medae}


def log_regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    mape = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)
    rmsle = float(np.sqrt(np.mean(np.square(np.log1p(y_pred) - np.log1p(y_true)))))
    medae = float(median_absolute_error(y_true, y_pred))

    metrics = {
        f"rmse": rmse,
        f"mae": mae,
        f"mape": mape,
        f"r2": r2,
        f"rmsle": rmsle,
        f"median_ae": medae,
    }
    mlflow.log_metrics(metrics)
    return metrics

def log_training_curves(model, metric_name: str = "rmsle", tag: str = "validation_0"):
    res = getattr(model, "evals_result", lambda: {})()
    if not res or tag not in res:
        return

    for split_name, series in res.items():
        if metric_name in series:
            for i, val in enumerate(series[metric_name]):
                mlflow.log_metric(f"{split_name}_{metric_name}", float(val), step=i)

    frames = {}
    for split_name, series in res.items():
        if metric_name in series:
            frames[split_name] = pd.Series(series[metric_name], name=f"{split_name}_{metric_name}")
    if frames:
        hist_df = pd.concat(frames.values(), axis=1)
        hist_df.index.name = "iteration"
        hist_df.to_csv("eval_history.csv")
        mlflow.log_artifact("eval_history.csv")

    plt.figure()
    for split_name, series in res.items():
        if metric_name in series:
            plt.plot(series[metric_name], label=split_name)
    plt.xlabel("Iteration")
    plt.ylabel(metric_name.upper())
    plt.title("Training curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig("training_curve.png")
    mlflow.log_artifact("training_curve.png")
    plt.close()

def log_feature_importance(model, importance_type: str = "gain", top_k: int | None = None):
    booster = model.get_booster()
    imp = booster.get_score(importance_type=importance_type)  
    if not imp:
        return

    imp_df = pd.DataFrame(sorted(imp.items(), key=lambda x: x[1], reverse=True),
                          columns=["feature", f"importance_{importance_type}"])
    if top_k is not None:
        plot_df = imp_df.head(top_k)
    else:
        plot_df = imp_df

    imp_df.to_csv("feature_importance.csv", index=False)
    mlflow.log_artifact("feature_importance.csv")

    height = max(4, int(len(plot_df) * 0.35))
    plt.figure(figsize=(8, height))
    plt.barh(plot_df["feature"], plot_df[f"importance_{importance_type}"])
    plt.gca().invert_yaxis()
    plt.xlabel(importance_type.capitalize())
    plt.title("Feature importance")
    plt.tight_layout()
    plt.savefig("feature_importance.png")
    mlflow.log_artifact("feature_importance.png")
    plt.close()

def log_pred_vs_actual(y_true, y_pred, fname: str = "pred_vs_actual.png"):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    plt.figure()
    plt.scatter(y_true, y_pred, alpha=0.4)
    line_min = float(min(y_true.min(), y_pred.min()))
    line_max = float(max(y_true.max(), y_pred.max()))
    plt.plot([line_min, line_max], [line_min, line_max], "r--")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Predicted vs Actual")
    plt.tight_layout()
    plt.savefig(fname)
    mlflow.log_artifact(fname)
    plt.close()

def log_residuals_hist(y_true, y_pred, fname: str = "residuals_hist.png", bins: int = 40):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    residuals = y_true - y_pred

    plt.figure()
    plt.hist(residuals, bins=bins, alpha=0.8)
    plt.xlabel("Residual (y_true - y_pred)")
    plt.ylabel("Count")
    plt.title("Residual distribution")
    plt.tight_layout()
    plt.savefig(fname)
    mlflow.log_artifact(fname)
    plt.close()
