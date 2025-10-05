.PHONY: run-server lgb cat xgb

run-server:
	mlflow server --host 0.0.0.0 --port 5002 --backend-store-uri sqlite:///mlflow.db  --default-artifact-root file:./mlartifacts

lgb:
	python3 -m src.models.train_lgbm

cat:
	python3 -m src.models.train_catboost

xgb:
	python3 -m src.models.train_xgboost