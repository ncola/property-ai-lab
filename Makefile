.PHONY: run-server xgb 

run-server:
	@echo "🚀 Starting MLflow tracking server on http://localhost:5002"
	mlflow server --host 0.0.0.0 --port 5002 --backend-store-uri sqlite:///mlflow.db  --default-artifact-root file:./mlartifacts

xgb:
	@echo "🏋️ Training XGBoost model..."
	python3 -m src.pipelines.train_xgboost

xgb-tune:
	@echo "🔍 Running XGBoost hyperparameter tuning and 🏋️ training on best parameters ..."
	python3 -m src.pipelines.train_xgboost --tune

app:
	@echo "Starting Streamlit inference app on http://localhost:8501"
	streamlit run app.py --server.port 8501 --server.headless true
