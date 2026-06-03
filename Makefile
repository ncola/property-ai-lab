MLFLOW_URI := http://localhost:5002


.PHONY: run-server app-automatic docker-build docker-up docker-down docker-logs

run-server:
	@echo "🚀 Starting MLflow tracking server on $(MLFLOW_URI)"
	mlflow server --host 0.0.0.0 --port 5002 --backend-store-uri sqlite:///mlflow.db  --default-artifact-root file:./mlartifacts

xgb:
	@echo "🏋️ Training XGBoost model..."
	python3 -m src.pipelines.train_xgboost

xgb-tune:
	@echo "🔍 Running XGBoost hyperparameter tuning and 🏋️ training on best parameters ..."
	python3 -m src.pipelines.train_xgboost --tune

app-manual:
	@echo "Starting Streamlit inference app (manual data input) on http://localhost:8503"
	PYTHONPATH=$(PWD) streamlit run app/app_manual.py --server.port 8503 --server.headless true

app-automatic:
	@echo "Starting Streamlit inference app (automatic based on PostgreSQL) on http://localhost:8504"
	PYTHONPATH=$(PWD) streamlit run app/app_automatic.py --server.port 8504 --server.headless true

docker-build:
	@echo "Building Docker images..."
	docker compose build

docker-up:
	@echo "Starting all services (mlflow + apps)..."
	docker compose up -d

docker-down:
	@echo "Stopping all services..."
	docker compose down

docker-logs:
	docker compose logs -f
