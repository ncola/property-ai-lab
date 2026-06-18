MLFLOW_URI := http://localhost:5002


.PHONY: run-server app train train-tune docker-build docker-up docker-down docker-logs

run-server:
	@echo "🚀 Starting MLflow tracking server on $(MLFLOW_URI)"
	mlflow server --host 0.0.0.0 --port 5002 --backend-store-uri sqlite:///mlflow.db  --default-artifact-root file:./mlartifacts

train:
	@echo "🏋️ Training price-per-m² XGBoost models..."
	python3 -m src.pipelines.train_model

train-tune:
	@echo "🔍 Running price-per-m² XGBoost hyperparameter tuning and 🏋️ training on best parameters ..."
	python3 -m src.pipelines.train_model --tune

app:
	@echo "Starting Streamlit multi-page app on http://localhost:8501"
	streamlit run app/streamlit_app.py --server.port 8501 --server.headless true

docker-build:
	@echo "Building Docker images..."
	docker compose build

docker-up:
	@echo "Starting all services (mlflow + app)..."
	docker compose up -d

docker-down:
	@echo "Stopping all services..."
	docker compose down

docker-logs:
	docker compose logs -f
