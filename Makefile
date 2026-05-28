.PHONY: install data train dashboard api docker test clean

install:
	pip install -r requirements.txt

data:
	python scripts/download_data.py

train:
	python src/model/train.py

api:
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

dashboard:
	streamlit run src/dashboard/app.py --server.port 8501

docker:
	docker compose up --build

docker-api:
	docker compose up api

test:
	pytest tests/ -v --cov=src --cov-report=term-missing

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .coverage

lint:
	ruff check src/ tests/
	black --check src/ tests/

format:
	black src/ tests/
	ruff check --fix src/ tests/
