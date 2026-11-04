.PHONY: build train score pipeline api

build:
	python -m src.data.batch_build

train:
	python -m src.training.train

score:
	python -m src.scoring.batch_score

pipeline:
	python run_pipeline.py

api:
	uvicorn src.api.app:app --host 0.0.0.0 --port 8000