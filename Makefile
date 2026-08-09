SERVICE := pyspark

.PHONY: help build up down logs shell test verify clean

help:
	@echo "make build   - build the Docker image"
	@echo "make up      - start the Jupyter+PySpark container (detached)"
	@echo "make down    - stop the container"
	@echo "make logs    - follow container logs (the Jupyter URL/token shows up here)"
	@echo "make shell   - open a shell inside the running container"
	@echo "make test    - run the pytest suite inside the container"
	@echo "make verify  - run the full pipeline, then the test suite"
	@echo "make clean   - remove regenerable data caches, logs, and chart images"

build:
	docker compose build

up:
	docker compose up -d

down:
	docker compose down

logs:
	docker compose logs -f

shell:
	docker compose exec $(SERVICE) bash

test:
	docker compose exec $(SERVICE) python -m pytest tests/

verify:
	docker compose exec $(SERVICE) python run_pipeline.py
	docker compose exec $(SERVICE) python -m pytest tests/

clean:
	rm -f data/raw/*.csv data/clean/*.csv logs/*.log images/*.png
