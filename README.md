# TMDB Movie Data Analysis (PySpark)

A PySpark port of a TMDB movie-analysis pipeline (ingestion → preprocessing → analysis → visualization), running in Docker.

## Setup

1. **Docker Desktop** must be installed and running.
2. Copy `.env.example` to `.env` and fill in your own TMDB v4 bearer token (get one at https://developer.themoviedb.org/reference/getting-started):
   ```bash
   cp .env.example .env
   ```
3. Build and start the container:
   ```bash
   make build
   make up
   ```
4. Get the Jupyter URL + access token:
   ```bash
   make logs
   ```
   Look for a line like `http://127.0.0.1:8888/lab?token=...` and open it in a browser — or point VS Code's Jupyter extension at it via **Jupyter: Specify Jupyter Server for Connections**.

## Running the analysis

- **Interactively**: open `MAIN.ipynb` in the Jupyter server above and run all cells top to bottom.
- **Alternative**: `python run_pipeline.py` from a shell inside the container (`make shell`), or `make verify` (see below).

Either way, the first run fetches all 19 movie ids from TMDB (id `0` is an intentionally-invalid placeholder and always 404s, leaving 18 real movies) and caches the raw response to `data/raw/movies_raw.csv`. Every subsequent run reuses that cache and makes **zero** further API calls for ids already fetched — this is why re-running ingestion doesn't re-spend API quota.

## Makefile targets

| Command | What it does |
| --- | --- |
| `make build` | Build the Docker image |
| `make up` / `make down` | Start / stop the container |
| `make logs` | Follow container logs (Jupyter URL + token show up here) |
| `make shell` | Open a shell inside the running container |
| `make test` | Run the pytest suite |
| `make verify` | Run the full pipeline (`run_pipeline.py`), then the test suite |
| `make clean` | Remove regenerable data caches, logs, and chart images |



## Notes

- No pandas anywhere in this pipeline — Spark handles ingestion through analysis, and `visualization.py` hands matplotlib plain Python lists via `.collect()`.
- Error handling and data-quality checks live inline inside `src/ingestion.py` and `src/preprocessing.py`.

- `tests/test_parity.py` requires `data/clean/movies_clean.csv` to exist (i.e., the pipeline has run at least once with a real token) — it skips cleanly if that cache is missing, rather than erroring.
