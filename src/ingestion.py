"""
ingestion.py
============
STEP 1: fetch raw movie data from the TMDB API, cached to CSV so repeated
runs never re-spend API calls on movies already fetched.
"""

import csv
import json
import os
import sys
import time

import requests

from src import config
from src.logger import get_logger

logger = get_logger("ingestion")
try:
    csv.field_size_limit(sys.maxsize)
except OverflowError:
    csv.field_size_limit(2**31 - 1)
REQUIRED_FIELDS = ["id", "title", "status", "budget", "revenue", "genres", "credits", "release_date"]
NESTED_FIELDS = ["genres", "belongs_to_collection", "production_companies",
                  "production_countries", "spoken_languages", "credits"]


def fetch_movie(movie_id, session=None):
    """
    Fetch a single movie (details + credits) from TMDB.

    Parameters
    ----------
    movie_id : int
    session : requests.Session, optional
        A re-used session makes repeated calls faster (keeps the connection
        open). If None, a plain requests.get is used.

    Returns
    -------
    dict or None
        The parsed JSON for the movie, or None when the id is invalid, the
        request failed outright (connection/timeout), or the API responded
        with a non-200 status.
    """
    url = f"{config.BASE_URL}/movie/{movie_id}"
    params = {"append_to_response": "credits"}
    getter = session.get if session is not None else requests.get

    try:
        response = getter(url, headers=config.HEADERS, params=params, timeout=10)
    except requests.exceptions.RequestException as exc:
        logger.warning("Skipping id %s (request failed: %s)", movie_id, exc)
        return None

    if response.status_code != 200:
        logger.warning("Skipping id %s (HTTP %s)", movie_id, response.status_code)
        return None

    data = response.json()
    logger.info("Fetched '%s' (id %s)", data.get("title"), movie_id)
    return data


def _missing_required_fields(record):
    """Return the REQUIRED_FIELDS keys absent from this record, if any."""
    return [field for field in REQUIRED_FIELDS if field not in record]


def fetch_movies(movie_ids=None, pause=0.25):
    """
    Fetch many movies and return the validated records ready to cache.

    Parameters
    ----------
    movie_ids : list of int, optional
        Which movies to fetch. Defaults to config.MOVIE_IDS.
    pause : float
        Seconds to wait between calls so we stay polite / under the rate limit.

    Returns
    -------
    list of dict

    Raises
    ------
    RuntimeError
        If zero usable records were fetched at all -- a handful of bad
        records among many good ones is expected and just logged, but a
        total failure (e.g. a bad token, no network) is worth stopping for.
    """
    if movie_ids is None:
        movie_ids = config.MOVIE_IDS

    records = []
    with requests.Session() as session:
        for movie_id in movie_ids:
            data = fetch_movie(movie_id, session=session)
            if data is not None:
                missing = _missing_required_fields(data)
                if missing:
                    logger.warning("Skipping id %s (missing required field(s): %s)", movie_id, missing)
                else:
                    records.append(data)
            time.sleep(pause)

    if not records:
        raise RuntimeError(
            "Fetched zero usable movie records -- check the TMDB_BEARER_TOKEN and network connection."
        )

    logger.info("Fetched %d/%d requested movies.", len(records), len(movie_ids))
    return records


def save_raw(records, path=None):
    """
    Save records to CSV. Any field whose value is a list/dict is
    JSON-encoded into its cell; everything else is written as-is. This is a
    generic rule over whatever keys actually appear in `records`, not a
    hardcoded column list, so it covers every nested TMDB field without
    preprocessing.py needing to know about this step at all.
    """
    if path is None:
        path = config.RAW_DATA_PATH

    fieldnames = sorted({key for record in records for key in record})

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            row = {
                key: json.dumps(value) if isinstance(value, (list, dict)) else value
                for key, value in record.items()
            }
            writer.writerow(row)

    logger.info("Saved %d records to %s", len(records), path)


def _load_cached_records(path):
    """
    Reload previously cached rows from CSV as-is: every value comes back as
    a plain string (CSV has no type system), and the nested fields are still
    JSON-encoded text. That's fine for merging back through save_raw, which
    only distinguishes real list/dict Python objects from strings -- an
    already-JSON-encoded string just passes through unchanged rather than
    being double-encoded.
    """
    if not os.path.exists(path):
        return []
    with open(path, "r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def ensure_raw_cache(path=None, movie_ids=None):
    """
    Cache-first entry point: fetch from TMDB only for the movie ids that
    aren't already in the cached CSV, merge them into what's cached, and
    rewrite the cache. This is what keeps repeated runs from re-spending API
    calls on movies already fetched.

    Returns
    -------
    list of dict
        Every cached record (old and newly-fetched combined), as plain
        dicts with string-typed scalar fields and JSON-encoded nested
        fields -- the same shape save_raw expects.
    """
    if path is None:
        path = config.RAW_DATA_PATH
    if movie_ids is None:
        movie_ids = config.MOVIE_IDS

    cached_records = _load_cached_records(path)
    cached_ids = {int(r["id"]) for r in cached_records if r.get("id")}
    missing_ids = [mid for mid in movie_ids if mid not in cached_ids]

    if not cached_records:
        logger.info("No cache found at %s -- fetching all %d movie(s).", path, len(movie_ids))
    elif not missing_ids:
        logger.info(
            "Cache hit: %s already covers all %d requested movie(s), no API calls made.",
            path, len(movie_ids),
        )
        return cached_records
    else:
        logger.info(
            "Partial cache at %s: %d/%d id(s) already present, fetching %d missing id(s): %s",
            path, len(cached_ids), len(movie_ids), len(missing_ids), missing_ids,
        )

    try:
        new_records = fetch_movies(missing_ids)
    except RuntimeError:
        # fetch_movies raises when every id it was given fails -- correct
        # for a from-scratch fetch (a bad token really should stop the
        # pipeline), but wrong here if we already have a good cache: e.g.
        # once the 18 real movies are cached, id 0 (the brief's own
        # intentionally-invalid placeholder, which always 404s) is
        # permanently "missing" and becomes the whole gap on every future
        # run. That shouldn't be fatal when there's already usable data.
        if cached_records:
            logger.warning(
                "Could not fetch any of the %d missing id(s) %s -- keeping the %d already-cached "
                "record(s) as-is (expected for ids TMDB doesn't recognize, e.g. the brief's own id 0).",
                len(missing_ids), missing_ids, len(cached_records),
            )
            return cached_records
        raise

    all_records = cached_records + new_records
    save_raw(all_records, path)
    return all_records
