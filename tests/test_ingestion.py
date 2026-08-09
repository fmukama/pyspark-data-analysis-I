"""
tests/test_ingestion.py
========================
Covers src/ingestion.py: fetch behavior (success, HTTP error, network
error), required-field validation, the CSV/JSON round-trip in save_raw
(including a title with an embedded quote and comma), and
ensure_raw_cache's cache-gap logic. No test here makes a real network call
-- every `requests` session used is a fake.
"""

import csv
import json

import pytest
import requests

from src import ingestion


class FakeResponse:
    def __init__(self, status_code=200, json_data=None):
        self.status_code = status_code
        self._json_data = json_data or {}

    def json(self):
        return self._json_data


class FakeSession:
    """Stand-in for requests.Session whose .get() is fully scripted, so
    fetch_movies' `with requests.Session() as session:` works unchanged."""

    def __init__(self, responses):
        # movie_id -> FakeResponse, or an Exception instance to raise
        self._responses = responses
        self.requested_ids = []

    def get(self, url, headers=None, params=None, timeout=None):
        movie_id = int(url.rsplit("/", 1)[-1])
        self.requested_ids.append(movie_id)
        result = self._responses[movie_id]
        if isinstance(result, Exception):
            raise result
        return result

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


def _movie_record(movie_id, title="Some Movie", **overrides):
    """A minimal-but-complete valid record covering every REQUIRED_FIELDS key."""
    record = {
        "id": movie_id,
        "title": title,
        "status": "Released",
        "budget": 1000000,
        "revenue": 5000000,
        "genres": [{"id": 28, "name": "Action"}],
        "credits": {"cast": [{"name": "Actor One"}], "crew": [{"name": "Director One", "job": "Director"}]},
        "release_date": "2020-01-01",
    }
    record.update(overrides)
    return record


# --- fetch_movie ---

def test_fetch_movie_success():
    session = FakeSession({299534: FakeResponse(200, _movie_record(299534, "Avengers: Endgame"))})
    result = ingestion.fetch_movie(299534, session=session)
    assert result["title"] == "Avengers: Endgame"


def test_fetch_movie_http_error_returns_none():
    session = FakeSession({0: FakeResponse(404, {})})
    result = ingestion.fetch_movie(0, session=session)
    assert result is None


def test_fetch_movie_connection_error_returns_none():
    session = FakeSession({299534: requests.exceptions.ConnectionError("boom")})
    result = ingestion.fetch_movie(299534, session=session)
    assert result is None


def test_fetch_movie_timeout_returns_none():
    session = FakeSession({299534: requests.exceptions.Timeout("too slow")})
    result = ingestion.fetch_movie(299534, session=session)
    assert result is None


# --- fetch_movies ---

def test_fetch_movies_skips_bad_ids_and_keeps_good_ones(monkeypatch):
    responses = {
        1: FakeResponse(200, _movie_record(1, "Good Movie")),
        2: FakeResponse(404, {}),
        3: requests.exceptions.ConnectionError("boom"),
        4: FakeResponse(200, _movie_record(4, "Another Good Movie")),
    }
    monkeypatch.setattr(ingestion.requests, "Session", lambda: FakeSession(responses))

    records = ingestion.fetch_movies([1, 2, 3, 4], pause=0)

    assert [r["id"] for r in records] == [1, 4]


def test_fetch_movies_skips_record_missing_required_field(monkeypatch):
    bad_record = _movie_record(5, "Missing Credits")
    del bad_record["credits"]
    responses = {5: FakeResponse(200, bad_record), 6: FakeResponse(200, _movie_record(6, "Fine"))}
    monkeypatch.setattr(ingestion.requests, "Session", lambda: FakeSession(responses))

    records = ingestion.fetch_movies([5, 6], pause=0)

    assert [r["id"] for r in records] == [6]


def test_fetch_movies_raises_when_everything_fails(monkeypatch):
    responses = {7: FakeResponse(404, {}), 8: requests.exceptions.ConnectionError("boom")}
    monkeypatch.setattr(ingestion.requests, "Session", lambda: FakeSession(responses))

    with pytest.raises(RuntimeError):
        ingestion.fetch_movies([7, 8], pause=0)


# --- save_raw / JSON round-trip ---

def test_save_raw_round_trips_nested_fields_and_special_characters(tmp_path):
    path = tmp_path / "movies_raw.csv"
    record = _movie_record(999, title='A "Quoted", Tricky Title')
    ingestion.save_raw([record], path=str(path))

    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 1
    row = rows[0]
    assert row["title"] == 'A "Quoted", Tricky Title'
    assert json.loads(row["genres"]) == record["genres"]
    assert json.loads(row["credits"]) == record["credits"]


def test_save_raw_round_trips_a_field_over_128kb(tmp_path):
    """Regression test: a real TMDB record's `credits` field (e.g. "The
    Avengers"' full cast+crew, JSON-encoded) can comfortably exceed Python
    csv's 131072-byte default field-size limit, which broke re-reading a
    real cache with `_csv.Error: field larger than field limit`."""
    huge_cast = [{"id": i, "name": f"Actor Number {i}", "character": "Someone"} for i in range(6000)]
    record = _movie_record(1000, credits={"cast": huge_cast, "crew": []})
    assert len(json.dumps(huge_cast)) > 131072  # confirm the fixture actually exercises the limit

    ingestion.save_raw([record], path=str(tmp_path / "movies_raw.csv"))
    reloaded = ingestion._load_cached_records(str(tmp_path / "movies_raw.csv"))

    assert len(reloaded) == 1
    assert json.loads(reloaded[0]["credits"])["cast"] == huge_cast


# --- ensure_raw_cache ---

def test_ensure_raw_cache_survives_a_permanently_unfetchable_gap(tmp_path, monkeypatch):
    """Regression test: once a cache holds every real movie and only a
    permanently-invalid id (like the brief's own id 0) remains "missing",
    fetch_movies correctly reports zero successes for that one-id gap and
    raises -- but ensure_raw_cache must not let that crash a run that
    already has perfectly good cached data."""
    path = tmp_path / "movies_raw.csv"
    ingestion.save_raw([_movie_record(1), _movie_record(2)], path=str(path))

    monkeypatch.setattr(ingestion.requests, "Session", lambda: FakeSession({0: FakeResponse(404, {})}))

    records = ingestion.ensure_raw_cache(path=str(path), movie_ids=[1, 2, 0])

    assert {int(r["id"]) for r in records} == {1, 2}


def test_ensure_raw_cache_still_raises_on_a_true_total_failure(tmp_path, monkeypatch):
    """The above must not swallow a genuine total failure (e.g. a bad
    token) on a from-scratch fetch, where there's no cache to fall back on."""
    path = tmp_path / "movies_raw.csv"  # no cache file created

    monkeypatch.setattr(ingestion.requests, "Session", lambda: FakeSession({1: FakeResponse(401, {})}))

    with pytest.raises(RuntimeError):
        ingestion.ensure_raw_cache(path=str(path), movie_ids=[1])


def test_ensure_raw_cache_fetches_only_the_gap(tmp_path, monkeypatch):
    path = tmp_path / "movies_raw.csv"
    ingestion.save_raw([_movie_record(1, "Already Cached")], path=str(path))

    responses = {2: FakeResponse(200, _movie_record(2, "Newly Fetched"))}
    fake_session = FakeSession(responses)
    monkeypatch.setattr(ingestion.requests, "Session", lambda: fake_session)

    records = ingestion.ensure_raw_cache(path=str(path), movie_ids=[1, 2])

    assert fake_session.requested_ids == [2]
    assert {int(r["id"]) for r in records} == {1, 2}


def test_ensure_raw_cache_full_hit_makes_no_api_calls(tmp_path, monkeypatch):
    path = tmp_path / "movies_raw.csv"
    ingestion.save_raw([_movie_record(1), _movie_record(2)], path=str(path))

    def _boom(*args, **kwargs):
        raise AssertionError("fetch_movies should not be called on a full cache hit")

    monkeypatch.setattr(ingestion, "fetch_movies", _boom)

    records = ingestion.ensure_raw_cache(path=str(path), movie_ids=[1, 2])

    assert {int(r["id"]) for r in records} == {1, 2}
