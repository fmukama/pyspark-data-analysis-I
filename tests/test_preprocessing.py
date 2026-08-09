"""
tests/test_preprocessing.py
============================
Covers src/preprocessing.py at two levels:
  - end-to-end through a synthetic raw CSV (via the real ingestion.save_raw)
    for parse_nested_columns/flatten_columns/fix_dtypes_and_values, since
    those rules are about how raw JSON shapes turn into clean values;
  - direct, hand-built DataFrames for clean_rows' own dedup/threshold/
    status-filter/reorder logic, since those don't depend on JSON shape at
    all and are clearer to test in isolation.
"""

import logging

import pytest
from pyspark.sql import Row
from pyspark.sql.types import StringType, StructField, StructType

from src import ingestion, preprocessing


def _raw_row(movie_id=1, title="Some Movie", **overrides):
    """A full 28-column raw record, shaped like a real TMDB response."""
    row = {
        "adult": False,
        "backdrop_path": "/backdrop.jpg",
        "belongs_to_collection": None,
        "budget": 1_000_000,
        "credits": {
            "cast": [{"name": "Actor One"}, {"name": "Actor Two"}],
            "crew": [{"name": "Director One", "job": "Director"}, {"name": "Writer One", "job": "Writer"}],
        },
        "genres": [{"id": 28, "name": "Action"}],
        "homepage": "https://example.com",
        "id": movie_id,
        "imdb_id": f"tt{movie_id:07d}",
        "origin_country": ["US"],
        "original_language": "en",
        "original_title": title,
        "overview": "A fine movie about things.",
        "popularity": 10.0,
        "poster_path": "/poster.jpg",
        "production_companies": [{"name": "Some Studio"}],
        "production_countries": [{"name": "United States of America"}],
        "release_date": "2020-01-01",
        "revenue": 5_000_000,
        "runtime": 120,
        "softcore": False,
        "spoken_languages": [{"english_name": "English", "name": "English"}],
        "status": "Released",
        "tagline": "A fine tagline.",
        "title": title,
        "video": False,
        "vote_average": 7.5,
        "vote_count": 100,
    }
    row.update(overrides)
    return row


def _flattened(spark, rows, tmp_path):
    """Write rows to CSV via the real ingestion.save_raw, then run them
    through load -> parse -> flatten -> fix_dtypes (everything except
    clean_rows, which is tested separately against hand-built frames)."""
    path = tmp_path / "movies_raw.csv"
    ingestion.save_raw(rows, path=str(path))
    df = preprocessing.load_raw_spark(spark, str(path))
    df = preprocessing.parse_nested_columns(df)
    df = preprocessing.flatten_columns(df)
    df = preprocessing.fix_dtypes_and_values(df)
    return df


def _row_by_id(df, movie_id):
    """id is already LongType by the time this is used -- fix_dtypes_and_values
    has already run inside _flattened."""
    return df.filter(df.id == movie_id).collect()[0]


# --- fix_dtypes_and_values ---

def test_zero_budget_revenue_runtime_become_null(spark, tmp_path):
    df = _flattened(spark, [_raw_row(1, budget=0, revenue=0, runtime=0)], tmp_path)
    row = _row_by_id(df, 1)
    assert row["budget_musd"] is None
    assert row["revenue_musd"] is None
    assert row["runtime"] is None


def test_zero_vote_count_nulls_rating(spark, tmp_path):
    df = _flattened(spark, [_raw_row(1, vote_count=0, vote_average=9.0)], tmp_path)
    assert _row_by_id(df, 1)["vote_average"] is None


def test_placeholder_text_becomes_null(spark, tmp_path):
    df = _flattened(spark, [_raw_row(1, overview="No Data", tagline="")], tmp_path)
    row = _row_by_id(df, 1)
    assert row["overview"] is None
    assert row["tagline"] is None


def test_real_overview_and_tagline_survive(spark, tmp_path):
    df = _flattened(spark, [_raw_row(1, overview="A real plot.", tagline="A real line.")], tmp_path)
    row = _row_by_id(df, 1)
    assert row["overview"] == "A real plot."
    assert row["tagline"] == "A real line."


def test_budget_musd_conversion(spark, tmp_path):
    df = _flattened(spark, [_raw_row(1, budget=250_000_000, revenue=1_500_000_000)], tmp_path)
    row = _row_by_id(df, 1)
    assert row["budget_musd"] == pytest.approx(250.0)
    assert row["revenue_musd"] == pytest.approx(1500.0)


# --- flatten_columns ---

def test_standalone_movie_has_null_collection(spark, tmp_path):
    df = _flattened(spark, [_raw_row(1, belongs_to_collection=None)], tmp_path)
    assert _row_by_id(df, 1)["belongs_to_collection"] is None


def test_collection_name_extracted(spark, tmp_path):
    collection = {"id": 1, "name": "Some Franchise Collection"}
    df = _flattened(spark, [_raw_row(1, belongs_to_collection=collection)], tmp_path)
    assert _row_by_id(df, 1)["belongs_to_collection"] == "Some Franchise Collection"


def test_genres_and_cast_pipe_joined(spark, tmp_path):
    row = _raw_row(
        1,
        genres=[{"id": 1, "name": "Action"}, {"id": 2, "name": "Comedy"}],
        credits={"cast": [{"name": "A"}, {"name": "B"}, {"name": "C"}], "crew": []},
    )
    df = _flattened(spark, [row], tmp_path)
    result = _row_by_id(df, 1)
    assert result["genres"] == "Action|Comedy"
    assert result["cast"] == "A|B|C"
    assert result["cast_size"] == 3


def test_empty_genre_list_becomes_null_not_empty_string(spark, tmp_path):
    # A second, normal row is included alongside the edge case: the
    # schema-tier check in parse_nested_columns looks for at least one row
    # with a real genres array across the whole batch (a real signal that
    # from_json actually parsed something, not a per-row rule), so an
    # all-empty single-row fixture would trip that check for an unrelated
    # reason -- this fixture matches how the edge case actually occurs in
    # a real dataset: most movies have genres, one particular one doesn't.
    df = _flattened(spark, [_raw_row(1, genres=[]), _raw_row(2)], tmp_path)
    assert _row_by_id(df, 1)["genres"] is None
    assert _row_by_id(df, 2)["genres"] == "Action"


def test_director_extracted_by_filtering_crew_job(spark, tmp_path):
    row = _raw_row(1, credits={
        "cast": [],
        "crew": [
            {"name": "Editor Person", "job": "Editor"},
            {"name": "Director Person", "job": "Director"},
            {"name": "Producer Person", "job": "Producer"},
        ],
    })
    df = _flattened(spark, [row], tmp_path)
    result = _row_by_id(df, 1)
    assert result["director"] == "Director Person"
    assert result["crew_size"] == 3


def test_co_directors_pipe_joined_in_crew_order(spark, tmp_path):
    row = _raw_row(1, credits={
        "cast": [],
        "crew": [
            {"name": "Second Director", "job": "Director"},
            {"name": "First Director", "job": "Director"},
        ],
    })
    df = _flattened(spark, [row], tmp_path)
    assert _row_by_id(df, 1)["director"] == "Second Director|First Director"


def test_no_director_credited_becomes_null(spark, tmp_path):
    row = _raw_row(1, credits={"cast": [], "crew": [{"name": "Someone", "job": "Producer"}]})
    df = _flattened(spark, [row], tmp_path)
    assert _row_by_id(df, 1)["director"] is None


# --- schema-tier check ---

def test_missing_raw_column_raises(spark, tmp_path):
    """A raw CSV genuinely lacking a 'credits' column raises Spark's own
    AnalysisException the moment parse_nested_columns references it --
    confirmed here since that's the actual failure mode, not a custom
    check (see the note on _check_parsed_schema)."""
    bad_row = _raw_row(1)
    del bad_row["credits"]
    path = tmp_path / "movies_raw.csv"
    ingestion.save_raw([bad_row], path=str(path))
    df = preprocessing.load_raw_spark(spark, str(path))

    with pytest.raises(Exception):
        preprocessing.parse_nested_columns(df)


def test_all_rows_null_after_parse_raises_for_credits(spark, tmp_path):
    """If every row's 'credits' cell fails to parse (e.g. TMDB's real shape
    doesn't match what this pipeline assumes), that's a code bug worth
    halting for, not messy data. Confirmed directly that from_json's
    PERMISSIVE mode turns unparseable text into a non-null struct with
    every field null ({cast: null, crew: null}) rather than a null struct
    -- this uses exactly that input to exercise the real failure shape."""
    path = tmp_path / "movies_raw.csv"
    ingestion.save_raw([_raw_row(1)], path=str(path))
    df = preprocessing.load_raw_spark(spark, str(path))
    df = df.withColumn("credits", preprocessing.F.lit("not valid json"))

    with pytest.raises(ValueError, match="credits.cast.*credits.crew"):
        preprocessing.parse_nested_columns(df)


def test_all_rows_null_after_parse_raises_for_genres(spark, tmp_path):
    """Unlike credits (a struct), genres is an array -- confirmed directly
    that from_json genuinely returns null for an array schema given
    unparseable text, a different failure shape than the struct case."""
    path = tmp_path / "movies_raw.csv"
    ingestion.save_raw([_raw_row(1)], path=str(path))
    df = preprocessing.load_raw_spark(spark, str(path))
    df = df.withColumn("genres", preprocessing.F.lit("not valid json"))

    with pytest.raises(ValueError, match="non-empty 'genres'"):
        preprocessing.parse_nested_columns(df)


# --- quality-tier check ---

def test_out_of_range_vote_average_warns_but_survives(spark, tmp_path, caplog):
    df = _flattened(spark, [_raw_row(1, vote_average=15.0, vote_count=100)], tmp_path)
    # fix_dtypes_and_values already ran (inside _flattened) and logged through
    # this check; re-running it here just to capture the warning explicitly.
    with caplog.at_level(logging.WARNING, logger="preprocessing"):
        preprocessing._check_value_ranges(df)

    assert any("vote_average outside [0, 10]" in message for message in caplog.messages)
    assert _row_by_id(df, 1)["vote_average"] == 15.0  # the check warns, it never filters


# --- clean_rows (hand-built frames -- dedup/threshold/status/reorder don't need real JSON shapes) ---

_CLEAN_ROWS_COLUMNS = preprocessing.FINAL_COLUMN_ORDER + ["status"]
_CLEAN_ROWS_SCHEMA = StructType([StructField(c, StringType()) for c in _CLEAN_ROWS_COLUMNS])


def _clean_rows_df(spark, rows):
    """rows: list of dicts with only the keys you care about; every other
    column defaults to null. clean_rows only reasons about presence/absence
    and the 'status' value, so plain strings are enough here regardless of
    what the real dtype would be post fix_dtypes_and_values."""
    full_rows = [Row(**{c: r.get(c) for c in _CLEAN_ROWS_COLUMNS}) for r in rows]
    return spark.createDataFrame(full_rows, schema=_CLEAN_ROWS_SCHEMA)


def _good_row(movie_id="1", **overrides):
    row = {
        "id": movie_id, "title": "Good Movie", "status": "Released",
        "tagline": "t", "release_date": "2020-01-01", "genres": "Action",
        "belongs_to_collection": "Some Collection", "original_language": "en",
        "budget_musd": "10.0", "revenue_musd": "50.0", "production_companies": "Studio",
    }
    row.update(overrides)
    return row


def test_clean_rows_dedupes_on_id(spark):
    df = _clean_rows_df(spark, [_good_row("1"), _good_row("1")])
    assert preprocessing.clean_rows(df).count() == 1


def test_clean_rows_drops_rows_missing_id_or_title(spark):
    df = _clean_rows_df(spark, [
        _good_row("1"),
        _good_row("2", id=None),
        _good_row("3", title=None),
    ])
    result = preprocessing.clean_rows(df)
    assert result.count() == 1
    assert result.collect()[0]["id"] == "1"


def test_clean_rows_drops_rows_below_non_null_threshold(spark):
    sparse_row = {c: None for c in _CLEAN_ROWS_COLUMNS}
    sparse_row.update({"id": "2", "title": "Sparse Movie", "status": "Released"})  # only 3 non-null
    df = _clean_rows_df(spark, [_good_row("1"), sparse_row])  # good row has 11 non-null

    result = preprocessing.clean_rows(df)

    assert result.count() == 1
    assert result.collect()[0]["id"] == "1"


def test_clean_rows_filters_to_released_only_and_drops_status(spark):
    df = _clean_rows_df(spark, [_good_row("1", status="Released"), _good_row("2", status="In Production")])
    result = preprocessing.clean_rows(df)
    assert result.count() == 1
    assert result.collect()[0]["id"] == "1"
    assert "status" not in result.columns


def test_clean_rows_reorders_to_final_column_order(spark):
    df = _clean_rows_df(spark, [_good_row("1")])
    assert preprocessing.clean_rows(df).columns == preprocessing.FINAL_COLUMN_ORDER


def test_check_final_schema_raises_on_duplicate_id(spark):
    """Direct unit test of _check_final_schema's own logic -- clean_rows
    itself already dedupes before this ever runs, so this deliberately
    bypasses clean_rows and hands _check_final_schema an already-final-
    shaped frame with a duplicate id, to confirm its own guard is sound."""
    schema = StructType([StructField(c, StringType()) for c in preprocessing.FINAL_COLUMN_ORDER])
    rows = [Row(**{c: ("1" if c == "id" else None) for c in preprocessing.FINAL_COLUMN_ORDER}) for _ in range(2)]
    df = spark.createDataFrame(rows, schema=schema)

    with pytest.raises(ValueError, match="distinct id"):
        preprocessing._check_final_schema(df)


def test_check_final_schema_raises_on_wrong_column_order(spark):
    reordered = list(reversed(preprocessing.FINAL_COLUMN_ORDER))
    schema = StructType([StructField(c, StringType()) for c in reordered])
    df = spark.createDataFrame([Row(**{c: None for c in reordered})], schema=schema)

    with pytest.raises(ValueError, match="do not match FINAL_COLUMN_ORDER"):
        preprocessing._check_final_schema(df)
