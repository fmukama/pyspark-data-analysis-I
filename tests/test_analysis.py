"""
tests/test_analysis.py
=======================
Covers src/analysis.py against a small hand-built fixture (clean-schema
shaped, not raw JSON -- analysis.py never touches JSON at all, so there's
nothing to gain from routing through ingestion/preprocessing here).
"""

import pytest
from pyspark.sql import Row
from pyspark.sql.types import DoubleType, LongType, StringType, StructField, StructType

from src import analysis

_SCHEMA = StructType([
    StructField("id", LongType()),
    StructField("title", StringType()),
    StructField("budget_musd", DoubleType()),
    StructField("revenue_musd", DoubleType()),
    StructField("vote_count", LongType()),
    StructField("vote_average", DoubleType()),
    StructField("popularity", DoubleType()),
    StructField("runtime", DoubleType()),
    StructField("genres", StringType()),
    StructField("cast", StringType()),
    StructField("director", StringType()),
    StructField("belongs_to_collection", StringType()),
])


def _movie(id, title, budget_musd=100.0, revenue_musd=200.0, vote_count=100, vote_average=7.0,
           popularity=10.0, runtime=100.0, genres="Action", cast="Someone",
           director="Some Director", belongs_to_collection=None):
    return Row(
        id=id, title=title, budget_musd=budget_musd, revenue_musd=revenue_musd,
        vote_count=vote_count, vote_average=vote_average, popularity=popularity,
        runtime=runtime, genres=genres, cast=cast, director=director,
        belongs_to_collection=belongs_to_collection,
    )


def _df(spark, rows):
    return spark.createDataFrame(rows, schema=_SCHEMA)


# --- add_metrics ---

def test_add_metrics_computes_profit_and_roi(spark):
    df = _df(spark, [_movie(1, "A", budget_musd=100.0, revenue_musd=250.0)])
    row = analysis.add_metrics(df).collect()[0]
    assert row["profit_musd"] == pytest.approx(150.0)
    assert row["roi"] == pytest.approx(2.5)


def test_add_metrics_null_budget_yields_null_roi_not_a_crash(spark):
    df = _df(spark, [_movie(1, "A", budget_musd=None, revenue_musd=250.0)])
    row = analysis.add_metrics(df).collect()[0]
    assert row["roi"] is None


# --- rank_movies ---

def test_rank_movies_respects_n_and_ascending(spark):
    rows = [_movie(i, f"Movie {i}", revenue_musd=float(i)) for i in range(1, 8)]
    df = _df(spark, rows)

    top3 = analysis.rank_movies(df, "revenue_musd", n=3)
    assert [r["title"] for r in top3.collect()] == ["Movie 7", "Movie 6", "Movie 5"]

    bottom2 = analysis.rank_movies(df, "revenue_musd", ascending=True, n=2)
    assert [r["title"] for r in bottom2.collect()] == ["Movie 1", "Movie 2"]


def test_rank_movies_min_budget_excludes_small_budgets(spark):
    df = _df(spark, [
        _movie(1, "Tiny Budget", budget_musd=1.0, revenue_musd=100.0),
        _movie(2, "Real Budget", budget_musd=50.0, revenue_musd=100.0),
    ])
    result = analysis.rank_movies(df, "roi", n=5, min_budget=10)
    assert [r["title"] for r in result.collect()] == ["Real Budget"]


def test_rank_movies_min_votes_excludes_low_vote_counts(spark):
    df = _df(spark, [
        _movie(1, "Few Votes", vote_count=2, vote_average=10.0),
        _movie(2, "Many Votes", vote_count=500, vote_average=5.0),
    ])
    result = analysis.rank_movies(df, "vote_average", n=5, min_votes=10)
    assert [r["title"] for r in result.collect()] == ["Many Votes"]


def test_rank_movies_never_exceeds_n_even_with_fewer_rows(spark):
    df = _df(spark, [_movie(1, "Only One")])
    result = analysis.rank_movies(df, "revenue_musd", n=5)
    assert result.count() == 1


def test_rank_movies_is_deterministic_on_ties(spark):
    """Movies tied on the ranking column would have no guaranteed order
    without the id tiebreak -- run the same ranking twice and confirm
    identical row order both times."""
    rows = [_movie(i, f"Tied {i}", revenue_musd=100.0) for i in [5, 3, 4, 1, 2]]
    df = _df(spark, rows)

    first = [r["title"] for r in analysis.rank_movies(df, "revenue_musd", n=5).collect()]
    second = [r["title"] for r in analysis.rank_movies(df, "revenue_musd", n=5).collect()]

    assert first == second
    assert first == ["Tied 1", "Tied 2", "Tied 3", "Tied 4", "Tied 5"]  # id ascending tiebreak


# --- search_movies ---

def test_search_movies_requires_all_genres_and_logic(spark):
    df = _df(spark, [
        _movie(1, "Both Genres", genres="Action|Science Fiction"),
        _movie(2, "Only Action", genres="Action|Comedy"),
    ])
    result = analysis.search_movies(df, genres=["Science Fiction", "Action"])
    assert [r["title"] for r in result.collect()] == ["Both Genres"]


def test_search_movies_case_insensitive(spark):
    df = _df(spark, [_movie(1, "Die Hard", cast="Bruce Willis|Alan Rickman")])
    result = analysis.search_movies(df, cast="bruce willis")
    assert result.count() == 1


def test_search_movies_null_columns_do_not_match_or_crash(spark):
    df = _df(spark, [_movie(1, "No Director Listed", director=None)])
    result = analysis.search_movies(df, director="Anyone")
    assert result.count() == 0


def test_search_scifi_action_bruce_willis(spark):
    df = _df(spark, [
        _movie(1, "The Fifth Element", genres="Science Fiction|Action", cast="Bruce Willis", vote_average=7.0),
        _movie(2, "Pulp Fiction", genres="Crime|Drama", cast="Uma Thurman|Bruce Willis", vote_average=9.0),
    ])
    result = analysis.search_scifi_action_bruce_willis(df)
    assert [r["title"] for r in result.collect()] == ["The Fifth Element"]


def test_search_thurman_tarantino_sorted_by_runtime_ascending(spark):
    df = _df(spark, [
        _movie(1, "Kill Bill", cast="Uma Thurman", director="Quentin Tarantino", runtime=180.0),
        _movie(2, "Pulp Fiction", cast="Uma Thurman|John Travolta", director="Quentin Tarantino", runtime=154.0),
        _movie(3, "Someone Else's Movie", cast="Uma Thurman", director="Someone Else", runtime=90.0),
    ])
    result = analysis.search_thurman_tarantino(df)
    assert [r["title"] for r in result.collect()] == ["Pulp Fiction", "Kill Bill"]


# --- group analysis ---

def test_franchise_vs_standalone_labels_and_grouping(spark):
    df = _df(spark, [
        _movie(1, "Franchise Movie", belongs_to_collection="Some Collection", revenue_musd=100.0),
        _movie(2, "Standalone Movie", belongs_to_collection=None, revenue_musd=200.0),
    ])
    result = {r["is_franchise"]: r["mean_revenue"] for r in analysis.franchise_vs_standalone(df).collect()}
    assert result == {"Franchise": 100.0, "Standalone": 200.0}


def test_franchise_summary_filters_by_min_movies(spark):
    df = _df(spark, [
        _movie(1, "Solo Franchise Entry", belongs_to_collection="Lonely Collection"),
        _movie(2, "Series Entry 1", belongs_to_collection="Big Collection"),
        _movie(3, "Series Entry 2", belongs_to_collection="Big Collection"),
    ])
    result = analysis.franchise_summary(df, min_movies=2).collect()
    assert [r["belongs_to_collection"] for r in result] == ["Big Collection"]
    assert result[0]["num_movies"] == 2


def test_director_summary_explodes_co_directors(spark):
    df = _df(spark, [
        _movie(1, "Co-Directed Movie", director="Director X|Director Y", revenue_musd=100.0),
        _movie(2, "Solo Movie", director="Director X", revenue_musd=50.0),
    ])
    result = {r["director"]: r for r in analysis.director_summary(df).collect()}

    assert result["Director X"]["num_movies"] == 2
    assert result["Director X"]["total_revenue"] == pytest.approx(150.0)
    assert result["Director Y"]["num_movies"] == 1
    assert result["Director Y"]["total_revenue"] == pytest.approx(100.0)


def test_roi_by_genre_summary_explodes_multi_genre(spark):
    df = _df(spark, [
        _movie(1, "Multi Genre", genres="Action|Drama", budget_musd=100.0, revenue_musd=200.0),
    ])
    result = {r["genres"]: r["median_roi"] for r in analysis.roi_by_genre_summary(df).collect()}
    assert result["Action"] == pytest.approx(2.0)
    assert result["Drama"] == pytest.approx(2.0)
