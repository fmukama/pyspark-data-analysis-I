"""
tests/test_parity.py
=====================
Compares this pipeline's KPI outputs against a checked-in snapshot of
dem-02-lab's (the pandas reference implementation) own known-good numbers
-- the strongest correctness signal available here, since a validated
reference implementation already exists.
"""

import json
import os

import pytest
from pyspark.sql.types import DoubleType, LongType

from src import analysis, config

_FIXTURE_PATH = os.path.join(os.path.dirname(__file__), "fixtures", "dem02_reference.json")

with open(_FIXTURE_PATH) as f:
    REFERENCE = json.load(f)


@pytest.fixture(scope="module")
def clean_df(spark):
    if not os.path.exists(config.CLEAN_DATA_PATH):
        pytest.skip(
            f"{config.CLEAN_DATA_PATH} not found -- run the pipeline first "
            "(make verify / run_pipeline.py) so there's real output to compare."
        )
    df = spark.read.option("header", True).csv(config.CLEAN_DATA_PATH)
    for c in ["budget_musd", "revenue_musd", "popularity", "vote_average"]:
        df = df.withColumn(c, df[c].cast(DoubleType()))
    for c in ["id", "vote_count", "cast_size", "crew_size"]:
        df = df.withColumn(c, df[c].cast(LongType()))
    df = df.withColumn("runtime", df["runtime"].cast(DoubleType()))
    return df


def _assert_ordered_ranking_matches(result_rows, expected, value_key):
    """Strict: both which movies rank and in what order must match."""
    actual_titles = [r["title"] for r in result_rows]
    expected_titles = [e["title"] for e in expected]
    assert actual_titles == expected_titles, f"got {actual_titles}, expected {expected_titles}"
    for actual, expected_row in zip(result_rows, expected):
        assert actual[value_key] == pytest.approx(expected_row[value_key], rel=1e-4), (
            f"{actual['title']}: {value_key}={actual[value_key]} != expected {expected_row[value_key]}"
        )


# --- stable, financial-figure-based KPIs: strict comparison ---

def test_highest_revenue_matches_reference(clean_df):
    rows = analysis.highest_revenue(clean_df).collect()
    _assert_ordered_ranking_matches(rows, REFERENCE["highest_revenue"], "revenue_musd")


def test_highest_budget_matches_reference_as_a_set(clean_df):
    """Compared as a set, not exact order: Star Wars: The Last Jedi and
    Avengers: Infinity War are exactly tied at budget_musd=300.0. This
    pipeline's deterministic id-ascending tiebreak (Confirmed Decision 2)
    orders the tie differently than the pandas reference's incidental
    original-row-order did -- neither order is more "correct", so this
    checks that the same 5 movies and values appear, not their tied
    sub-order. Every other ranking below has no ties in its top 5 (checked
    directly against the reference values), so those use exact order."""
    rows = {r["title"]: r["budget_musd"] for r in analysis.highest_budget(clean_df).collect()}
    expected = {e["title"]: e["budget_musd"] for e in REFERENCE["highest_budget"]}
    assert rows.keys() == expected.keys()
    for title in expected:
        assert rows[title] == pytest.approx(expected[title], rel=1e-4)


def test_highest_profit_matches_reference(clean_df):
    rows = analysis.highest_profit(clean_df).collect()
    _assert_ordered_ranking_matches(rows, REFERENCE["highest_profit"], "profit_musd")


def test_lowest_profit_matches_reference(clean_df):
    rows = analysis.lowest_profit(clean_df).collect()
    _assert_ordered_ranking_matches(rows, REFERENCE["lowest_profit"], "profit_musd")


def test_highest_roi_matches_reference(clean_df):
    rows = analysis.highest_roi(clean_df).collect()
    _assert_ordered_ranking_matches(rows, REFERENCE["highest_roi"], "roi")


def test_lowest_roi_matches_reference(clean_df):
    rows = analysis.lowest_roi(clean_df).collect()
    _assert_ordered_ranking_matches(rows, REFERENCE["lowest_roi"], "roi")


def test_franchise_summary_financials_match_reference(clean_df):
    rows = {r["belongs_to_collection"]: r for r in analysis.franchise_summary(clean_df).collect()}
    for expected in REFERENCE["franchise_summary"]:
        actual = rows[expected["belongs_to_collection"]]
        assert actual["num_movies"] == expected["num_movies"]
        assert actual["total_budget"] == pytest.approx(expected["total_budget"], rel=1e-4)
        assert actual["total_revenue"] == pytest.approx(expected["total_revenue"], rel=1e-4)


def test_director_summary_revenue_matches_reference(clean_df):
    rows = {r["director"]: r for r in analysis.director_summary(clean_df).collect()}
    for expected in REFERENCE["director_summary"]:
        actual = rows[expected["director"]]
        assert actual["num_movies"] == expected["num_movies"]
        assert actual["total_revenue"] == pytest.approx(expected["total_revenue"], rel=1e-4)


def test_franchise_vs_standalone_financials_match_reference(clean_df):
    rows = {r["is_franchise"]: r for r in analysis.franchise_vs_standalone(clean_df).collect()}
    for label, expected in REFERENCE["franchise_vs_standalone"].items():
        actual = rows[label]
        assert actual["mean_revenue"] == pytest.approx(expected["mean_revenue"], rel=1e-4)
        assert actual["mean_budget"] == pytest.approx(expected["mean_budget"], rel=1e-4)
        assert actual["median_roi"] == pytest.approx(expected["median_roi"], rel=1e-4)


def test_both_searches_return_zero_rows_matching_reference(clean_df):
    """Both searches are known to return zero rows against this exact
    18-movie sample (all major franchise blockbusters) -- confirmed
    correct against the pandas reference, not a bug."""
    assert analysis.search_scifi_action_bruce_willis(clean_df).count() == 0
    assert analysis.search_thurman_tarantino(clean_df).count() == 0


# --- volatile metrics: structural sanity only, not compared to the pandas
# snapshot -- see the module docstring and fixture file for why.

@pytest.mark.parametrize("ranking_fn", [
    analysis.most_voted,
    analysis.highest_rated,
    analysis.lowest_rated,
    analysis.most_popular,
])
def test_volatile_rankings_run_and_return_five_rows(clean_df, ranking_fn):
    result = ranking_fn(clean_df).collect()
    assert len(result) == 5
