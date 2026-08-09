"""
tests/test_visualization.py
============================
Smoke tests for src/visualization.py against a small hand-built fixture:
each chart function should return an Axes without raising, and write a
file when a save_path is given. Forces the Agg backend before importing
the module under test, since this runs headless (no display).
"""

import datetime
import os

import matplotlib
matplotlib.use("Agg")

import matplotlib.axes
import pytest
from pyspark.sql import Row
from pyspark.sql.types import DateType, DoubleType, LongType, StringType, StructField, StructType

from src import visualization

_SCHEMA = StructType([
    StructField("id", LongType()),
    StructField("title", StringType()),
    StructField("budget_musd", DoubleType()),
    StructField("revenue_musd", DoubleType()),
    StructField("vote_average", DoubleType()),
    StructField("popularity", DoubleType()),
    StructField("release_date", DateType()),
    StructField("genres", StringType()),
    StructField("belongs_to_collection", StringType()),
])


def _movie(id, title, budget_musd=100.0, revenue_musd=200.0, vote_average=7.0,
           popularity=10.0, release_date=datetime.date(2020, 1, 1), genres="Action",
           belongs_to_collection=None):
    return Row(
        id=id, title=title, budget_musd=budget_musd, revenue_musd=revenue_musd,
        vote_average=vote_average, popularity=popularity, release_date=release_date,
        genres=genres, belongs_to_collection=belongs_to_collection,
    )


@pytest.fixture
def sample_df(spark):
    return spark.createDataFrame([
        _movie(1, "A", budget_musd=100.0, revenue_musd=300.0, genres="Action|Drama",
               release_date=datetime.date(2019, 1, 1)),
        _movie(2, "B", budget_musd=50.0, revenue_musd=80.0, genres="Comedy",
               belongs_to_collection="Some Collection", release_date=datetime.date(2020, 1, 1)),
        _movie(3, "C", budget_musd=200.0, revenue_musd=150.0, genres="Drama",
               release_date=datetime.date(2020, 6, 1)),
    ], schema=_SCHEMA)


@pytest.mark.parametrize("chart_fn", [
    visualization.revenue_vs_budget,
    visualization.roi_by_genre,
    visualization.popularity_vs_rating,
    visualization.yearly_box_office,
    visualization.franchise_vs_standalone_plot,
])
def test_chart_returns_axes_without_raising(sample_df, chart_fn):
    ax = chart_fn(sample_df)
    assert isinstance(ax, matplotlib.axes.Axes)


@pytest.mark.parametrize("chart_fn", [
    visualization.revenue_vs_budget,
    visualization.roi_by_genre,
    visualization.popularity_vs_rating,
    visualization.yearly_box_office,
    visualization.franchise_vs_standalone_plot,
])
def test_chart_writes_file_when_save_path_given(sample_df, chart_fn, tmp_path):
    path = tmp_path / "chart.png"
    chart_fn(sample_df, save_path=str(path))
    assert path.exists()
    assert path.stat().st_size > 0


def test_generate_all_charts_writes_five_numbered_files(sample_df, tmp_path):
    paths = visualization.generate_all_charts(sample_df, output_dir=str(tmp_path))

    assert len(paths) == 5
    for path in paths:
        assert os.path.exists(path)
        assert os.path.getsize(path) > 0

    written = sorted(p.name for p in tmp_path.iterdir())
    assert written == [
        "01-revenue-vs-budget.png",
        "02-roi-by-genre.png",
        "03-popularity-vs-rating.png",
        "04-yearly-box-office.png",
        "05-franchise-vs-standalone.png",
    ]
