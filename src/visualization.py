"""
visualization.py
=================
STEP 4: visual summaries of the cleaned dataset.
"""

import os

from pyspark.sql import functions as F

import matplotlib.pyplot as plt

from src import analysis, config
from src.logger import get_logger

logger = get_logger("visualization")


def _new_ax(ax, figsize):
    """Create a fresh Axes if the caller didn't pass one in."""
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    return ax


def _save_if_requested(ax, save_path):
    """
    Save the figure if a path was given, then close it -- appropriate for a
    batch/headless caller (like generate_all_charts) that won't reference
    the figure again. When save_path is None (the interactive/notebook
    case), the figure is left open for the caller -- or Jupyter's own
    inline backend -- to display and manage.
    """
    if save_path:
        ax.figure.savefig(save_path, bbox_inches="tight")
        plt.close(ax.figure)


# 1) Revenue vs Budget

def revenue_vs_budget(df, ax=None, save_path=None):
    """
    Scatter of revenue against budget, with a dashed break-even line
    (points above the line made money, points below lost money).
    """
    rows = df.filter(F.col("budget_musd").isNotNull() & F.col("revenue_musd").isNotNull()) \
              .select("budget_musd", "revenue_musd").collect()
    budgets = [r["budget_musd"] for r in rows]
    revenues = [r["revenue_musd"] for r in rows]

    ax = _new_ax(ax, (8, 6))
    ax.scatter(budgets, revenues, alpha=0.7, edgecolor="k")

    top = max(budgets + revenues)
    ax.plot([0, top], [0, top], "r--", label="Break-even (revenue = budget)")

    ax.set_xlabel("Budget (million USD)")
    ax.set_ylabel("Revenue (million USD)")
    ax.set_title("Revenue vs Budget")
    ax.legend()
    _save_if_requested(ax, save_path)
    return ax


# 2) ROI distribution by genre

def roi_by_genre(df, ax=None, save_path=None):
    """
    Median ROI per genre. The explode + median-per-genre shaping lives in
    analysis.roi_by_genre_summary (not duplicated here, unlike the pandas
    original) -- this function just plots the already-small result.
    """
    rows = analysis.roi_by_genre_summary(df).collect()
    genres = [r["genres"] for r in rows]
    medians = [r["median_roi"] for r in rows]

    ax = _new_ax(ax, (9, 6))
    ax.barh(genres, medians, color="teal")
    ax.set_xlabel("Median ROI (revenue / budget)")
    ax.set_ylabel("Genre")
    ax.set_title("ROI Distribution by Genre (median)")
    _save_if_requested(ax, save_path)
    return ax


# 3) Popularity vs Rating

def popularity_vs_rating(df, ax=None, save_path=None):
    """Scatter of popularity against average rating."""
    rows = df.filter(F.col("vote_average").isNotNull() & F.col("popularity").isNotNull()) \
              .select("vote_average", "popularity").collect()
    ratings = [r["vote_average"] for r in rows]
    popularity = [r["popularity"] for r in rows]

    ax = _new_ax(ax, (8, 6))
    ax.scatter(ratings, popularity, alpha=0.7, color="darkorange", edgecolor="k")
    ax.set_xlabel("Average Rating")
    ax.set_ylabel("Popularity")
    ax.set_title("Popularity vs Rating")
    _save_if_requested(ax, save_path)
    return ax


# 4) Yearly box-office performance

def yearly_box_office(df, ax=None, save_path=None):
    """Total revenue per release year (bar chart)."""
    data = df.filter(F.col("release_date").isNotNull() & F.col("revenue_musd").isNotNull())
    data = data.withColumn("year", F.year(F.col("release_date")))
    yearly = data.groupBy("year").agg(F.sum("revenue_musd").alias("total_revenue")).orderBy("year")

    rows = yearly.collect()
    years = [str(r["year"]) for r in rows]
    revenues = [r["total_revenue"] for r in rows]

    ax = _new_ax(ax, (10, 6))
    ax.bar(years, revenues, color="steelblue")
    ax.set_xlabel("Release Year")
    ax.set_ylabel("Total Revenue (million USD)")
    ax.set_title("Yearly Box-Office Performance")
    _save_if_requested(ax, save_path)
    return ax


# 5) Franchise vs Standalone success

def franchise_vs_standalone_plot(df, ax=None, save_path=None):
    """
    Grouped bar chart comparing mean revenue and mean budget for franchise
    vs standalone movies (uses analysis.franchise_vs_standalone, which
    orders its 2 rows deterministically so this chart's bar order is
    consistent run to run).
    """
    rows = analysis.franchise_vs_standalone(df).collect()
    labels = [r["is_franchise"] for r in rows]
    mean_revenue = [r["mean_revenue"] for r in rows]
    mean_budget = [r["mean_budget"] for r in rows]

    ax = _new_ax(ax, (8, 6))
    x = range(len(labels))
    width = 0.35
    ax.bar([i - width / 2 for i in x], mean_revenue, width, label="Mean Revenue")
    ax.bar([i + width / 2 for i in x], mean_budget, width, label="Mean Budget")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=0)
    ax.set_ylabel("Million USD")
    ax.set_title("Franchise vs Standalone: Mean Revenue & Budget")
    ax.legend()
    _save_if_requested(ax, save_path)
    return ax


# Orchestrator: generate and save all 5 charts with deterministic numbered
# filenames -- closes the gap the pandas project left open (its
# visualization.py never called savefig; the images/ PNGs there were
# exported by hand from the notebook).

def generate_all_charts(df, output_dir=None):
    """Generate and save all 5 charts, numbered in the brief's own Step 4
    order. Returns the list of file paths written."""
    if output_dir is None:
        output_dir = config.IMAGES_DIR
    os.makedirs(output_dir, exist_ok=True)

    chart_fns = [
        ("01-revenue-vs-budget.png", revenue_vs_budget),
        ("02-roi-by-genre.png", roi_by_genre),
        ("03-popularity-vs-rating.png", popularity_vs_rating),
        ("04-yearly-box-office.png", yearly_box_office),
        ("05-franchise-vs-standalone.png", franchise_vs_standalone_plot),
    ]

    paths = []
    for filename, chart_fn in chart_fns:
        path = os.path.join(output_dir, filename)
        chart_fn(df, save_path=path)
        paths.append(path)

    logger.info("Wrote %d chart(s) to %s", len(paths), output_dir)
    return paths
