"""
analysis.py
===========
STEP 3: KPIs, rankings, searches and group analysis.
"""

from pyspark.sql import Window
from pyspark.sql import functions as F

from src.logger import get_logger

logger = get_logger("analysis")


def add_metrics(df):
    """
    Add the two derived KPI columns the rankings need:
        profit_musd = revenue_musd - budget_musd
        roi         = revenue_musd / budget_musd
    roi is null when budget_musd is null (division by null), matching the
    pandas version's NaN-on-missing-budget behavior.
    """
    df = df.withColumn("profit_musd", F.col("revenue_musd") - F.col("budget_musd"))
    df = df.withColumn("roi", F.col("revenue_musd") / F.col("budget_musd"))
    return df


# Task 3.1: the ranking helper (one function drives all 10 rankings)

def rank_movies(df, by, ascending=False, n=5, min_budget=None, min_votes=None, extra_cols=None):
    """
    Generic ranking helper -- the project's UDF-equivalent for streamlining
    ranking (see the module docstring on why this is a Window function, not
    a literal registered udf()).

    Parameters
    ----------
    by : str
        Column to sort by. A normal column ('revenue_musd', 'budget_musd',
        'vote_count', 'vote_average', 'popularity') or a derived metric
        ('profit_musd' / 'roi').
    ascending : bool
        False -> highest first (best). True -> lowest first (worst).
    n : int
        How many movies to return.
    min_budget : float, optional
        Keep only movies with budget_musd >= this (used for ROI rankings).
    min_votes : int, optional
        Keep only movies with vote_count >= this (used for rating rankings).
    extra_cols : list of str, optional
        Extra columns to show next to the ranking metric.

    Returns
    -------
    pyspark.sql.DataFrame
        A tidy table: title + the ranking metric (+ any extra columns),
        already ordered so `.show()`/`.collect()` reflects the ranking.
    """
    data = add_metrics(df)

    if min_budget is not None:
        data = data.filter(F.col("budget_musd") >= min_budget)
    if min_votes is not None:
        data = data.filter(F.col("vote_count") >= min_votes)

    data = data.filter(F.col(by).isNotNull())

    # id ascending as a secondary sort key: Spark's orderBy has no
    # guaranteed stable tie-order across partitions the way pandas'
    # sort_values does, so ties would otherwise not be reproducible run to
    # run. This same key drives both the ranking window and the final
    # physical sort below, since row_number() alone doesn't guarantee the
    # output rows are physically returned in that order.
    order = F.col(by).asc() if ascending else F.col(by).desc()
    tiebreak = F.col("id").asc()

    window = Window.orderBy(order, tiebreak)
    data = data.withColumn("_rank", F.row_number().over(window))
    data = data.filter(F.col("_rank") <= n).drop("_rank")

    cols = ["title", by]
    if extra_cols:
        cols += [c for c in extra_cols if c not in cols]

    result = data.select(*cols).orderBy(order, tiebreak)

    assert result.count() <= n, f"rank_movies: returned more than the requested {n} rows"
    logger.info("rank_movies(by=%s, ascending=%s, n=%s): %d row(s)", by, ascending, n, result.count())
    return result


# Task 3.1 wrappers: the 10 required "best / worst" rankings

def highest_revenue(df, n=5):
    """Top movies by revenue."""
    return rank_movies(df, "revenue_musd", n=n)


def highest_budget(df, n=5):
    """Top movies by budget."""
    return rank_movies(df, "budget_musd", n=n)


def highest_profit(df, n=5):
    """Top movies by profit (revenue - budget)."""
    return rank_movies(df, "profit_musd", n=n, extra_cols=["revenue_musd", "budget_musd"])


def lowest_profit(df, n=5):
    """Biggest money-losers (lowest profit)."""
    return rank_movies(df, "profit_musd", ascending=True, n=n,
                        extra_cols=["revenue_musd", "budget_musd"])


def highest_roi(df, n=5):
    """Best ROI -- only movies with budget >= 10M (avoids tiny-budget noise)."""
    return rank_movies(df, "roi", n=n, min_budget=10,
                        extra_cols=["revenue_musd", "budget_musd"])


def lowest_roi(df, n=5):
    """Worst ROI -- only movies with budget >= 10M."""
    return rank_movies(df, "roi", ascending=True, n=n, min_budget=10,
                        extra_cols=["revenue_musd", "budget_musd"])


def most_voted(df, n=5):
    """Movies with the most votes."""
    return rank_movies(df, "vote_count", n=n)


def highest_rated(df, n=5):
    """Highest rated -- only movies with at least 10 votes."""
    return rank_movies(df, "vote_average", n=n, min_votes=10, extra_cols=["vote_count"])


def lowest_rated(df, n=5):
    """Lowest rated -- only movies with at least 10 votes."""
    return rank_movies(df, "vote_average", ascending=True, n=n, min_votes=10, extra_cols=["vote_count"])


def most_popular(df, n=5):
    """Most popular movies (TMDB popularity score)."""
    return rank_movies(df, "popularity", n=n)


# Task 3.2: advanced search queries

def search_movies(df, genres=None, cast=None, director=None, sort_by="vote_average", ascending=False):
    """
    Flexible search over the cleaned dataset.

    genres/cast/director are matched as case-insensitive substrings against
    the pipe-joined columns. `genres` may be a single string or a list of
    strings (ALL must be present). Null-safety is automatic: `.contains()`
    on a null column evaluates to null, and Spark's filter already excludes
    null (not just false) predicates -- the same effect as pandas'
    `na=False`, with no extra code needed.

    Returns the matching rows sorted by `sort_by`.
    """
    data = df

    if genres:
        genre_list = [genres] if isinstance(genres, str) else genres
        for genre in genre_list:
            data = data.filter(F.lower(F.col("genres")).contains(genre.lower()))
    if cast:
        data = data.filter(F.lower(F.col("cast")).contains(cast.lower()))
    if director:
        data = data.filter(F.lower(F.col("director")).contains(director.lower()))

    order = F.col(sort_by).asc() if ascending else F.col(sort_by).desc()
    result = data.orderBy(order)
    logger.info("search_movies(genres=%s, cast=%s, director=%s): %d row(s)", genres, cast, director, result.count())
    return result


def search_scifi_action_bruce_willis(df):
    """
    Search 1: best-rated Science-Fiction Action movies starring Bruce
    Willis, sorted by rating (highest to lowest).
    """
    result = search_movies(
        df,
        genres=["Science Fiction", "Action"],
        cast="Bruce Willis",
        sort_by="vote_average",
        ascending=False,
    )
    return result.select("title", "vote_average", "genres", "cast")


def search_thurman_tarantino(df):
    """
    Search 2: movies starring Uma Thurman directed by Quentin Tarantino,
    sorted by runtime (shortest to longest).
    """
    result = search_movies(
        df,
        cast="Uma Thurman",
        director="Quentin Tarantino",
        sort_by="runtime",
        ascending=True,
    )
    return result.select("title", "runtime", "director", "cast")


# Task 3.3: franchise vs standalone comparison

def franchise_vs_standalone(df):
    """
    Compare franchise movies (belongs_to_collection is set) against
    standalone movies across the KPIs the brief lists.

    Returns a 2-row table with an `is_franchise` column holding the labels
    "Franchise"/"Standalone" -- named the same as the pandas version's own
    groupby index, which (despite the boolean-sounding name) also ends up
    holding these same string labels once relabeled.
    """
    data = add_metrics(df)
    data = data.withColumn("is_franchise", F.col("belongs_to_collection").isNotNull())

    summary = data.groupBy("is_franchise").agg(
        F.mean("revenue_musd").alias("mean_revenue"),
        F.median("roi").alias("median_roi"),
        F.mean("budget_musd").alias("mean_budget"),
        F.mean("popularity").alias("mean_popularity"),
        F.mean("vote_average").alias("mean_rating"),
    )
    summary = summary.withColumn(
        "is_franchise",
        F.when(F.col("is_franchise"), F.lit("Franchise")).otherwise(F.lit("Standalone")),
    )
    # groupBy's own row order isn't guaranteed deterministic run to run;
    # this only has 2 rows, but franchise_vs_standalone_plot (Phase 7)
    # relies on a stable label order to bar-chart it consistently.
    return summary.orderBy("is_franchise")


# Task 3.4: most successful franchises

def franchise_summary(df, min_movies=2):
    """
    Aggregate KPIs per franchise (belongs_to_collection).

    Parameters
    ----------
    min_movies : int
        Only keep franchises with at least this many movies in the dataset.

    Returns a table sorted by total revenue (descending).
    """
    data = add_metrics(df).filter(F.col("belongs_to_collection").isNotNull())

    summary = data.groupBy("belongs_to_collection").agg(
        F.count("title").alias("num_movies"),
        F.sum("budget_musd").alias("total_budget"),
        F.mean("budget_musd").alias("mean_budget"),
        F.sum("revenue_musd").alias("total_revenue"),
        F.mean("revenue_musd").alias("mean_revenue"),
        F.mean("vote_average").alias("mean_rating"),
    )
    summary = summary.filter(F.col("num_movies") >= min_movies)
    return summary.orderBy(F.col("total_revenue").desc())


# Task 3.5: most successful directors

def director_summary(df):
    """
    Aggregate KPIs per director.

    A movie can list more than one director (joined by '|'), so the column
    is split and exploded first, giving each director their own row(s) --
    Spark's split()+explode() map almost verbatim to the pandas
    .str.split("|")+.explode() this replaces. Note the pipe is escaped in
    the split pattern: unlike pandas (which treats a single-character
    separator literally), Spark's split() always takes a regex, and "|"
    unescaped is the regex alternation operator.

    Returns a table sorted by total revenue (descending).
    """
    data = add_metrics(df).filter(F.col("director").isNotNull())
    data = data.withColumn("director", F.explode(F.split(F.col("director"), "\\|")))

    summary = data.groupBy("director").agg(
        F.count("title").alias("num_movies"),
        F.sum("revenue_musd").alias("total_revenue"),
        F.mean("vote_average").alias("mean_rating"),
    )
    return summary.orderBy(F.col("total_revenue").desc())


def roi_by_genre_summary(df):
    """
    Median ROI per genre, exploding the pipe-joined genres column so a
    movie with multiple genres contributes to each. Lives here rather than
    in visualization.py -- an architectural improvement over the pandas
    original, which mixed this same shaping logic into its plotting
    function -- reusing the same split+explode pattern as director_summary.

    Returns a table sorted by median ROI ascending, matching the pandas
    version's own sort (a horizontal bar chart reads largest-on-top when
    the underlying data is sorted ascending).
    """
    data = add_metrics(df).filter(F.col("genres").isNotNull() & F.col("roi").isNotNull())
    data = data.withColumn("genres", F.explode(F.split(F.col("genres"), "\\|")))

    summary = data.groupBy("genres").agg(F.median("roi").alias("median_roi"))
    return summary.orderBy(F.col("median_roi").asc())
