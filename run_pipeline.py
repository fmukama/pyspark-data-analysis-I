import sys

import matplotlib
matplotlib.use("Agg")  # headless entry point -- no display, no notebook

from src import analysis, config, visualization
from src.ingestion import ensure_raw_cache
from src.logger import get_logger
from src.preprocessing import preprocess
from src.spark_session import get_spark_session

logger = get_logger("run_pipeline")


def _show(title, df, truncate=False):
    print(f"\n{title}")
    df.show(truncate=truncate)


def main():
    # The try/except lives inside main(), not around the call to it, so
    # `stage` stays in scope for the except block below -- it needs to be
    # the *same* variable that's been updated throughout the run, not a
    # separate one reconstructed from an outer scope's locals().
    stage = "startup"
    try:
        spark = get_spark_session()

        stage = "ingestion"
        print("Checking TMDB token...")
        config.check_token()
        ensure_raw_cache()

        stage = "preprocessing"
        df = preprocess(spark)

        stage = "analysis"
        _show("Highest Revenue", analysis.highest_revenue(df))
        _show("Highest Budget", analysis.highest_budget(df))
        _show("Highest Profit (revenue - budget)", analysis.highest_profit(df))
        _show("Lowest Profit", analysis.lowest_profit(df))
        _show("Highest ROI (budget >= 10M)", analysis.highest_roi(df))
        _show("Lowest ROI (budget >= 10M)", analysis.lowest_roi(df))
        _show("Most Voted", analysis.most_voted(df))
        _show("Highest Rated (>= 10 votes)", analysis.highest_rated(df))
        _show("Lowest Rated (>= 10 votes)", analysis.lowest_rated(df))
        _show("Most Popular", analysis.most_popular(df))
        _show("Search 1: Sci-Fi Action, Bruce Willis", analysis.search_scifi_action_bruce_willis(df))
        _show("Search 2: Uma Thurman + Quentin Tarantino", analysis.search_thurman_tarantino(df))
        _show("Franchise vs Standalone", analysis.franchise_vs_standalone(df))
        _show("Franchise Summary", analysis.franchise_summary(df))
        _show("Director Summary", analysis.director_summary(df))

        stage = "visualization"
        chart_paths = visualization.generate_all_charts(df)
        print(f"\nWrote {len(chart_paths)} chart(s):")
        for path in chart_paths:
            print(f"  {path}")

        spark.stop()
        print("\nPipeline completed successfully.")
    except Exception:
        logger.exception("Pipeline failed during the '%s' stage", stage)
        print(f"\nPipeline FAILED during the '{stage}' stage -- "
              f"see logs/pipeline.log for the full traceback.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
