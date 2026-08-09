"""
preprocessing.py
================
STEP 2: clean and transform the raw cached CSV with Spark.

The raw CSV has TMDB's nested fields (genres, cast, production companies,
...) JSON-encoded into single cells by ingestion.save_raw. we:
  1. read the cache with an explicit all-string schema, then parse those
     JSON-string columns into real struct/array columns via from_json,
  2. flatten the parsed structures into simple pipe-separated text columns,
  3. fix data types and replace impossible values (0 budget, etc.),
  4. drop duplicates/bad rows and keep only released movies,
  5. reorder to the final schema the brief specifies,
checking the data at each boundary along the way (see the module-level
docstrings on the _check_* functions for what each one guards against).

The public entry point is preprocess(spark), which runs the whole pipeline
and writes data/clean/movies_clean.csv.
"""

import os
import shutil

from pyspark.sql import functions as F
from pyspark.sql.types import (
    ArrayType, DoubleType, LongType, StringType, StructField, StructType,
)
from pyspark.sql.functions import udf

from src import config
from src.logger import get_logger

logger = get_logger("preprocessing")

_NAME_ARRAY_SCHEMA = ArrayType(StructType([StructField("name", StringType())]))
_COLLECTION_SCHEMA = StructType([StructField("name", StringType())])
_LANGUAGE_ARRAY_SCHEMA = ArrayType(StructType([StructField("english_name", StringType())]))
_CREDITS_SCHEMA = StructType([
    StructField("cast", ArrayType(StructType([StructField("name", StringType())]))),
    StructField("crew", ArrayType(StructType([
        StructField("name", StringType()),
        StructField("job", StringType()),
    ]))),
])

# Task 2.1: drop irrelevant columns.
DROP_COLUMNS = ["adult", "imdb_id", "original_title", "video", "homepage"]

# The exact final column order the brief specifies.
FINAL_COLUMN_ORDER = [
    "id", "title", "tagline", "release_date", "genres", "belongs_to_collection",
    "original_language", "budget_musd", "revenue_musd", "production_companies",
    "production_countries", "vote_count", "vote_average", "popularity", "runtime",
    "overview", "spoken_languages", "poster_path", "cast", "cast_size",
    "director", "crew_size",
]

def load_raw_spark(spark, path=None):
    """
    Read the cached raw CSV with every column defaulting to string -- no
    inferSchema, and deliberately no explicit StructType either. A fixed
    StructType looked like the obvious "explicit schema" choice, but Spark's
    CSV reader matches a declared schema against the header *by position*,
    not by name (confirmed directly: a 3-column file read against a
    hardcoded schema whose column order didn't line up came back all
    NULLs, not a name-matched read or a loud error) -- so a fixed schema
    is actually the fragile choice here, silently misaligning data the
    moment a file's column order or set doesn't exactly match what was
    hardcoded. Omitting both inferSchema and an explicit schema instead
    makes Spark use the header row for column *names* (robust to reordering)
    while still defaulting every column to StringType (satisfying "explicit,
    no type inference" the same way a StructType would, just not
    position-dependent). Nothing gets cast to a real type until
    fix_dtypes_and_values, mirroring the pandas version's own module
    boundary (its flatten_columns never touched dtypes either). Quoting
    matches Python's csv module default (RFC 4180: quotechar='"',
    doubled-quote escaping), which is what ingestion.save_raw actually
    wrote, so it's set explicitly here rather than left to Spark's own
    default in case that default ever changes.
    """
    if path is None:
        path = config.RAW_DATA_PATH

    df = (
        spark.read
        .option("header", True)
        .option("quote", '"')
        .option("escape", '"')
        .csv(path)
    )
    logger.info("Loaded %d raw rows from %s", df.count(), path)
    return df


def parse_nested_columns(df):
    """
    Parse the JSON-string columns into real struct/array columns. This
    fully replaces the pandas version's _as_object/ast.literal_eval dance --
    that existed only to cope with pandas' own dual input shape (a native
    object straight from the API vs. a stringified one after a CSV reload).
    Here there's exactly one input shape: always a freshly-read CSV string.
    """
    df = (
        df
        .withColumn("genres", F.from_json(F.col("genres"), _NAME_ARRAY_SCHEMA))
        .withColumn("belongs_to_collection", F.from_json(F.col("belongs_to_collection"), _COLLECTION_SCHEMA))
        .withColumn("production_companies", F.from_json(F.col("production_companies"), _NAME_ARRAY_SCHEMA))
        .withColumn("production_countries", F.from_json(F.col("production_countries"), _NAME_ARRAY_SCHEMA))
        .withColumn("spoken_languages", F.from_json(F.col("spoken_languages"), _LANGUAGE_ARRAY_SCHEMA))
        .withColumn("credits", F.from_json(F.col("credits"), _CREDITS_SCHEMA))
    )
    _check_parsed_schema(df)
    return df


def _check_parsed_schema(df):
    """
    Schema tier (raises): confirm genres/credits actually parsed into
    something usable on at least one row. Note what this deliberately does
    *not* check: a genuinely missing source column (e.g. the raw CSV
    lacking a "credits" header) already raises its own clear
    AnalysisException the moment the withColumn calls above reference it --
    so "column missing" isn't a reachable failure mode to check for here.

    What Spark won't catch on its own is every row silently parsing to
    nothing because TMDB's real field names don't match what this pipeline
    assumed. Confirmed directly (not assumed) that from_json's failure
    shape differs by type, so the two columns need different checks:
    an ArrayType (genres) can come back as a genuine null on a parse
    failure, but a StructType (credits) never does -- from_json's
    PERMISSIVE default instead returns a non-null struct with every field
    null (e.g. {cast: null, crew: null}), so checking "is credits null"
    would silently never fire; checking its actual fields is what works.
    """
    total_rows = df.count()
    if total_rows == 0:
        return

    genres_present = df.filter(F.size(F.col("genres")) > 0).count()
    if genres_present == 0:
        raise ValueError(
            "parse_nested_columns: no row has a non-empty 'genres' array after from_json -- "
            "this should be present on essentially every real movie, so the parse likely "
            "silently failed rather than the source data genuinely lacking it everywhere."
        )

    credits_present = df.filter(F.col("credits.cast").isNotNull() | F.col("credits.crew").isNotNull()).count()
    if credits_present == 0:
        raise ValueError(
            "parse_nested_columns: no row has a non-null 'credits.cast' or 'credits.crew' after "
            "from_json -- this should be present on essentially every real movie, so the parse "
            "likely silently failed rather than the source data genuinely lacking it everywhere."
        )


@udf(returnType=StringType())
def _extract_director(crew):
    """
    Pipe-join the name(s) of crew members whose job is 'Director'.

    This is the one deliberate registered UDF in this module (Confirmed
    Decision 2): satisfies the brief's literal "define a UDF" requirement
    where it's honestly defensible (filter-then-join branching logic), not
    because it's the only way to do this in Spark -- the built-in
    alternative would be:
        array_join(transform(filter(col("credits.crew"),
                                     lambda x: x["job"] == "Director"),
                              lambda x: x["name"]), "|")
    Returns None (not "") when there's no crew or no one credited as
    Director, matching the pandas version's _director() helper.
    """
    if not crew:
        return None
    directors = [member["name"] for member in crew if member["job"] == "Director"]
    return "|".join(directors) if directors else None


def _joined_names(array_column, sep="|"):
    """
    Pipe-join an array<struct<...>>-shaped column already projected down to
    the field of interest (e.g. col("genres.name")). Both a null array and
    an *empty* array become null, not an empty string -- matching the
    pandas version's _names()/_cast_names() helpers, which treat "no items"
    as NaN. F.array_join alone would only null out the null case; the
    F.size guard is what handles the empty-array case the same way.
    """
    return F.when(F.size(array_column) > 0, F.array_join(array_column, sep)).otherwise(F.lit(None).cast(StringType()))


def flatten_columns(df):
    """
    Drop irrelevant columns and turn every parsed nested field into a clean
    text column: genres, collection, languages, countries, companies, cast,
    director, plus cast_size/crew_size.
    """
    df = df.drop(*[c for c in DROP_COLUMNS if c in df.columns])

    df = df.withColumn("genres", _joined_names(F.col("genres.name")))
    df = df.withColumn("belongs_to_collection", F.col("belongs_to_collection.name"))
    df = df.withColumn("production_companies", _joined_names(F.col("production_companies.name")))
    df = df.withColumn("production_countries", _joined_names(F.col("production_countries.name")))
    df = df.withColumn("spoken_languages", _joined_names(F.col("spoken_languages.english_name")))

    df = df.withColumn("cast", _joined_names(F.col("credits.cast.name")))
    df = df.withColumn("cast_size", F.size(F.col("credits.cast")))
    df = df.withColumn("director", _extract_director(F.col("credits.crew")))
    df = df.withColumn("crew_size", F.size(F.col("credits.crew")))
    df = df.drop("credits")

    return df


# Task 2.5/2.6: dtypes, unrealistic values, unit conversion.
_NUMERIC_DOUBLE_COLUMNS = ["budget", "popularity", "revenue", "runtime", "vote_average"]
_NUMERIC_LONG_COLUMNS = ["id", "vote_count"]

_PLACEHOLDER_TEXT = ["No Data", "No overview found.", ""]


def fix_dtypes_and_values(df):
    """
    Convert data types, replace impossible values with null, and express
    money in millions of USD.
    """
    for column in _NUMERIC_DOUBLE_COLUMNS:
        df = df.withColumn(column, F.col(column).cast(DoubleType()))
    for column in _NUMERIC_LONG_COLUMNS:
        df = df.withColumn(column, F.col(column).cast(LongType()))

    df = df.withColumn("release_date", F.to_date(F.col("release_date")))

    # A budget/revenue/runtime of 0 is not real data -> mark as missing.
    for column in ["budget", "revenue", "runtime"]:
        df = df.withColumn(column, F.when(F.col(column) == 0, F.lit(None)).otherwise(F.col(column)))

    df = df.withColumn("budget_musd", F.col("budget") / 1_000_000)
    df = df.withColumn("revenue_musd", F.col("revenue") / 1_000_000)

    # A movie with 0 votes has a meaningless 0.0 average rating -> null.
    df = df.withColumn(
        "vote_average",
        F.when(F.col("vote_count") == 0, F.lit(None)).otherwise(F.col("vote_average")),
    )

    # Replace known placeholder text with null.
    for column in ["overview", "tagline"]:
        df = df.withColumn(
            column,
            F.when(F.col(column).isin(_PLACEHOLDER_TEXT), F.lit(None)).otherwise(F.col(column)),
        )

    _check_value_ranges(df)
    return df


def _check_value_ranges(df):
    """
    Quality tier (warns, doesn't raise): real-world API data is expected to
    have occasional blemishes, so a violation here is logged and the
    pipeline continues rather than halting. Each check counts offending
    rows with a single aggregation rather than collecting them, since only
    the count is needed to decide whether to log.
    """
    today = F.current_date()
    checks = {
        "budget_musd negative": F.col("budget_musd") < 0,
        "revenue_musd negative": F.col("revenue_musd") < 0,
        "runtime <= 0": F.col("runtime") <= 0,
        "vote_average outside [0, 10]": (F.col("vote_average") < 0) | (F.col("vote_average") > 10),
        "vote_average not null despite vote_count == 0": (F.col("vote_count") == 0) & F.col("vote_average").isNotNull(),
        "release_date after today": F.col("release_date") > today,
        "release_date before 1888-01-01": F.col("release_date") < F.lit("1888-01-01").cast("date"),
    }

    counts = df.select([F.sum(condition.cast("int")).alias(name) for name, condition in checks.items()]).first()

    for name in checks:
        count = counts[name] or 0
        if count > 0:
            logger.warning("Value-range check failed for %d row(s): %s", count, name)


def clean_rows(df):
    """
    Remove duplicates and bad rows, keep only well-populated released
    movies, then reorder to the final schema.
    """
    before = df.count()

    # Task 2.7: drop rows with no id or title, then dedupe on id (the
    # correct natural key regardless -- unlike the pandas version, Spark's
    # typed columns were never at risk of the "unhashable list column"
    # problem a whole-row dropDuplicates() would have hit there).
    df = df.dropna(subset=["id", "title"])
    df = df.dropDuplicates(["id"])
    after_dedup = df.count()

    # Task 2.8: keep only rows with >= 10 non-null values. pandas'
    # dropna(thresh=10) has no direct Spark equivalent, so it's hand-built:
    # sum a 0/1 indicator per column and filter on the row total.
    non_null_count = sum(F.col(c).isNotNull().cast("int") for c in df.columns)
    df = df.withColumn("_non_null_count", non_null_count)
    df = df.filter(F.col("_non_null_count") >= 10).drop("_non_null_count")
    after_threshold = df.count()

    # Task 2.9: keep only 'Released' movies, then drop the status column.
    if "status" in df.columns:
        df = df.filter(F.col("status") == "Released")
        df = df.drop("status")
    after_status = df.count()

    logger.info(
        "clean_rows: %d -> %d (dedup) -> %d (>=10 non-null) -> %d (Released only)",
        before, after_dedup, after_threshold, after_status,
    )

    # Task 2.10: reorder to the exact layout the brief asks for. No
    # .reset_index() -- Spark has no positional row index to reset.
    df = df.select(*[c for c in FINAL_COLUMN_ORDER if c in df.columns])

    _check_final_schema(df)
    return df


_EXPECTED_ROW_COUNT = 18  # dem-02-lab's own known-good result on this exact 19-id list


def _check_final_schema(df):
    """
    Schema tier (raises) for structure, quality tier (warns) for the count.
    A column-order/duplicate-id break here is a code bug, not messy data --
    it means clean_rows itself regressed, not that TMDB's data is unusual.
    """
    if df.columns != FINAL_COLUMN_ORDER:
        raise ValueError(
            f"clean_rows: final columns {df.columns} do not match FINAL_COLUMN_ORDER {FINAL_COLUMN_ORDER}."
        )

    total = df.count()
    distinct_ids = df.select("id").distinct().count()
    if distinct_ids != total:
        raise ValueError(f"clean_rows: {total} rows but only {distinct_ids} distinct id(s) -- duplicates survived.")

    null_ids = df.filter(F.col("id").isNull()).count()
    if null_ids > 0:
        raise ValueError(f"clean_rows: {null_ids} row(s) have a null id.")

    if abs(total - _EXPECTED_ROW_COUNT) > 2:
        logger.warning(
            "clean_rows: final row count is %d, expected close to %d -- worth investigating "
            "(TMDB data can shift slightly over time, so this warns rather than raises).",
            total, _EXPECTED_ROW_COUNT,
        )


def _write_single_csv(df, path):
    """
    Write `df` as a single, plainly-named CSV file at `path` rather than
    the part-file directory Spark's writer normally produces -- a
    human-readable single file matches the pandas project's own
    movies_clean.csv artifact and stays easy to open directly.
    """
    tmp_dir = path + ".tmp"
    df.coalesce(1).write.mode("overwrite").option("header", True).csv(tmp_dir)

    part_file = next(name for name in os.listdir(tmp_dir) if name.startswith("part-") and name.endswith(".csv"))
    if os.path.exists(path):
        os.remove(path)
    shutil.move(os.path.join(tmp_dir, part_file), path)
    shutil.rmtree(tmp_dir)
    logger.info("Wrote clean dataset to %s", path)


def preprocess(spark, raw_path=None, clean_path=None):
    """Run the whole cleaning pipeline and write the result to CSV."""
    if clean_path is None:
        clean_path = config.CLEAN_DATA_PATH

    df = load_raw_spark(spark, raw_path)
    df = parse_nested_columns(df)
    df = flatten_columns(df)
    df = fix_dtypes_and_values(df)
    df = clean_rows(df)

    _write_single_csv(df, clean_path)
    logger.info("Clean dataset: %d movies x %d columns.", df.count(), len(df.columns))
    return df
