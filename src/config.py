import os

from dotenv import load_dotenv

load_dotenv()

TMDB_BEARER_TOKEN = os.getenv("TMDB_BEARER_TOKEN")

BASE_URL = "https://api.themoviedb.org/3"
HEADERS = {
    "accept": "application/json",
    "Authorization": f"Bearer {TMDB_BEARER_TOKEN}",
}

MOVIE_IDS = [
    0, 299534, 19995, 140607, 299536, 597, 135397,
    420818, 24428, 168259, 99861, 284054, 12445,
    181808, 330457, 351286, 109445, 321612, 260513,
]


def check_token() -> bool:
    """Small helper so the notebook can confirm the token was loaded."""
    if not TMDB_BEARER_TOKEN:
        print("No Bearer token found.")
        return False
    print("TMDB token loaded successfully.")
    return True

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "movies_raw.csv")
CLEAN_DATA_PATH = os.path.join(BASE_DIR, "data", "clean", "movies_clean.csv")
LOG_DIR = os.path.join(BASE_DIR, "logs")
LOG_FILE = os.path.join(LOG_DIR, "pipeline.log")
IMAGES_DIR = os.path.join(BASE_DIR, "images")

os.makedirs(os.path.dirname(RAW_DATA_PATH), exist_ok=True)
os.makedirs(os.path.dirname(CLEAN_DATA_PATH), exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)


SPARK_APP_NAME = "tmdb-movie-analysis"
SPARK_MASTER = "local[*]"
SPARK_SHUFFLE_PARTITIONS = 4  # the 200-partition default is pure overhead on ~18 rows
