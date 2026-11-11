"""Dataset and evaluation constants.

Split values follow benchmarks/yambda/constants.py and
benchmarks/yambda/processing/timesplit.py of the yandex/yambda repository.
Timestamps are seconds since the first event, binned to 5 seconds.
"""

HF_REPO_ID = "yandex/yambda"
DATASET_SIZE = "50m"

# Files read from the Hub, relative to the repository root.
FLAT_FILES = [
    "flat/50m/listens.parquet",
    "flat/50m/likes.parquet",
    "flat/50m/dislikes.parquet",
    "flat/50m/unlikes.parquet",
    "flat/50m/undislikes.parquet",
]
EMBEDDINGS_FILE = "embeddings.parquet"
ALBUM_FILE = "album_item_mapping.parquet"
ARTIST_FILE = "artist_item_mapping.parquet"

# Global temporal split.
DAY_SECONDS = 86_400
GAP_SECONDS = 1_800
LAST_TIMESTAMP = 26_000_000
TEST_START = LAST_TIMESTAMP - DAY_SECONDS
VAL_END = TEST_START - GAP_SECONDS
VAL_START = VAL_END - DAY_SECONDS
TRAIN_END = VAL_START - GAP_SECONDS

# Listen+ : a listen counts when at least half of the track was played.
LISTEN_THRESHOLD_PCT = 50

# Ranking.
K_VALUES = [10, 50, 100]
NUM_RANKED = 100
NUM_CANDIDATES = 200
PADDING_INDEX = 0
