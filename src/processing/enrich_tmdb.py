import json
import time
from pathlib import Path

import pandas as pd

from src.ingestion.tmdb_client import (
    find_movie_by_imdb_id,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "imdb_movies.parquet"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "imdb_tmdb_mapping.jsonl"
)


def load_processed_ids() -> set[str]:
    """Load IMDb IDs already processed."""

    if not OUTPUT_PATH.exists():
        return set()

    processed = set()

    with OUTPUT_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:

        for line in file:
            record = json.loads(line)
            processed.add(record["tconst"])

    return processed


movies = pd.read_parquet(INPUT_PATH)

processed_ids = load_processed_ids()

print(
    f"Already processed: "
    f"{len(processed_ids):,}"
)

remaining = movies[
    ~movies["tconst"].isin(processed_ids)
]

remaining = remaining.head(1000)

print(
    f"Remaining: "
    f"{len(remaining):,}"
)


with OUTPUT_PATH.open(
    "a",
    encoding="utf-8",
) as output_file:

    for index, imdb_id in enumerate(
        remaining["tconst"],
        start=1,
    ):

        try:

            tmdb_id = find_movie_by_imdb_id(
                imdb_id
            )

            record = {
                "tconst": imdb_id,
                "tmdb_id": tmdb_id,
                "status": (
                    "matched"
                    if tmdb_id is not None
                    else "not_found"
                ),
            }

        except Exception as error:

            record = {
                "tconst": imdb_id,
                "tmdb_id": None,
                "status": "error",
                "error": str(error),
            }

        output_file.write(
            json.dumps(
                record,
                ensure_ascii=False,
            )
            + "\n"
        )

        output_file.flush()

        if index % 100 == 0:

            print(
                f"Processed "
                f"{index:,}/{len(remaining):,}"
            )

        time.sleep(0.1)