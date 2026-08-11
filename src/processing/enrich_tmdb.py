import json
import time
from pathlib import Path

import pandas as pd

from src.ingestion.tmdb_client import find_movie_by_imdb_id


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
    """Load IMDb IDs that have already been processed."""

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

remaining = movies[
    ~movies["tconst"].isin(processed_ids)
].copy()

print(f"Already processed: {len(processed_ids):,}")
print(f"Remaining: {len(remaining):,}")

if len(remaining) == 0:
    print("Nothing to process.")
    raise SystemExit


matched = 0
not_found = 0
errors = 0

start_time = time.time()

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

            if tmdb_id is not None:

                status = "matched"
                matched += 1

            else:

                status = "not_found"
                not_found += 1

            record = {
                "tconst": imdb_id,
                "tmdb_id": tmdb_id,
                "status": status,
            }

        except Exception as error:

            errors += 1

            record = {
                "tconst": imdb_id,
                "tmdb_id": None,
                "status": "error",
                "error": str(error),
            }

            print(
                f"ERROR {imdb_id}: {error}"
            )

        output_file.write(
            json.dumps(
                record,
                ensure_ascii=False,
            )
            + "\n"
        )

        output_file.flush()

        if index % 100 == 0:

            elapsed = time.time() - start_time

            print(
                f"Processed "
                f"{index:,}/{len(remaining):,} "
                f"| "
                f"Matched: {matched:,} "
                f"| "
                f"Not found: {not_found:,} "
                f"| "
                f"Errors: {errors:,} "
                f"| "
                f"Time: {elapsed / 60:.1f} min"
            )

        # Small pause between requests.
        time.sleep(0.25)


elapsed = time.time() - start_time

print("\n=== PROCESS FINISHED ===")

print(
    f"Processed: {len(remaining):,}"
)

print(
    f"Matched: {matched:,}"
)

print(
    f"Not found: {not_found:,}"
)

print(
    f"Errors: {errors:,}"
)

print(
    f"Time: {elapsed / 60:.1f} minutes"
)