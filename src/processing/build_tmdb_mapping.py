from pathlib import Path
import time

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
    / "imdb_tmdb_mapping.parquet"
)


movies = pd.read_parquet(INPUT_PATH)

print(f"Movies to process: {len(movies):,}")

results = []

for index, imdb_id in enumerate(movies["tconst"], start=1):

    try:
        tmdb_id = find_movie_by_imdb_id(imdb_id)

    except Exception as error:
        print(
            f"ERROR {imdb_id}: {error}"
        )
        tmdb_id = None

    results.append(
        {
            "tconst": imdb_id,
            "tmdb_id": tmdb_id,
        }
    )

    if index % 100 == 0:
        print(
            f"Processed: {index:,}/{len(movies):,}"
        )

    time.sleep(0.1)


mapping = pd.DataFrame(results)

mapping.to_parquet(
    OUTPUT_PATH,
    index=False,
)

matched = mapping["tmdb_id"].notna().sum()

print("\nFinished.")
print(f"Total: {len(mapping):,}")
print(f"Matched: {matched:,}")
print(
    f"Coverage: {matched / len(mapping) * 100:.2f}%"
)