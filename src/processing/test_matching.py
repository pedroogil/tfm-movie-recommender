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


movies = pd.read_parquet(INPUT_PATH)

sample = movies.head(10)

print("Testing IMDb → TMDB matching\n")

for imdb_id in sample["tconst"]:
    tmdb_id = find_movie_by_imdb_id(imdb_id)

    print(
        f"{imdb_id} -> {tmdb_id}"
    )