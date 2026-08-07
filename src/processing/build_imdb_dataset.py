from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

BASICS_PATH = PROJECT_ROOT / "data" / "raw" / "imdb" / "title.basics.tsv"
RATINGS_PATH = PROJECT_ROOT / "data" / "raw" / "imdb" / "title.ratings.tsv"

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "imdb_movies.parquet"
)


print("Loading IMDb basics...")

basics = pd.read_csv(
    BASICS_PATH,
    sep="\t",
    na_values="\\N",
    dtype={
        "tconst": "string",
        "titleType": "string",
        "primaryTitle": "string",
        "originalTitle": "string",
        "isAdult": "string",
        "startYear": "string",
        "endYear": "string",
        "runtimeMinutes": "string",
        "genres": "string",
    },
)

print("Filtering movies...")

movies = basics[
    (basics["titleType"] == "movie")
    & (basics["isAdult"] == "0")
].copy()

movies["startYear"] = pd.to_numeric(
    movies["startYear"],
    errors="coerce",
)

movies["runtimeMinutes"] = pd.to_numeric(
    movies["runtimeMinutes"],
    errors="coerce",
)


print("Loading IMDb ratings...")

ratings = pd.read_csv(
    RATINGS_PATH,
    sep="\t",
)


print("Joining datasets...")

movies = movies.merge(
    ratings,
    on="tconst",
    how="inner",
)


print("Applying vote threshold...")

movies = movies[
    movies["numVotes"] >= 10_000
].copy()


columns = [
    "tconst",
    "primaryTitle",
    "originalTitle",
    "startYear",
    "runtimeMinutes",
    "genres",
    "averageRating",
    "numVotes",
]

movies = movies[columns]


OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

movies.to_parquet(
    OUTPUT_PATH,
    index=False,
)


print(f"\nDataset created: {OUTPUT_PATH}")
print(f"Movies: {len(movies):,}")
print(f"Columns: {len(movies.columns)}")