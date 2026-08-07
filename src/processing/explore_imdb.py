from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

BASICS_PATH = PROJECT_ROOT / "data" / "raw" / "imdb" / "title.basics.tsv"
RATINGS_PATH = PROJECT_ROOT / "data" / "raw" / "imdb" / "title.ratings.tsv"


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

print(f"Total titles: {len(basics):,}")

print("\nTitle types:")
print(basics["titleType"].value_counts().head(15))

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

print(f"\nMovies (non-adult): {len(movies):,}")

print("\nLoading IMDb ratings...")

ratings = pd.read_csv(
    RATINGS_PATH,
    sep="\t",
)

print(f"Total ratings: {len(ratings):,}")

movies = movies.merge(
    ratings,
    on="tconst",
    how="inner",
)

print(f"\nMovies after ratings join: {len(movies):,}")

movies = movies[movies["numVotes"] >= 10_000].copy()

print(f"Movies with >= 10,000 votes: {len(movies):,}")