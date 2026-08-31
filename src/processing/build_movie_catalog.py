from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

IMDB_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "imdb_movies.parquet"
)

TMDB_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "tmdb_movies_enriched.jsonl"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "movies_enriched.parquet"
)


def main() -> None:
    print("Loading IMDb dataset...")

    imdb = pd.read_parquet(IMDB_PATH)

    print(f"IMDb rows: {len(imdb):,}")

    print("Loading TMDB enriched dataset...")

    tmdb = pd.read_json(
        TMDB_PATH,
        lines=True,
    )

    tmdb = tmdb[
        tmdb["status"] == "matched"
    ].copy()

    print(f"TMDB matched rows: {len(tmdb):,}")

    # Remove TMDB fields that duplicate information
    # already available from IMDb and that we do not
    # need in the final catalog.
    tmdb = tmdb.drop(
        columns=[
            "status",
            "title",
            "original_title",
            "release_date",
            "runtime",
        ],
        errors="ignore",
    )

    print("Joining IMDb and TMDB...")

    movies = imdb.merge(
        tmdb,
        on="tconst",
        how="inner",
        suffixes=("_imdb", "_tmdb"),
    )

    print(
        f"Final rows after join: "
        f"{len(movies):,}"
    )

    # Both IMDb and TMDB contain genres.
    # For the semantic recommendation system,
    # we keep the TMDB version because it is
    # consistent with overview, keywords,
    # director and cast.
    movies = movies.rename(
        columns={
            "genres_tmdb": "genres",
        }
    )

    columns = [
        "tconst",
        "tmdb_id",
        "primaryTitle",
        "originalTitle",
        "startYear",
        "runtimeMinutes",
        "averageRating",
        "numVotes",
        "overview",
        "genres",
        "keywords",
        "director",
        "cast",
    ]

    missing_columns = [
        column
        for column in columns
        if column not in movies.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing columns after join: "
            f"{missing_columns}"
        )

    movies = movies[columns].copy()

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    movies.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    print()
    print("=== MOVIE CATALOG CREATED ===")
    print(f"Rows: {len(movies):,}")
    print(
        f"Columns: "
        f"{len(movies.columns)}"
    )
    print(f"Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()