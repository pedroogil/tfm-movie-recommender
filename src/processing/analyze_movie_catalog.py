from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "movies_enriched.parquet"
)


def main() -> None:
    movies = pd.read_parquet(INPUT_PATH)

    print("=== MOVIE CATALOG QUALITY REPORT ===")
    print()

    print(f"Rows: {len(movies):,}")
    print(f"Columns: {len(movies.columns)}")

    print("\n=== MISSING VALUES ===")

    missing = (
        movies.isna()
        .sum()
        .sort_values(ascending=False)
    )

    print(missing)

    print("\n=== EMPTY TEXT VALUES ===")

    text_columns = [
        "overview",
        "genres",
        "keywords",
        "director",
        "cast",
    ]

    for column in text_columns:
        empty_count = (
            movies[column]
            .fillna("")
            .astype(str)
            .str.strip()
            .eq("")
            .sum()
        )

        percentage = (
            empty_count
            / len(movies)
            * 100
        )

        print(
            f"{column:>12}: "
            f"{empty_count:>6,} "
            f"({percentage:.2f}%)"
        )

    print("\n=== DUPLICATES ===")

    tconst_duplicates = (
        movies["tconst"]
        .duplicated()
        .sum()
    )

    tmdb_duplicates = (
        movies["tmdb_id"]
        .duplicated()
        .sum()
    )

    print(
        f"tconst duplicates: "
        f"{tconst_duplicates}"
    )

    print(
        f"tmdb_id duplicates: "
        f"{tmdb_duplicates}"
    )

    print("\n=== IMDb RATING STATISTICS ===")

    print(
        movies["averageRating"]
        .describe()
    )

    print("\n=== IMDb VOTE STATISTICS ===")

    print(
        movies["numVotes"]
        .describe()
    )

    print("\n=== YEAR STATISTICS ===")

    print(
        movies["startYear"]
        .describe()
    )

    print("\n=== MOVIES WITHOUT OVERVIEW ===")

    without_overview = movies[
        movies["overview"]
        .fillna("")
        .astype(str)
        .str.strip()
        .eq("")
    ]

    print(
        f"Movies without overview: "
        f"{len(without_overview):,}"
    )

    if len(without_overview) > 0:
        print(
            without_overview[
                [
                    "tconst",
                    "primaryTitle",
                    "startYear",
                ]
            ]
            .head(20)
            .to_string(index=False)
        )

    print("\n=== SAMPLE MOVIES ===")

    sample_columns = [
        "primaryTitle",
        "startYear",
        "averageRating",
        "genres",
        "director",
    ]

    print(
        movies[
            sample_columns
        ]
        .head(10)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()