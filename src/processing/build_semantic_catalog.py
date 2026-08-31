from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "movies_enriched.parquet"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "movies_semantic.parquet"
)


def clean_text(value) -> str:
    """
    Normalize a potentially missing text value.
    """

    if pd.isna(value):
        return ""

    return str(value).strip()


def build_overview_text(row: pd.Series) -> str:
    """
    Representation A:
    movie overview only.
    """

    return clean_text(row["overview"])


def build_content_text(row: pd.Series) -> str:
    """
    Representation B:
    overview + genres + keywords.

    Focuses on plot and thematic information while
    avoiding person-related metadata.
    """

    parts = []

    overview = clean_text(row["overview"])
    genres = clean_text(row["genres"])
    keywords = clean_text(row["keywords"])

    if overview:
        parts.append(
            f"Plot: {overview}"
        )

    if genres:
        parts.append(
            f"Genres: {genres}"
        )

    if keywords:
        parts.append(
            f"Keywords: {keywords}"
        )

    return "\n".join(parts)


def build_enriched_text(row: pd.Series) -> str:
    """
    Representation C:
    overview + genres + keywords + director + cast.
    """

    parts = []

    overview = clean_text(row["overview"])
    genres = clean_text(row["genres"])
    keywords = clean_text(row["keywords"])
    director = clean_text(row["director"])
    cast = clean_text(row["cast"])

    if overview:
        parts.append(
            f"Plot: {overview}"
        )

    if genres:
        parts.append(
            f"Genres: {genres}"
        )

    if keywords:
        parts.append(
            f"Keywords: {keywords}"
        )

    if director:
        parts.append(
            f"Director: {director}"
        )

    if cast:
        parts.append(
            f"Cast: {cast}"
        )

    return "\n".join(parts)


def main() -> None:
    print("Loading enriched movie catalog...")

    movies = pd.read_parquet(
        INPUT_PATH
    )

    print(
        f"Initial movies: "
        f"{len(movies):,}"
    )

    text_columns = [
        "overview",
        "genres",
        "keywords",
        "director",
        "cast",
    ]

    for column in text_columns:
        movies[column] = (
            movies[column]
            .apply(clean_text)
        )

    # An overview is required for all experiments.
    movies = movies[
        movies["overview"] != ""
    ].copy()

    print(
        f"Movies with overview: "
        f"{len(movies):,}"
    )

    movies["overview_text"] = (
        movies.apply(
            build_overview_text,
            axis=1,
        )
    )

    movies["content_text"] = (
        movies.apply(
            build_content_text,
            axis=1,
        )
    )

    movies["enriched_text"] = (
        movies.apply(
            build_enriched_text,
            axis=1,
        )
    )

    semantic_columns = [
        "overview_text",
        "content_text",
        "enriched_text",
    ]

    for column in semantic_columns:

        empty_count = (
            movies[column]
            .str.strip()
            .eq("")
            .sum()
        )

        if empty_count > 0:
            raise ValueError(
                f"{column} contains "
                f"{empty_count} empty rows."
            )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    movies.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    print()
    print(
        "=== SEMANTIC CATALOG CREATED ==="
    )

    print(
        f"Rows: {len(movies):,}"
    )

    print(
        f"Columns: "
        f"{len(movies.columns)}"
    )

    print(
        f"Output: {OUTPUT_PATH}"
    )

    print()
    print(
        "Representations created:"
    )

    print(
        "A - overview_text: "
        "overview only"
    )

    print(
        "B - content_text: "
        "overview + genres + keywords"
    )

    print(
        "C - enriched_text: "
        "overview + genres + keywords "
        "+ director + cast"
    )


if __name__ == "__main__":
    main()