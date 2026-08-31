from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

METADATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "embeddings"
    / "embedding_metadata.parquet"
)

OVERVIEW_EMBEDDINGS_PATH = (
    PROJECT_ROOT
    / "data"
    / "embeddings"
    / "overview_embeddings.npy"
)

CONTENT_EMBEDDINGS_PATH = (
    PROJECT_ROOT
    / "data"
    / "embeddings"
    / "content_embeddings.npy"
)

ENRICHED_EMBEDDINGS_PATH = (
    PROJECT_ROOT
    / "data"
    / "embeddings"
    / "enriched_embeddings.npy"
)


def find_movie_index(
    title: str,
    metadata: pd.DataFrame,
) -> int:
    """
    Find a movie row by title.

    Matching is case insensitive.
    """

    matches = metadata[
        metadata["primaryTitle"]
        .str.lower()
        .eq(title.lower())
    ]

    if matches.empty:
        raise ValueError(
            f"Movie not found: {title}"
        )

    if len(matches) > 1:
        print(
            f"Warning: found {len(matches)} "
            f"movies called '{title}'."
        )

        print(
            matches[
                [
                    "primaryTitle",
                    "startYear",
                    "tmdb_id",
                ]
            ]
            .to_string(index=False)
        )

        print(
            "Using the first match."
        )

    return matches.index[0]


def recommend_similar_movies(
    movie_index: int,
    embeddings: np.ndarray,
    metadata: pd.DataFrame,
    top_k: int = 5,
) -> pd.DataFrame:
    """
    Recommend movies using cosine similarity.

    Embeddings are already normalized, so the
    dot product is equivalent to cosine similarity.
    """

    movie_embedding = embeddings[
        movie_index
    ]

    similarities = (
        embeddings
        @ movie_embedding
    )

    ranked_indices = np.argsort(
        similarities
    )[::-1]

    ranked_indices = [
        index
        for index in ranked_indices
        if index != movie_index
    ]

    top_indices = ranked_indices[
        :top_k
    ]

    results = metadata.iloc[
        top_indices
    ].copy()

    results["similarity"] = (
        similarities[top_indices]
    )

    return results


def print_results(
    label: str,
    results: pd.DataFrame,
) -> None:
    """
    Print recommendation results.
    """

    print()
    print(f"--- {label} ---")

    columns = [
        "primaryTitle",
        "startYear",
        "averageRating",
        "genres",
        "director",
        "similarity",
    ]

    print(
        results[
            columns
        ]
        .to_string(index=False)
    )


def compare_representations(
    movie_title: str,
    metadata: pd.DataFrame,
    overview_embeddings: np.ndarray,
    content_embeddings: np.ndarray,
    enriched_embeddings: np.ndarray,
) -> None:
    """
    Compare all semantic representations for one movie.
    """

    movie_index = find_movie_index(
        movie_title,
        metadata,
    )

    selected_movie = metadata.iloc[
        movie_index
    ]

    print()
    print("=" * 100)

    print(
        f"{selected_movie['primaryTitle']} "
        f"({int(selected_movie['startYear'])})"
    )

    print("=" * 100)

    overview_results = (
        recommend_similar_movies(
            movie_index=movie_index,
            embeddings=overview_embeddings,
            metadata=metadata,
            top_k=5,
        )
    )

    content_results = (
        recommend_similar_movies(
            movie_index=movie_index,
            embeddings=content_embeddings,
            metadata=metadata,
            top_k=5,
        )
    )

    enriched_results = (
        recommend_similar_movies(
            movie_index=movie_index,
            embeddings=enriched_embeddings,
            metadata=metadata,
            top_k=5,
        )
    )

    print_results(
        "A - OVERVIEW ONLY",
        overview_results,
    )

    print_results(
        "B - OVERVIEW + GENRES + KEYWORDS",
        content_results,
    )

    print_results(
        "C - OVERVIEW + GENRES + KEYWORDS + DIRECTOR + CAST",
        enriched_results,
    )


def main() -> None:
    print("Loading metadata...")

    metadata = pd.read_parquet(
        METADATA_PATH
    )

    print("Loading embeddings...")

    overview_embeddings = np.load(
        OVERVIEW_EMBEDDINGS_PATH
    )

    content_embeddings = np.load(
        CONTENT_EMBEDDINGS_PATH
    )

    enriched_embeddings = np.load(
        ENRICHED_EMBEDDINGS_PATH
    )

    test_movies = [
        "Inception",
        "The Godfather",
        "Toy Story",
        "The Shining",
        "La La Land",
    ]

    for movie_title in test_movies:
        try:
            compare_representations(
                movie_title=movie_title,
                metadata=metadata,
                overview_embeddings=overview_embeddings,
                content_embeddings=content_embeddings,
                enriched_embeddings=enriched_embeddings,
            )

        except ValueError as error:
            print()
            print(
                f"Skipping {movie_title}: "
                f"{error}"
            )


if __name__ == "__main__":
    main()