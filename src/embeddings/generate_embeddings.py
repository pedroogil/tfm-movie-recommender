from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "movies_semantic.parquet"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "embeddings"
)

OVERVIEW_OUTPUT_PATH = (
    OUTPUT_DIR
    / "overview_embeddings.npy"
)

CONTENT_OUTPUT_PATH = (
    OUTPUT_DIR
    / "content_embeddings.npy"
)

ENRICHED_OUTPUT_PATH = (
    OUTPUT_DIR
    / "enriched_embeddings.npy"
)

METADATA_OUTPUT_PATH = (
    OUTPUT_DIR
    / "embedding_metadata.parquet"
)

MODEL_NAME = (
    "sentence-transformers/"
    "all-MiniLM-L6-v2"
)


def generate_embeddings(
    model: SentenceTransformer,
    texts: list[str],
    label: str,
) -> np.ndarray:
    """
    Generate normalized sentence embeddings.
    """

    print()
    print(
        f"Generating {label} embeddings..."
    )

    return model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
    )


def main() -> None:
    print("Loading semantic catalog...")

    movies = pd.read_parquet(
        INPUT_PATH
    )

    print(
        f"Movies loaded: "
        f"{len(movies):,}"
    )

    print()
    print(
        f"Loading embedding model: "
        f"{MODEL_NAME}"
    )

    model = SentenceTransformer(
        MODEL_NAME
    )

    overview_embeddings = (
        generate_embeddings(
            model,
            movies[
                "overview_text"
            ].tolist(),
            "overview-only",
        )
    )

    content_embeddings = (
        generate_embeddings(
            model,
            movies[
                "content_text"
            ].tolist(),
            "content",
        )
    )

    enriched_embeddings = (
        generate_embeddings(
            model,
            movies[
                "enriched_text"
            ].tolist(),
            "enriched",
        )
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.save(
        OVERVIEW_OUTPUT_PATH,
        overview_embeddings,
    )

    np.save(
        CONTENT_OUTPUT_PATH,
        content_embeddings,
    )

    np.save(
        ENRICHED_OUTPUT_PATH,
        enriched_embeddings,
    )

    metadata_columns = [
        "tconst",
        "tmdb_id",
        "primaryTitle",
        "originalTitle",
        "startYear",
        "averageRating",
        "numVotes",
        "genres",
        "director",
    ]

    metadata = movies[
        metadata_columns
    ].copy()

    metadata.to_parquet(
        METADATA_OUTPUT_PATH,
        index=False,
    )

    print()
    print(
        "=== EMBEDDINGS CREATED ==="
    )

    print(
        "Overview shape:",
        overview_embeddings.shape,
    )

    print(
        "Content shape:",
        content_embeddings.shape,
    )

    print(
        "Enriched shape:",
        enriched_embeddings.shape,
    )


if __name__ == "__main__":
    main()