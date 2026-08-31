from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


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

ENRICHED_EMBEDDINGS_PATH = (
    PROJECT_ROOT
    / "data"
    / "embeddings"
    / "enriched_embeddings.npy"
)

MODEL_NAME = (
    "sentence-transformers/"
    "all-MiniLM-L6-v2"
)


def search(
    query: str,
    embeddings: np.ndarray,
    metadata: pd.DataFrame,
    model: SentenceTransformer,
    top_k: int = 10,
) -> pd.DataFrame:
    """
    Search movies using semantic similarity.
    """

    query_embedding = model.encode(
        query,
        normalize_embeddings=True,
    )

    scores = embeddings @ query_embedding

    top_indices = np.argsort(
        scores
    )[::-1][:top_k]

    results = metadata.iloc[
        top_indices
    ].copy()

    results["similarity"] = (
        scores[top_indices]
    )

    return results


def main() -> None:
    print("Loading model...")

    model = SentenceTransformer(
        MODEL_NAME
    )

    print("Loading metadata...")

    metadata = pd.read_parquet(
        METADATA_PATH
    )

    print("Loading embeddings...")

    overview_embeddings = np.load(
        OVERVIEW_EMBEDDINGS_PATH
    )

    enriched_embeddings = np.load(
        ENRICHED_EMBEDDINGS_PATH
    )

    query = (
        "A dark science fiction movie "
        "about dreams, memory and "
        "manipulation of reality"
    )

    print()
    print(
        f"Query: {query}"
    )

    print()
    print(
        "=== OVERVIEW RESULTS ==="
    )

    overview_results = search(
        query=query,
        embeddings=overview_embeddings,
        metadata=metadata,
        model=model,
        top_k=10,
    )

    print(
        overview_results[
            [
                "primaryTitle",
                "startYear",
                "genres",
                "similarity",
            ]
        ]
        .to_string(index=False)
    )

    print()
    print(
        "=== ENRICHED RESULTS ==="
    )

    enriched_results = search(
        query=query,
        embeddings=enriched_embeddings,
        metadata=metadata,
        model=model,
        top_k=10,
    )

    print(
        enriched_results[
            [
                "primaryTitle",
                "startYear",
                "genres",
                "similarity",
            ]
        ]
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()