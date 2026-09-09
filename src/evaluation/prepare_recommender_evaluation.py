from pathlib import Path
import random

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

METADATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "embeddings"
    / "embedding_metadata.parquet"
)

OVERVIEW_PATH = (
    PROJECT_ROOT
    / "data"
    / "embeddings"
    / "overview_embeddings.npy"
)

CONTENT_PATH = (
    PROJECT_ROOT
    / "data"
    / "embeddings"
    / "content_embeddings.npy"
)

ENRICHED_PATH = (
    PROJECT_ROOT
    / "data"
    / "embeddings"
    / "enriched_embeddings.npy"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "recommender_evaluation_v2.csv"
)

MAPPING_PATH = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "system_mapping_v2.txt"
)

TOP_K = 5
RANDOM_SEED = 42


TEST_MOVIES = [
    ("Inception", 2010),
    ("The Godfather", 1972),
    ("Toy Story", 1995),
    ("The Shining", 1980),
    ("La La Land", 2016),
    ("The Matrix", 1999),
    ("Pulp Fiction", 1994),
    ("Finding Nemo", 2003),
    ("Alien", 1979),
    ("Titanic", 1997),
    ("The Social Network", 2010),
    ("Gladiator", 2000),
    ("The Silence of the Lambs", 1991),
    ("Groundhog Day", 1993),
    ("Blade Runner", 1982),
]


def find_movie_index(
    title: str,
    year: int,
    metadata: pd.DataFrame,
) -> int:
    """
    Find a unique movie using title and release year.
    """

    matches = metadata[
        metadata["primaryTitle"]
        .str.lower()
        .eq(title.lower())
        &
        metadata["startYear"]
        .eq(year)
    ]

    if matches.empty:
        raise ValueError(
            f"Movie not found: {title} ({year})"
        )

    if len(matches) > 1:
        raise ValueError(
            f"Multiple matches found for "
            f"{title} ({year})"
        )

    return matches.index[0]


def get_recommendations(
    movie_index: int,
    embeddings: np.ndarray,
    metadata: pd.DataFrame,
    top_k: int,
) -> pd.DataFrame:
    """
    Retrieve the most similar movies.
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


def main() -> None:
    print(
        "Loading metadata and embeddings..."
    )

    metadata = pd.read_parquet(
        METADATA_PATH
    )

    embeddings = {
        "A": np.load(
            OVERVIEW_PATH
        ),
        "B": np.load(
            CONTENT_PATH
        ),
        "C": np.load(
            ENRICHED_PATH
        ),
    }

    system_names = [
        "System X",
        "System Y",
        "System Z",
    ]

    random.seed(
        RANDOM_SEED
    )

    shuffled_names = (
        system_names.copy()
    )

    random.shuffle(
        shuffled_names
    )

    system_mapping = dict(
        zip(
            embeddings.keys(),
            shuffled_names,
        )
    )

    rows = []

    for movie_title, movie_year in TEST_MOVIES:

        print(
            f"Preparing: "
            f"{movie_title} "
            f"({movie_year})"
        )

        movie_index = find_movie_index(
            title=movie_title,
            year=movie_year,
            metadata=metadata,
        )

        anchor = metadata.iloc[
            movie_index
        ]

        for system_code, matrix in embeddings.items():

            recommendations = (
                get_recommendations(
                    movie_index=movie_index,
                    embeddings=matrix,
                    metadata=metadata,
                    top_k=TOP_K,
                )
            )

            for rank, recommendation in enumerate(
                recommendations.itertuples(
                    index=False
                ),
                start=1,
            ):

                rows.append(
                    {
                        "anchor_movie":
                            anchor[
                                "primaryTitle"
                            ],

                        "anchor_year":
                            int(
                                anchor[
                                    "startYear"
                                ]
                            ),

                        "system":
                            system_mapping[
                                system_code
                            ],

                        "rank":
                            rank,

                        "recommended_movie":
                            recommendation.primaryTitle,

                        "recommended_year":
                            int(
                                recommendation.startYear
                            ),

                        "genres":
                            recommendation.genres,

                        "similarity":
                            recommendation.similarity,

                        "relevance_score":
                            "",

                        "comments":
                            "",
                    }
                )

    evaluation = pd.DataFrame(
        rows
    )

    evaluation = (
        evaluation
        .sort_values(
            [
                "anchor_movie",
                "system",
                "rank",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    evaluation.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    with MAPPING_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:

        for code, name in (
            system_mapping.items()
        ):

            file.write(
                f"{name} = {code}\n"
            )

    print()
    print(
        "=== EVALUATION DATASET CREATED ==="
    )

    print(
        f"Rows: {len(evaluation):,}"
    )

    print(
        f"Evaluation file: "
        f"{OUTPUT_PATH}"
    )

    print(
        f"Mapping file: "
        f"{MAPPING_PATH}"
    )


if __name__ == "__main__":
    main()