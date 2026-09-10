import os
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg

from pgvector.psycopg import register_vector


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MOVIES_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "movies_semantic.parquet"
)

EMBEDDINGS_PATH = (
    PROJECT_ROOT
    / "data"
    / "embeddings"
    / "content_embeddings.npy"
)

DATABASE_URL = os.getenv(
    "DATABASE_URL"
)


def clean_optional_text(
    value,
) -> str | None:
    """
    Convert missing text values to None.
    """

    if pd.isna(value):
        return None

    value = str(value).strip()

    if not value:
        return None

    return value


def clean_optional_int(
    value,
) -> int | None:
    """
    Convert missing numeric values to None.
    """

    if pd.isna(value):
        return None

    return int(value)


def clean_optional_float(
    value,
) -> float | None:
    """
    Convert missing numeric values to None.
    """

    if pd.isna(value):
        return None

    return float(value)


def main() -> None:
    if not DATABASE_URL:
        raise ValueError(
            "DATABASE_URL environment variable "
            "is not configured."
        )

    print(
        "Loading semantic movie catalog..."
    )

    movies = pd.read_parquet(
        MOVIES_PATH
    )

    print(
        f"Movies loaded: "
        f"{len(movies):,}"
    )

    print(
        "Loading content embeddings..."
    )

    embeddings = np.load(
        EMBEDDINGS_PATH
    )

    print(
        f"Embeddings shape: "
        f"{embeddings.shape}"
    )

    if len(movies) != len(embeddings):
        raise ValueError(
            "Number of movies and embeddings "
            "does not match."
        )

    if embeddings.ndim != 2:
        raise ValueError(
            "Embeddings must be a 2D matrix."
        )

    if embeddings.shape[1] != 384:
        raise ValueError(
            "Expected 384-dimensional embeddings, "
            f"found {embeddings.shape[1]}."
        )

    print(
        "Connecting to Supabase PostgreSQL..."
    )

    with psycopg.connect(
        DATABASE_URL,
        sslmode="require",
    ) as connection:

        register_vector(
            connection
        )

        with connection.cursor() as cursor:

            print(
                "Checking movies table..."
            )

            cursor.execute(
                """
                SELECT COUNT(*)
                FROM public.movies;
                """
            )

            existing_rows = (
                cursor.fetchone()[0]
            )

            print(
                f"Existing rows: "
                f"{existing_rows:,}"
            )

            if existing_rows > 0:
                print(
                    "Clearing existing rows..."
                )

                cursor.execute(
                    """
                    TRUNCATE TABLE public.movies
                    RESTART IDENTITY;
                    """
                )

            print(
                "Loading movies into Supabase..."
            )

            for index, row in enumerate(
                movies.itertuples(
                    index=False
                ),
                start=0,
            ):

                embedding = (
                    embeddings[index]
                    .astype(
                        np.float32
                    )
                )

                cursor.execute(
                    """
                    INSERT INTO public.movies (
                        tconst,
                        tmdb_id,
                        title,
                        original_title,
                        release_year,
                        runtime_minutes,
                        average_rating,
                        num_votes,
                        overview,
                        genres,
                        keywords,
                        director,
                        cast_names,
                        semantic_text,
                        embedding
                    )
                    VALUES (
                        %s,
                        %s,
                        %s,
                        %s,
                        %s,
                        %s,
                        %s,
                        %s,
                        %s,
                        %s,
                        %s,
                        %s,
                        %s,
                        %s,
                        %s
                    );
                    """,
                    (
                        row.tconst,
                        int(row.tmdb_id),
                        row.primaryTitle,
                        clean_optional_text(
                            row.originalTitle
                        ),
                        clean_optional_int(
                            row.startYear
                        ),
                        clean_optional_int(
                            row.runtimeMinutes
                        ),
                        clean_optional_float(
                            row.averageRating
                        ),
                        clean_optional_int(
                            row.numVotes
                        ),
                        row.overview,
                        clean_optional_text(
                            row.genres
                        ),
                        clean_optional_text(
                            row.keywords
                        ),
                        clean_optional_text(
                            row.director
                        ),
                        clean_optional_text(
                            row.cast
                        ),
                        row.content_text,
                        embedding,
                    ),
                )

                processed = (
                    index + 1
                )

                if processed % 500 == 0:
                    print(
                        f"Inserted: "
                        f"{processed:,}/"
                        f"{len(movies):,}"
                    )

        connection.commit()

    print()
    print(
        "=== SUPABASE DATABASE LOADED ==="
    )

    print(
        f"Movies inserted: "
        f"{len(movies):,}"
    )


if __name__ == "__main__":
    main()