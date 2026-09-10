import os

import numpy as np
import psycopg

from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer


DATABASE_URL = os.getenv(
    "DATABASE_URL",
    (
        "postgresql://"
        "tfm_user:tfm_password"
        "@localhost:5432/tfm_movies"
    ),
)

MODEL_NAME = (
    "sentence-transformers/"
    "all-MiniLM-L6-v2"
)


class RecommendationService:
    """
    Service layer for semantic movie recommendation.

    The service can connect either to:

    - Supabase / cloud PostgreSQL through DATABASE_URL
    - Local Docker PostgreSQL as fallback
    """

    def __init__(self) -> None:
        print("Loading embedding model...")

        self.model = SentenceTransformer(
            MODEL_NAME
        )

    def _connect(self):
        """
        Create a PostgreSQL connection and register
        pgvector support.
        """

        connection = psycopg.connect(
            DATABASE_URL,
            sslmode=(
                "require"
                if os.getenv("DATABASE_URL")
                else "prefer"
            ),
            connect_timeout=10,
        )

        register_vector(
            connection
        )

        return connection

    def search_by_text(
        self,
        query: str,
        top_k: int = 10,
        min_rating: float = 0.0,
        min_year: int | None = None,
        max_year: int | None = None,
    ) -> list[dict]:
        """
        Search movies from a natural-language query.
        """

        query_embedding = (
            self.model.encode(
                query,
                normalize_embeddings=True,
            )
            .astype(np.float32)
        )

        filters = [
            "average_rating >= %s"
        ]

        params = [
            min_rating
        ]

        if min_year is not None:
            filters.append(
                "release_year >= %s"
            )

            params.append(
                min_year
            )

        if max_year is not None:
            filters.append(
                "release_year <= %s"
            )

            params.append(
                max_year
            )

        where_clause = (
            " AND ".join(filters)
        )

        with self._connect() as connection:

            with connection.cursor() as cursor:

                cursor.execute(
                    "SET statement_timeout = '15s';"
                )

                query_sql = f"""
                    SELECT
                        id,
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
                        1 - (
                            embedding <=> %s
                        ) AS similarity
                    FROM public.movies
                    WHERE {where_clause}
                    ORDER BY
                        embedding <=> %s
                    LIMIT %s;
                """

                query_params = (
                    [query_embedding]
                    + params
                    + [
                        query_embedding,
                        top_k,
                    ]
                )

                cursor.execute(
                    query_sql,
                    query_params,
                )

                rows = cursor.fetchall()

        return [
            self._row_to_dict(
                row
            )
            for row in rows
        ]

    def recommend_by_movie(
        self,
        movie_id: int,
        top_k: int = 10,
        min_rating: float = 0.0,
        min_year: int | None = None,
        max_year: int | None = None,
    ) -> list[dict]:
        """
        Recommend movies similar to a selected movie.
        """

        filters = [
            "id <> %s",
            "average_rating >= %s",
        ]

        with self._connect() as connection:

            with connection.cursor() as cursor:

                cursor.execute(
                    "SET statement_timeout = '15s';"
                )

                cursor.execute(
                    """
                    SELECT embedding
                    FROM public.movies
                    WHERE id = %s;
                    """,
                    (
                        movie_id,
                    ),
                )

                result = cursor.fetchone()

                if result is None:
                    raise ValueError(
                        "Movie not found."
                    )

                embedding = result[0]

                params = [
                    movie_id,
                    min_rating,
                ]

                if min_year is not None:
                    filters.append(
                        "release_year >= %s"
                    )

                    params.append(
                        min_year
                    )

                if max_year is not None:
                    filters.append(
                        "release_year <= %s"
                    )

                    params.append(
                        max_year
                    )

                where_clause = (
                    " AND ".join(filters)
                )

                query_sql = f"""
                    SELECT
                        id,
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
                        1 - (
                            embedding <=> %s
                        ) AS similarity
                    FROM public.movies
                    WHERE {where_clause}
                    ORDER BY
                        embedding <=> %s
                    LIMIT %s;
                """

                query_params = (
                    [embedding]
                    + params
                    + [
                        embedding,
                        top_k,
                    ]
                )

                cursor.execute(
                    query_sql,
                    query_params,
                )

                rows = cursor.fetchall()

        return [
            self._row_to_dict(
                row
            )
            for row in rows
        ]

    def search_movies_by_title(
        self,
        query: str,
        limit: int = 20,
    ) -> list[dict]:
        """
        Search movies by title for UI selection.
        """

        query = query.strip()

        if not query:
            return []

        with self._connect() as connection:

            with connection.cursor() as cursor:

                cursor.execute(
                    "SET statement_timeout = '15s';"
                )

                cursor.execute(
                    """
                    SELECT
                        id,
                        tconst,
                        tmdb_id,
                        title,
                        release_year,
                        average_rating,
                        genres
                    FROM public.movies
                    WHERE title ILIKE %s
                    ORDER BY
                        num_votes DESC
                    LIMIT %s;
                    """,
                    (
                        f"%{query}%",
                        limit,
                    ),
                )

                rows = cursor.fetchall()

        return [
            {
                "id": row[0],
                "tconst": row[1],
                "tmdb_id": row[2],
                "title": row[3],
                "release_year": row[4],
                "average_rating": row[5],
                "genres": row[6],
            }
            for row in rows
        ]

    def get_movie(
        self,
        movie_id: int,
    ) -> dict | None:
        """
        Retrieve full movie information.
        """

        with self._connect() as connection:

            with connection.cursor() as cursor:

                cursor.execute(
                    "SET statement_timeout = '15s';"
                )

                cursor.execute(
                    """
                    SELECT
                        id,
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
                        cast_names
                    FROM public.movies
                    WHERE id = %s;
                    """,
                    (
                        movie_id,
                    ),
                )

                row = cursor.fetchone()

        if row is None:
            return None

        return {
            "id": row[0],
            "tconst": row[1],
            "tmdb_id": row[2],
            "title": row[3],
            "original_title": row[4],
            "release_year": row[5],
            "runtime_minutes": row[6],
            "average_rating": row[7],
            "num_votes": row[8],
            "overview": row[9],
            "genres": row[10],
            "keywords": row[11],
            "director": row[12],
            "cast_names": row[13],
        }

    @staticmethod
    def _row_to_dict(
        row,
    ) -> dict:
        """
        Convert a recommendation SQL row into a dictionary.
        """

        return {
            "id": row[0],
            "tconst": row[1],
            "tmdb_id": row[2],
            "title": row[3],
            "original_title": row[4],
            "release_year": row[5],
            "runtime_minutes": row[6],
            "average_rating": row[7],
            "num_votes": row[8],
            "overview": row[9],
            "genres": row[10],
            "keywords": row[11],
            "director": row[12],
            "cast_names": row[13],
            "similarity": (
                float(row[14])
                if row[14] is not None
                else None
            ),
        }