import numpy as np
import psycopg

from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer


DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "dbname": "tfm_movies",
    "user": "tfm_user",
    "password": "tfm_password",
}

MODEL_NAME = (
    "sentence-transformers/"
    "all-MiniLM-L6-v2"
)


class RecommendationService:
    """
    Service layer for semantic movie recommendation.
    """

    def __init__(self) -> None:
        print("Loading embedding model...")

        self.model = SentenceTransformer(
            MODEL_NAME
        )

    def _connect(self):
        """
        Create a PostgreSQL connection
        and register pgvector support.
        """

        connection = psycopg.connect(
            **DB_CONFIG
        )

        register_vector(
            connection
        )

        return connection

    def search_by_text(
        self,
        query: str,
        top_k: int = 10,
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

        with self._connect() as connection:

            with connection.cursor() as cursor:

                cursor.execute(
                    """
                    SELECT
                        id,
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
                    FROM movies
                    ORDER BY
                        embedding <=> %s
                    LIMIT %s;
                    """,
                    (
                        query_embedding,
                        query_embedding,
                        top_k,
                    ),
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
    ) -> list[dict]:
        """
        Recommend movies similar to a selected movie.
        """

        with self._connect() as connection:

            with connection.cursor() as cursor:

                cursor.execute(
                    """
                    SELECT
                        embedding
                    FROM movies
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

                cursor.execute(
                    """
                    SELECT
                        id,
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
                    FROM movies
                    WHERE id <> %s
                    ORDER BY
                        embedding <=> %s
                    LIMIT %s;
                    """,
                    (
                        embedding,
                        movie_id,
                        embedding,
                        top_k,
                    ),
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
                    """
                    SELECT
                        id,
                        title,
                        release_year,
                        average_rating,
                        genres
                    FROM movies
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
                "title": row[1],
                "release_year": row[2],
                "average_rating": row[3],
                "genres": row[4],
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
                    """
                    SELECT
                        id,
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
                    FROM movies
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
            "title": row[1],
            "original_title": row[2],
            "release_year": row[3],
            "runtime_minutes": row[4],
            "average_rating": row[5],
            "num_votes": row[6],
            "overview": row[7],
            "genres": row[8],
            "keywords": row[9],
            "director": row[10],
            "cast_names": row[11],
        }

    @staticmethod
    def _row_to_dict(
        row,
    ) -> dict:
        """
        Convert a recommendation SQL row
        into a dictionary.
        """

        return {
            "id": row[0],
            "title": row[1],
            "original_title": row[2],
            "release_year": row[3],
            "runtime_minutes": row[4],
            "average_rating": row[5],
            "num_votes": row[6],
            "overview": row[7],
            "genres": row[8],
            "keywords": row[9],
            "director": row[10],
            "cast_names": row[11],
            "similarity": (
                float(row[12])
                if row[12] is not None
                else None
            ),
        }