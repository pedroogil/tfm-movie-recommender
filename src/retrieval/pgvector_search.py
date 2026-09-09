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


def search_movies(
    query: str,
    top_k: int = 10,
) -> None:
    """
    Convert a natural-language query into an embedding
    and retrieve the most similar movies from PostgreSQL.
    """

    print("Loading embedding model...")

    model = SentenceTransformer(
        MODEL_NAME
    )

    print("Generating query embedding...")

    query_embedding = model.encode(
        query,
        normalize_embeddings=True,
    ).astype(np.float32)

    print("Connecting to PostgreSQL...")

    with psycopg.connect(
        **DB_CONFIG
    ) as connection:

        register_vector(connection)

        with connection.cursor() as cursor:

            cursor.execute(
                """
                SELECT
                    title,
                    release_year,
                    average_rating,
                    genres,
                    director,
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

    print()
    print(
        f"Query: {query}"
    )

    print()
    print(
        "=== VECTOR SEARCH RESULTS ==="
    )

    for position, row in enumerate(
        rows,
        start=1,
    ):

        (
            title,
            year,
            rating,
            genres,
            director,
            similarity,
        ) = row

        print(
            f"{position:>2}. "
            f"{title} "
            f"({year})"
        )

        print(
            f"    Rating: "
            f"{rating}"
        )

        print(
            f"    Genres: "
            f"{genres}"
        )

        print(
            f"    Director: "
            f"{director}"
        )

        print(
            f"    Similarity: "
            f"{similarity:.4f}"
        )

        print()


def main() -> None:
    query = (
        "A dark science fiction movie "
        "about dreams, memory and "
        "manipulation of reality"
    )

    search_movies(
        query=query,
        top_k=10,
    )


if __name__ == "__main__":
    main()