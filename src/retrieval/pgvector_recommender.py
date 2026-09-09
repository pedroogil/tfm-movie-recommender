import psycopg
from pgvector.psycopg import register_vector


DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "dbname": "tfm_movies",
    "user": "tfm_user",
    "password": "tfm_password",
}


def recommend_by_title(
    movie_title: str,
    top_k: int = 10,
    exact_search: bool = False,
) -> None:
    """
    Retrieve movies similar to a selected movie
    using its embedding stored in PostgreSQL.

    If exact_search is True, PostgreSQL index scans
    are disabled so that the complete dataset is
    evaluated. This is useful for validating the
    approximate vector index.
    """

    with psycopg.connect(
        **DB_CONFIG
    ) as connection:

        register_vector(connection)

        with connection.cursor() as cursor:

            if exact_search:
                cursor.execute(
                    "SET LOCAL enable_indexscan = off;"
                )

            cursor.execute(
                """
                SELECT
                    id,
                    title,
                    release_year,
                    embedding
                FROM movies
                WHERE LOWER(title) = LOWER(%s)
                LIMIT 1;
                """,
                (movie_title,),
            )

            selected = cursor.fetchone()

            if selected is None:
                raise ValueError(
                    f"Movie not found: {movie_title}"
                )

            (
                movie_id,
                title,
                year,
                embedding,
            ) = selected

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

    mode = (
        "EXACT"
        if exact_search
        else "HNSW"
    )

    print()
    print(
        f"Recommendations for: "
        f"{title} ({year})"
    )

    print(
        f"Search mode: {mode}"
    )

    print()

    for position, row in enumerate(
        rows,
        start=1,
    ):

        (
            rec_title,
            rec_year,
            rating,
            genres,
            director,
            similarity,
        ) = row

        print(
            f"{position:>2}. "
            f"{rec_title} "
            f"({rec_year})"
        )

        print(
            f"    Rating: {rating}"
        )

        print(
            f"    Genres: {genres}"
        )

        print(
            f"    Director: {director}"
        )

        print(
            f"    Similarity: "
            f"{similarity:.4f}"
        )

        print()


def main() -> None:
    movie_title = "Inception"

    print(
        "=" * 70
    )

    print(
        "EXACT VECTOR SEARCH"
    )

    print(
        "=" * 70
    )

    recommend_by_title(
        movie_title=movie_title,
        top_k=10,
        exact_search=True,
    )

    print()
    print(
        "=" * 70
    )

    print(
        "HNSW VECTOR SEARCH"
    )

    print(
        "=" * 70
    )

    recommend_by_title(
        movie_title=movie_title,
        top_k=10,
        exact_search=False,
    )


if __name__ == "__main__":
    main()