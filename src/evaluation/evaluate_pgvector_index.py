import random
import statistics
import time

import psycopg
from pgvector.psycopg import register_vector


DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "dbname": "tfm_movies",
    "user": "tfm_user",
    "password": "tfm_password",
}

SAMPLE_SIZE = 100
TOP_K = 10
RANDOM_SEED = 42


def get_sample_movies() -> list[tuple[int, str]]:
    """
    Select a reproducible random sample of movies
    from PostgreSQL.
    """

    with psycopg.connect(
        **DB_CONFIG
    ) as connection:

        with connection.cursor() as cursor:

            cursor.execute(
                """
                SELECT id, title
                FROM movies
                ORDER BY id;
                """
            )

            movies = cursor.fetchall()

    random.seed(RANDOM_SEED)

    sample_size = min(
        SAMPLE_SIZE,
        len(movies),
    )

    return random.sample(
        movies,
        sample_size,
    )


def get_movie_embedding(
    movie_id: int,
):
    """
    Retrieve one stored movie embedding.
    """

    with psycopg.connect(
        **DB_CONFIG
    ) as connection:

        register_vector(connection)

        with connection.cursor() as cursor:

            cursor.execute(
                """
                SELECT embedding
                FROM movies
                WHERE id = %s;
                """,
                (movie_id,),
            )

            result = cursor.fetchone()

    if result is None:
        raise ValueError(
            f"Movie ID not found: {movie_id}"
        )

    return result[0]


def exact_search(
    movie_id: int,
    embedding,
    top_k: int,
) -> tuple[list[int], float]:
    """
    Perform exhaustive vector search.

    Index scans are disabled so PostgreSQL evaluates
    the complete collection.
    """

    with psycopg.connect(
        **DB_CONFIG
    ) as connection:

        register_vector(connection)

        with connection.cursor() as cursor:

            cursor.execute(
                "SET enable_indexscan = off;"
            )

            cursor.execute(
                "SET enable_bitmapscan = off;"
            )

            start = time.perf_counter()

            cursor.execute(
                """
                SELECT id
                FROM movies
                WHERE id <> %s
                ORDER BY embedding <=> %s
                LIMIT %s;
                """,
                (
                    movie_id,
                    embedding,
                    top_k,
                ),
            )

            results = [
                row[0]
                for row in cursor.fetchall()
            ]

            elapsed = (
                time.perf_counter()
                - start
            )

    return results, elapsed


def hnsw_search(
    movie_id: int,
    embedding,
    top_k: int,
) -> tuple[list[int], float]:
    """
    Perform approximate nearest-neighbour search
    using the HNSW index.
    """

    with psycopg.connect(
        **DB_CONFIG
    ) as connection:

        register_vector(connection)

        with connection.cursor() as cursor:

            # Increase search quality while keeping
            # approximate ANN behaviour.
            cursor.execute(
                "SET hnsw.ef_search = 100;"
            )

            start = time.perf_counter()

            cursor.execute(
                """
                SELECT id
                FROM movies
                WHERE id <> %s
                ORDER BY embedding <=> %s
                LIMIT %s;
                """,
                (
                    movie_id,
                    embedding,
                    top_k,
                ),
            )

            results = [
                row[0]
                for row in cursor.fetchall()
            ]

            elapsed = (
                time.perf_counter()
                - start
            )

    return results, elapsed


def calculate_recall(
    exact_results: list[int],
    approximate_results: list[int],
) -> float:
    """
    Calculate Recall@K using exhaustive search
    as ground truth.
    """

    exact_set = set(
        exact_results
    )

    approximate_set = set(
        approximate_results
    )

    relevant_found = len(
        exact_set.intersection(
            approximate_set
        )
    )

    return (
        relevant_found
        / len(exact_results)
    )


def main() -> None:
    print(
        "=== PGVECTOR INDEX EVALUATION ==="
    )

    print(
        f"Sample size: {SAMPLE_SIZE}"
    )

    print(
        f"Top-K: {TOP_K}"
    )

    print(
        f"Random seed: {RANDOM_SEED}"
    )

    print()

    sample_movies = (
        get_sample_movies()
    )

    recalls = []

    exact_times = []
    hnsw_times = []

    for position, (
        movie_id,
        title,
    ) in enumerate(
        sample_movies,
        start=1,
    ):

        embedding = (
            get_movie_embedding(
                movie_id
            )
        )

        exact_results, exact_time = (
            exact_search(
                movie_id=movie_id,
                embedding=embedding,
                top_k=TOP_K,
            )
        )

        hnsw_results, hnsw_time = (
            hnsw_search(
                movie_id=movie_id,
                embedding=embedding,
                top_k=TOP_K,
            )
        )

        recall = calculate_recall(
            exact_results,
            hnsw_results,
        )

        recalls.append(
            recall
        )

        exact_times.append(
            exact_time
        )

        hnsw_times.append(
            hnsw_time
        )

        if position % 10 == 0:

            print(
                f"Evaluated "
                f"{position:>3}/"
                f"{len(sample_movies)}"
            )

    mean_recall = (
        statistics.mean(
            recalls
        )
    )

    min_recall = min(
        recalls
    )

    perfect_queries = sum(
        recall == 1.0
        for recall in recalls
    )

    mean_exact_ms = (
        statistics.mean(
            exact_times
        )
        * 1000
    )

    mean_hnsw_ms = (
        statistics.mean(
            hnsw_times
        )
        * 1000
    )

    median_exact_ms = (
        statistics.median(
            exact_times
        )
        * 1000
    )

    median_hnsw_ms = (
        statistics.median(
            hnsw_times
        )
        * 1000
    )

    if mean_hnsw_ms > 0:

        speedup = (
            mean_exact_ms
            / mean_hnsw_ms
        )

    else:

        speedup = float("inf")

    print()
    print(
        "=== RESULTS ==="
    )

    print(
        f"Mean Recall@{TOP_K}: "
        f"{mean_recall:.4f}"
    )

    print(
        f"Minimum Recall@{TOP_K}: "
        f"{min_recall:.4f}"
    )

    print(
        "Queries with perfect "
        f"Recall@{TOP_K}: "
        f"{perfect_queries}/"
        f"{len(recalls)}"
    )

    print()

    print(
        "Mean exact search time: "
        f"{mean_exact_ms:.3f} ms"
    )

    print(
        "Mean HNSW search time: "
        f"{mean_hnsw_ms:.3f} ms"
    )

    print()

    print(
        "Median exact search time: "
        f"{median_exact_ms:.3f} ms"
    )

    print(
        "Median HNSW search time: "
        f"{median_hnsw_ms:.3f} ms"
    )

    print()

    print(
        f"Mean speedup: "
        f"{speedup:.2f}x"
    )


if __name__ == "__main__":
    main()