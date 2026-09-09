from src.services.recommendation_service import (
    RecommendationService,
)


def main() -> None:
    service = RecommendationService()

    print()
    print(
        "=== TITLE SEARCH ==="
    )

    movies = (
        service.search_movies_by_title(
            "Inception"
        )
    )

    for movie in movies:
        print(
            movie
        )

    if not movies:
        raise ValueError(
            "No movies found."
        )

    selected_id = (
        movies[0]["id"]
    )

    print()
    print(
        "=== MOVIE RECOMMENDATIONS ==="
    )

    recommendations = (
        service.recommend_by_movie(
            selected_id,
            top_k=5,
        )
    )

    for movie in recommendations:
        print(
            movie["title"],
            movie["release_year"],
            round(
                movie[
                    "similarity"
                ],
                4,
            ),
        )

    print()
    print(
        "=== TEXT SEARCH ==="
    )

    results = (
        service.search_by_text(
            (
                "A dark science fiction "
                "movie about dreams, "
                "memory and reality"
            ),
            top_k=5,
        )
    )

    for movie in results:
        print(
            movie["title"],
            movie["release_year"],
            round(
                movie[
                    "similarity"
                ],
                4,
            ),
        )


if __name__ == "__main__":
    main()