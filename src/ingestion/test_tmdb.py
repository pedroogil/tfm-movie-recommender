from src.ingestion.tmdb_client import find_movie_by_imdb_id


imdb_ids = [
    "tt1375666",  # Inception
    "tt0468569",  # The Dark Knight
    "tt0133093",  # The Matrix
]


for imdb_id in imdb_ids:
    tmdb_id = find_movie_by_imdb_id(imdb_id)

    print(
        f"IMDb: {imdb_id} -> TMDB: {tmdb_id}"
    )
    