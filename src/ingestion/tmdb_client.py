import os

import requests
from dotenv import load_dotenv


load_dotenv()

TMDB_BASE_URL = "https://api.themoviedb.org/3"
TMDB_ACCESS_TOKEN = os.getenv("TMDB_ACCESS_TOKEN")


def get_movie(movie_id: int, append_to_response: str | None = None) -> dict:
    """Get movie details from TMDB."""

    url = f"{TMDB_BASE_URL}/movie/{movie_id}"

    headers = {
        "Authorization": f"Bearer {TMDB_ACCESS_TOKEN}",
        "accept": "application/json",
    }

    params = {}

    if append_to_response:
        params["append_to_response"] = append_to_response

    response = requests.get(
        url,
        headers=headers,
        params=params,
        timeout=30,
    )

    response.raise_for_status()

    return response.json()

def find_movie_by_imdb_id(imdb_id: str) -> int | None:
    """Find the TMDB movie ID associated with an IMDb ID."""

    url = f"{TMDB_BASE_URL}/find/{imdb_id}"

    headers = {
        "Authorization": f"Bearer {TMDB_ACCESS_TOKEN}",
        "accept": "application/json",
    }

    params = {
        "external_source": "imdb_id",
    }

    response = requests.get(
        url,
        headers=headers,
        params=params,
        timeout=30,
    )

    response.raise_for_status()

    data = response.json()

    movie_results = data.get("movie_results", [])

    if not movie_results:
        return None

    return movie_results[0]["id"]