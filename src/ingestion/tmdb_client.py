import os
import time

import requests
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


load_dotenv()

TMDB_BASE_URL = "https://api.themoviedb.org/3"
TMDB_ACCESS_TOKEN = os.getenv("TMDB_ACCESS_TOKEN")


if not TMDB_ACCESS_TOKEN:
    raise ValueError(
        "TMDB_ACCESS_TOKEN no está configurado."
    )


def create_session() -> requests.Session:
    """Create a requests session with automatic retries."""

    retry_strategy = Retry(
        total=5,
        connect=5,
        read=5,
        status=5,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        respect_retry_after_header=True,
    )

    adapter = HTTPAdapter(
        max_retries=retry_strategy
    )

    session = requests.Session()

    session.mount(
        "https://",
        adapter,
    )

    session.headers.update(
        {
            "Authorization": (
                f"Bearer {TMDB_ACCESS_TOKEN}"
            ),
            "accept": "application/json",
        }
    )

    return session


SESSION = create_session()


def get_movie(
    movie_id: int,
    append_to_response: str | None = None,
) -> dict:
    """Get movie details from TMDB."""

    url = f"{TMDB_BASE_URL}/movie/{movie_id}"

    params = {}

    if append_to_response:
        params["append_to_response"] = (
            append_to_response
        )

    response = SESSION.get(
        url,
        params=params,
        timeout=30,
    )

    response.raise_for_status()

    return response.json()


def find_movie_by_imdb_id(
    imdb_id: str,
) -> int | None:
    """Find the TMDB movie ID associated with an IMDb ID."""

    url = f"{TMDB_BASE_URL}/find/{imdb_id}"

    params = {
        "external_source": "imdb_id",
    }

    response = SESSION.get(
        url,
        params=params,
        timeout=30,
    )

    response.raise_for_status()

    data = response.json()

    movie_results = data.get(
        "movie_results",
        [],
    )

    if not movie_results:
        return None

    return movie_results[0]["id"]