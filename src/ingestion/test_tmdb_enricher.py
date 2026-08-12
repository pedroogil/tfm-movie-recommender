import json

from src.ingestion.tmdb_client import get_movie
from src.ingestion.tmdb_enricher import extract_movie_data


data = get_movie(
    27205,
    "credits,keywords",
)

movie = extract_movie_data(data)

print(
    json.dumps(
        movie,
        indent=2,
        ensure_ascii=False,
    )
)