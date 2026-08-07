import json
from dataclasses import asdict
from pathlib import Path

from src.models.movie import Movie


def save_movie(movie: Movie, output_path: Path) -> None:
    """Save a movie as a JSON object."""

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("a", encoding="utf-8") as file:
        json.dump(
            asdict(movie),
            file,
            ensure_ascii=False,
        )
        file.write("\n")