from dataclasses import dataclass
from typing import Optional


@dataclass
class Movie:
    tmdb_id: int
    title: str
    original_title: str
    overview: str
    release_date: Optional[str]
    runtime: Optional[int]
    vote_average: float
    vote_count: int
    popularity: float
    original_language: str
    genres: list[str]
    keywords: list[str]
    director: Optional[str]
    cast: list[str]