from src.models.movie import Movie


def parse_movie(data: dict) -> Movie:
    """Transform a TMDB API response into our Movie model."""

    genres = [
        genre["name"]
        for genre in data.get("genres", [])
    ]

    keywords = [
        keyword["name"]
        for keyword in data.get("keywords", {}).get("keywords", [])
    ]

    cast = [
        actor["name"]
        for actor in data.get("credits", {}).get("cast", [])[:10]
    ]

    director = next(
        (
            person["name"]
            for person in data.get("credits", {}).get("crew", [])
            if person.get("job") == "Director"
        ),
        None,
    )

    return Movie(
        tmdb_id=data["id"],
        title=data.get("title", ""),
        original_title=data.get("original_title", ""),
        overview=data.get("overview", ""),
        release_date=data.get("release_date") or None,
        runtime=data.get("runtime"),
        vote_average=data.get("vote_average", 0.0),
        vote_count=data.get("vote_count", 0),
        popularity=data.get("popularity", 0.0),
        original_language=data.get("original_language", ""),
        genres=genres,
        keywords=keywords,
        director=director,
        cast=cast,
    )