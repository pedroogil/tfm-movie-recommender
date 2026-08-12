from typing import Any


def extract_movie_data(data: dict[str, Any]) -> dict[str, Any]:
    """
    Extract and normalize the relevant information
    from a TMDB movie response.
    """

    # Basic movie information
    tmdb_id = data.get("id")

    title = data.get("title")
    original_title = data.get("original_title")

    overview = data.get("overview")

    release_date = data.get("release_date")

    runtime = data.get("runtime")

    # Genres
    genres = data.get("genres", [])

    genre_names = [
        genre.get("name")
        for genre in genres
        if genre.get("name")
    ]

    # Keywords
    keywords_data = data.get("keywords", {})

    keywords = keywords_data.get("keywords", [])

    keyword_names = [
        keyword.get("name")
        for keyword in keywords
        if keyword.get("name")
    ]

    # Credits
    credits = data.get("credits", {})

    cast = credits.get("cast", [])
    crew = credits.get("crew", [])

    # Main cast: first 10 actors
    cast_names = [
        person.get("name")
        for person in cast[:10]
        if person.get("name")
    ]

    # Director
    directors = [
        person.get("name")
        for person in crew
        if person.get("job") == "Director"
        and person.get("name")
    ]

    director = directors[0] if directors else None

    return {
        "tmdb_id": tmdb_id,
        "title": title,
        "original_title": original_title,
        "overview": overview,
        "release_date": release_date,
        "runtime": runtime,
        "genres": " | ".join(genre_names),
        "keywords": " | ".join(keyword_names),
        "director": director,
        "cast": " | ".join(cast_names),
    }