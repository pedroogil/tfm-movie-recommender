import sys
from pathlib import Path

import streamlit as st


# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.services.recommendation_service import (
    RecommendationService,
)


st.set_page_config(
    page_title="Movie Semantic Recommender",
    page_icon="🎬",
    layout="wide",
)


@st.cache_resource
def get_service() -> RecommendationService:
    """
    Load the recommendation service once.
    """
    return RecommendationService()


service = get_service()


def render_movie_card(
    movie: dict,
    position: int | None = None,
) -> None:
    """
    Render one movie recommendation.
    """

    title = movie["title"]
    year = movie["release_year"] or "N/A"

    if position is not None:
        st.subheader(
            f"{position}. {title} ({year})"
        )
    else:
        st.subheader(
            f"{title} ({year})"
        )

    col1, col2, col3 = st.columns(3)

    with col1:
        rating = movie["average_rating"]

        st.metric(
            "IMDb rating",
            (
                f"{rating:.1f}/10"
                if rating is not None
                else "N/A"
            ),
        )

    with col2:
        runtime = movie["runtime_minutes"]

        st.metric(
            "Runtime",
            (
                f"{runtime} min"
                if runtime
                else "N/A"
            ),
        )

    with col3:
        similarity = movie.get("similarity")

        st.metric(
            "Similarity",
            (
                f"{similarity:.3f}"
                if similarity is not None
                else "—"
            ),
        )

    if movie.get("genres"):
        st.write(
            f"**Genres:** {movie['genres']}"
        )

    if movie.get("director"):
        st.write(
            f"**Director:** {movie['director']}"
        )

    if movie.get("overview"):
        st.write(
            movie["overview"]
        )

    st.divider()


st.title(
    "🎬 Movie Semantic Recommender"
)

st.write(
    """
    Sistema de recomendación de películas
    basado en embeddings semánticos y
    búsqueda vectorial con PostgreSQL
    y pgvector.
    """
)

mode = st.radio(
    "¿Cómo quieres buscar?",
    [
        "Películas similares",
        "Describe qué te apetece ver",
    ],
    horizontal=True,
)


if mode == "Películas similares":

    st.header(
        "Encuentra películas parecidas"
    )

    title_query = st.text_input(
        "Busca una película",
        placeholder="Ej.: Inception",
    )

    if title_query:

        matches = (
            service.search_movies_by_title(
                title_query
            )
        )

        if not matches:

            st.warning(
                "No se encontraron películas."
            )

        else:

            options = {
                (
                    f"{movie['title']} "
                    f"({movie['release_year']})"
                ):
                    movie["id"]
                for movie in matches
            }

            selected_label = st.selectbox(
                "Selecciona la película",
                options.keys(),
            )

            selected_id = options[
                selected_label
            ]

            selected_movie = (
                service.get_movie(
                    selected_id
                )
            )

            if selected_movie:

                st.subheader(
                    "Película seleccionada"
                )

                render_movie_card(
                    selected_movie
                )

            if st.button(
                "Recomendar películas",
                type="primary",
            ):

                with st.spinner(
                    "Buscando películas similares..."
                ):

                    recommendations = (
                        service.recommend_by_movie(
                            selected_id,
                            top_k=10,
                        )
                    )

                st.header(
                    "Recomendaciones"
                )

                for position, movie in enumerate(
                    recommendations,
                    start=1,
                ):
                    render_movie_card(
                        movie,
                        position,
                    )


else:

    st.header(
        "Describe qué te apetece ver"
    )

    query = st.text_area(
        "Descripción",
        placeholder=(
            "Ej.: Quiero una película oscura "
            "de ciencia ficción sobre memoria, "
            "sueños y realidades alternativas."
        ),
        height=120,
    )

    if st.button(
        "Buscar",
        type="primary",
    ):

        if not query.strip():

            st.warning(
                "Escribe primero una descripción."
            )

        else:

            with st.spinner(
                "Interpretando tu búsqueda..."
            ):

                recommendations = (
                    service.search_by_text(
                        query=query,
                        top_k=10,
                    )
                )

            st.header(
                "Resultados"
            )

            for position, movie in enumerate(
                recommendations,
                start=1,
            ):
                render_movie_card(
                    movie,
                    position,
                )