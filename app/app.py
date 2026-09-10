import sys
from pathlib import Path

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from src.ingestion.tmdb_client import get_movie
from src.services.recommendation_service import (
    RecommendationService,
)


TMDB_IMAGE_BASE_URL = (
    "https://image.tmdb.org/t/p/w500"
)


st.set_page_config(
    page_title="Recomendador Semántico de Películas",
    page_icon="🎬",
    layout="wide",
)


@st.cache_resource
def get_service() -> RecommendationService:
    """
    Load the recommendation service only once.
    """

    return RecommendationService()


@st.cache_data(
    show_spinner=False
)
def get_poster_url(
    tmdb_id: int,
) -> str | None:
    """
    Retrieve and cache the TMDB poster URL.
    """

    try:
        data = get_movie(
            tmdb_id
        )

        poster_path = data.get(
            "poster_path"
        )

        if not poster_path:
            return None

        return (
            f"{TMDB_IMAGE_BASE_URL}"
            f"{poster_path}"
        )

    except Exception:
        return None


service = get_service()


def format_votes(
    num_votes: int | None,
) -> str:
    """
    Format IMDb vote count for display.
    """

    if num_votes is None:
        return "N/A"

    if num_votes >= 1_000_000:
        return (
            f"{num_votes / 1_000_000:.1f} M"
        )

    if num_votes >= 1_000:
        return (
            f"{num_votes / 1_000:.1f} k"
        )

    return str(num_votes)


def format_runtime(
    runtime: int | None,
) -> str:
    """
    Convert runtime in minutes to a compact
    hours/minutes format.
    """

    if not runtime:
        return "N/A"

    hours = runtime // 60
    minutes = runtime % 60

    if hours > 0:
        return f"{hours}h {minutes}m"

    return f"{minutes}m"


def render_movie_card(
    movie: dict,
    position: int | None = None,
) -> None:
    """
    Render a movie recommendation card.
    """

    title = movie["title"]

    year = (
        movie["release_year"]
        or "N/A"
    )

    if position is not None:
        st.subheader(
            f"{position}. "
            f"{title} ({year})"
        )
    else:
        st.subheader(
            f"{title} ({year})"
        )

    tmdb_id = movie.get(
        "tmdb_id"
    )

    poster_url = None

    if tmdb_id:
        poster_url = get_poster_url(
            int(tmdb_id)
        )

    poster_column, info_column = (
        st.columns(
            [
                1,
                3,
            ]
        )
    )

    with poster_column:

        if poster_url:
            st.image(
                poster_url,
                width="stretch",
            )
        else:
            st.info(
                "🎬 Póster no disponible"
            )

    with info_column:

        rating = movie.get(
            "average_rating"
        )

        runtime = movie.get(
            "runtime_minutes"
        )

        num_votes = movie.get(
            "num_votes"
        )

        similarity = movie.get(
            "similarity"
        )

        metric_1, metric_2 = (
            st.columns(2)
        )

        with metric_1:
            st.metric(
                "IMDb",
                (
                    f"{rating:.1f}/10"
                    if rating is not None
                    else "N/A"
                ),
            )

        with metric_2:
            st.metric(
                "Duración",
                format_runtime(
                    runtime
                ),
            )

        metric_3, metric_4 = (
            st.columns(2)
        )

        with metric_3:
            st.metric(
                "Votos",
                format_votes(
                    num_votes
                ),
            )

        with metric_4:
            st.metric(
                "Similitud",
                (
                    f"{similarity:.3f}"
                    if similarity is not None
                    else "—"
                ),
            )

        if movie.get(
            "genres"
        ):
            st.write(
                "**Géneros:** "
                f"{movie['genres']}"
            )

        if movie.get(
            "director"
        ):
            st.write(
                "**Director:** "
                f"{movie['director']}"
            )

        if movie.get(
            "overview"
        ):
            st.write(
                "**Sinopsis**"
            )

            st.write(
                movie["overview"]
            )

        cast = movie.get(
            "cast_names"
        )

        if cast:

            with st.expander(
                "Ver reparto"
            ):
                st.write(
                    cast
                )

        link_col1, link_col2 = (
            st.columns(2)
        )

        tconst = movie.get(
            "tconst"
        )

        if tconst:

            with link_col1:
                st.link_button(
                    "Ver en IMDb",
                    (
                        "https://www.imdb.com/"
                        f"title/{tconst}/"
                    ),
                    use_container_width=True,
                )

        if tmdb_id:

            with link_col2:
                st.link_button(
                    "Ver en TMDB",
                    (
                        "https://www.themoviedb.org/"
                        f"movie/{tmdb_id}"
                    ),
                    use_container_width=True,
                )

    st.divider()


st.title(
    "🎬 Recomendador Semántico de Películas"
)

st.caption(
    "TFM · Ciencia de Datos e Ingeniería de Datos"
)

st.write(
    """
    Sistema de recomendación basado en **embeddings semánticos**
    y búsqueda vectorial con **PostgreSQL, pgvector y HNSW**.

    Cada película se representa mediante su **sinopsis, géneros
    y palabras clave**, configuración seleccionada tras comparar
    distintas estrategias de representación.
    """
)


with st.sidebar:

    st.header(
        "⚙️ Configuración"
    )

    top_k = st.slider(
        "Número de resultados",
        min_value=5,
        max_value=20,
        value=10,
        step=1,
    )

    min_rating = st.slider(
        "Valoración IMDb mínima",
        min_value=0.0,
        max_value=9.0,
        value=0.0,
        step=0.5,
    )

    st.subheader(
        "Año de estreno"
    )

    use_year_filter = st.checkbox(
        "Filtrar por año"
    )

    if use_year_filter:

        year_range = st.slider(
            "Rango",
            min_value=1915,
            max_value=2026,
            value=(
                1980,
                2026,
            ),
        )

        min_year = year_range[0]
        max_year = year_range[1]

    else:

        min_year = None
        max_year = None

    st.divider()

    st.caption(
        "Catálogo: 12.227 películas"
    )

    st.caption(
        "Modelo: all-MiniLM-L6-v2"
    )

    st.caption(
        "Dimensión del embedding: 384"
    )


mode = st.radio(
    "¿Cómo quieres buscar?",
    [
        "🎞️ Películas similares",
        "💬 Describe qué te apetece ver",
    ],
    horizontal=True,
)


if mode == "🎞️ Películas similares":

    st.header(
        "Encuentra películas parecidas"
    )

    title_query = st.text_input(
        "Busca una película",
        placeholder="Ej.: Inception",
    )

    if title_query:

        matches = (
            service
            .search_movies_by_title(
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

            selected_label = (
                st.selectbox(
                    "Selecciona la película",
                    options.keys(),
                )
            )

            selected_id = (
                options[
                    selected_label
                ]
            )

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
                "🔎 Recomendar películas",
                type="primary",
            ):

                with st.spinner(
                    "Buscando películas similares..."
                ):

                    recommendations = (
                        service
                        .recommend_by_movie(
                            movie_id=selected_id,
                            top_k=top_k,
                            min_rating=min_rating,
                            min_year=min_year,
                            max_year=max_year,
                        )
                    )

                st.header(
                    "Recomendaciones"
                )

                if not recommendations:

                    st.info(
                        "No hay resultados "
                        "con los filtros actuales."
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

    st.write(
        """
        Describe con tus propias palabras el tipo de
        película que buscas. El texto se transforma
        en un embedding y se compara con el catálogo.
        """
    )

    query = st.text_area(
        "Descripción",
        placeholder=(
            "Ej.: Quiero una película oscura "
            "de ciencia ficción sobre memoria, "
            "sueños y realidades alternativas."
        ),
        height=130,
    )

    if st.button(
        "🔎 Buscar películas",
        type="primary",
    ):

        if not query.strip():

            st.warning(
                "Escribe primero "
                "una descripción."
            )

        else:

            with st.spinner(
                "Interpretando tu búsqueda..."
            ):

                recommendations = (
                    service.search_by_text(
                        query=query,
                        top_k=top_k,
                        min_rating=min_rating,
                        min_year=min_year,
                        max_year=max_year,
                    )
                )

            st.header(
                "Resultados"
            )

            if not recommendations:

                st.info(
                    "No hay resultados "
                    "con los filtros actuales."
                )

            for position, movie in enumerate(
                recommendations,
                start=1,
            ):

                render_movie_card(
                    movie,
                    position,
                )