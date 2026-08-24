TFM — Sistema inteligente de recomendación cinematográfica

Trabajo Fin de Máster del Máster en Ciencia de Datos e Ingeniería de Datos en la Nube.

Autor: Pedro Gil Olivares
Estado: versión de seguimiento 1.0 (agosto de 2026)

Objetivo

Construir un sistema de recomendación cinematográfica basado en contenido semántico. El proyecto integra datos de IMDb y TMDB, construye un catálogo enriquecido y, en las siguientes fases, generará embeddings, realizará búsqueda vectorial, incorporará un modelo de lenguaje para explicar o interpretar recomendaciones y expondrá el sistema mediante una aplicación Streamlit.

Arquitectura

IMDb title.basics + title.ratings
            ↓
      Filtrado + ETL
            ↓
  imdb_movies.parquet
            ↓
     IMDb → TMDB ID
            ↓
 imdb_tmdb_mapping.jsonl
            ↓
 TMDB details + credits + keywords
            ↓
 tmdb_movies_enriched.jsonl
            ↓
       [siguiente fase]
 perfil semántico → embeddings → vector DB → recomendador → LLM → Streamlit

Estado actual

738.348 títulos IMDb con titleType=movie.

729.093 películas no adultas.

333.894 películas con información de rating.

12.231 películas candidatas tras exigir al menos 10.000 votos.

12.228 correspondencias válidas IMDb → TMDB.

Cobertura del matching: 99,98 %.

3 títulos sin correspondencia TMDB.

Extracción normalizada de overview, genres, keywords, director y cast.

Pipeline de enriquecimiento incremental y reanudable validado con una muestra controlada de 5 películas.

Estructura principal

src/
├── ingestion/
│   ├── tmdb_client.py
│   ├── tmdb_enricher.py
│   ├── enrich_movies.py
│   ├── test_tmdb.py
│   ├── test_tmdb_enricher.py
│   ├── inspect_imdb.py
│   └── inspect_imdb_ratings.py
├── processing/
│   ├── explore_imdb.py
│   ├── build_imdb_dataset.py
│   ├── test_matching.py
│   ├── build_tmdb_mapping.py
│   ├── enrich_tmdb.py
│   ├── analyze_mapping.py
│   ├── movie_parser.py
│   └── storage.py
└── models/
    └── movie.py

Datos

Los datasets y resultados pesados no se versionan en Git. La estructura local esperada es:

data/
├── raw/imdb/
│   ├── title.basics.tsv
│   └── title.ratings.tsv
└── processed/
    ├── imdb_movies.parquet
    ├── imdb_tmdb_mapping.jsonl
    └── tmdb_movies_enriched.jsonl

Instalación

python -m venv tfm
# Windows
tfm\Scripts\activate
pip install -r requirements.txt

Comandos principales

Construcción del catálogo IMDb:

python -m src.processing.build_imdb_dataset

Análisis del mapping IMDb → TMDB:

python -m src.processing.analyze_mapping

Validación del extractor TMDB:

python -m src.ingestion.test_tmdb_enricher

Prueba controlada del enriquecimiento:

python -m src.ingestion.enrich_movies --limit 5

Enriquecimiento del conjunto pendiente:

python -m src.ingestion.enrich_movies

Próximos pasos

Completar/verificar el enriquecimiento del catálogo completo.

Analizar calidad: nulos, duplicados y cobertura de metadatos.

Consolidar el dataset maestro en Parquet.

Construir el perfil textual de cada película.

Comparar TF-IDF con modelos de embeddings preentrenados.

Implementar almacenamiento y búsqueda vectorial.

Evaluar formalmente la calidad de las recomendaciones.

Integrar el LLM como capa de explicación/interacción.

Desarrollar y desplegar la aplicación Streamlit.
