CREATE EXTENSION IF NOT EXISTS vector;

DROP TABLE IF EXISTS movies;

CREATE TABLE movies (
    id BIGSERIAL PRIMARY KEY,

    -- Identifiers
    tconst TEXT UNIQUE NOT NULL,
    tmdb_id INTEGER UNIQUE NOT NULL,

    -- Basic metadata
    title TEXT NOT NULL,
    original_title TEXT,
    release_year INTEGER,
    runtime_minutes INTEGER,

    -- IMDb information
    average_rating DOUBLE PRECISION,
    num_votes BIGINT,

    -- TMDB semantic information
    overview TEXT NOT NULL,
    genres TEXT,
    keywords TEXT,
    director TEXT,
    cast_names TEXT,

    -- Exact text used to generate the embedding
    semantic_text TEXT NOT NULL,

    -- all-MiniLM-L6-v2 embedding
    embedding VECTOR(384) NOT NULL
);

CREATE INDEX idx_movies_embedding
ON movies
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

CREATE INDEX idx_movies_title
ON movies (title);

CREATE INDEX idx_movies_release_year
ON movies (release_year);

CREATE INDEX idx_movies_rating
ON movies (average_rating);