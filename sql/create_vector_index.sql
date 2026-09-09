DROP INDEX IF EXISTS idx_movies_embedding;

CREATE INDEX idx_movies_embedding
ON movies
USING hnsw (
    embedding vector_cosine_ops
);

ANALYZE movies;