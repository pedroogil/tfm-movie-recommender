import json
import time
from pathlib import Path

import pandas as pd

from src.ingestion.tmdb_client import get_movie
from src.ingestion.tmdb_enricher import extract_movie_data


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MOVIES_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "imdb_movies.parquet"
)

MAPPING_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "imdb_tmdb_mapping.jsonl"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "tmdb_movies_enriched.jsonl"
)


def load_mapping() -> pd.DataFrame:
    """Load IMDb → TMDB mapping."""

    mapping = pd.read_json(
        MAPPING_PATH,
        lines=True,
    )

    mapping = mapping[
        mapping["status"] == "matched"
    ].copy()

    mapping["tmdb_id"] = (
        mapping["tmdb_id"]
        .astype(int)
    )

    return mapping[
        ["tconst", "tmdb_id"]
    ]


def load_processed_ids() -> set[str]:
    """Load IMDb IDs already enriched."""

    if not OUTPUT_PATH.exists():
        return set()

    processed = set()

    with OUTPUT_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:

        for line in file:

            record = json.loads(line)

            if record.get("status") == "matched":
                processed.add(
                    record["tconst"]
                )

    return processed


def main(limit: int | None = None) -> None:

    print("Loading IMDb movies...")

    movies = pd.read_parquet(
        MOVIES_PATH
    )

    print(
        f"IMDb movies loaded: "
        f"{len(movies):,}"
    )

    print("Loading IMDb → TMDB mapping...")

    mapping = load_mapping()

    print(
        f"Matched movies: "
        f"{len(mapping):,}"
    )

    movies = movies.merge(
        mapping,
        on="tconst",
        how="inner",
    )

    print(
        f"Movies available for enrichment: "
        f"{len(movies):,}"
    )

    processed_ids = load_processed_ids()

    print(
        f"Already enriched: "
        f"{len(processed_ids):,}"
    )

    movies = movies[
        ~movies["tconst"].isin(
            processed_ids
        )
    ].copy()

    if limit is not None:
        movies = movies.head(limit)

    print(
        f"Remaining to process: "
        f"{len(movies):,}"
    )

    if len(movies) == 0:
        print("Nothing to process.")
        return

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    matched = 0
    errors = 0

    start_time = time.time()

    with OUTPUT_PATH.open(
        "a",
        encoding="utf-8",
    ) as output_file:

        for index, row in enumerate(
            movies.itertuples(
                index=False
            ),
            start=1,
        ):

            tconst = row.tconst
            tmdb_id = int(row.tmdb_id)

            try:

                data = get_movie(
                    tmdb_id,
                    "credits,keywords",
                )

                enriched = extract_movie_data(
                    data
                )

                record = {
                    "tconst": tconst,
                    "status": "matched",
                    **enriched,
                }

                matched += 1

            except Exception as error:

                record = {
                    "tconst": tconst,
                    "tmdb_id": tmdb_id,
                    "status": "error",
                    "error": str(error),
                }

                errors += 1

                print(
                    f"ERROR {tconst} "
                    f"(TMDB {tmdb_id}): "
                    f"{error}"
                )

            output_file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )

            output_file.flush()

            if index % 10 == 0:

                elapsed = (
                    time.time()
                    - start_time
                )

                print(
                    f"Processed "
                    f"{index:,}/"
                    f"{len(movies):,} "
                    f"| "
                    f"Matched: "
                    f"{matched:,} "
                    f"| "
                    f"Errors: "
                    f"{errors:,} "
                    f"| "
                    f"Time: "
                    f"{elapsed / 60:.1f} min"
                )

            time.sleep(0.25)

    elapsed = time.time() - start_time

    print()
    print("=== ENRICHMENT FINISHED ===")
    print(
        f"Processed: {len(movies):,}"
    )
    print(
        f"Matched: {matched:,}"
    )
    print(
        f"Errors: {errors:,}"
    )
    print(
        f"Time: {elapsed / 60:.1f} minutes"
    )


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Enrich IMDb movies "
            "with TMDB data."
        )
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Maximum number of movies "
            "to process."
        ),
    )

    args = parser.parse_args()

    main(
        limit=args.limit
    )