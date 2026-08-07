from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MAPPING_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "imdb_tmdb_mapping.jsonl"
)


mapping = pd.read_json(
    MAPPING_PATH,
    lines=True,
)

print("\n=== TMDB MAPPING ===")

print(f"Total processed: {len(mapping):,}")

print("\nStatus:")
print(mapping["status"].value_counts())

matched = (
    mapping["status"] == "matched"
).sum()

not_found = (
    mapping["status"] == "not_found"
).sum()

errors = (
    mapping["status"] == "error"
).sum()

print("\nResults:")
print(f"Matched:   {matched:,}")
print(f"Not found: {not_found:,}")
print(f"Errors:    {errors:,}")

if len(mapping) > 0:
    print(
        f"\nCoverage: "
        f"{matched / len(mapping) * 100:.2f}%"
    )