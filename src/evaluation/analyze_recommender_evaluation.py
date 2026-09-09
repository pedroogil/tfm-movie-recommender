from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

EVALUATION_PATH = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "recommender_evaluation.csv"
)

MAPPING_PATH = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "system_mapping.txt"
)


def load_system_mapping() -> dict[str, str]:
    """
    Load the mapping between blind system names
    and semantic representations.

    Expected format:
        System X = B
        System Y = A
        System Z = C

    Returns:
        {
            "System X": "B",
            "System Y": "A",
            "System Z": "C",
        }
    """

    mapping = {}

    with MAPPING_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:

        for line in file:
            line = line.strip()

            if not line:
                continue

            system_name, code = line.split("=")

            mapping[
                system_name.strip()
            ] = code.strip()

    return mapping


def main() -> None:
    print(
        "=== RECOMMENDER EVALUATION ANALYSIS ==="
    )
    print()

    print(
        f"Loading: {EVALUATION_PATH}"
    )

    evaluation = pd.read_csv(
        EVALUATION_PATH
    )

    print(
        f"Rows loaded: "
        f"{len(evaluation):,}"
    )

    if len(evaluation) != 225:
        raise ValueError(
            "Expected 225 evaluation rows, "
            f"but found {len(evaluation)}."
        )

    print()
    print(
        "Checking relevance scores..."
    )

    missing_scores = (
        evaluation[
            "relevance_score"
        ]
        .isna()
        .sum()
    )

    print(
        f"Missing relevance scores: "
        f"{missing_scores}"
    )

    if missing_scores > 0:
        raise ValueError(
            f"There are {missing_scores} "
            "missing relevance scores. "
            "Check that you saved the scored CSV."
        )

    evaluation[
        "relevance_score"
    ] = pd.to_numeric(
        evaluation[
            "relevance_score"
        ],
        errors="raise",
    ).astype(int)

    invalid_scores = (
        ~evaluation[
            "relevance_score"
        ]
        .isin(
            [0, 1, 2]
        )
    )

    if invalid_scores.any():
        invalid_values = (
            evaluation.loc[
                invalid_scores,
                "relevance_score",
            ]
            .unique()
            .tolist()
        )

        raise ValueError(
            "Scores must be 0, 1 or 2. "
            f"Invalid values: {invalid_values}"
        )

    print(
        "All relevance scores are valid."
    )

    print()
    print(
        "=== BLIND SYSTEM RESULTS ==="
    )

    blind_summary = (
        evaluation
        .groupby(
            "system"
        )
        .agg(
            recommendations=(
                "relevance_score",
                "count",
            ),
            mean_score=(
                "relevance_score",
                "mean",
            ),
            score_2_rate=(
                "relevance_score",
                lambda values:
                    values.eq(2).mean(),
            ),
            relevant_rate=(
                "relevance_score",
                lambda values:
                    values.ge(1).mean(),
            ),
        )
        .sort_values(
            "mean_score",
            ascending=False,
        )
    )

    print(
        blind_summary
        .round(4)
        .to_string()
    )

    print()
    print(
        "=== SYSTEM MAPPING ==="
    )

    system_mapping = (
        load_system_mapping()
    )

    representation_names = {
        "A": (
            "Overview only"
        ),
        "B": (
            "Overview + genres + keywords"
        ),
        "C": (
            "Overview + genres + keywords "
            "+ director + cast"
        ),
    }

    for system, code in (
        system_mapping.items()
    ):
        print(
            f"{system} -> "
            f"{code} -> "
            f"{representation_names[code]}"
        )

    evaluation[
        "representation"
    ] = (
        evaluation[
            "system"
        ]
        .map(
            system_mapping
        )
    )

    if (
        evaluation[
            "representation"
        ]
        .isna()
        .any()
    ):
        raise ValueError(
            "Some systems could not be mapped "
            "to representations."
        )

    print()
    print(
        "=== REPRESENTATION RESULTS ==="
    )

    summary = (
        evaluation
        .groupby(
            "representation"
        )
        .agg(
            recommendations=(
                "relevance_score",
                "count",
            ),
            mean_score=(
                "relevance_score",
                "mean",
            ),
            score_2_rate=(
                "relevance_score",
                lambda values:
                    values.eq(2).mean(),
            ),
            relevant_rate=(
                "relevance_score",
                lambda values:
                    values.ge(1).mean(),
            ),
        )
        .sort_values(
            "mean_score",
            ascending=False,
        )
    )

    print(
        summary
        .round(4)
        .to_string()
    )

    print()
    print(
        "Legend:"
    )

    print(
        "A = overview only"
    )

    print(
        "B = overview + genres + keywords"
    )

    print(
        "C = overview + genres + keywords "
        "+ director + cast"
    )

    print()
    print(
        "=== MEAN SCORE BY MOVIE ==="
    )

    by_movie = (
        evaluation
        .pivot_table(
            index=[
                "anchor_movie",
                "anchor_year",
            ],
            columns="representation",
            values="relevance_score",
            aggfunc="mean",
        )
        .round(2)
    )

    print(
        by_movie.to_string()
    )

    print()
    print(
        "=== MEAN SCORE BY RANK ==="
    )

    by_rank = (
        evaluation
        .pivot_table(
            index="rank",
            columns="representation",
            values="relevance_score",
            aggfunc="mean",
        )
        .round(3)
    )

    print(
        by_rank.to_string()
    )

    print()
    print(
        "=== BEST REPRESENTATION PER MOVIE ==="
    )

    movie_scores = (
        evaluation
        .groupby(
            [
                "anchor_movie",
                "representation",
            ]
        )[
            "relevance_score"
        ]
        .mean()
        .reset_index()
    )

    for movie in (
        movie_scores[
            "anchor_movie"
        ]
        .unique()
    ):

        subset = (
            movie_scores[
                movie_scores[
                    "anchor_movie"
                ]
                == movie
            ]
            .sort_values(
                "relevance_score",
                ascending=False,
            )
        )

        best_score = (
            subset[
                "relevance_score"
            ]
            .max()
        )

        best = (
            subset[
                subset[
                    "relevance_score"
                ]
                == best_score
            ][
                "representation"
            ]
            .tolist()
        )

        print(
            f"{movie}: "
            f"{', '.join(best)} "
            f"({best_score:.2f})"
        )


if __name__ == "__main__":
    main()