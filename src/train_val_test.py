#!/usr/bin/env python3
"""Create a leakage-safe train/validation/test split for BreakHis.

The split is performed at case/slide level using ``year + slide_id`` so that
related images cannot appear in different dataset splits.
"""

from pathlib import Path
import re
import sys

import pandas as pd
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "BreakHis"
    / "dataset_cancer_v1"
    / "dataset_cancer_v1"
    / "classificacao_binaria"
)
OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "breakhis_split.csv"

RANDOM_STATE = 42
VALIDATION_TEST_FRACTION = 0.30
TEST_FRACTION_OF_HOLDOUT = 0.50

FILENAME_PATTERN = re.compile(
    r"^(SOB)_([BM])_([A-Z]+)-(\d{2})-([A-Za-z0-9]+)-(\d+)-(\d+)\.png$",
    re.IGNORECASE,
)


def find_image_files(data_dir: Path) -> list[Path]:
    """Return all PNG files below the BreakHis binary-classification folder."""
    if not data_dir.exists():
        raise FileNotFoundError(
            f"Dataset directory not found:\n{data_dir}\n"
            "Expected the dataset under data/raw/BreakHis/."
        )

    return sorted(
        {
            path
            for pattern in ("*.png", "*.PNG")
            for path in data_dir.rglob(pattern)
        }
    )


def parse_filenames(image_files: list[Path]) -> pd.DataFrame:
    """Parse metadata encoded in BreakHis filenames."""
    records = []
    unmatched = []

    for image_path in image_files:
        match = FILENAME_PATTERN.match(image_path.name)

        if not match:
            unmatched.append(image_path.name)
            continue

        _, tumor_class, tumor_type, year, slide_id, magnification, sequence = (
            match.groups()
        )

        records.append(
            {
                "image_path": str(image_path),
                "filename": image_path.name,
                "class": (
                    "benign"
                    if tumor_class.upper() == "B"
                    else "malignant"
                ),
                "tumor_type": tumor_type,
                "year": year,
                "slide_id": slide_id,
                "magnification": magnification,
                "sequence": sequence,
                "group_id": f"{year}-{slide_id}",
            }
        )

    if unmatched:
        print(
            f"Warning: {len(unmatched)} file(s) did not match the "
            "expected BreakHis filename pattern."
        )
        for filename in unmatched[:5]:
            print(f"  - {filename}")

    df = pd.DataFrame(records)

    if df.empty:
        raise ValueError(
            "No filenames matched the expected BreakHis pattern."
        )

    return df


def build_group_table(df: pd.DataFrame) -> pd.DataFrame:
    """Create one row per case/slide group."""
    return df[["group_id", "class"]].drop_duplicates("group_id")


def split_groups(groups: pd.DataFrame):
    """Create stratified train/validation/test splits at group level."""
    train_groups, holdout_groups = train_test_split(
        groups,
        test_size=VALIDATION_TEST_FRACTION,
        stratify=groups["class"],
        random_state=RANDOM_STATE,
    )

    validation_groups, test_groups = train_test_split(
        holdout_groups,
        test_size=TEST_FRACTION_OF_HOLDOUT,
        stratify=holdout_groups["class"],
        random_state=RANDOM_STATE,
    )

    return train_groups, validation_groups, test_groups


def assign_splits(
    df: pd.DataFrame,
    train_groups: pd.DataFrame,
    validation_groups: pd.DataFrame,
    test_groups: pd.DataFrame,
) -> pd.DataFrame:
    """Assign each image to the split of its group."""
    split_by_group = {
        **{group_id: "train" for group_id in train_groups["group_id"]},
        **{
            group_id: "validation"
            for group_id in validation_groups["group_id"]
        },
        **{group_id: "test" for group_id in test_groups["group_id"]},
    }

    result = df.copy()
    result["split"] = result["group_id"].map(split_by_group)

    if result["split"].isna().any():
        raise ValueError("Some groups could not be assigned to a split.")

    return result


def verify_no_leakage(df: pd.DataFrame) -> None:
    """Verify that every case/slide group belongs to one split only."""
    split_counts = df.groupby("group_id")["split"].nunique()
    leaking_groups = split_counts[split_counts > 1]

    if not leaking_groups.empty:
        raise ValueError(
            f"Data leakage detected in {len(leaking_groups)} group(s)."
        )

    print("No group leakage detected.")


def print_summary(df: pd.DataFrame) -> None:
    """Print image, group, and class distributions."""
    summary = (
        df.groupby("split")
        .agg(images=("filename", "count"), groups=("group_id", "nunique"))
        .reindex(["train", "validation", "test"])
    )

    class_counts = (
        df.groupby(["split", "class"])
        .size()
        .unstack(fill_value=0)
        .reindex(["train", "validation", "test"])
    )

    print("\n" + "=" * 60)
    print("BREAKHIS SPLIT SUMMARY")
    print("=" * 60)
    print(summary)

    print("\nClass counts:")
    print(class_counts)

    print("\nClass distribution (%):")
    print(class_counts.div(class_counts.sum(axis=1), axis=0).mul(100).round(2))


def main() -> pd.DataFrame:
    """Build, verify, save, and report the dataset split."""
    image_files = find_image_files(DATA_DIR)
    print(f"Images found: {len(image_files)}")

    df = parse_filenames(image_files)
    print(f"Images parsed: {len(df)}")
    print(f"Unique groups: {df['group_id'].nunique()}")

    groups = build_group_table(df)
    train_groups, validation_groups, test_groups = split_groups(groups)

    df = assign_splits(
        df,
        train_groups,
        validation_groups,
        test_groups,
    )

    verify_no_leakage(df)
    print_summary(df)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)

    print(f"\nSplit file saved to:\n{OUTPUT_FILE}")
    return df


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError) as error:
        print(f"\nERROR: {error}", file=sys.stderr)
        sys.exit(1)
