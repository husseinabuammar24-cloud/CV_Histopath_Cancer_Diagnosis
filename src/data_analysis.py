"""Exploratory data analysis for the BreakHis dataset.

The script inspects the raw dataset, parses metadata encoded in filenames,
and reports class, magnification, and case/group statistics.
"""

from pathlib import Path
import re

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_ROOT = PROJECT_ROOT / "data" / "raw" / "BreakHis"
DATA_DIR = (
    DATA_ROOT
    / "dataset_cancer_v1"
    / "dataset_cancer_v1"
    / "classificacao_binaria"
)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}

FILENAME_PATTERN = re.compile(
    r"^(SOB)_([BM])_([A-Z]+)-"
    r"(\d{2})-([A-Za-z0-9]+)-(\d+)-(\d+)\.png$",
    re.IGNORECASE,
)


def find_images(data_dir: Path) -> list[Path]:
    """Return all supported image files."""
    return sorted(
        path
        for path in data_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def parse_image_records(
    image_files: list[Path],
) -> tuple[pd.DataFrame, list[Path]]:
    """Parse metadata encoded in BreakHis filenames."""
    records = []
    parsed_paths = set()

    for image_path in image_files:
        match = FILENAME_PATTERN.match(image_path.name)

        if not match:
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
        parsed_paths.add(image_path)

    unparsed = [
        path for path in image_files if path not in parsed_paths
    ]

    return pd.DataFrame(records), unparsed


def summarize_dataset(df: pd.DataFrame) -> None:
    """Print high-level image and case statistics."""
    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)
    print(f"Total images:  {len(df)}")
    print(f"Unique groups: {df['group_id'].nunique()}")

    print("\nImages per class:")
    print(df["class"].value_counts())

    print("\nGroups per class:")
    print(df.groupby("class")["group_id"].nunique())


def summarize_magnifications(df: pd.DataFrame) -> None:
    """Print image counts by magnification and class."""
    summary = (
        df.groupby(["magnification", "class"])
        .size()
        .unstack(fill_value=0)
        .sort_index()
    )

    print("\nImages by magnification and class:")
    print(summary)


def summarize_groups(df: pd.DataFrame) -> None:
    """Print descriptive statistics for image counts per group."""
    group_counts = df.groupby("group_id").size()

    print("\nImages per group:")
    print(group_counts.describe())


def check_group_labels(df: pd.DataFrame) -> None:
    """Verify that each group corresponds to a single class."""
    classes_per_group = df.groupby("group_id")["class"].nunique()
    invalid_groups = classes_per_group[classes_per_group > 1]

    print(
        f"\nGroups containing multiple classes: "
        f"{len(invalid_groups)}"
    )

    if not invalid_groups.empty:
        print(invalid_groups)


def main() -> None:
    """Run the BreakHis exploratory analysis."""
    if not DATA_DIR.exists():
        raise FileNotFoundError(
            f"BreakHis dataset not found:\n{DATA_DIR}"
        )

    image_files = find_images(DATA_DIR)
    print(f"Image files found: {len(image_files)}")

    df, unparsed = parse_image_records(image_files)

    print(f"Images successfully parsed: {len(df)}")
    print(f"Unparsed images: {len(unparsed)}")

    if unparsed:
        print("\nFirst five unparsed files:")
        for path in unparsed[:5]:
            print(f"  - {path.name}")

    summarize_dataset(df)
    summarize_magnifications(df)
    summarize_groups(df)
    check_group_labels(df)


if __name__ == "__main__":
    main()
