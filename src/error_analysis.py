"""Visual analysis of final-test false positives and false negatives.

The selected model is evaluated on the unchanged test set. The script saves
the most confident false-positive and false-negative examples for inspection.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = PROJECT_ROOT / "models" / "final_model.keras"
CSV_PATH = PROJECT_ROOT / "data" / "processed" / "breakhis_split.csv"
RESULTS_DIR = PROJECT_ROOT / "results" / "error_analysis"

IMAGE_SIZE = (224, 224)
THRESHOLD = 0.5
MAX_EXAMPLES = 12


def load_test_metadata() -> pd.DataFrame:
    """Load and validate the final test-set metadata."""
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Split CSV not found:\n{CSV_PATH}")

    df = pd.read_csv(CSV_PATH)

    required_columns = {
        "image_path",
        "filename",
        "class",
        "magnification",
        "tumor_type",
        "split",
    }

    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing CSV columns: {sorted(missing)}")

    return df[df["split"] == "test"].copy()


def preprocess_image(image_path: str) -> np.ndarray:
    """Load one image using the same preprocessing as MobileNetV2 training."""
    image = tf.keras.utils.load_img(
        image_path,
        target_size=IMAGE_SIZE,
    )
    image = tf.keras.utils.img_to_array(image)
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)

    return np.expand_dims(image, axis=0)


def predict_test_errors(
    model: tf.keras.Model,
    test_df: pd.DataFrame,
) -> tuple[list[dict], list[dict]]:
    """Identify false positives and false negatives on the test set."""
    false_positives = []
    false_negatives = []

    for _, row in test_df.iterrows():
        probability = float(
            model.predict(
                preprocess_image(row["image_path"]),
                verbose=0,
            )[0][0]
        )

        actual = row["class"].strip().lower()
        predicted = (
            "malignant"
            if probability >= THRESHOLD
            else "benign"
        )

        example = {
            "image_path": row["image_path"],
            "filename": row["filename"],
            "actual": actual,
            "predicted": predicted,
            "probability": probability,
            "magnification": row["magnification"],
            "tumor_type": row["tumor_type"],
        }

        if actual == "benign" and predicted == "malignant":
            false_positives.append(example)

        elif actual == "malignant" and predicted == "benign":
            false_negatives.append(example)

    return false_positives, false_negatives


def select_most_confident_errors(
    false_positives: list[dict],
    false_negatives: list[dict],
) -> tuple[list[dict], list[dict]]:
    """Keep the most confidently misclassified examples."""
    false_positives = sorted(
        false_positives,
        key=lambda item: item["probability"],
        reverse=True,
    )[:MAX_EXAMPLES]

    false_negatives = sorted(
        false_negatives,
        key=lambda item: item["probability"],
    )[:MAX_EXAMPLES]

    return false_positives, false_negatives


def save_error_grid(
    examples: list[dict],
    title: str,
    output_file: Path,
) -> None:
    """Save a grid of incorrectly classified test images."""
    if not examples:
        print(f"No examples available for: {title}")
        return

    output_file.parent.mkdir(parents=True, exist_ok=True)

    columns = 4
    rows = int(np.ceil(len(examples) / columns))

    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(14, 3.5 * rows),
    )
    axes = np.asarray(axes).reshape(-1)

    for ax in axes:
        ax.axis("off")

    for ax, example in zip(axes, examples):
        image = tf.keras.utils.load_img(
            example["image_path"],
            target_size=IMAGE_SIZE,
        )
        image = tf.keras.utils.img_to_array(image) / 255.0

        ax.imshow(image)
        ax.set_title(
            f"Actual: {example['actual'].capitalize()}\n"
            f"Predicted: {example['predicted'].capitalize()}\n"
            f"P(Malignant): {example['probability']:.2f}\n"
            f"Magnification: {example['magnification']}",
            fontsize=9,
        )
        ax.axis("off")

    fig.suptitle(title, fontsize=15)
    fig.tight_layout()
    fig.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    print(f"Saved: {output_file}")


def save_error_table(
    false_positives: list[dict],
    false_negatives: list[dict],
) -> None:
    """Save metadata for the visualized errors."""
    records = [
        {**example, "error_type": "false_positive"}
        for example in false_positives
    ] + [
        {**example, "error_type": "false_negative"}
        for example in false_negatives
    ]

    output_file = RESULTS_DIR / "final_error_examples.csv"
    pd.DataFrame(records).to_csv(output_file, index=False)
    print(f"Saved: {output_file}")


def main() -> None:
    """Run final-test error analysis."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Final model not found:\n{MODEL_PATH}"
        )

    test_df = load_test_metadata()
    model = tf.keras.models.load_model(MODEL_PATH)

    false_positives, false_negatives = predict_test_errors(
        model,
        test_df,
    )

    total_fp = len(false_positives)
    total_fn = len(false_negatives)

    false_positives, false_negatives = select_most_confident_errors(
        false_positives,
        false_negatives,
    )

    save_error_grid(
        false_positives,
        "False Positive Examples - Benign Predicted as Malignant",
        RESULTS_DIR / "final_false_positive_examples.png",
    )

    save_error_grid(
        false_negatives,
        "False Negative Examples - Malignant Predicted as Benign",
        RESULTS_DIR / "final_false_negative_examples.png",
    )

    save_error_table(false_positives, false_negatives)

    print("\n" + "=" * 60)
    print("ERROR ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"Test images:             {len(test_df)}")
    print(f"Total false positives:   {total_fp}")
    print(f"Total false negatives:   {total_fn}")
    print(f"Visual FP examples:      {len(false_positives)}")
    print(f"Visual FN examples:      {len(false_negatives)}")
    print(f"Results directory:       {RESULTS_DIR}")


if __name__ == "__main__":
    main()
