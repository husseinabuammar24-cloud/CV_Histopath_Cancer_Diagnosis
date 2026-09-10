"""Preprocessing pipeline for BreakHis and MobileNetV2.

Pipeline:
    PNG -> RGB -> 224x224 -> float32 -> MobileNetV2 preprocess_input [-1, 1]

Augmentation is applied only to the training set. Validation and test images
remain unaugmented.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SPLIT_FILE = PROJECT_ROOT / "data" / "processed" / "breakhis_split.csv"
PREVIEW_FILE = (
    PROJECT_ROOT
    / "results"
    / "preprocessing"
    / "augmented_training_preview.png"
)

TARGET_SIZE = (224, 224)
BATCH_SIZE = 32
RANDOM_SEED = 42

CLASS_NAMES = ["benign", "malignant"]
LABEL_MAP = {"benign": 0, "malignant": 1}

REQUIRED_COLUMNS = [
    "image_path",
    "filename",
    "class",
    "tumor_type",
    "year",
    "slide_id",
    "magnification",
    "sequence",
    "group_id",
    "split",
]


def load_split_file() -> pd.DataFrame:
    """Load and validate the processed split CSV."""
    if not SPLIT_FILE.exists():
        raise FileNotFoundError(
            f"Split file not found:\n{SPLIT_FILE}\n"
            "Run src/train_val_test.py first."
        )

    df = pd.read_csv(SPLIT_FILE)

    missing_columns = [
        column for column in REQUIRED_COLUMNS if column not in df.columns
    ]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    return df


def split_dataframe(df: pd.DataFrame):
    """Create train, validation, and test DataFrames with numeric labels."""
    split_data = {}

    for split in ("train", "validation", "test"):
        dataframe = df[df["split"] == split].copy()
        dataframe["label"] = dataframe["class"].map(LABEL_MAP)

        if dataframe["label"].isna().any():
            raise ValueError(f"Unknown class label found in {split} split.")

        split_data[split] = dataframe

    return (
        split_data["train"],
        split_data["validation"],
        split_data["test"],
    )


def verify_no_group_leakage(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:
    """Verify that case/slide groups do not overlap."""
    train_groups = set(train_df["group_id"])
    validation_groups = set(validation_df["group_id"])
    test_groups = set(test_df["group_id"])

    overlaps = {
        "train/validation": train_groups & validation_groups,
        "train/test": train_groups & test_groups,
        "validation/test": validation_groups & test_groups,
    }

    for name, overlap in overlaps.items():
        if overlap:
            raise ValueError(
                f"Group leakage detected between {name}: "
                f"{len(overlap)} overlapping group(s)."
            )


def load_and_preprocess_image(image_path, label):
    """Load one image and apply the MobileNetV2 preprocessing."""
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.resize(image, TARGET_SIZE)
    image = tf.cast(image, tf.float32)

    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)

    return image, label


def create_dataset(
    dataframe: pd.DataFrame,
    shuffle: bool = False,
) -> tf.data.Dataset:
    """Create a batched TensorFlow dataset."""
    image_paths = dataframe["image_path"].to_numpy()
    labels = dataframe["label"].to_numpy(dtype=np.float32)

    dataset = tf.data.Dataset.from_tensor_slices(
        (image_paths, labels)
    )

    dataset = dataset.map(
        load_and_preprocess_image,
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    if shuffle:
        dataset = dataset.shuffle(
            buffer_size=len(dataframe),
            seed=RANDOM_SEED,
            reshuffle_each_iteration=True,
        )

    return dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


data_augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip(
            mode="horizontal_and_vertical"
        ),
        tf.keras.layers.RandomRotation(factor=0.5),
        tf.keras.layers.RandomZoom(
            height_factor=0.10,
            width_factor=0.10,
        ),
    ],
    name="data_augmentation",
)


def augment_training_dataset(dataset: tf.data.Dataset) -> tf.data.Dataset:
    """Apply augmentation to training images only."""
    return dataset.map(
        lambda images, labels: (
            data_augmentation(images, training=True),
            labels,
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    ).prefetch(tf.data.AUTOTUNE)


def save_training_preview(dataset: tf.data.Dataset) -> None:
    """Save a visual preview of augmented training images."""
    PREVIEW_FILE.parent.mkdir(parents=True, exist_ok=True)

    for images, labels in dataset.take(1):
        fig, axes = plt.subplots(3, 3, figsize=(9, 9))

        for ax, image, label in zip(
            axes.flat,
            images[:9],
            labels[:9],
        ):
            display_image = tf.clip_by_value(
                (image + 1.0) / 2.0,
                0.0,
                1.0,
            )

            ax.imshow(display_image.numpy())
            ax.set_title(CLASS_NAMES[int(label.numpy())])
            ax.axis("off")

        fig.suptitle("Augmented Training Samples", fontsize=14)
        fig.tight_layout()
        fig.savefig(PREVIEW_FILE, dpi=150, bbox_inches="tight")
        plt.close(fig)
        break


# Public datasets used by the training/evaluation scripts.
df = load_split_file()
train_df, validation_df, test_df = split_dataframe(df)

verify_no_group_leakage(
    train_df,
    validation_df,
    test_df,
)

train_dataset = augment_training_dataset(
    create_dataset(train_df, shuffle=True)
)
validation_dataset = create_dataset(validation_df, shuffle=False)
test_dataset = create_dataset(test_df, shuffle=False)

print("\n" + "=" * 60)
print("PREPROCESSING PIPELINE READY")
print("=" * 60)
print(f"Image size:      {TARGET_SIZE[0]} x {TARGET_SIZE[1]}")
print("Color format:    RGB")
print("Normalization:   MobileNetV2 preprocess_input [-1, 1]")
print(f"Batch size:      {BATCH_SIZE}")
print("Augmentation:    Training only")
print(f"Training images: {len(train_df)}")
print(f"Validation:      {len(validation_df)}")
print(f"Test images:     {len(test_df)}")
print("Group leakage:   None detected")
print("=" * 60)
