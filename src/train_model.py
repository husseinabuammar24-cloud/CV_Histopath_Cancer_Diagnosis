"""Train the frozen MobileNetV2 baseline on BreakHis.

Only the new binary classification head is trained; the ImageNet-pretrained
MobileNetV2 backbone remains frozen.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import tensorflow as tf

from preprocessing import train_dataset, validation_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_OUTPUT = (
    PROJECT_ROOT
    / "models"
    / "experiments"
    / "best_frozen_mobilenetv2.keras"
)
CURVE_OUTPUT_DIR = PROJECT_ROOT / "results" / "training_curves"

LEARNING_RATE = 1e-3
EPOCHS = 20


def build_model() -> tf.keras.Model:
    """Build and compile the frozen MobileNetV2 classifier."""
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    model = tf.keras.Sequential(
        [
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ],
        name="MobileNetV2_BreakHis",
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=LEARNING_RATE
        ),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )

    return model


def build_callbacks() -> list[tf.keras.callbacks.Callback]:
    """Create callbacks based on validation AUC."""
    MODEL_OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    return [
        tf.keras.callbacks.ModelCheckpoint(
            MODEL_OUTPUT,
            monitor="val_auc",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc",
            mode="max",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_auc",
            mode="max",
            factor=0.5,
            patience=2,
            min_lr=1e-7,
            verbose=1,
        ),
    ]


def plot_training_history(history: tf.keras.callbacks.History) -> None:
    """Save training/validation curves for tracked metrics."""
    CURVE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for metric in ("loss", "accuracy", "precision", "recall", "auc"):
        fig, ax = plt.subplots(figsize=(8, 5))

        ax.plot(
            history.history[metric],
            label=f"Training {metric}",
        )
        ax.plot(
            history.history[f"val_{metric}"],
            label=f"Validation {metric}",
        )

        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric.capitalize())
        ax.set_title(
            f"Training vs Validation {metric.capitalize()}"
        )
        ax.legend()
        ax.grid(True)
        fig.tight_layout()

        fig.savefig(
            CURVE_OUTPUT_DIR / f"{metric}_training_curve.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)


def main() -> None:
    """Train the baseline model and save its best checkpoint."""
    model = build_model()
    model.summary()

    print("\nStarting frozen MobileNetV2 training...")

    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=EPOCHS,
        callbacks=build_callbacks(),
    )

    plot_training_history(history)

    best_val_auc = max(history.history["val_auc"])
    best_epoch = history.history["val_auc"].index(best_val_auc) + 1

    print("\n" + "=" * 60)
    print("FROZEN MOBILENETV2 TRAINING COMPLETED")
    print("=" * 60)
    print(f"Best validation AUC: {best_val_auc:.4f}")
    print(f"Best epoch:          {best_epoch}")
    print(f"Model saved to:      {MODEL_OUTPUT}")
    print(f"Curves saved to:     {CURVE_OUTPUT_DIR}")


if __name__ == "__main__":
    main()
