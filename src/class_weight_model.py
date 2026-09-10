"""Class-weighted fine-tuning of the MobileNetV2 BreakHis model.

The experiment fine-tunes the top 30 MobileNetV2 layers while using balanced
class weights calculated from the training split.
"""

from pathlib import Path

import tensorflow as tf

from preprocessing import train_dataset, train_df, validation_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "experiments"
    / "best_frozen_mobilenetv2.keras"
)
OUTPUT_PATH = (
    PROJECT_ROOT
    / "models"
    / "experiments"
    / "best_classweighted_30_finetune_layers.keras"
)

LEARNING_RATE = 1e-4
UNFREEZE_TOP_LAYERS = 30
EPOCHS = 20


def calculate_class_weights() -> dict[int, float]:
    """Calculate balanced class weights from the training split."""
    class_counts = train_df["label"].value_counts().sort_index()
    total = class_counts.sum()
    number_of_classes = len(class_counts)

    return {
        int(label): total / (number_of_classes * count)
        for label, count in class_counts.items()
    }


def find_mobilenet_base(model: tf.keras.Model) -> tf.keras.Model:
    """Return the MobileNetV2 backbone."""
    for layer in model.layers:
        if (
            isinstance(layer, tf.keras.Model)
            and "mobilenetv2" in layer.name.lower()
        ):
            return layer

    raise ValueError("Could not find the MobileNetV2 base model.")


def configure_fine_tuning(
    base_model: tf.keras.Model,
) -> tuple[int, int]:
    """Unfreeze the selected top layers while keeping BatchNorm frozen."""
    base_model.trainable = True

    fine_tune_start = len(base_model.layers) - UNFREEZE_TOP_LAYERS

    for layer in base_model.layers[:fine_tune_start]:
        layer.trainable = False

    for layer in base_model.layers[fine_tune_start:]:
        layer.trainable = True

    for layer in base_model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False

    trainable_count = sum(
        layer.trainable for layer in base_model.layers
    )

    return fine_tune_start, trainable_count


def compile_model(model: tf.keras.Model) -> None:
    """Compile the class-weighted fine-tuning model."""
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


def build_callbacks() -> list[tf.keras.callbacks.Callback]:
    """Create validation-AUC callbacks."""
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    return [
        tf.keras.callbacks.ModelCheckpoint(
            OUTPUT_PATH,
            monitor="val_auc",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc",
            mode="max",
            patience=6,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_auc",
            mode="max",
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1,
        ),
    ]


def main() -> None:
    """Run class-weighted fine-tuning and save the best checkpoint."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    class_weights = calculate_class_weights()

    print("\n" + "=" * 60)
    print("CLASS WEIGHTS")
    print("=" * 60)
    print(f"Benign weight:    {class_weights[0]:.4f}")
    print(f"Malignant weight: {class_weights[1]:.4f}")

    model = tf.keras.models.load_model(MODEL_PATH)
    base_model = find_mobilenet_base(model)

    frozen_until, trainable_count = configure_fine_tuning(base_model)
    compile_model(model)

    print("\n" + "=" * 60)
    print("CLASS-WEIGHTED FINE-TUNING")
    print("=" * 60)
    print(f"Base model:          {base_model.name}")
    print(f"Total base layers:   {len(base_model.layers)}")
    print(f"Frozen base layers:  {frozen_until}")
    print(f"Trainable layers:    {trainable_count}")
    print(f"Learning rate:       {LEARNING_RATE}")
    print(f"Epochs:              {EPOCHS}")

    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=EPOCHS,
        callbacks=build_callbacks(),
        class_weight=class_weights,
    )

    best_val_auc = max(history.history["val_auc"])
    best_epoch = history.history["val_auc"].index(best_val_auc) + 1

    print("\n" + "=" * 60)
    print("CLASS-WEIGHTED FINE-TUNING COMPLETED")
    print("=" * 60)
    print(f"Best validation AUC: {best_val_auc:.4f}")
    print(f"Best epoch:          {best_epoch}")
    print(f"Model saved to:      {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
