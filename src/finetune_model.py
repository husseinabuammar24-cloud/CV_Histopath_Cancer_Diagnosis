"""Fine-tune the top 10 MobileNetV2 layers on BreakHis."""

from pathlib import Path

import tensorflow as tf

from preprocessing import train_dataset, validation_dataset


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
    / "best_finetuned_10layers.keras"
)

LEARNING_RATE = 1e-5
UNFREEZE_TOP_LAYERS = 10
EPOCHS = 20


def find_mobilenet_base(model: tf.keras.Model) -> tf.keras.Model:
    """Return the MobileNetV2 backbone."""
    for layer in model.layers:
        if (
            isinstance(layer, tf.keras.Model)
            and "mobilenetv2" in layer.name.lower()
        ):
            return layer

    raise ValueError("Could not find the MobileNetV2 base model.")


def configure_fine_tuning(base_model: tf.keras.Model) -> int:
    """Freeze all layers except the selected top layers and BatchNorm."""
    base_model.trainable = True

    for layer in base_model.layers:
        layer.trainable = False

    for layer in base_model.layers[-UNFREEZE_TOP_LAYERS:]:
        layer.trainable = True

    for layer in base_model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False

    return sum(layer.trainable for layer in base_model.layers)


def compile_model(model: tf.keras.Model) -> None:
    """Compile the fine-tuning model."""
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


def main() -> None:
    """Fine-tune the frozen baseline model."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    model = tf.keras.models.load_model(MODEL_PATH)
    base_model = find_mobilenet_base(model)
    trainable_count = configure_fine_tuning(base_model)

    compile_model(model)

    print("\n" + "=" * 60)
    print("10-LAYER FINE-TUNING")
    print("=" * 60)
    print(f"Base model:           {base_model.name}")
    print(f"Total base layers:    {len(base_model.layers)}")
    print(f"Requested top layers: {UNFREEZE_TOP_LAYERS}")
    print(f"Trainable layers:     {trainable_count}")
    print(f"Learning rate:        {LEARNING_RATE}")
    print(f"Epochs:               {EPOCHS}")

    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=EPOCHS,
        callbacks=build_callbacks(),
    )

    best_val_auc = max(history.history["val_auc"])
    best_epoch = history.history["val_auc"].index(best_val_auc) + 1

    print("\n" + "=" * 60)
    print("10-LAYER FINE-TUNING COMPLETED")
    print("=" * 60)
    print(f"Best validation AUC: {best_val_auc:.4f}")
    print(f"Best epoch:          {best_epoch}")
    print(f"Model saved to:      {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
