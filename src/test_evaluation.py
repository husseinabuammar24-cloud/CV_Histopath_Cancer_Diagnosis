"""Final evaluation of the selected BreakHis model on the test set.

The test set is evaluated only after model selection. No test-set result is
used for further tuning.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from preprocessing import test_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = PROJECT_ROOT / "models" / "final_model.keras"
RESULTS_DIR = PROJECT_ROOT / "results" / "evaluation"

THRESHOLD = 0.5


def collect_predictions(
    model: tf.keras.Model,
) -> tuple[np.ndarray, np.ndarray]:
    """Collect true labels and malignant probabilities from the test set."""
    y_true = []
    y_prob = []

    for images, labels in test_dataset:
        probabilities = model.predict(images, verbose=0).ravel()

        y_true.extend(labels.numpy())
        y_prob.extend(probabilities)

    return (
        np.asarray(y_true, dtype=int),
        np.asarray(y_prob, dtype=float),
    )


def save_confusion_matrix(cm: np.ndarray) -> None:
    """Save the final confusion matrix."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(cm)

    ax.set_title("Confusion Matrix - Final Test Set")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")

    ax.set_xticks([0, 1], ["Benign", "Malignant"])
    ax.set_yticks([0, 1], ["Benign", "Malignant"])

    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center",
            )

    fig.tight_layout()
    fig.savefig(
        RESULTS_DIR / "final_test_confusion_matrix.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def save_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    auc: float,
) -> None:
    """Save the ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(
        fpr,
        tpr,
        label=f"MobileNetV2 (AUC = {auc:.4f})",
    )
    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Random classifier",
    )

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve - Final Test Set")
    ax.legend()
    ax.grid(True)

    fig.tight_layout()
    fig.savefig(
        RESULTS_DIR / "final_test_roc_curve.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def save_precision_recall_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> None:
    """Save the precision-recall curve."""
    precision, recall, _ = precision_recall_curve(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(
        recall,
        precision,
        label="MobileNetV2",
    )

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve - Final Test Set")
    ax.legend()
    ax.grid(True)

    fig.tight_layout()
    fig.savefig(
        RESULTS_DIR / "final_test_precision_recall_curve.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def main() -> None:
    """Load the final model, evaluate it, and save evaluation plots."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Final model not found:\n{MODEL_PATH}"
        )

    model = tf.keras.models.load_model(MODEL_PATH)

    y_true, y_prob = collect_predictions(model)
    y_pred = (y_prob >= THRESHOLD).astype(int)

    accuracy = float(np.mean(y_true == y_pred))
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_prob)

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    save_confusion_matrix(cm)
    save_roc_curve(y_true, y_prob, auc)
    save_precision_recall_curve(y_true, y_prob)

    print("\n" + "=" * 60)
    print("FINAL TEST EVALUATION")
    print("=" * 60)
    print(f"Model:        {MODEL_PATH}")
    print(f"Test samples: {len(y_true)}")
    print(f"Threshold:    {THRESHOLD:.2f}")

    print("\nMetrics:")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")
    print(f"AUC      : {auc:.4f}")

    print("\nConfusion matrix:")
    print(cm)
    print(f"TN={tn}, FP={fp}, FN={fn}, TP={tp}")

    print("\nClassification report:")
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=["Benign", "Malignant"],
            digits=4,
            zero_division=0,
        )
    )

    print("\nSaved evaluation plots to:")
    print(RESULTS_DIR)


if __name__ == "__main__":
    main()
