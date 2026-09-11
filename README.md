# BreakHis Breast Cancer Classification

## 📌 Description

This project classifies breast histopathology images as **benign** or **malignant** using the BreakHis dataset. It uses transfer learning with a MobileNetV2 backbone (pretrained on ImageNet), with a strong focus on proper evaluation: leakage-safe data splitting and close attention to false negatives, since missing a malignant case matters most in a medical context.

---

## 🚀 Installation

```bash
git clone https://github.com/<your-username>/CV_Histopath_Cancer_Diagnosis.git
cd CV_Histopath_Cancer_Diagnosis

python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

Download the [BreakHis dataset](https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/) and place it under `data/raw/BreakHis/` (the raw dataset is git-ignored).

---

## 🧪 Usage

```bash
python src/data_analysis.py       # explore the dataset
python src/train_val_test.py      # create the leakage-safe split
python src/train_model.py         # train the frozen MobileNetV2 baseline
python src/finetune_model.py      # fine-tune the top layers
python src/class_weight_model.py  # class-weighted fine-tuning
python src/test_evaluation.py     # final evaluation on the test set
python src/error_analysis.py      # inspect misclassifications
```

> ⚠️ The test set is evaluated only after model selection, and never used for further tuning.

---

## 📊 Visuals

| Visualization | Location |
|---|---|
| Training curves | `results/training_curves/` |
| Confusion matrix | `results/evaluation/final_test_confusion_matrix.png` |
| ROC curve | `results/evaluation/final_test_roc_curve.png` |
| Precision-recall curve | `results/evaluation/final_test_precision_recall_curve.png` |
| False positive / negative grids | `results/error_analysis/` |

---

## ⚠️ Important notes

* **Class imbalance** — the dataset is not perfectly balanced between benign and malignant cases; the strongest experiment uses balanced class weights during training to correct for this.
* **Group-level splitting** — images from the same case/slide are kept together in one split only, to avoid data leakage between train, validation, and test.
* **Validation vs. test gap** — validation AUC (0.8208) is noticeably higher than final test AUC (0.7622), a reminder that validation performance doesn't always fully transfer.
* **Accuracy is not enough** — for a medical task, recall on the malignant class matters more than overall accuracy, since a false negative is more costly than a false positive.
* **Combined experiment** — the best result combines class weighting and fine-tuning depth together, so their individual effects can't be separated from this experiment alone.
* **No test-set tuning** — the test set is used exactly once, after model selection, and is never used to adjust hyperparameters or thresholds.

---

## 👥 Contributors

* [Your name](https://github.com/your-username)
* [Teammate's name](https://github.com/teammate-username)

---

## 🗓️ Timeline

* **Day 1** — Dataset exploration, leakage-safe split, preprocessing, frozen baseline.
* **Day 2** — Fine-tuning experiments, final model selection, evaluation, error analysis.

---

## 🙋 Personal situation

*(Short note on your background going into the challenge and any constraints you worked under.)*
