# 🔬 Histopathology Cancer Diagnosis

## Explainable Deep Learning for Automated Quantitative Analysis of Histopathology Images for Cancer Diagnosis

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-Latest-red?logo=keras&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

A deep-learning computer vision project for binary classification of breast histopathology images as **benign** or **malignant** using the **BreaKHis** dataset and **MobileNetV2 transfer learning**.

> ⚠️ **Important:** This project is intended for educational and research purposes only. It is **not a clinically validated diagnostic system** and must not be used for independent medical diagnosis.

---

## 📌 Project Overview

The original computer-vision challenge was to design a CNN capable of recognizing pneumonia in X-ray images. This project adapts the same methodology to a different medical-imaging problem:

> **Design and evaluate a CNN capable of recognizing malignant and benign breast histopathology images.**

### Objectives

- Train a CNN capable of recognizing previously unseen histopathology images.
- Apply rigorous training, validation, and testing methodology.
- Use transfer learning with a pretrained CNN.
- Perform hyperparameter and fine-tuning experiments.
- Address class imbalance using class weighting.
- Evaluate the final model using multiple classification metrics.
- Generate training and evaluation visualizations.
- Analyze false positives and false negatives.
- Explore explainability and future feature-map visualization.
- Demonstrate why leakage prevention and careful test-set discipline are important in medical machine learning.

### Key Features

- ✨ **Transfer Learning** — MobileNetV2 pretrained on ImageNet
- 🎯 **Binary Classification** — Benign vs. Malignant
- 🔒 **Group-Level Splitting** — Prevents related cases/slides from appearing across different splits
- ⚖️ **Class Weighting** — Compensates for class imbalance
- 📈 **Comprehensive Evaluation** — Accuracy, Precision, Recall, F1-score, ROC-AUC, PR curve, and confusion matrix
- 🔍 **Error Analysis** — Visual investigation of false positives and false negatives
- 💡 **Explainability Focus** — Foundation for feature-map and Grad-CAM analysis

---

# 📊 Dataset

## BreaKHis — Breast Cancer Histopathological Image Classification

The project uses the **BreaKHis** dataset, a publicly available collection of breast histopathology images developed in collaboration with the **P&D Laboratory – Pathological Anatomy and Cytopathology, Paraná, Brazil**.

### Dataset Statistics

| Metric | Value |
|---|---:|
| Total images | **7,909** |
| Benign images | **2,480** |
| Malignant images | **5,429** |
| Patients | **82** |
| Magnifications | **40X, 100X, 200X, 400X** |
| Image representation | **224 × 224 × 3 RGB** |

The project uses the 7,909-image BreaKHis 1.0 set. The broader BreaKHis database is reported as containing 9,109 images, while the publicly downloadable set used here contains 7,909 images.

### Data Sources

- **Official BreaKHis database:**  
  https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/
- **Kaggle dataset:**  
  https://www.kaggle.com/datasets/waseemalastal/breakhis-breast-cancer-histopathological-dataset

---

## 🧬 Benign vs. Malignant

This project performs **binary classification**:

| Class | Label |
|---|---:|
| Benign | **0** |
| Malignant | **1** |

The model produces a single probability representing the likelihood that an image belongs to the malignant class.

### Histological Subtypes

Although the model performs binary classification, BreaKHis contains eight histological types.

**Benign**

1. Adenosis
2. Fibroadenoma
3. Phyllodes Tumor
4. Tubular Adenoma

**Malignant**

5. Ductal Carcinoma
6. Lobular Carcinoma
7. Mucinous Carcinoma
8. Papillary Carcinoma

A future extension could investigate the more difficult **eight-class classification problem**.

---

## 🔬 Magnification Levels

BreaKHis images are available at four magnifications:

- **40X**
- **100X**
- **200X**
- **400X**

| Magnification | Benign | Malignant | Total |
|---|---:|---:|---:|
| 40X | 652 | 1,370 | 1,995 |
| 100X | 644 | 1,437 | 2,081 |
| 200X | 623 | 1,390 | 2,013 |
| 400X | 588 | 1,232 | 1,820 |
| **Total** | **2,480** | **5,429** | **7,909** |

---

# 🔒 Dataset Split and Leakage Prevention

Histopathology datasets can contain multiple images originating from the same underlying case or slide. Randomly splitting images can therefore cause **data leakage**, where visually related images appear in both training and evaluation sets.

This project uses a **group-level split** based on the case/slide identifier derived from the image metadata.

### Final Split

| Split | Images |
|---|---:|
| Training | **5,303** |
| Validation | **1,120** |
| Testing | **1,486** |
| **Total** | **7,909** |

All images belonging to the same group are assigned to exactly one split.

### Leakage Checks

- Train/validation group overlap: **0**
- Train/test group overlap: **0**
- Validation/test group overlap: **0**

This is an important methodological strength because it makes validation and test performance a more meaningful estimate of generalization.

The reproducible split is stored in:

```text
data/processed/breakhis_split.csv
```

The split uses `random_state=42`.

---

# ⚙️ Preprocessing

The original challenge suggested OpenCV preprocessing. This project instead uses **TensorFlow/Keras and PIL**, integrating the preprocessing directly with the MobileNetV2 workflow.

### Pipeline

```text
Input image
     │
     ▼
Load image
     │
     ▼
Convert to RGB
     │
     ▼
Resize to 224 × 224
     │
     ▼
MobileNetV2 preprocess_input
     │
     ▼
Approximately [-1, 1]
     │
     ▼
TensorFlow dataset
     │
     ▼
Batching + Prefetching
```

### Important: Model Input vs. Visualization

MobileNetV2 preprocessing converts image values to approximately:

```text
[-1, 1]
```

The model receives these values during training, validation, and testing.

For visualization only, the transformation is approximately reversed:

```python
(image + 1) / 2
```

and clipped to `[0, 1]`.

Therefore:

```text
Model input:    [-1, 1]
Visualization:  [0, 1]
```

The model is **not trained on `[0, 1]` images**.

---

# 🔄 Data Augmentation

Augmentation is applied **only to the training dataset**.

Current augmentation:

- Horizontal flipping
- Vertical flipping
- Random rotation with `factor=0.5`
- Random zoom of 10%

Validation and test images remain unaugmented.

> **Note:** The rotation factor is relatively large. It was preserved as part of the recorded experiment rather than silently changed during code cleanup. Changing augmentation strength should be treated as a new experiment.

---

# 🧠 Model Architecture

## MobileNetV2 Transfer Learning

The project uses **MobileNetV2 pretrained on ImageNet** as the feature extractor.

The original classification head is replaced with a small binary classification head:

```text
Input Image
224 × 224 × 3
      │
      ▼
MobileNetV2
ImageNet pretrained
      │
      ▼
Global Average Pooling
      │
      ▼
Dropout(0.5)
      │
      ▼
Dense(1)
      │
      ▼
Sigmoid
      │
      ▼
P(Malignant)
```

The sigmoid output is appropriate for the two-class problem, where malignant is represented as class `1`.

---

# 🔁 Transfer Learning and Fine-Tuning

The training workflow consists of two stages.

### Stage 1 — Frozen Baseline

- MobileNetV2 convolutional base is frozen.
- Only the newly added classification head is trained.
- Adam optimizer is used.
- Binary cross-entropy is used as the loss.
- Validation AUC is used for model comparison.

### Stage 2 — Fine-Tuning

Selected layers of MobileNetV2 are unfrozen and fine-tuned.

Batch Normalization layers are kept frozen for training stability.

```text
ImageNet-pretrained MobileNetV2
              │
              ▼
      Freeze base model
              │
              ▼
   Train classification head
              │
              ▼
      Evaluate validation
              │
              ▼
   Unfreeze selected layers
              │
              ▼
          Fine-tune
              │
              ▼
      Evaluate validation
              │
              ▼
      Select final model
              │
              ▼
     Test once on test set
```

---

# ⚖️ Class Weighting

The dataset is imbalanced, with more malignant than benign images.

The strongest experiment uses approximately:

```text
Benign:      1.6606
Malignant:   0.7156
```

Class weighting changes the contribution of each class to the training loss.

The final experiment combines:

- **Class weighting**
- **30-layer fine-tuning**
- **Learning rate = 1e-4**

> ⚠️ Because class weighting and fine-tuning depth were changed together, the improvement cannot be attributed to class weighting alone. The reported result represents the **combined configuration**.

---

# 🧪 Hyperparameter Experiments

Several configurations were evaluated using validation AUC.

| Experiment | Best Validation AUC |
|---|---:|
| Frozen MobileNetV2, low LR (~1e-6) | ~0.567 |
| Frozen MobileNetV2, LR 1e-4 | ~0.565 |
| Frozen MobileNetV2, LR 0.001 | ~0.7003 |
| 30-layer fine-tuning, no class weights | ~0.6246 |
| 10-layer fine-tuning, no class weights | **0.7006** |
| Class-weighted + 30-layer fine-tuning | ~0.8112 |
| Class-weighted + 15-layer fine-tuning | **0.8002** |
| Class-weighted + 30-layer fine-tuning rerun | **0.8208** |

### Selected Model

```text
best_classweighted_30_finetune_layers.keras
```

**Best validation AUC: 0.8208**

The selected model is the **class-weighted 30-layer fine-tuned MobileNetV2**.

---

# 🧪 Evaluation Methodology

The test set is kept completely separate from model selection.

```text
Training Set
     │
     ▼
Train model
     │
     ▼
Validation Set
     │
     ▼
Hyperparameter experiments
     │
     ▼
Select best model
     │
     ▼
Held-out Test Set
     │
     ▼
Final evaluation
```

### Test-Set Discipline

- Training data is used to fit model parameters.
- Validation data is used to compare experiments and select the model.
- The test set is reserved for final evaluation.
- The final test results must not be used to tune architecture, learning rate, class weights, or threshold.

This is particularly important for medical imaging, where repeatedly adapting a model based on test performance can effectively turn the test set into an indirect training resource.

---

# 🏆 Final Test Results

The selected model was evaluated on **1,486 previously unseen test images**.

| Metric | Result |
|---|---:|
| Accuracy | **0.6743** |
| Malignant Precision | **0.7860** |
| Malignant Recall | **0.6016** |
| Malignant F1-score | **0.6816** |
| ROC-AUC | **0.7622** |

### Validation vs. Test

```text
Validation AUC: 0.8208
Test AUC:       0.7622
Difference:     0.0586
```

The gap indicates that validation performance did not fully transfer to the held-out test set and should be considered when discussing model generalization.

---

# 📊 Confusion Matrix

```text
                         Predicted
                    Benign    Malignant
Actual Benign         484        141
Actual Malignant      343        518
```

Therefore:

```text
True Negatives  = 484
False Positives = 141
False Negatives = 343
True Positives  = 518
```

---

# 🏥 Medical Interpretation

Medical image classification should not be evaluated using accuracy alone because different types of errors can have different consequences.

## False Negatives

A false negative occurs when:

```text
Actual:    Malignant
Predicted: Benign
```

The final model produced **343 false negatives**.

The malignant recall was:

```text
0.6016
```

This means the model identified approximately 60% of the malignant test images at the evaluation threshold, while a substantial number of malignant images were classified as benign.

In a real diagnostic setting, missed malignant cases could be particularly consequential. Therefore, **malignant recall/sensitivity** deserves close attention.

## False Positives

A false positive occurs when:

```text
Actual:    Benign
Predicted: Malignant
```

The final model produced **141 false positives**.

Malignant precision was:

```text
0.7860
```

This indicates that a relatively high proportion of images predicted as malignant were actually malignant, although some benign images were still incorrectly classified.

## Important Metrics

For a medical classification problem, no single metric is sufficient.

- **Recall / Sensitivity** — proportion of actual malignant samples detected.
- **Precision** — proportion of malignant predictions that are actually malignant.
- **F1-score** — balances precision and recall.
- **ROC-AUC** — measures discrimination across classification thresholds.
- **Precision-Recall curve** — shows the precision/recall trade-off.
- **Confusion matrix** — shows TP, TN, FP, and FN directly.

In a real clinical application, the classification threshold should be selected according to the clinical consequences of false negatives and false positives rather than automatically assuming that `0.5` is optimal.

---

# 📈 Visualizations

The project generates the following evaluation artifacts.

## Training Curves

```text
results/training_curves/
├── loss_training_curve.png
├── accuracy_training_curve.png
├── precision_training_curve.png
├── recall_training_curve.png
└── auc_training_curve.png
```

These help investigate:

- convergence
- overfitting
- underfitting
- training stability
- validation performance

## Evaluation

```text
results/evaluation/
├── final_test_confusion_matrix.png
├── final_test_roc_curve.png
└── final_test_precision_recall_curve.png
```

The ROC curve illustrates the sensitivity/specificity trade-off across thresholds.

The Precision-Recall curve illustrates the precision/recall trade-off, which is particularly informative when the positive class is important and the classes are imbalanced.

These plots are descriptive evaluation artifacts and should not be used to tune the final test threshold after observing test performance.

---

# 🔍 Error Analysis

The project performs visual analysis of incorrectly classified test images.

### False Positives

```text
Benign → Malignant
```

### False Negatives

```text
Malignant → Benign
```

Results are stored in:

```text
results/error_analysis/
```

The analysis can help investigate whether errors are associated with:

- visually similar tissue structures
- staining variations
- magnification
- difficult histological patterns
- image artifacts
- ambiguous cases

The most confident errors can also be selected for inspection, such as:

- high malignant probability among false positives
- low malignant probability among false negatives

> ⚠️ Test-set errors are used for qualitative analysis only and should not be used to tune the model.

---

# 💡 Explainability

The project uses the term **Explainable Deep Learning** because understanding model predictions is particularly important in medical applications.

The current project includes:

- quantitative evaluation
- confusion-matrix analysis
- false-positive analysis
- false-negative analysis
- model-performance visualization

### Future Explainability

Potential extensions include:

- Feature-map visualization
- Grad-CAM
- Grad-CAM++
- Saliency maps
- Activation visualization

These techniques could help investigate which image regions contribute most strongly to a model prediction.

---

# 💻 Installation

## Prerequisites

- Python 3.8+
- pip or conda
- Approximately 2 GB free disk space for the dataset

## 1. Clone the Repository

```bash
git clone https://github.com/husseinabuammar24-cloud/CV_Histopath_Cancer_Diagnosis.git
cd CV_Histopath_Cancer_Diagnosis
```

## 2. Create a Virtual Environment

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## 4. Download the Dataset

Download BreaKHis from Kaggle or the official UFPR database and place it under:

```text
data/raw/
```

The raw dataset is intentionally not committed to GitHub because of its size.

---

# 🚀 Usage

Run the scripts sequentially.

## 1. Analyze the Dataset

```bash
python src/data_analysis.py
```

This analyzes:

- image classes
- class distribution
- magnification levels
- groups/cases
- dataset structure
- potential leakage

## 2. Create the Train/Validation/Test Split

```bash
python src/train_val_test.py
```

Creates:

```text
data/processed/breakhis_split.csv
```

The split is performed at group level.

## 3. Prepare the Preprocessing Pipeline

```bash
python src/preprocessing.py
```

Creates TensorFlow datasets, applies MobileNetV2 preprocessing, and prepares training augmentation.

## 4. Train the Baseline Model

```bash
python src/train_model.py
```

This trains the initial frozen MobileNetV2 model.

Output:

```text
models/experiments/frozen_mobilenetv2.keras
```

## 5. Fine-Tune the Model

### 10-layer fine-tuning

```bash
python src/finetune_model.py
```

### Class-weighted 30-layer fine-tuning

```bash
python src/class_weight_model.py
```

The latter corresponds to the selected final configuration.

## 6. Evaluate the Final Model

```bash
python src/test_evaluation.py
```

Generates:

- accuracy
- precision
- recall
- F1-score
- ROC-AUC
- confusion matrix
- ROC curve
- Precision-Recall curve

## 7. Analyze Errors

```bash
python src/error_analysis.py
```

Identifies and visualizes false positives and false negatives.

---

# 📁 Repository Structure

```text
CV_Histopath_Cancer_Diagnosis/
│
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
├── BreakHis_Project_Future_Reference_Documentation.pdf
│
├── data/
│   ├── raw/                         # Local dataset; not committed
│   │   └── BreakHis/
│   └── processed/
│       └── breakhis_split.csv
│
├── models/
│   └── experiments/
│       ├── frozen_mobilenetv2.keras
│       ├── 10layer_finetune.keras
│       ├── 30layer_finetune.keras
│       └── best_classweighted_30_finetune_layers.keras
│
├── results/
│   ├── training_curves/
│   ├── evaluation/
│   ├── error_analysis/
│   └── preprocessing/
│
└── src/
    ├── data_analysis.py
    ├── train_val_test.py
    ├── preprocessing.py
    ├── train_model.py
    ├── finetune_model.py
    ├── class_weight_model.py
    ├── test_evaluation.py
    └── error_analysis.py
```

### Script Responsibilities

| Script | Purpose |
|---|---|
| `data_analysis.py` | Dataset inspection, statistics, magnification and group analysis |
| `train_val_test.py` | Leakage-safe group-level split |
| `preprocessing.py` | TensorFlow datasets, preprocessing and augmentation |
| `train_model.py` | Frozen MobileNetV2 baseline |
| `finetune_model.py` | Fine-tuning experiment |
| `class_weight_model.py` | Class-weighted fine-tuning experiment |
| `test_evaluation.py` | Final held-out test evaluation |
| `error_analysis.py` | False-positive and false-negative analysis |

---

# 🗓️ Project Timeline

## Day 1 — Dataset and Model Development

- Download BreaKHis dataset
- Inspect folder structure
- Analyze images, patients, classes and subtypes
- Analyze magnification levels
- Create group-level train/validation/test split
- Implement preprocessing
- Implement data augmentation
- Build MobileNetV2 transfer-learning model
- Train frozen baseline
- Begin hyperparameter experiments

## Day 2 — Optimization and Evaluation

- Fine-tune MobileNetV2
- Experiment with different numbers of trainable layers
- Experiment with class weighting
- Select the best model using validation performance
- Evaluate once on the held-out test set
- Generate training curves
- Generate confusion matrix
- Generate ROC and Precision-Recall curves
- Analyze false positives and false negatives
- Document the project
- Prepare the GitHub repository

---

# ⚠️ Limitations

The final model demonstrates useful classification ability, but its performance is **not sufficient for independent clinical diagnosis**.

In particular:

- 343 malignant test images were classified as benign.
- Test ROC-AUC (0.7622) is lower than validation AUC (0.8208).
- The dataset is limited to the available BreaKHis population.
- External generalization has not been established.
- The classification threshold has not been optimized using an independent clinical objective.
- Probability calibration has not been performed.
- Advanced explainability methods have not yet been implemented.

A real clinical system would require, among other things:

- independent external validation
- evaluation on additional patient populations
- careful threshold selection
- probability calibration
- robustness testing
- explainability
- clinical validation
- comparison with qualified medical professionals
- appropriate regulatory approval

The current system should therefore be considered an:

> **Educational and research prototype**

rather than a medical diagnostic device.

---

# ⭐ Future Work / Nice-to-Have Features

### 1. Feature-Map Visualization

Inspect intermediate MobileNetV2 activations to understand what visual structures are detected at different network stages.

**Status:** Not yet implemented.

### 2. Grad-CAM / Advanced Explainability

Implement Grad-CAM, Grad-CAM++, saliency maps, or related methods to visualize image regions influencing predictions.

**Status:** Not yet implemented.

### 3. Compare CNN Architectures

Compare MobileNetV2 with architectures such as:

- ResNet
- EfficientNet
- VGG
- other MobileNet variants

Potential comparison metrics:

| Metric | MobileNetV2 | ResNet | EfficientNet |
|---|---:|---:|---:|
| Accuracy | | | |
| Precision | | | |
| Recall | | | |
| F1-score | | | |
| ROC-AUC | | | |
| Parameters | | | |
| Training time | | | |

**Status:** Not yet implemented.

### 4. Threshold Optimization

If threshold optimization is pursued, it should be performed using the **validation set**, not the final test set. Once selected and frozen, the threshold can be applied once to the held-out test set.

### 5. Eight-Class Classification

Extend the binary task to classification of the eight BreaKHis histological subtypes.

---

# ✅ Challenge Checklist

## Must-Have

- [x] CNN trained on the dataset
- [x] Classification of unseen images
- [x] Training/validation/test split
- [x] Group-level splitting
- [x] Leakage checks
- [x] Transfer learning
- [x] Hyperparameter experimentation
- [x] Class weighting experiment
- [x] Confusion matrix
- [x] Accuracy
- [x] Precision
- [x] Recall
- [x] F1-score
- [x] ROC-AUC
- [x] ROC curve
- [x] Precision-Recall curve
- [x] Training/validation curves
- [x] False-positive analysis
- [x] False-negative analysis
- [x] Discussion of medical evaluation metrics

## Nice-to-Have

- [ ] Feature-map visualization
- [ ] Grad-CAM / advanced explainability
- [ ] Comparison with other CNN architectures
- [ ] Eight-class classification

---

# 📚 Sources and References

### Dataset

1. **BreaKHis Official Database — Universidade Federal do Paraná**  
   https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/

2. **BreaKHis on Kaggle**  
   https://www.kaggle.com/datasets/waseemalastal/breakhis-breast-cancer-histopathological-dataset

3. **Original BreaKHis Research Paper**  
   Spanhol, F. A., Oliveira, L. S., Petitjean, C., & Heutte, L.  
   *A Dataset for Breast Cancer Histopathological Image Classification.*  
   IEEE Transactions on Biomedical Engineering, 63(7), 1455–1462, 2016.

4. **BreaKHis 224×224 Kaggle Dataset Version**  
   https://www.kaggle.com/datasets/tathagatbanerjee/breakhis-breast-cancer-histopathological

---

# 🔬 End-to-End Workflow

```text
BreaKHis Dataset
       │
       ▼
Dataset Inspection
       │
       ▼
Group-Level Split
       │
       ▼
Preprocessing
       │
       ▼
Training Augmentation
       │
       ▼
MobileNetV2 Transfer Learning
       │
       ▼
Hyperparameter Experiments
       │
       ▼
Fine-Tuning + Class Weighting
       │
       ▼
Validation-Based Model Selection
       │
       ▼
Held-Out Test Evaluation
       │
       ▼
ROC / PR / Confusion Matrix
       │
       ▼
False Positive / False Negative Analysis
       │
       ▼
Explainability / Future Work
```

---

# 📌 Key Results at a Glance

| Item | Value |
|---|---|
| Dataset | BreaKHis |
| Total images | **7,909** |
| Train / Validation / Test | **5,303 / 1,120 / 1,486** |
| Classes | **Benign / Malignant** |
| Input | **224 × 224 RGB** |
| Model | **MobileNetV2 + GAP + Dropout(0.5) + Dense(1, sigmoid)** |
| Fine-tuning | **30 layers** |
| Class weighting | **Yes** |
| Learning rate | **1e-4** |
| Best validation AUC | **0.8208** |
| Test accuracy | **0.6743** |
| Test malignant precision | **0.7860** |
| Test malignant recall | **0.6016** |
| Test malignant F1 | **0.6816** |
| Test ROC-AUC | **0.7622** |
| TN / FP / FN / TP | **484 / 141 / 343 / 518** |

---

# 👤 Project Context

This project was developed as part of a **Computer Vision consolidation challenge**. The original challenge focused on pneumonia detection in chest X-rays; the methodology was adapted here to breast histopathology classification.

The project provided practical experience with:

- medical image classification
- convolutional neural networks
- transfer learning
- MobileNetV2
- image preprocessing
- data augmentation
- hyperparameter tuning
- class weighting
- model evaluation
- error analysis
- explainable AI concepts

A particular focus was placed on understanding that medical machine-learning systems should not be evaluated using accuracy alone and that **data leakage prevention, model-selection discipline, multiple evaluation metrics, visualization, and error analysis** are essential.

---

## Made with ❤️ for learning and research

If you find the project useful, consider ⭐ starring the repository.

Repository:  
https://github.com/husseinabuammar24-cloud/CV_Histopath_Cancer_Diagnosis
