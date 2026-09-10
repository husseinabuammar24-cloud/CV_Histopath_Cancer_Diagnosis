# 🔬 Histopathology Cancer Diagnosis

<div align="center">

**Explainable Deep Learning for Automated Quantitative Analysis of Histopathology Images for Cancer Diagnosis**

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg?logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-Latest-red.svg?logo=keras&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Active-brightgreen.svg)

[Project Overview](#-project-overview) • [Dataset](#-dataset) • [Installation](#-installation) • [Usage](#-usage) • [Results](#-results) • [Contributors](#-contributors)

</div>

---

## 📌 Project Overview

This project develops a **deep-learning computer vision system** for the automated classification of breast histopathology images into **benign** and **malignant** categories.

### Key Features

✨ **Transfer Learning** — MobileNetV2 pretrained on ImageNet  
🎯 **Binary Classification** — Benign vs. Malignant tissue detection  
📊 **Group-Level Splitting** — Prevents data leakage between train/val/test sets  
📈 **Comprehensive Evaluation** — Accuracy, Precision, Recall, F1-score, ROC-AUC, and more  
🔍 **Error Analysis** — Visual investigation of false positives and false negatives  
💡 **Explainability Focus** — Understanding model predictions in medical contexts  
⚖️ **Class Weighting** — Handles imbalanced dataset effectively  

### Motivation

The original challenge was to design a CNN for recognizing pneumonia in X-ray images. This project adapts that methodology to a different medical-imaging problem:

> **Design a CNN capable of recognizing malignant and benign breast histopathology images from the BreaKHis dataset.**

### Objectives

- Train a CNN capable of recognizing previously unseen histopathology images
- Apply rigorous training, validation, and testing methodology
- Use transfer learning with pretrained CNNs
- Perform hyperparameter and fine-tuning experiments
- Evaluate using multiple classification metrics beyond accuracy
- Generate clearly labeled visualizations and confusion matrices
- Analyze false positives and false negatives
- Discuss the importance of evaluation metrics in medical environments
- Explore explainability and feature-map visualization

> **⚠️ Important:** This project is intended for **educational and research purposes only**. It is **not** a clinically validated diagnostic system.

---

## 📊 Dataset

### BreaKHis — Breast Cancer Histopathological Image Classification

The project uses the **BreaKHis** dataset, a publicly available collection of breast histopathology images developed in collaboration with the **P&D Laboratory – Pathological Anatomy and Cytopathology, Paraná, Brazil**.

#### Dataset Statistics

| Metric | Value |
|--------|-------|
| **Total Images** | 7,909 |
| **Benign Images** | 2,480 |
| **Malignant Images** | 5,429 |
| **Number of Patients** | 82 |
| **Magnification Levels** | 40X, 100X, 200X, 400X |
| **Image Dimensions** | 224 × 224 × 3 (RGB) |

#### Magnification Distribution

| Magnification | Benign | Malignant | Total |
|---------------|--------|-----------|-------|
| 40X | 652 | 1,370 | 1,995 |
| 100X | 644 | 1,437 | 2,081 |
| 200X | 623 | 1,390 | 2,013 |
| 400X | 588 | 1,232 | 1,820 |
| **Total** | **2,480** | **5,429** | **7,909** |

#### Histological Subtypes

**Benign (4 types):**
- Adenosis
- Fibroadenoma
- Phyllodes Tumor
- Tubular Adenoma

**Malignant (4 types):**
- Ductal Carcinoma
- Lobular Carcinoma
- Mucinous Carcinoma
- Papillary Carcinoma

#### Dataset Split Strategy

To prevent data leakage, splitting is performed at the **group/case level**:

| Split | Images |
|-------|--------|
| **Training** | 5,303 |
| **Validation** | 1,120 |
| **Testing** | 1,486 |
| **Total** | 7,909 |

This ensures that images from the same patient are not distributed across different sets.

#### Data Sources

- **Official BreaKHis Database:** [UFPR BreaKHis Database](https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/)
- **Kaggle Dataset:** [BreaKHis on Kaggle](https://www.kaggle.com/datasets/waseemalastal/breakhis-breast-cancer-histopathological-dataset)

---

## 🏗️ Project Structure

```
Histopathology_Cancer_Diagnosis/
│
├── 📄 README.md                          # This file
├── 📋 requirements.txt                   # Python dependencies
├── 📄 .gitignore                         # Git ignore configuration
├── 📑 BreakHis_Project_Future_Reference_Documentation.pdf
│
├── 📁 data/
│   ├── raw/                              # Raw dataset (not committed)
│   │   └── BreakHis/
│   │       ├── benign/
│   │       └── malignant/
│   │
│   └── processed/
│       └── breakhis_split.csv            # Train/Val/Test split manifest
│
├── 📁 models/
│   └── experiments/
│       ├── frozen_mobilenetv2.keras
│       ├── 10layer_finetune.keras
│       ├── 30layer_finetune.keras
│       └── best_classweighted_30_finetune_layers.keras  # ⭐ Final model
│
├── 📁 src/                               # Python source code
│   ├── data_analysis.py                  # Dataset inspection & statistics
│   ├── train_val_test.py                 # Create train/val/test split
│   ├── preprocessing.py                  # Image preprocessing pipeline
│   ├── train_model.py                    # Train baseline model
│   ├── finetune_model.py                 # Fine-tune model
│   ├── class_weight_model.py             # Class-weighted training
│   ├── test_evaluation.py                # Final model evaluation
│   └── error_analysis.py                 # False positive/negative analysis
│
└── 📁 results/                           # Generated outputs
    ├── training_curves/
    │   ├── loss_training_curve.png
    │   ├── accuracy_training_curve.png
    │   ├── precision_training_curve.png
    │   ├── recall_training_curve.png
    │   └── auc_training_curve.png
    │
    ├── evaluation/
    │   ├── final_test_confusion_matrix.png
    │   ├── final_test_roc_curve.png
    │   └── final_test_precision_recall_curve.png
    │
    ├── error_analysis/
    │   ├── false_positives/
    │   └── false_negatives/
    │
    └── preprocessing/
        └── sample_augmented_images.png
```

---

## 🧠 Model Architecture

### MobileNetV2 Transfer Learning

The project uses **MobileNetV2 pretrained on ImageNet** for efficient, lightweight feature extraction.

```
Input Image
224 × 224 × 3
        │
        ▼
MobileNetV2 (ImageNet Pretrained)
        │
        ▼
Global Average Pooling
        │
        ▼
Dropout(0.5)
        │
        ▼
Dense Layer (1 neuron)
        │
        ▼
Sigmoid Activation
        │
        ▼
Output: P(Malignant)
```

### Training Configuration

| Parameter | Value |
|-----------|-------|
| **Architecture** | MobileNetV2 |
| **Pretrained Weights** | ImageNet |
| **Input Size** | 224 × 224 × 3 |
| **Batch Size** | 32 |
| **Optimizer** | Adam |
| **Loss Function** | Binary Cross-Entropy |
| **Output Layer** | 1 Sigmoid Neuron |
| **Dropout** | 0.5 |
| **Selection Metric** | Validation AUC |

### Data Augmentation

Applied **only to training data** to improve generalization:

- Horizontal and vertical flipping
- Random rotation
- Random zoom

Validation and test sets are **not augmented** to ensure realistic evaluation.

---

## 💻 Installation

### Prerequisites

- Python 3.8+
- pip or conda
- ~2GB free disk space for the dataset

### Step 1: Clone the Repository

```bash
git clone https://github.com/husseinabuammar24-cloud/CV_Histopath_Cancer_Diagnosis.git
cd CV_Histopath_Cancer_Diagnosis
```

### Step 2: Create Virtual Environment

**Windows:**
```bash
python -m venv .venv
.venv\Scripts\activate
```

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Download Dataset

1. Download the BreaKHis dataset from [Kaggle](https://www.kaggle.com/datasets/waseemalastal/breakhis-breast-cancer-histopathological-dataset)
2. Extract the dataset
3. Place it in the `data/raw/` directory:

```
data/raw/
└── BreakHis/
    ├── benign/
    └── malignant/
```

### Step 5: Verify Installation

```bash
python -c "import tensorflow as tf; print(f'TensorFlow version: {tf.__version__}')"
```

---

## 🚀 Usage

Follow these steps sequentially to reproduce the project:

### 1️⃣ Analyze the Dataset

```bash
python src/data_analysis.py
```

**What it does:**
- Inspects image classes and distribution
- Analyzes magnification levels
- Identifies patient/case groups
- Checks for potential data leakage
- Generates dataset statistics

**Output:**
- Console summary of dataset characteristics
- Saves statistics to `results/`

---

### 2️⃣ Create Train/Validation/Test Split

```bash
python src/train_val_test.py
```

**What it does:**
- Performs group-level splitting (prevents data leakage)
- Creates balanced splits across benign/malignant classes
- Saves split manifest to `data/processed/breakhis_split.csv`

**Output:**
- `data/processed/breakhis_split.csv` — Split assignments for all images

---

### 3️⃣ Prepare Preprocessing Pipeline

```bash
python src/preprocessing.py
```

**What it does:**
- Creates TensorFlow datasets
- Applies MobileNetV2 preprocessing ([-1, 1] normalization)
- Implements data augmentation for training set
- Sets up batching and prefetching

**Output:**
- Processed datasets ready for training
- Sample augmented images to `results/preprocessing/`

---

### 4️⃣ Train Baseline Model

```bash
python src/train_model.py
```

**What it does:**
- Trains frozen MobileNetV2 with new classification head
- Logs training progress and metrics
- Saves model checkpoints

**Configuration:**
```
- Frozen convolutional base
- Only classification layers trainable
- Baseline for comparison
```

**Output:**
- Model saved to `models/experiments/frozen_mobilenetv2.keras`
- Training curves to `results/training_curves/`

---

### 5️⃣ Fine-Tune the Model

**Option A: 10-layer fine-tuning**
```bash
python src/finetune_model.py
```

**Option B: Class-weighted 30-layer fine-tuning** (recommended)
```bash
python src/class_weight_model.py
```

**What it does:**
- Unfreezes selected layers of MobileNetV2
- Fine-tunes on the training data
- Uses class weights to handle data imbalance
- Keeps Batch Normalization layers frozen for stability

**Recommended Configuration:**
```
- 30-layer fine-tuning
- Class weights: Benign=1.6606, Malignant=0.7156
- Validation AUC: 0.8208 (best)
```

**Output:**
- Best model saved to `models/experiments/best_classweighted_30_finetune_layers.keras`
- Updated training curves

---

### 6️⃣ Evaluate on Test Set

```bash
python src/test_evaluation.py
```

**What it does:**
- Evaluates final model on held-out test set
- Computes all classification metrics
- Generates confusion matrix and ROC curves
- Creates Precision-Recall curve

**Metrics Generated:**
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion matrix
- ROC curve
- Precision-Recall curve

**Output:**
- `results/evaluation/final_test_confusion_matrix.png`
- `results/evaluation/final_test_roc_curve.png`
- `results/evaluation/final_test_precision_recall_curve.png`

---

### 7️⃣ Analyze Errors

```bash
python src/error_analysis.py
```

**What it does:**
- Identifies false positives and false negatives
- Visualizes misclassified histopathology images
- Helps investigate error patterns

**Output:**
- `results/error_analysis/false_positives/` — Benign predicted as malignant
- `results/error_analysis/false_negatives/` — Malignant predicted as benign

---

## 📈 Results

### Hyperparameter Experiments Summary

| Experiment | Best Validation AUC |
|-----------|---------------------|
| Frozen MobileNetV2 | ~0.7003 |
| 30-layer fine-tuning | ~0.6246 |
| 10-layer fine-tuning | 0.7006 |
| Class-weighted + 15-layer | 0.8002 |
| Class-weighted + 30-layer | 0.8112 |
| **Class-weighted + 30-layer (Final)** | **0.8208** ⭐ |

### Final Test Results

| Metric | Value |
|--------|-------|
| **Accuracy** | 0.6743 |
| **Malignant Precision** | 0.7860 |
| **Malignant Recall** | 0.6016 |
| **Malignant F1-score** | 0.6816 |
| **ROC-AUC** | 0.7622 |

### Confusion Matrix

```
                    Predicted
                  Benign  Malignant

Actual Benign       484       141      (TN=484, FP=141)

Actual Malignant    343       518      (FN=343, TP=518)
```

#### Interpretation

- **True Negatives (TN):** 484 — Correctly identified benign images
- **False Positives (FP):** 141 — Benign wrongly predicted as malignant
- **False Negatives (FN):** 343 — Malignant wrongly predicted as benign ⚠️
- **True Positives (TP):** 518 — Correctly identified malignant images

### Medical Interpretation

#### False Negatives (343 cases)

A false negative occurs when:
```
Actual:    Malignant
Predicted: Benign
```

This is **particularly critical** in cancer detection because a malignant sample being classified as benign could delay diagnosis and treatment.

**Malignant Recall = 0.6016** means the model successfully identifies about 60% of malignant cases but misses 40%.

#### False Positives (141 cases)

A false positive occurs when:
```
Actual:    Benign
Predicted: Malignant
```

This may result in unnecessary follow-up examinations and patient anxiety.

**Malignant Precision = 0.7860** means approximately 79% of positive predictions are correct.

#### Critical Metrics for Medical Diagnosis

1. **Recall/Sensitivity** — How many actual malignant cases are detected?
2. **Precision** — How many positive predictions are truly malignant?
3. **F1-score** — Balanced measure of precision and recall
4. **ROC-AUC** — Performance across all classification thresholds
5. **Confusion Matrix** — Direct overview of all error types

---

## 🎨 Visualizations

The project generates comprehensive visualizations:

### Training Curves

```
results/training_curves/
├── loss_training_curve.png              # Model convergence
├── accuracy_training_curve.png
├── precision_training_curve.png
├── recall_training_curve.png
└── auc_training_curve.png
```

These plots reveal:
- Convergence patterns
- Overfitting/underfitting
- Training stability
- Validation performance

### Evaluation Plots

```
results/evaluation/
├── final_test_confusion_matrix.png      # Classification breakdown
├── final_test_roc_curve.png             # ROC-AUC visualization
└── final_test_precision_recall_curve.png # Precision-Recall tradeoff
```

### Error Analysis

```
results/error_analysis/
├── false_positives/                     # Benign → Malignant
├── false_negatives/                     # Malignant → Benign
└── error_summary.png
```

---

## 👥 Contributors

| Contributor | Role | Contact |
|-------------|------|---------|
| Hussein Abu Ammar | Computer Vision / Deep Learning | [@husseinabuammar24-cloud](https://github.com/husseinabuammar24-cloud) |

**Contributions & Feedback Welcome!** 🙌

This project is open for collaboration. If you'd like to contribute:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

---

## 📅 Timeline

### Day 1 — Dataset & Model Development
- ✅ Download BreaKHis dataset
- ✅ Inspect folder structure and metadata
- ✅ Analyze images, patients, and magnification levels
- ✅ Create group-level train/validation/test splits
- ✅ Implement preprocessing pipeline
- ✅ Implement data augmentation
- ✅ Build MobileNetV2 transfer-learning model
- ✅ Train frozen baseline
- ✅ Begin hyperparameter experiments

### Day 2 — Optimization & Evaluation
- ✅ Fine-tune MobileNetV2 with various configurations
- ✅ Experiment with different numbers of trainable layers
- ✅ Implement class weighting
- ✅ Select best model using validation performance
- ✅ Evaluate on held-out test set
- ✅ Generate comprehensive evaluation plots
- ✅ Perform error analysis
- ✅ Document findings
- ✅ Prepare GitHub repository

**Total Development Time:** ~2 days

---

## 👤 Personal Situation

This project was developed as part of a **Computer Vision consolidation challenge** to deepen expertise in deep learning applications.

### Learning Objectives Achieved

✅ Medical image classification with CNNs  
✅ Transfer learning with pretrained models  
✅ MobileNetV2 architecture and optimization  
✅ Image preprocessing and data augmentation  
✅ Hyperparameter tuning and fine-tuning strategies  
✅ Class weighting for imbalanced datasets  
✅ Comprehensive model evaluation beyond accuracy  
✅ Understanding false positives vs. false negatives  
✅ Error analysis and visualization techniques  
✅ Explainable AI concepts in medical contexts  
✅ Risk management in healthcare AI systems  

### Key Insights

1. **Accuracy is Not Enough** — In medical applications, false negatives and false positives have very different consequences. A single metric cannot capture model safety.

2. **Data Leakage Risk** — Group-level splitting prevented patient data from being scattered across train/val/test, ensuring realistic generalization estimates.

3. **Class Imbalance Matters** — The 2:1 ratio of malignant to benign images significantly impacts model bias. Class weighting improved validation AUC from ~0.70 to 0.82.

4. **Validation ≠ Test** — The model achieved 0.8208 AUC on validation but 0.7622 on test, highlighting the importance of held-out evaluation.

5. **Threshold Selection is Critical** — The default 0.5 threshold may not be optimal for clinical use. The ROC and Precision-Recall curves enable informed threshold selection based on clinical priorities.

---

## ⭐ Nice-to-Have Features

### Future Enhancements

- [ ] **Feature-Map Visualization** — Inspect intermediate MobileNetV2 activations
- [ ] **Grad-CAM/Grad-CAM++** — Generate visual explanations for predictions
- [ ] **Saliency Maps** — Highlight image regions influencing predictions
- [ ] **Architecture Comparison** — Benchmark ResNet, EfficientNet, VGG
- [ ] **Eight-Class Classification** — Classify histological subtypes separately
- [ ] **Multi-Magnification Fusion** — Combine 40X, 100X, 200X, 400X predictions
- [ ] **Confidence Calibration** — Improve probability estimate reliability
- [ ] **Deployment Pipeline** — API for inference on new images
- [ ] **Web Interface** — User-friendly demo application

---

## ⚠️ Limitations & Disclaimer

### Model Performance Limitations

The final model demonstrates useful classification ability but is **not sufficient for independent clinical diagnosis**:

- **343 false negatives** — Misses a substantial number of malignant cases
- **Validation-Test Gap** — 0.8208 AUC (validation) → 0.7622 AUC (test) suggests some overfitting
- **Limited Generalization** — Trained only on BreaKHis dataset; performance on other datasets unknown

### Disclaimer

> **This is an EDUCATIONAL AND RESEARCH PROTOTYPE, not a medical diagnostic device.**

A real clinical system would require:
- ✅ Independent external validation on diverse populations
- ✅ Evaluation by qualified pathologists
- ✅ Careful classification-threshold selection based on clinical costs
- ✅ Probability calibration for reliable confidence estimates
- ✅ Robustness testing (adversarial examples, distribution shift)
- ✅ Comprehensive explainability analysis
- ✅ Regulatory approval (FDA, CE mark, etc.)
- ✅ Comparison with specialist performance
- ✅ Human-in-the-loop deployment strategy

**DO NOT use this model for clinical decision-making without proper validation and regulatory approval.**

---

## 📚 References

### Dataset Sources

1. **Official BreaKHis Database** — Universidade Federal do Paraná  
   https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/

2. **BreaKHis on Kaggle**  
   https://www.kaggle.com/datasets/waseemalastal/breakhis-breast-cancer-histopathological-dataset

3. **Original BreaKHis Research Paper**  
   Spanhol, F. A., Oliveira, L. S., Petitjean, C., & Heutte, L. (2016).  
   *"A Dataset for Breast Cancer Histopathological Image Classification."*  
   IEEE Transactions on Biomedical Engineering, 63(7), 1455–1462.

### Related Resources

- [MobileNetV2 Paper](https://arxiv.org/abs/1801.04381) — Architecture and pretrained weights
- [Transfer Learning Guide](https://www.tensorflow.org/tutorials/images/transfer_learning) — TensorFlow documentation
- [Imbalanced Learning](https://imbalanced-learn.org/) — Handling class imbalance
- [Medical Image Analysis](https://www.sciencedirect.com/journal/medical-image-analysis) — State-of-the-art research

---

## 📝 License

This project is licensed under the **MIT License** — see the LICENSE file for details.

---

## 🤝 Acknowledgments

- **BreaKHis Database Contributors** — P&D Laboratory (UFPR), Brazil
- **TensorFlow & Keras Communities** — For excellent deep learning frameworks
- **Computer Vision Educators** — For the original CNN challenge

---

## 📞 Support & Questions

If you have questions or encounter issues:

1. **Check the documentation** — This README and the project PDF guide
2. **Review the code comments** — Each script is well-documented
3. **Inspect error logs** — Check console output and saved logs
4. **Open an Issue** — Create a GitHub issue with detailed error information

---

## 🎯 Project Summary

```
Input: Histopathology images (BreaKHis dataset, 7,909 images)
         │
         ▼
Processing: MobileNetV2 transfer learning + fine-tuning
         │
         ▼
Training: Adam optimizer, binary cross-entropy loss, class weighting
         │
         ▼
Evaluation: Validation AUC 0.8208 → Test ROC-AUC 0.7622
         │
         ▼
Output: Benign/Malignant classification with probability estimates
         │
         ▼
Deliverables: Model, visualizations, error analysis, documentation
```

**Final Status:** ✅ Complete and fully documented

**Best Model:** `best_classweighted_30_finetune_layers.keras`  
**Best Validation AUC:** 0.8208  
**Test Performance:** ROC-AUC 0.7622, Accuracy 0.6743

---

<div align="center">

**Made with ❤️ by Hussein Abu Ammar**

[⭐ Star this repo](https://github.com/husseinabuammar24-cloud/CV_Histopath_Cancer_Diagnosis) if you found it helpful!

</div>
