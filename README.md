# Histopathology_Cancer_Diagnosis

## Explainable Deep Learning for Automated Quantitative Analysis of Histopathology Images for Cancer Diagnosis

---

## 📌 Project Overview

This project develops a deep-learning computer vision system for the automated classification of breast histopathology images into **benign** and **malignant** categories.

The project uses the **BreaKHis (Breast Cancer Histopathological Image Classification)** dataset and applies **MobileNetV2 transfer learning** to perform binary breast cancer classification.

The original computer vision challenge was:

> **Design a CNN capable of recognizing pneumonia in X-rays of patients.**

For this project, the same computer vision methodology has been adapted to a different medical-imaging problem:

> **Design and evaluate a CNN capable of recognizing malignant and benign breast histopathology images.**

The main objectives are to:

* Train a CNN capable of recognizing previously unseen histopathology images.
* Apply a rigorous training, validation and testing methodology.
* Use transfer learning with a pretrained CNN.
* Perform hyperparameter and fine-tuning experiments.
* Evaluate the final model using multiple classification metrics.
* Generate clearly labeled visualizations.
* Analyze false positives and false negatives.
* Discuss the importance of different evaluation metrics in a medical environment.
* Explore explainability and future feature-map visualization.

> **Important:** This project is intended for educational and research purposes. It is not a clinically validated diagnostic system.

---

# 📊 Dataset

## BreaKHis — Breast Cancer Histopathological Image Classification

The project uses the **BreaKHis** dataset, a publicly available collection of breast histopathology images.

The original BreaKHis database was developed in collaboration with the **P&D Laboratory – Pathological Anatomy and Cytopathology, Paraná, Brazil** and is made available for research purposes.

### Official BreaKHis database

The official database and dataset information can be found here:

[BreaKHis — Official UFPR Database](https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/?utm_source=chatgpt.com)

### Kaggle dataset

The dataset used for this project was obtained from Kaggle:

[BreaKHis — Kaggle Dataset](https://www.kaggle.com/datasets/waseemalastal/breakhis-breast-cancer-histopathological-dataset?utm_source=chatgpt.com)

The Kaggle version provides the BreaKHis dataset with both binary and multi-class labels and includes images at 40X, 100X, 200X and 400X magnifications.

---

# 🔎 Dataset Inspection

The first step of the project was to download the dataset and inspect its folder structure and metadata.

The following characteristics were investigated:

* Number of images
* Number of patients
* Binary classification categories
* Histological subtypes
* Magnification levels
* Image dimensions
* Slide/case identifiers
* Potential data leakage between train, validation and test sets

---

## How many images?

The publicly downloadable BreaKHis dataset used in this project contains:

### **7,909 images**

The official BreaKHis page reports:

| Category  | Number of images |
| --------- | ---------------: |
| Benign    |            2,480 |
| Malignant |            5,429 |
| **Total** |        **7,909** |

The official page describes the broader database as containing 9,109 images, while the BreaKHis 1.0 publicly available/downloadable set contains 7,909 images.

For this project, the relevant dataset is therefore:

> **7,909 images**

---

## How many patients?

The dataset contains images collected from:

### **82 patients**

Patient/case information is important because images from the same patient or slide can be visually related.

For this reason, this project does not simply perform a random image-level split.

Instead, a **group-level split** is used to reduce the risk of data leakage.

---

# 🧬 Benign vs Malignant

The main objective of this project is **binary classification**.

There are two classes:

| Class     | Label |
| --------- | ----: |
| Benign    |     0 |
| Malignant |     1 |

The model therefore produces a single probability indicating how likely an image is to belong to the malignant class.

The original BreaKHis dataset contains 2,480 benign images and 5,429 malignant images.

---

# 🔬 Histological Subtypes

Although this project performs **binary classification**, the BreaKHis dataset contains eight histological tumor types.

## Benign

1. Adenosis
2. Fibroadenoma
3. Phyllodes Tumor
4. Tubular Adenoma

## Malignant

5. Ductal Carcinoma
6. Lobular Carcinoma
7. Mucinous Carcinoma
8. Papillary Carcinoma

The official BreaKHis documentation identifies four benign and four malignant histological types.

The current project combines these eight subtypes into two diagnostic categories:

```text
                         BreaKHis
                            │
              ┌─────────────┴─────────────┐
              │                           │
           Benign                      Malignant
              │                           │
       ┌──────┼──────┐             ┌──────┼──────┐
       │      │      │             │      │      │
    Adenosis Fibro- Phyllodes    Ductal  Lobular Mucinous
             adenoma Tumor       Carcinoma Carcinoma Carcinoma
              │                           │
        Tubular Adenoma             Papillary Carcinoma
```

A future extension could investigate the more difficult **eight-class classification problem**.

---

# 🔬 Magnification Levels

BreaKHis images are available at four magnification levels:

* **40X**
* **100X**
* **200X**
* **400X**

These magnifications provide different levels of histological information.

The distribution in the official BreaKHis 1.0 dataset is:

| Magnification |    Benign | Malignant |     Total |
| ------------- | --------: | --------: | --------: |
| 40X           |       652 |     1,370 |     1,995 |
| 100X          |       644 |     1,437 |     2,081 |
| 200X          |       623 |     1,390 |     2,013 |
| 400X          |       588 |     1,232 |     1,820 |
| **Total**     | **2,480** | **5,429** | **7,909** |

---

# 🖼️ Image Dimensions

The image dimensions used by this project are:

### **224 × 224 pixels**

The model receives images as:

```text
224 × 224 × 3
```

where:

* 224 = image height
* 224 = image width
* 3 = RGB channels

The Kaggle version of the BreaKHis dataset also describes its images as resized to **224×224 pixels**.

This resolution is suitable for the MobileNetV2 architecture used in this project.

### Model input pipeline

```text
Input image
     │
     ▼
224 × 224 × 3
     │
     ▼
MobileNetV2 preprocessing
     │
     ▼
Approximately [-1, 1]
     │
     ▼
MobileNetV2
```

> **Important:** The `224 × 224` dimension refers to the image representation used by this project/model pipeline.

---

# 📁 Dataset Split

The final dataset split used in the project is:

| Split      |    Images |
| ---------- | --------: |
| Training   |     5,303 |
| Validation |     1,120 |
| Testing    |     1,486 |
| **Total**  | **7,909** |

The split is performed at the **group/case level**.

This means that images belonging to the same underlying group are kept within the same split.

The project verifies that there is no overlap between the groups assigned to:

* training
* validation
* testing

This reduces the risk of data leakage and provides a more realistic evaluation of model generalization.

---

# ⚙️ Preprocessing

The original challenge suggested performing preprocessing using OpenCV.

For this project, preprocessing was implemented using **TensorFlow/Keras and PIL** instead of OpenCV.

This choice was made because the selected MobileNetV2 architecture has a specific preprocessing function that integrates directly with the TensorFlow/Keras workflow.

The preprocessing pipeline consists of:

1. Load the image.
2. Convert the image to RGB.
3. Resize the image to 224×224.
4. Apply MobileNetV2 `preprocess_input`.
5. Create TensorFlow datasets.
6. Batch the images.
7. Prefetch the data.

---

## MobileNetV2 preprocessing

The model receives image values approximately in:

```text
[-1, 1]
```

The `[0, 1]` representation is only used for displaying images in visualizations.

Therefore:

```text
Model input:
[-1, 1]

Visualization:
[0, 1]
```

This distinction is important because the model is **not trained on `[0,1]` images**.

---

# 🔄 Data Augmentation

Data augmentation is applied **only to the training dataset**.

The current augmentation pipeline includes:

* horizontal and vertical flipping
* random rotation
* random zoom

The validation and test datasets are not augmented.

This ensures that validation and test performance represents the model's performance on the actual evaluation data.

---

# 🧠 Model Architecture

## MobileNetV2

The main CNN architecture used in this project is **MobileNetV2 pretrained on ImageNet**.

The original classification head is removed and replaced with a binary classification head.

The architecture is:

```text
Input
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
Dropout
0.5
       │
       ▼
Dense
1 neuron
       │
       ▼
Sigmoid
       │
       ▼
Benign / Malignant
```

---

# 🔁 Transfer Learning

The project uses a transfer-learning workflow.

Initially, the pretrained MobileNetV2 base is frozen and only the newly added classification layers are trained.

After establishing a baseline, selected layers of the pretrained network are unfrozen for fine-tuning.

Batch Normalization layers are kept frozen during fine-tuning for training stability.

The workflow is:

```text
Pretrained MobileNetV2
          │
          ▼
Freeze convolutional base
          │
          ▼
Train new classification head
          │
          ▼
Evaluate on validation set
          │
          ▼
Unfreeze selected layers
          │
          ▼
Fine-tune
          │
          ▼
Evaluate validation performance
          │
          ▼
Select final model
          │
          ▼
Test once on test set
```

---

# 🎯 Training Configuration

| Parameter             | Value                |
| --------------------- | -------------------- |
| Architecture          | MobileNetV2          |
| Pretrained weights    | ImageNet             |
| Input size            | 224×224×3            |
| Batch size            | 32                   |
| Optimizer             | Adam                 |
| Loss                  | Binary Cross-Entropy |
| Output                | 1 sigmoid neuron     |
| Dropout               | 0.5                  |
| Main selection metric | Validation AUC       |

---

# 🧪 Hyperparameter Experiments

Several configurations were tested.

| Experiment                                        | Best Validation AUC |
| ------------------------------------------------- | ------------------: |
| Frozen MobileNetV2                                |             ~0.7003 |
| 30-layer fine-tuning                              |             ~0.6246 |
| 10-layer fine-tuning                              |              0.7006 |
| Class-weighted + 15-layer fine-tuning             |              0.8002 |
| Class-weighted + 30-layer fine-tuning             |              0.8112 |
| **Class-weighted + 30-layer fine-tuning — final** |          **0.8208** |

The final selected model is:

```text
best_classweighted_30_finetune_layers.keras
```

The best validation AUC achieved was:

```text
0.8208
```

---

# ⚖️ Class Weighting

The final experiment uses class weighting to compensate for the imbalance between benign and malignant samples.

The approximate weights were:

```text
Benign:     1.6606
Malignant:  0.7156
```

The final experiment combined:

* class weighting
* 30-layer fine-tuning

This produced the best validation AUC.

Because both class weighting and the number of trainable layers were changed together, the result should be interpreted as the performance of the **combined configuration**, rather than evidence that class weighting alone caused the improvement.

---

# 🧪 Evaluation Methodology

The test set is kept completely separate from model selection.

The workflow is:

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
Hyperparameter tuning
     │
     ▼
Select final model
     │
     ▼
Test Set
     │
     ▼
Final evaluation
```

The test set is **not used to tune the model**.

This is essential to avoid test-set leakage.

---

# 📈 Evaluation Metrics

The final model is evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC
* Confusion matrix
* ROC curve
* Precision-Recall curve
* Training/validation curves
* False-positive analysis
* False-negative analysis

---

# 🏆 Final Test Results

The selected model was evaluated on the previously unseen test set.

| Metric              |     Result |
| ------------------- | ---------: |
| Accuracy            | **0.6743** |
| Malignant Precision | **0.7860** |
| Malignant Recall    | **0.6016** |
| Malignant F1-score  | **0.6816** |
| ROC-AUC             | **0.7622** |

---

# 📊 Confusion Matrix

The final confusion matrix is:

```text
                    Predicted
                  Benign  Malignant

Actual Benign       484       141

Actual Malignant    343       518
```

Therefore:

```text
True Negatives  = 484
False Positives = 141
False Negatives = 343
True Positives   = 518
```

---

# 🏥 Medical Interpretation

Medical image classification requires more than simply measuring accuracy.

Different types of errors have different consequences.

## False Negatives

A false negative occurs when:

```text
Actual:    Malignant
Predicted: Benign
```

The final model produced:

**343 false negatives.**

This is particularly important in a cancer-detection scenario because a malignant sample being classified as benign could potentially delay further investigation.

Therefore, **malignant recall/sensitivity** is an important metric.

The final malignant recall was:

```text
0.6016
```

---

## False Positives

A false positive occurs when:

```text
Actual:    Benign
Predicted: Malignant
```

The final model produced:

**141 false positives.**

False positives may result in unnecessary follow-up examinations, additional testing and patient anxiety.

Therefore, malignant precision is also important.

The final malignant precision was:

```text
0.7860
```

---

# 📌 Which Metrics Are Most Important?

For a medical diagnostic system, no single metric is sufficient.

The most important metrics depend on the clinical objective and the consequences of different errors.

For cancer detection, particular attention should be given to:

### Recall / Sensitivity

Measures how many actual malignant cases are successfully detected.

A low recall means that too many malignant cases are missed.

### Precision

Measures how many predictions of malignant tissue are actually malignant.

### F1-score

Balances precision and recall.

### ROC-AUC

Measures the model's ability to discriminate between benign and malignant samples across classification thresholds.

### Precision-Recall Curve

Shows the trade-off between precision and recall at different thresholds.

### Confusion Matrix

Provides a direct overview of:

* True Positives
* True Negatives
* False Positives
* False Negatives

In a real clinical application, the decision threshold should be chosen according to the clinical cost of false negatives and false positives rather than automatically assuming that 0.5 is optimal.

---

# 📉 Visualizations

The project generates several visualizations.

## Training Curves

The following plots are generated:

```text
results/training_curves/
├── loss_training_curve.png
├── accuracy_training_curve.png
├── precision_training_curve.png
├── recall_training_curve.png
└── auc_training_curve.png
```

These plots allow us to investigate:

* convergence
* overfitting
* underfitting
* training stability
* validation performance

---

# 📈 ROC Curve

The ROC curve is generated from the final test predictions.

Output:

```text
results/evaluation/final_test_roc_curve.png
```

Final test ROC-AUC:

```text
0.7622
```

---

# 📉 Precision-Recall Curve

The Precision-Recall curve is generated from the final test predictions.

Output:

```text
results/evaluation/final_test_precision_recall_curve.png
```

This visualization helps demonstrate the trade-off between precision and recall at different classification thresholds.

---

# 📊 Confusion Matrix Visualization

The final confusion matrix visualization is saved to:

```text
results/evaluation/final_test_confusion_matrix.png
```

---

# 🔍 Error Analysis

The project also performs visual analysis of incorrectly classified images.

## False Positives

```text
Benign → Malignant
```

## False Negatives

```text
Malignant → Benign
```

The analysis identifies representative errors and visualizes the corresponding histopathology images.

This can help investigate whether errors are related to:

* visually similar tissue structures
* staining variations
* magnification
* difficult histological patterns
* image artifacts
* ambiguous cases

Results are stored in:

```text
results/error_analysis/
```

---

# 💡 Explainability

The project title includes **Explainable Deep Learning** because understanding model predictions is particularly important in medical applications.

The current project includes:

* quantitative evaluation
* confusion-matrix analysis
* false-positive analysis
* false-negative analysis
* visualization of model performance

Future explainability work could include:

* feature-map visualization
* Grad-CAM
* Grad-CAM++
* saliency maps
* activation visualization

These techniques could help investigate which image regions contribute most strongly to the model's prediction.

---

# ⭐ Nice-to-Have Features

## Feature Map Visualization

Feature-map visualization would allow us to inspect intermediate activations within the MobileNetV2 network.

This could provide insight into which visual structures the network detects at different stages.

**Status: Not yet implemented.**

---

## Comparison with Other CNN Architectures

A future extension would compare MobileNetV2 against alternative CNN architectures such as:

* ResNet
* EfficientNet
* VGG
* other MobileNet variants

Possible comparison metrics:

| Metric        | MobileNetV2 | ResNet | EfficientNet |
| ------------- | ----------: | -----: | -----------: |
| Accuracy      |             |        |              |
| Precision     |             |        |              |
| Recall        |             |        |              |
| F1-score      |             |        |              |
| ROC-AUC       |             |        |              |
| Parameters    |             |        |              |
| Training time |             |        |              |

**Status: Not yet implemented.**

---

# 💻 Installation

## 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd Histopathology_Cancer_Diagnosis
```

## 2. Create a virtual environment

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

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

# 📥 Dataset Installation

Download the dataset from Kaggle:

[Download BreaKHis from Kaggle](https://www.kaggle.com/datasets/waseemalastal/breakhis-breast-cancer-histopathological-dataset?utm_source=chatgpt.com)

The official source is also available through the UFPR BreaKHis database:

[BreaKHis official database](https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/?utm_source=chatgpt.com)

Place the downloaded dataset inside:

```text
data/raw/
```

The raw dataset is not committed to GitHub because of its size.

---

# 🚀 Usage

## 1. Analyze the dataset

```bash
python src/data_analysis.py
```

This analyzes:

* image classes
* magnification levels
* groups
* dataset structure
* class distribution

---

## 2. Create the train/validation/test split

```bash
python src/train_val_test.py
```

This creates:

```text
data/processed/breakhis_split.csv
```

The split is performed at the group level to reduce data leakage.

---

## 3. Prepare preprocessing

```bash
python src/preprocessing.py
```

This creates the TensorFlow datasets and applies the MobileNetV2 preprocessing pipeline.

---

## 4. Train the baseline model

```bash
python src/train_model.py
```

This trains the initial frozen MobileNetV2 model.

---

## 5. Fine-tune the model

For the 10-layer fine-tuning experiment:

```bash
python src/finetune_model.py
```

For the class-weighted 30-layer fine-tuning experiment:

```bash
python src/class_weight_model.py
```

---

## 6. Evaluate the final model

```bash
python src/test_evaluation.py
```

This generates:

* accuracy
* precision
* recall
* F1-score
* ROC-AUC
* confusion matrix
* ROC curve
* Precision-Recall curve

---

## 7. Analyze errors

```bash
python src/error_analysis.py
```

This identifies and visualizes false positives and false negatives.

---

# 📁 Repository Structure

```text
Histopathology_Cancer_Diagnosis/
│
├── data/
│   ├── raw/
│   │   └── BreakHis/
│   │
│   └── processed/
│       └── breakhis_split.csv
│
├── models/
│   └── experiments/
│
├── results/
│   ├── training_curves/
│   ├── evaluation/
│   ├── error_analysis/
│   └── preprocessing/
│
├── src/
│   ├── data_analysis.py
│   ├── train_val_test.py
│   ├── preprocessing.py
│   ├── train_model.py
│   ├── finetune_model.py
│   ├── class_weight_model.py
│   ├── test_evaluation.py
│   └── error_analysis.py
│
├── README.md
├── requirements.txt
├── .gitignore
└── LICENSE
```

---

# 📅 Timeline

## Day 1 — Dataset and Model Development

* Download the BreaKHis dataset.
* Inspect the folder structure.
* Analyze the number of images.
* Analyze the number of patients.
* Analyze benign and malignant classes.
* Analyze histological subtypes.
* Analyze magnification levels.
* Create group-level train/validation/test splits.
* Implement preprocessing.
* Implement data augmentation.
* Build the MobileNetV2 transfer-learning model.
* Train the frozen baseline model.
* Begin hyperparameter experiments.

## Day 2 — Optimization and Evaluation

* Fine-tune MobileNetV2.
* Experiment with different numbers of trainable layers.
* Experiment with class weighting.
* Select the best model using validation performance.
* Evaluate the final model on the test set.
* Generate training and validation curves.
* Generate the confusion matrix.
* Generate ROC and Precision-Recall curves.
* Analyze false positives and false negatives.
* Document the project.
* Prepare the GitHub repository.

---

# 👥 Contributors

| Contributor   | Role                            |
| ------------- | ------------------------------- |
| Contributor 1 | Computer Vision / Deep Learning |
| Contributor 2 | Computer Vision / Deep Learning |

Replace these placeholders with the actual team members before publishing the repository.

---

# 👤 Personal Situation

This project was developed as part of a **Computer Vision consolidation challenge**.

The original challenge focused on designing a CNN capable of recognizing pneumonia in X-ray images.

For this project, the same machine-learning workflow was adapted to breast cancer histopathology.

The project provided practical experience with:

* medical image classification
* convolutional neural networks
* transfer learning
* MobileNetV2
* image preprocessing
* data augmentation
* hyperparameter tuning
* class weighting
* model evaluation
* error analysis
* explainable AI concepts

A particular focus was placed on understanding that medical machine-learning systems should not be evaluated using accuracy alone.

In a medical environment, false negatives and false positives can have very different consequences. Therefore, precision, recall, F1-score, ROC-AUC and the confusion matrix are all important when evaluating the model.

---

# ✅ Challenge Checklist

## Must-Have Features

* [x] CNN trained on the dataset
* [x] Classification of new/unseen images
* [x] Training/validation/test split
* [x] Group-level splitting to reduce data leakage
* [x] Transfer learning
* [x] Hyperparameter experimentation
* [x] Confusion matrix
* [x] Accuracy
* [x] Precision
* [x] Recall
* [x] F1-score
* [x] ROC-AUC
* [x] ROC curve
* [x] Precision-Recall curve
* [x] Training/validation curves
* [x] False-positive analysis
* [x] False-negative analysis
* [x] Discussion of medical evaluation metrics

## Nice-to-Have Features

* [ ] Feature-map visualization
* [ ] Comparison with other CNN architectures
* [ ] Grad-CAM / advanced explainability

---

# 🏆 Results Summary

## Final Model

```text
MobileNetV2
+ ImageNet pretrained weights
+ Global Average Pooling
+ Dropout(0.5)
+ Dense(1, sigmoid)
+ 30-layer fine-tuning
+ Class weighting
```

## Validation

```text
Best Validation AUC: 0.8208
```

## Final Test Performance

```text
Accuracy:             0.6743
Malignant Precision:  0.7860
Malignant Recall:     0.6016
Malignant F1-score:   0.6816
Test ROC-AUC:         0.7622
```

## Confusion Matrix

```text
TN = 484
FP = 141
FN = 343
TP = 518
```

The validation AUC was higher than the final test AUC:

```text
Validation AUC: 0.8208
Test AUC:       0.7622
Difference:     0.0586
```

This difference is considered when discussing model generalization.

---

# ⚠️ Limitations

The final model demonstrates useful classification ability, but its performance is not sufficient for independent clinical diagnosis.

In particular, the number of false negatives demonstrates that the model still misses a substantial number of malignant samples.

The model should therefore be considered an:

> **Educational and research prototype**

rather than a medical diagnostic device.

A real clinical system would require:

* independent external validation
* evaluation on additional patient populations
* careful classification-threshold selection
* probability calibration
* robustness testing
* explainability
* clinical validation
* comparison with qualified medical professionals
* appropriate regulatory approval

---

# 📚 Sources and References

## Dataset Sources

### 1. Official BreaKHis Database — Universidade Federal do Paraná

[BreaKHis Official Database](https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/?utm_source=chatgpt.com)

The official source provides information about the BreaKHis database, its classes, patients, magnification levels, dataset statistics and original publications.

### 2. Kaggle — BreaKHis Breast Cancer Histopathological Dataset

[BreaKHis on Kaggle](https://www.kaggle.com/datasets/waseemalastal/breakhis-breast-cancer-histopathological-dataset?utm_source=chatgpt.com)

This is the Kaggle dataset used as the project's dataset source.

### 3. Original BreaKHis Research Paper

Spanhol, F. A., Oliveira, L. S., Petitjean, C., & Heutte, L.

**A Dataset for Breast Cancer Histopathological Image Classification.**

IEEE Transactions on Biomedical Engineering, 63(7), 1455–1462, 2016.

The official BreaKHis database requests that researchers acknowledge the original publication when using the database.

### 4. Kaggle BreaKHis Dataset Version with 224×224 Images

[BreaKHis — 224×224 Kaggle Dataset Version](https://www.kaggle.com/datasets/tathagatbanerjee/breakhis-breast-cancer-histopathological?utm_source=chatgpt.com)

This dataset description explicitly states that the BreaKHis images were resized to 224×224 pixels and organized for binary and multi-class classification.

---

# 📌 Final Note

This repository demonstrates an end-to-end workflow for deep-learning-based histopathology image classification:

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
Data Augmentation
       │
       ▼
MobileNetV2 Transfer Learning
       │
       ▼
Hyperparameter Experiments
       │
       ▼
Fine-Tuning
       │
       ▼
Validation-Based Model Selection
       │
       ▼
Final Test Evaluation
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

The final model achieved a **validation AUC of 0.8208** and a **test ROC-AUC of 0.7622**.

The project demonstrates not only how to train a CNN for medical image classification, but also why **data leakage prevention, model selection, multiple evaluation metrics, visualization, and error analysis are essential when developing machine-learning systems for medical applications**.
