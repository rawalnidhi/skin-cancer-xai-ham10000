# Skin Cancer Classification with Explainable AI (HAM10000)

Deep learning model for **skin lesion classification** using the **HAM10000 dermoscopy dataset**, implemented with **PyTorch transfer learning** and interpreted using **Grad-CAM visualizations**.

This project explores multiple modeling approaches and compares them against baseline models to identify the most effective method for classifying dermoscopic images.

---

# Problem Statement

Early detection of skin cancer is critical for improving treatment outcomes. Dermatologists often rely on dermoscopic images to diagnose different types of skin lesions.

The goal of this project is to build a **deep learning model that can classify dermoscopic images into multiple skin lesion categories** and provide **visual explanations** for model predictions.

---

# Dataset

Dataset used: **HAM10000 (Human Against Machine with 10000 training images)**

The dataset contains **10,015 dermoscopic images** labeled into **7 diagnostic categories**:

- akiec — Actinic keratoses
- bcc — Basal cell carcinoma
- bkl — Benign keratosis
- df — Dermatofibroma
- mel — Melanoma
- nv — Melanocytic nevi
- vasc — Vascular lesions

Dataset characteristics:

- Highly **imbalanced**
- Large visual similarity between some lesion types
- Requires **CNN-based feature extraction**

---

# Approach

The project follows a complete deep learning workflow:

1. Exploratory Data Analysis (EDA)
2. Data preprocessing and augmentation
3. Baseline model evaluation
4. Transfer learning experiments
5. Final model selection
6. Model interpretability using Grad-CAM
7. Error analysis

---

# Baseline Models

| Model | Accuracy | Macro F1 |
|------|------|------|
Random baseline | 0.15 | 0.086 |
Majority baseline | 0.67 | 0.11 |
Logistic Regression | 0.43 | 0.20 |

---

# Deep Learning Experiments

Three experiments were conducted using **ResNet18 transfer learning**.

| Experiment | Description | Validation Macro F1 |
|------|------|------|
Experiment 1 | Frozen ResNet18 backbone | ~0.47 |
Experiment 2 | Fine-tuned ResNet18 | **0.70** |
Experiment 3 | Weighted sampler | 0.59 |

**Experiment 2 produced the best performance and was selected as the final model.**

---

# Final Model Performance

Evaluation on a **held-out test set**.

| Metric | Score |
|------|------|
Accuracy | **0.7814** |
Macro F1 | **0.6548** |

Test set size: **1002 images**

---

# Training Curves

## Experiment 1

<p align="center">
<img src="results/exp1_accuracy_curve.png" width="400">
<img src="results/exp1_loss_curve.png" width="400">
</p>

<p align="center">
<img src="results/exp1_macro_f1_curve.png" width="600">
</p>

---

## Experiment 2 (Best Model)

<p align="center">
<img src="results/exp2_accuracy_curve.png" width="400">
<img src="results/exp2_loss_curve.png" width="400">
</p>

<p align="center">
<img src="results/exp2_macro_f1_curve.png" width="600">
</p>

---

## Experiment 3

<p align="center">
<img src="results/exp3_accuracy_curve.png" width="400">
<img src="results/exp3_loss_curve.png" width="400">
</p>

<p align="center">
<img src="results/exp3_macro_f1_curve.png" width="600">
</p>

---

# Model Explainability (Grad-CAM)

To improve interpretability, **Grad-CAM** was used to visualize which regions of the image influenced the model's predictions.

These visualizations highlight the **important lesion regions used by the CNN during classification**.

Observations:

- Correct predictions usually focus on the lesion area
- Misclassifications often occur when lesions have ambiguous or overlapping visual patterns
- Grad-CAM confirms that the model primarily uses lesion features rather than background information

---

# Key Insights

- Transfer learning significantly improves performance compared to baseline models.
- Dataset imbalance impacts model performance on minority classes.
- Fine-tuning deeper layers of ResNet18 improves feature extraction.
- Grad-CAM provides useful interpretability for CNN predictions in medical imaging.

---

---

# Technologies Used

- Python
- PyTorch
- Scikit-learn
- Matplotlib
- NumPy
- Grad-CAM
- Jupyter Notebook

---

# Future Improvements

Potential improvements for this project include:

- Handling class imbalance using advanced techniques
- Testing deeper architectures such as EfficientNet
- Performing hyperparameter optimization
- Increasing dataset size with additional medical datasets
- Deploying the model as a clinical decision support tool

---

# License

This project is released under the **MIT License**.
