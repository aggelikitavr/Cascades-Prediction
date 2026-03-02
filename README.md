# Cascade Prediction


This project implements the methodology proposed in the article "Can Cascades Be Predicted? (2014)" for predicting information cascades in social networks using the **Weibo** dataset.


## Description
The objective is to predict whether an information cascade will double in size after observing its first k reshares. The project compares multiple machine learning models and analyzes the significance of structural and temporal features.


## Machine Learning Models
The implementation evaluates the following models as described in the original paper:
* **Logistic Regression**: Used as the primary baseline for its interpretability and ability to show the direction (positive/negative) of feature influence.
* **Random Forest (RF)**: A non-linear ensemble method that captures complex interactions between structural and temporal features.
* **Support Vector Machine (SVM)**: A linear kernel SVM used to test the robustness of the features across different classification algorithms.


## File Structure
* **main.py**: Entry point for running comparative experiments across different k values.
* **experiments/train.py**: Model training logic using 10-fold Cross-Validation.
* **src/cascade.py**: Class representing the propagation tree structure.
* **src/feature_extraction.py**: Extraction of key structural and temporal features.
* **src/parser.py**: Logic for converting raw Weibo text data into Cascade objects.
* **data/weibo_dataset.txt**: Raw dataset file.


## Prerequisites
Python 3.x and the following libraries are required:
```bash
pip install numpy scikit-learn scipy