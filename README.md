# Breast Cancer Predictor

## About

I built this project to learn how the K-Nearest Neighbors (KNN) algorithm works and how different hyperparameter values can affect model performance.

The goal of the project is to predict whether a breast tumor is benign or malignant using medical measurements from the Breast Cancer Wisconsin dataset.

Instead of simply training a model and stopping there, I experimented with different values of **k** to understand how hyperparameter tuning impacts prediction accuracy.

---

## Technologies Used

* Python
* Pandas
* Scikit-learn
* K-Nearest Neighbors (KNN)

---

## Dataset

This project uses the Breast Cancer Wisconsin dataset available through Scikit-learn.

The dataset contains 30 medical measurements including:

* Mean Radius
* Mean Texture
* Mean Perimeter
* Mean Area
* Mean Smoothness
* Mean Compactness
* Mean Concavity
* Mean Symmetry
* Worst Radius
* Worst Texture

and several other related features.

Target Values:

* 0 = Malignant
* 1 = Benign

---

## Project Workflow

```text
Breast Cancer Dataset
        ↓
Load Dataset
        ↓
Create Features (X) and Target (y)
        ↓
Train-Test Split
        ↓
Train KNN Model
        ↓
Evaluate Accuracy
        ↓
Hyperparameter Tuning
        ↓
Sample Prediction
```

---

## Model Performance

### KNN with k = 3

Accuracy:

92.00%

---

### KNN with k = 5

Accuracy:

95.61%

---

### KNN with k = 7

Accuracy:

95.61%

---

## Observation

The model achieved its best performance when using:

```text
k = 5
```

Increasing the number of neighbors beyond 5 did not improve the results.

This experiment helped me understand how hyperparameter tuning can influence model performance even when the underlying algorithm remains the same.

---

## Features

* Uses the Breast Cancer Wisconsin dataset
* Trains a K-Nearest Neighbors classifier
* Evaluates model accuracy
* Compares multiple k values
* Demonstrates hyperparameter tuning
* Performs sample patient prediction

---

## What I Learned

Through this project I learned:

* How K-Nearest Neighbors works
* The importance of distance-based learning
* How classification differs from regression
* How to evaluate classification models using accuracy
* How hyperparameter tuning affects performance
* Why selecting the right value of k is important

---

## How to Run

Install the required libraries:

```bash
pip install pandas scikit-learn
```

Run the project:

```bash
python main.py
```

The program trains the model, evaluates its accuracy, and performs a sample prediction.

---

## Future Improvements

* Apply feature scaling using StandardScaler
* Compare KNN with Decision Tree and Random Forest
* Use cross-validation for evaluation
* Build a web application interface
* Test additional hyperparameter values

---

## Author

Adithya Prasad
