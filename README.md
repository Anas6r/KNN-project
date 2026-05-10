# MNIST Classification with k-Nearest Neighbors (k-NN)

This project explores the use of the **k-Nearest Neighbors** algorithm to classify handwritten digits from the MNIST dataset. It utilizes Scikit-Learn’s `GridSearchCV` to automate the process of hyperparameter tuning to achieve the highest possible accuracy.

## 🚀 Project Overview

The goal is to build a model that can take a $28 \times 28$ pixel image of a handwritten digit and correctly identify it as a number between 0 and 9.

### Key Features:

- **Dataset:** MNIST-784 (70,000 samples).
    
- **Algorithm:** k-Nearest Neighbors (k-NN).
    
- **Optimization:** Grid Search cross-validation to tune `n_neighbors` and `weights`.

## 🛠️ Implementation Details

### Hyperparameter Tuning

The project searches through the following parameter grid to find the optimal configuration:

Parameter - Values Checked ;
n_neighbors =2, 3, 4, 5, 6, 7, 8, 9;
weights "uniform", "distance"

### Prerequisites

You will need Python 3.x and the following libraries:
`pip install numpy pandas scikit-learn`

## 📂 Code Structure

The script performs the following steps:

1. **Data Acquisition:** Fetches the MNIST dataset using `fetch_openml`.
    
2. **Data Splitting:** Separates the data into 60,000 training images and 10,000 test images.
    
3. **Model Selection:** Initializes a `KNeighborsClassifier`.
    
4. **Grid Search:** Runs an 8-fold cross-validation (`cv=8`) to find the best parameters.
    
5. **Evaluation:** Validates the model performance on the unseen test set.

📊 Performance
The GridSearchCV identifies the "best estimator," which is then used to make final predictions.
Using weights='distance' typically yields significantly higher accuracy on MNIST compared to uniform weights.
