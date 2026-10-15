# PyTorch Wine Classification

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-ee4c2c?logo=pytorch&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-Data%20Processing-f7931e?logo=scikitlearn&logoColor=white)
![Jupyter](https://img.shields.io/badge/Environment-Google%20Colab-orange?logo=googlecolab&logoColor=white)
![Status](https://img.shields.io/badge/Status-Completed-success)
![License](https://img.shields.io/badge/License-MIT-green)

An end-to-end multiclass classification project built with PyTorch and the Wine dataset from scikit-learn. The project demonstrates a complete machine learning workflow, from exploratory data analysis and preprocessing to neural network training, evaluation, and model artifact saving.

## Project Overview

The objective is to classify wine samples into one of three classes using 13 numerical input features. A feedforward neural network is implemented and trained using PyTorch.

The project was developed in Google Colab to practice fundamental deep learning concepts and understand the practical steps involved in building a classification model.

## Key Features

- Exploratory data analysis (EDA)
- Data quality checks for missing values and duplicate records
- Feature correlation analysis using a heatmap
- Stratified training, validation, and test splits
- Feature standardization using `StandardScaler`
- PyTorch tensor conversion and DataLoader configuration
- Feedforward neural network implementation
- Model training with Adam optimizer and CrossEntropyLoss
- Validation loss and accuracy monitoring
- Classification report and confusion matrix
- Saving model weights and the fitted feature scaler
- Reloading artifacts for prediction verification

## Dataset

The project uses the Wine dataset provided by `sklearn.datasets`.

| Property | Value |
|---|---:|
| Total samples | 178 |
| Input features | 13 |
| Target classes | 3 |
| Training samples | 106 |
| Validation samples | 36 |
| Test samples | 36 |

## Model Architecture

The classifier is a feedforward neural network with the following structure:

```text
Input Layer: 13 features
       |
Linear Layer: 13 → 64
       |
ReLU Activation
       |
Output Layer: 64 → 3
       |
Class Logits
```

The network contains 1,091 trainable parameters.

### Training Configuration

| Parameter | Configuration |
|---|---|
| Loss function | CrossEntropyLoss |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Training epochs | 100 |
| Batch size | 16 |
| Random seed | 42 |
| Device | CPU |

## Results

The trained model achieved the following results on the selected data split:

| Metric | Result |
|---|---:|
| Training accuracy | 100% |
| Validation accuracy at best checkpoint | 97.22% |
| Test accuracy | 100% |
| Correct test predictions | 36 out of 36 |

The confusion matrix showed that all 36 test samples were classified correctly.

**Important:** The test set contains only 36 samples. Perfect accuracy on this split does not guarantee equivalent performance on unseen data. The difference between training and validation loss should also be considered when assessing generalization. Stratified cross-validation and additional independent testing are recommended.

## Project Structure

```text
pytorch-wine-classification/
├── README.md
├── wine_classification.ipynb
├── wine_classifier.pth
└── wine_scaler.joblib
```

The filenames above are suggested. Update them to match the files included in your repository.

## Installation

Create a virtual environment if running locally, then install the required libraries:

```bash
pip install torch numpy pandas matplotlib seaborn scikit-learn joblib jupyter
```

## How to Run

1. Open `wine_classification.ipynb` in Google Colab or Jupyter Notebook.
2. Run the notebook cells in order.
3. Review the EDA and preprocessing outputs.
4. Train the neural network and inspect validation metrics.
5. Evaluate the final model on the test set.
6. Save and verify the model weights and feature scaler.

## Saved Artifacts

- `wine_classifier.pth`  trained PyTorch model state dictionary.
- `wine_scaler.joblib`  fitted `StandardScaler` used to preprocess input features.

For inference, apply the saved scaler to input features before passing the resulting values to the model. The input must contain the same 13 features in the same order as during training.

**Security note:** Only load trusted model and Joblib files. Joblib artifacts can execute code when deserialized.

## Technologies Used

- Python
- PyTorch
- NumPy
- Pandas
- Matplotlib
- Seaborn
- scikit-learn
- Joblib
- Google Colab

## Future Improvements

- Perform stratified cross-validation.
- Evaluate performance on additional independent data.
- Add a reusable inference module.
- Add automated tests for preprocessing and prediction.
- Compare the neural network with baseline models such as Logistic Regression and Random Forest.
- Add a reproducible local execution setup.

## Learning Outcomes

This project provided practical experience with exploratory data analysis, feature scaling, data splitting, PyTorch tensors, DataLoaders, neural network training, classification metrics, and model artifact management.

It also reinforced the importance of evaluating generalization instead of relying only on accuracy from a single test split.

## License

This project is intended for educational and portfolio purposes. If you choose the MIT License, include a `LICENSE` file containing the standard MIT License text.



**Author:** Your GitHub username

