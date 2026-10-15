# PyTorch Wine Classification

![Python](https://img.shields.io/badge/Python-3.13%2B-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-ee4c2c?logo=pytorch&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-Data%20Processing-f7931e?logo=scikitlearn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-FF4B4B?logo=streamlit&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-Visualization-3F4F75?logo=plotly&logoColor=white)
![Jupyter](https://img.shields.io/badge/Notebook-Google%20Colab-orange?logo=googlecolab&logoColor=white)
![Status](https://img.shields.io/badge/Status-Completed-success)
![License](https://img.shields.io/badge/License-MIT-green)

An end-to-end multiclass wine classification project built with PyTorch and the Wine dataset from scikit-learn. The project covers exploratory data analysis, data preprocessing, neural network training, model evaluation, artifact management, and an interactive Streamlit application for predictions.

## Project Overview

The objective is to classify wine samples into one of three target classes using 13 numerical chemical measurements.

A feedforward neural network is trained with PyTorch. The fitted `StandardScaler` and model weights are saved as reusable artifacts and loaded into a Streamlit interface for interactive inference.

The project was developed in Google Colab for model development and in a local Python environment for application development.

## Key Features

- Exploratory data analysis (EDA)
- Missing-value and duplicate-record checks
- Feature correlation heatmap
- Stratified training, validation, and test splits
- Feature standardization with `StandardScaler`
- PyTorch tensors and DataLoader configuration
- Feedforward neural network with ReLU activation
- Model training with Adam and CrossEntropyLoss
- Training and validation metric monitoring
- Classification report and confusion matrix
- Model state dictionary and scaler persistence
- Interactive Streamlit prediction interface
- Ten reference samples distributed across all three classes
- Editable chemical feature inputs
- Predicted class, confidence, and class probability visualization
- Reference-label comparison for sample testing

## Dataset

The project uses the Wine dataset provided by `sklearn.datasets.load_wine`.

| Property | Value |
|---|---:|
| Total samples | 178 |
| Numerical input features | 13 |
| Target classes | 3 |
| Training samples | 106 |
| Validation samples | 36 |
| Test samples | 36 |
| Missing values | 0 |
| Duplicate records | 0 |

### Reference Sample Distribution

The Streamlit application provides ten reference samples for testing the inference workflow.

| Wine class | Reference samples |
|---|---:|
| Wine Class 0 | 4 |
| Wine Class 1 | 3 |
| Wine Class 2 | 3 |
| **Total** | **10** |

The class labels preserve the dataset's original target categories. They are not intended to represent verified grape-variety names.

## Model Architecture

The classifier is a feedforward neural network implemented using `torch.nn`.

```text
Input: 13 numerical features
             |
      Linear (13 → 64)
             |
         ReLU
             |
       Linear (64 → 3)
             |
        Class Logits
             |
      Softmax for Inference
```

The network contains **1,091 trainable parameters**.

### Training Configuration

| Parameter | Configuration |
|---|---|
| Loss function | CrossEntropyLoss |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Training epochs | 100 |
| Batch size | 16 |
| Random seed | 42 |
| Execution device | CPU |

The feature scaler is fitted using training data only and then reused for validation, testing, and inference.

## Results

The model achieved the following results on the selected data split:

| Metric | Result |
|---|---:|
| Final training accuracy | 100% |
| Best recorded validation accuracy | 97.22% |
| Test accuracy | 100% |
| Correct test predictions | 36 / 36 |

The confusion matrix showed that all 36 test samples were classified correctly.

**Evaluation limitation:** The test set contains only 36 samples. Perfect accuracy on this split does not establish guaranteed generalization to unseen data. The final training loss was lower than the validation loss, so overfitting should be considered. Stratified cross-validation and evaluation on independent data are recommended.

## Streamlit Application

The interactive interface allows users to select a reference sample, review or edit its 13 numerical features, and run a prediction using the saved model.

### Interface Features

- Minimal sidebar with essential project information
- Clean layout for the 13 input features
- Ten reference samples distributed across three classes
- Predicted wine class and model confidence
- Probability chart for all three classes
- Comparison between the reference label and predicted class
- Error handling for missing or incompatible model artifacts

Model confidence represents the model's predicted probability and should not be interpreted as a guarantee of correctness.

## Project Structure

```text
pytorch-wine-classification/
├── app.py
├── Introduction_to_Pytorch.ipynb
├── wine_classifier.pth
├── wine_scaler.joblib
├── requirements.txt
├── .gitignore
└── README.md
```

Keep the notebook filename consistent with the file committed to the repository. The `requirements.txt` and `.gitignore` files are recommended for reproducible setup and repository hygiene.

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/pytorch-wine-classification.git
cd pytorch-wine-classification
```

Replace `YOUR_USERNAME` with your GitHub username.

### 2. Create a Virtual Environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```bash
python -m pip install --upgrade pip
python -m pip install torch numpy pandas scikit-learn joblib streamlit plotly matplotlib seaborn jupyter
```

For a reproducible environment, use the project's tested and pinned package versions in `requirements.txt`.

## Run the Application

Ensure the following files are in the project directory:

- `app.py`
- `wine_classifier.pth`
- `wine_scaler.joblib`

Start the application:

```bash
python -m streamlit run app.py
```

Streamlit will provide a local URL, typically:

```text
http://localhost:8501
```

Open the URL in a browser to use the application.

## Run the Notebook

1. Open `Introduction_to_Pytorch.ipynb` in Google Colab or Jupyter Notebook.
2. Execute the notebook cells in order.
3. Review dataset exploration and preprocessing.
4. Train the neural network and inspect training metrics.
5. Evaluate the model on the test split.
6. Save and verify the model weights and fitted scaler.

## Saved Artifacts

| File | Purpose |
|---|---|
| `wine_classifier.pth` | Trained PyTorch model state dictionary |
| `wine_scaler.joblib` | Fitted feature standardization pipeline |

During inference, input values must be transformed using the saved scaler before being passed to the model. Feature order and model architecture must match the training configuration.

**Security:** Load only trusted model and Joblib artifacts. Deserializing untrusted Joblib files can execute arbitrary code.

## Technologies Used

- Python
- PyTorch
- NumPy
- Pandas
- scikit-learn
- Joblib
- Matplotlib and Seaborn
- Plotly
- Streamlit
- Google Colab
- Jupyter Notebook

## Future Improvements

- Implement stratified cross-validation
- Evaluate on additional independent data
- Add automated tests for preprocessing and inference
- Compare performance against Logistic Regression and Random Forest baselines
- Add model versioning and reproducible experiment configuration
- Add continuous integration for automated testing

## Learning Outcomes

This project provided practical experience with exploratory data analysis, feature scaling, stratified data splitting, PyTorch tensors and DataLoaders, neural network training, classification metrics, model serialization, and deployment through a Streamlit interface.

It also demonstrates the importance of distinguishing performance on a small test split from reliable generalization to unseen samples.


