# Diabetes Prediction using Random Forest

This project aims to predict the presence of diabetes in patients using the Random Forest algorithm. The dataset contains various health-related features of patients, and the goal is to build a model that can accurately predict whether a patient has diabetes based on these features.

## Table of Contents

1. [Introduction](#introduction)
2. [Data Loading and Preprocessing](#data-loading-and-preprocessing)
   - [Loading the Data](#loading-the-data)
   - [Handling Missing Values](#handling-missing-values)
   - [Data Transformation](#data-transformation)
3. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
   - [Distribution of Heart Disease](#distribution-of-heart-disease)
   - [Distribution of Cholesterol](#distribution-of-cholesterol)
4. [Modeling](#modeling)
   - [Train-Test Split](#train-test-split)
   - [Model Training and Evaluation](#model-training-and-evaluation)
     - [Random Forest Classifier](#random-forest-classifier)
     - [XGBoost Classifier](#xgboost-classifier)
5. [Conclusion](#conclusion)

## Project Overview

In this project, we use a dataset containing various health-related features of patients to predict the presence of diabetes. We employ the Random Forest algorithm for this task and evaluate the model's performance using various metrics.

## Dataset

The dataset used in this project is `diabetes_prediction_dataset.csv`. It contains the following columns [1]:

- `Age`: Age of the patient [years]
- `Sex`: Sex of the patient [M: Male, F: Female]
- `ChestPainType`: Chest pain type [TA: Typical Angina, ATA: Atypical Angina, NAP: Non-Anginal Pain, ASY: Asymptomatic]
- `RestingBP`: Resting blood pressure [mm Hg]
- `Cholesterol`: Serum cholesterol [mm/dl]
- `FastingBS`: Fasting blood sugar [1: if FastingBS > 120 mg/dl, 0: otherwise]
- `RestingECG`: Resting electrocardiogram results [Normal: Normal, ST: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV), LVH: showing probable or definite left ventricular hypertrophy by Estes' criteria]
- `MaxHR`: Maximum heart rate achieved [Numeric value between 60 and 202]
- `ExerciseAngina`: Exercise-induced angina [Y: Yes, N: No]
- `Oldpeak`: Oldpeak = ST [Numeric value measured in depression]
- `ST_Slope`: The slope of the peak exercise ST segment [Up: upsloping, Flat: flat, Down: downsloping]
- `HeartDisease`: Output class [1: heart disease, 0: Normal]

## Installation

First, the Poetry need to be installed. If you don't have Poetry installed, you can install it by following the instructions on the [Poetry website](https://python-poetry.org/docs/#installation).


First, configure Poetry to create virtual environments inside the project directory:

```bash
poetry config virtualenvs.in-project true
```

Then, install the dependencies (based on pyptoject.toml and corresponding poetry.lock):

```bash
poetry install
```

## Usage

To run the notebook, use the following command:

```bash
poetry run jupyter notebook
```

## Setting the Kernel to Use Poetry Virtual Environment:

To use the Poetry virtual environment as the kernel in Jupyter Notebook, use created .venv environment as kernel.

```bash
poetry shell
ipython kernel install --name "capstone" --user 
```
![alt text](image-1.png)
![alt text](image-2.png)



## Exploratory Data Analysis (EDA) and Feature Engineering



Before building the model, we perform exploratory data analysis to understand the structure of the data and check for any inconsistencies or missing values. This includes visualizing the distribution of the target variable (`diabetes`) and other features.

We explore creating new features or transforming existing ones to provide the model with more informative inputs.
During EDA, we create new features such as `bmi_category` and `age_range`:
- **BMI Category**: Categorizes the BMI values into groups such as Underweight, Normal, Overweight, and Obese.
- **Age Range**: Categorizes the age values into ranges such as 0-18, 19-30, 31-45, 46-60, and 60+.

## Model Training

We use the Random Forest algorithm to train our model on the training data. The model is trained using the `RandomForestClassifier` from `sklearn`.

## Model Evaluation

We evaluate the model's performance using various metrics such as accuracy, confusion matrix, classification report, and ROC curve. We also analyze the feature importance to understand which features contribute most to the prediction.

## Hyperparameter Tuning

We use `GridSearchCV` to perform hyperparameter tuning and find the optimal hyperparameters for the Random Forest model. This helps in enhancing the model's performance.

## Handling Imbalanced Data

If the dataset has an imbalanced distribution of classes, we use techniques like SMOTE (Synthetic Minority Over-sampling Technique) to balance the classes and improve the model's performance.

## Conclusion

The Random Forest model achieved an accuracy of approximately 95%, indicating a high level of performance in predicting diabetes based on the provided features. The confusion matrix and classification report further confirm the model's reliability, showing minimal false positives and false negatives. Feature importance analysis reveals which factors are most influential in the predictions, providing valuable insights for healthcare professionals.

## Additional Consideration

- **Hyperparameter Tuning:** We utilize techniques like `GridSearchCV` to find the optimal hyperparameters for the Random Forest model, potentially enhancing its performance.
- **Handling Imbalanced Data:** If the dataset has an imbalanced distribution of classes, we may consider using techniques like SMOTE or undersampling to balance the classes.
- **Feature Engineering:** Explore creating new features or transforming existing ones to provide the model with more informative inputs.

By following these recommendations, we further improve the model's performance and gain deeper insights into the factors influencing diabetes.

## Training the model
To train the model, run the following command:

```bash
python train.py
```

**Using an Existing Model**

If you want to use an existing model and DictVectorizer, you can use them by moving the files from the model directory to the current directory.


If you have an existing model and DictVectorizer, you can use them by moving the files from the model directory to the current directory:

This will generate the model and save it along with the DictVectorizer in the current directory.

## Using the model

To use the model, follow these steps:

1. Build the Docker image using the following command:

```bash
docker build -t predict:latest .
```

2. Run the Docker container using the following command:

```bash
docker run -d -p 9696:9696 predict:latest
```

This will start the FastAPI application inside the Docker container, and it will be accessible at http://localhost:9696.

3. Run the request.py script to send data to the FastAPI endpoint and get predictions:

```bash
python request.py
```


## Bibliography

For more information, please refer to the following sources:

1. [Heart Disease Prediction using Machine Learning](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction/data)
