# Credit-Wise-Loan-Approval-System
Predict the price of Houses based on the location and house description.

## Project Overview

This project focuses on predicting whether a loan application will be approved or rejected using machine learning classification algorithms.

The project uses applicant information such as income, credit score, employment status, loan amount, debt-to-income ratio, and other financial and personal attributes to explore the factors associated with loan approval.

It covers the complete machine learning workflow, from data preprocessing and exploratory data analysis to model training, hyperparameter tuning, and performance evaluation.

## Objectives

- Clean the dataset and handle missing values.
- Perform Exploratory Data Analysis (EDA) to understand the data and identify patterns.
- Visualize relationships between applicant characteristics and loan approval.
- Convert categorical variables into numerical representations.
- Analyze correlations between features and the target variable.
- Train and evaluate multiple classification models.
- Improve model performance through feature engineering and hyperparameter tuning.

## Technologies and Libraries

- **Python** — Programming language
- **Pandas** — Data manipulation and analysis
- **NumPy** — Numerical operations
- **Matplotlib** — Data visualization
- **Seaborn** — Statistical data visualization
- **Scikit-learn** — Data preprocessing, model training, hyperparameter tuning, and evaluation

## Project Workflow

### 1. Data Loading and Inspection

The dataset is loaded using Pandas. Initial inspection is performed to understand its structure, column names, data types, and dimensions.

### 2. Data Cleaning and Preprocessing

Missing values are handled using `SimpleImputer`:

- Numerical columns: Missing values are replaced with the mean.
- Categorical columns: Missing values are replaced with the most frequent value.

The `Applicant_ID` column is removed because it is an identifier rather than a meaningful predictive feature.

### 3. Exploratory Data Analysis (EDA)

Data visualization is used to explore the dataset and identify patterns related to loan approval.

The analysis includes:

- **Credit Score and Collateral:** Examining their distributions in relation to loan approval.
- **Income Analysis:** Exploring applicant and co-applicant income distributions.
- **Employment Status:** Analyzing the distribution of employment categories.
- **Outlier Analysis:** Using box plots to examine applicant income, credit score, debt-to-income ratio, and loan amount.
- **Class Distribution:** Examining the distribution of approved and rejected applications.
- **Property Area and Loan Purpose:** Analyzing the distribution of property locations and loan purposes.
- **Gender Distribution:** Exploring the distribution of applicants by gender.
- **Correlation Analysis:** Using a correlation matrix and heatmap to examine relationships between numerical features.

### 4. Categorical Encoding

Categorical variables are converted into numerical representations so they can be used by machine learning algorithms.

The project uses:

- **Label Encoding:** Applied to `Education_Level` and `Loan_Approved`.
- **One-Hot Encoding:** Applied to `Employment_Status`, `Marital_Status`, `Loan_Purpose`, `Property_Area`, `Gender`, and `Employer_Category`.

### 5. Train-Test Split and Feature Scaling

The dataset is divided into training and testing sets using an 80:20 split, with `random_state=42`.

`StandardScaler` is used to standardize the input features for model training.

### 6. Machine Learning Models

The following classification algorithms are implemented:

**Logistic Regression**

Used as a classification model to predict loan approval based on applicant features.

**K-Nearest Neighbors (KNN)**

Classifies loan applications based on the similarity between applicant records.

**Gaussian Naive Bayes**

Applies the Naive Bayes classification approach to predict loan approval.

### 7. Model Evaluation

The models are evaluated using the following metrics:

- **Accuracy:** Measures the proportion of correctly classified applications.
- **Precision:** Measures the proportion of predicted approvals that are actual approvals.
- **Recall:** Measures the proportion of actual approvals correctly identified.
- **F1-Score:** Combines precision and recall into a single metric.
- **Mean Squared Error (MSE):** Calculates the average squared difference between encoded target values and predicted class labels.
- **Confusion Matrix:** Shows correct and incorrect classifications for each class.

These metrics help evaluate different aspects of model performance.

### 8. Hyperparameter Tuning

`GridSearchCV` is used to tune the KNN classifier.

The project tests different values of `n_neighbors`:

- 3
- 5
- 7
- 9

Five-fold cross-validation is used with precision as the scoring metric to select the best-performing parameter configuration.

### 9. Feature Engineering and Model Re-evaluation

Additional features are created by squaring two numerical variables:

- `DTI_Ratio_sq`
- `Credit_Score_sq`

The original `DTI_Ratio` and `Credit_Score` columns are then removed from the input features in this modeling stage.

The models are trained and evaluated again to examine their performance after feature engineering.

## Installation and Setup

### Prerequisites

- Python 3.9 or later
- Jupyter Notebook or JupyterLab
- Basic knowledge of Python and machine learning

### Step 1: Clone the Repository

```bash
git clone <your-github-repository-url>
```

Navigate to the project directory:

```bash
cd <your-repository-folder>
```

### Step 2: Install the Required Libraries

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### Step 3: Add the Dataset

Place the `loan_approval_data.csv` dataset in the appropriate project directory.

Update the dataset path in the code to match your local file location.

For example, if the CSV file is inside a `Data` folder:

```python
df = pd.read_csv("Data/loan_approval_data.csv")
```

### Step 4: Run the Project

Open the project notebook in Jupyter Notebook or JupyterLab and execute the cells in order.

If the code is saved as a Python script, run it using:

```bash
python loan_approval_prediction.py
```

Replace the filename with the actual name of your Python script.

## Dataset Features

The project uses applicant and loan-related attributes, including:

| Feature | Description |
|---|---|
| `Applicant_ID` | Applicant identifier; removed during preprocessing |
| `Applicant_Income` | Income of the applicant |
| `Coapplicant_Income` | Income of the co-applicant |
| `Credit_Score` | Credit score of the applicant |
| `Collateral_Value` | Value of the collateral |
| `DTI_Ratio` | Debt-to-income ratio |
| `Loan_Amount` | Requested loan amount |
| `Education_Level` | Applicant's education category |
| `Employment_Status` | Applicant's employment category |
| `Marital_Status` | Applicant's marital status |
| `Loan_Purpose` | Purpose of the loan |
| `Property_Area` | Property location category |
| `Gender` | Applicant's gender |
| `Employer_Category` | Employer category |
| `Loan_Approved` | Target variable indicating loan approval status |

*Note: This table lists the columns identifiable from the project code. The dataset's complete schema and individual feature definitions should be verified against the original CSV file.*

## Key Learnings

Through this project, I gained practical experience with:

- Data cleaning and missing-value imputation.
- Exploratory data analysis and data visualization.
- Categorical feature encoding.
- Correlation analysis and outlier detection.
- Train-test splitting and feature scaling.
- Implementing classification algorithms with Scikit-learn.
- Evaluating models using multiple classification metrics.
- Hyperparameter tuning using GridSearchCV.
- Feature engineering and model re-evaluation.

## Limitations and Future Improvements

Potential improvements to the project include:

- Comparing all models using a consolidated evaluation table.
- Addressing class imbalance if it is present in the dataset.
- Using a Scikit-learn pipeline to prevent preprocessing leakage between training and testing data.
- Exploring additional classification algorithms, such as Decision Trees and Random Forest.
- Saving the best-performing model and scaler for predictions on new applications.
- Performing feature importance or model interpretation analysis.

## Disclaimer

This project is developed for educational purposes and demonstrates a machine learning approach to loan approval prediction.

The predictions are based on the available dataset and should not be treated as financial advice or used as the sole basis for real-world lending decisions.

## Author

**Ananda**

Aspiring Data Scientist | Python | Data Analysis | Machine Learning
