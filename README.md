# Employee Salary Prediction Using Lasso Regression

## Project Overview

Employee Salary Prediction is a Machine Learning project that predicts an employee's estimated salary based on their professional and educational details.

This project uses **Lasso Regression**, a supervised Machine Learning algorithm, to predict salaries based on factors such as job title, years of experience, education level, skills, industry, company size, location, remote work, and certifications.

A web application is developed using **Streamlit**, allowing users to enter employee information and get salary predictions through an interactive interface.

## Project Objectives

- Predict employee salaries using Machine Learning.
- Understand the relationship between employee characteristics and salary.
- Apply data preprocessing and feature engineering.
- Implement Lasso Regression for salary prediction.
- Understand how Lasso performs feature selection using regularization.
- Build an interactive web application using Streamlit.

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| Pandas | Data manipulation |
| NumPy | Numerical operations |
| Matplotlib | Data visualization |
| Seaborn | Exploratory data analysis |
| Scikit-learn | Machine Learning and preprocessing |
| Lasso Regression | Salary prediction |
| Joblib | Saving and loading trained models |
| Streamlit | Web application development |

## Dataset Information

**Dataset:** `job_salary_prediction_dataset.csv`

The dataset contains 250,000 records and 10 columns.

### Dataset Features

| Feature | Description |
|---|---|
| job_title | Employee's job title |
| experience_years | Years of work experience |
| education_level | Employee's education level |
| skills_count | Number of skills |
| industry | Industry category |
| company_size | Size of the company |
| location | Employee's location |
| remote_work | Remote work arrangement |
| certifications | Number of certifications |
| salary | Target variable to predict |

**Target Variable:** `salary`

## Project Workflow

1. Data Collection
2. Data Loading
3. Exploratory Data Analysis (EDA)
4. Data Cleaning and Validation
5. Feature Selection
6. Train-Test Split
7. Categorical Feature Encoding
8. Numerical Feature Scaling
9. Lasso Regression Model Training
10. Model Evaluation
11. Alpha Parameter Experimentation
12. Model Saving using Joblib
13. Streamlit Web Application Development

## Machine Learning Algorithm

### Lasso Regression

Lasso Regression (Least Absolute Shrinkage and Selection Operator) is a regularized linear regression algorithm that uses L1 regularization.

It reduces the impact of less important features by shrinking their coefficients. Some coefficients can become exactly zero, which helps in feature selection.

**Why Lasso Regression?**

- Helps reduce overfitting.
- Performs automatic feature selection.
- Produces sparse models.
- Helps identify less important features.

### Alpha Parameter

The alpha parameter controls the strength of regularization.

- Lower alpha: Weaker regularization.
- Higher alpha: Stronger regularization.
- Very high alpha: Can cause underfitting.

Different alpha values were experimented with to understand their effect on model performance.

## Model Evaluation

The trained model was evaluated using the following regression metrics:

| Evaluation Metric | Result | Interpretation |
|---|---:|---|
| MAE | 5,435.81 | Average absolute prediction error |
| MSE | 50,771,165.43 | Mean squared prediction error |
| RMSE | 7,125.39 | Root mean squared prediction error |
| R² Score | 0.96347 | Approximately 96.35% of target variance explained |

**Note:** These are the evaluation results from the model run. Results may vary if the dataset, train-test split, or model parameters change.

## Model Comparison

Linear Regression and Lasso Regression were compared using the same test dataset.

| Metric | Linear Regression | Lasso Regression |
|---|---:|---:|
| MAE | 5,436.10 | 5,435.81 |
| RMSE | 7,125.52 | 7,125.39 |
| R² Score | 0.963469 | 0.963470 |

Lasso Regression achieved slightly better results in this experiment.

## Streamlit Web Application

The trained model is integrated into an interactive Streamlit application.

### Application Features

- Employee job title selection
- Experience input
- Education level selection
- Skills count input
- Industry selection
- Company size selection
- Location selection
- Remote work selection
- Certification count input
- Real-time salary prediction after form submission

### Application Workflow

1. User enters employee details.
2. Streamlit collects the input values.
3. The saved preprocessor transforms the input data.
4. The trained Lasso Regression model predicts the salary.
5. The application displays the predicted salary.

## Project Structure

```text
Employee_Salary_Prediction/
│
├── app.py
│
├── final_lasso_salary_model.pkl
│
├── salary_preprocessor.pkl
│
├── README.md
│
└── job_salary_prediction_dataset.csv
```

The dataset and Jupyter Notebook can also be included in the repository if redistribution is permitted.

## Installation and Setup

### 1. Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### 2. Navigate to the Project Folder

```bash
cd Employee_Salary_Prediction
```

### 3. Install Required Libraries

```bash
pip install pandas numpy matplotlib seaborn scikit-learn streamlit joblib
```

### 4. Run the Streamlit Application

```bash
streamlit run app.py
```

### 5. Open the Application

Open the following URL in your browser:

```text
http://localhost:8501
```

## Saved Model Files

Two trained objects are saved using Joblib:

- `final_lasso_salary_model.pkl` – Contains the trained Lasso Regression model.
- `salary_preprocessor.pkl` – Contains the preprocessing transformations used during model training.

Both files are required to run the application.

## Future Improvements

- Improve the application interface and user experience.
- Add downloadable salary prediction reports.
- Add interactive salary analysis dashboards.
- Experiment with Ridge Regression and Elastic Net.
- Optimize model hyperparameters using cross-validation.
- Deploy the application online.
- Add more relevant features and evaluate performance on additional data.

## Key Learnings

Through this project, I learned:

- Exploratory Data Analysis using Python.
- Data preprocessing and feature encoding.
- Feature scaling using StandardScaler.
- Linear Regression and Lasso Regression.
- L1 regularization and feature selection.
- Model evaluation using regression metrics.
- Saving and loading Machine Learning models using Joblib.
- Integrating a trained Machine Learning model with Streamlit.
- Building an interactive Machine Learning web application.

## Author

**Poorva Dhananjay Chavan**

Machine Learning | Python | Data Analytics

---

*This project was developed for learning and educational purposes. Predicted salaries are estimates based on the training dataset and should not be treated as guaranteed salary offers. Salary units, currency, and pay period should be verified before interpreting predictions.*
