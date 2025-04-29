# Titanic Survival Prediction 🚢
This project was completed as part of the GrowthLink Data Science Internship selection process.

## 📌 Objective
Develop a machine learning model to predict whether a passenger survived the Titanic disaster using historical data.

## 📁 Files
- `titanic_survival_prediction.py`: Python script with full data preprocessing, model training, and evaluation.
- `tested.csv`: Titanic dataset used for the task (provided by GrowthLink).
- `Titanic_Survival_Prediction.ipynb`: (Optional) Jupyter Notebook version of the solution.

## 🧪 Steps Performed
1. **Data Cleaning**: Removed irrelevant columns and handled missing values.
2. **Feature Engineering**: Encoded categorical variables and scaled numerical values.
3. **Modeling**: Trained Logistic Regression and Random Forest classifiers.
4. **Evaluation**: Models achieved 100% accuracy on the test set (possible due to simplified dataset).

## 🔧 Requirements
- Python 3.7+
- pandas, scikit-learn, numpy

Install requirements with:
```bash
pip install -r requirements.txt
```

## ▶️ How to Run
### Option 1: Jupyter Notebook
```bash
jupyter notebook Titanic_Survival_Prediction.ipynb
```

### Option 2: Python Script
```bash
python titanic_survival_prediction.py
```

## 📊 Model Performance (on test set)
| Model               | Accuracy | Precision | Recall | F1 Score |
|--------------------|----------|-----------|--------|----------|
| Logistic Regression| 100%     | 100%      | 100%   | 100%     |
| Random Forest       | 100%     | 100%      | 100%   | 100%     |

> Note: Real-world performance may differ. Use original Kaggle Titanic dataset for more robust training and evaluation.

## 📬 Contact
For queries or issues, please contact: [help.growthlink@gmail.com](mailto:help.growthlink@gmail.com)
