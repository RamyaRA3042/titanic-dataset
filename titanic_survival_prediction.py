# Titanic Survival Prediction - GrowthLink Internship Task

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load dataset
df = pd.read_csv("tested.csv")

# Drop irrelevant or sparse columns
df_clean = df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"])

# Fill missing values
imputer = SimpleImputer(strategy='median')
df_clean[["Age", "Fare"]] = imputer.fit_transform(df_clean[["Age", "Fare"]])

# Encode categorical features
label_encoders = {}
for col in ["Sex", "Embarked"]:
    le = LabelEncoder()
    df_clean[col] = le.fit_transform(df_clean[col])
    label_encoders[col] = le

# Separate features and target
X = df_clean.drop(columns="Survived")
y = df_clean["Survived"]

# Normalize numeric columns
scaler = StandardScaler()
X[["Age", "Fare"]] = scaler.fit_transform(X[["Age", "Fare"]])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
}

# Train and evaluate
results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred)
    })

# Display results
results_df = pd.DataFrame(results)
print(results_df)
