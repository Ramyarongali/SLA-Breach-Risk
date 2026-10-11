import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

from xgboost import XGBClassifier


# 1. Load prepared dataset
df = pd.read_csv("prepared_incident_data.csv")

print("Dataset loaded successfully")
print("Shape:", df.shape)


# 2. Separate features and target
X = df.drop(columns=["number", "sla_breach"])
y = df["sla_breach"]


# 3. Define categorical features
categorical_features = [
    "impact",
    "urgency",
    "priority",
    "category",
    "subcategory",
    "contact_type",
    "assignment_group"
]


# 4. Define numerical features
numeric_features = [
    "reassignment_count",
    "reopen_count",
    "sys_mod_count",
    "knowledge",
    "u_priority_confirmation"
]


# 5. Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        ),
        (
            "numeric",
            "passthrough",
            numeric_features
        )
    ]
)


# 6. Create XGBoost model
model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric="logloss"
)


# 7. Create complete ML pipeline
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model)
])


# 8. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\nTraining model...")


# 9. Train model
pipeline.fit(X_train, y_train)


print("Training completed!")


# 10. Make predictions
y_pred = pipeline.predict(X_test)

y_probability = pipeline.predict_proba(X_test)[:, 1]


# 11. Calculate performance
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_probability)


# 12. Display results
print("\n==============================")
print("       MODEL RESULTS")
print("==============================")

print("Accuracy :", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall   :", round(recall, 4))
print("F1 Score :", round(f1, 4))
print("ROC-AUC  :", round(roc_auc, 4))


# 13. Confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# 14. Example breach-risk predictions
print("\nExample Breach Risk Predictions:")

for probability in y_probability[:10]:
    risk = probability * 100
    print("Breach Risk:", round(risk, 2), "%")


# 15. Save trained model
joblib.dump(pipeline, "sla_breach_model.pkl")

print("\n==============================")
print("Model saved successfully!")
print("File: sla_breach_model.pkl")
print("==============================")