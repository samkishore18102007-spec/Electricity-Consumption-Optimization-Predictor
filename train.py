import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Load Kaggle dataset
data = pd.read_csv("electricity_bill_dataset.csv")

print("Dataset Loaded Successfully!")
print("Dataset Shape:", data.shape)
print("\nDataset Columns:")
print(data.columns.tolist())

# Separate features and target
X = data.drop("ElectricityBill", axis=1)
y = data["ElectricityBill"]

# Identify categorical and numerical columns
categorical_features = ["City", "Company"]

numerical_features = [
    column for column in X.columns
    if column not in categorical_features
]

# Preprocess categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ("categorical",
         OneHotEncoder(handle_unknown="ignore"),
         categorical_features)
    ],
    remainder="passthrough"
)

# Create machine learning pipeline
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    ))
])

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Train model
print("\nTraining Random Forest model...")
model.fit(X_train, y_train)

# Evaluate model
predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("\nModel Training Completed!")
print("Mean Absolute Error:", round(mae, 2))
print("R2 Score:", round(r2, 4))

# Save trained model
joblib.dump(model, "electricity_model.pkl")

print("\nTrained model saved successfully!")