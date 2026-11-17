
import pandas as pd
import joblib

# Load saved model
loaded_model = joblib.load("models/churn_model.pkl")

print("Model loaded successfully!")

# Load dataset
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Clean TotalCharges exactly as we did during training
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"], errors="coerce"
)
df["TotalCharges"] = df["TotalCharges"].fillna(0)

# Remove customerID
df = df.drop("customerID", axis=1)

# Separate features and target
X = df.drop("Churn", axis=1)
y = df["Churn"]

# Select 5 customer records
sample_customers = X.iloc[:5]

# Predict using loaded model
predictions = loaded_model.predict(sample_customers)
probabilities = loaded_model.predict_proba(sample_customers)[:, 1]

# Display results
for i in range(len(sample_customers)):
    print(f"\nCustomer {i + 1}")
    print("Actual Churn:", y.iloc[i])
    print("Predicted Churn:", predictions[i])
    print("Churn Probability:", round(probabilities[i] * 100, 2), "%")
