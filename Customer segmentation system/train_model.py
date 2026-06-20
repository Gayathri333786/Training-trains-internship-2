import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib

# Load dataset
df = pd.read_csv("dataset/customers.csv")

X = df[['Age','AnnualIncome','SpendingScore']]

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# KMeans
model = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=10
)

model.fit(X_scaled)

# Save model
joblib.dump(model,"model.pkl")
joblib.dump(scaler,"scaler.pkl")

print("Model Trained Successfully")
