import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

df = pd.read_csv("dataset/customers.csv")

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

X = df[['Age','AnnualIncome','SpendingScore']]
X_scaled = scaler.transform(X)

clusters = model.predict(X_scaled)

# PCA Visualization
pca = PCA(n_components=2)
data_2d = pca.fit_transform(X_scaled)

plt.figure(figsize=(10,6))

plt.scatter(
    data_2d[:,0],
    data_2d[:,1],
    c=clusters,
    cmap='viridis',
    s=100
)

plt.title("Customer Segmentation Clusters")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.colorbar()

plt.savefig("screenshots/clusters.png")
plt.show()
