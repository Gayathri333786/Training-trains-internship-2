import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

df = pd.read_csv(
    "dataset/customers.csv"
)

X = df[
    ["AnnualIncome","SpendingScore"]
]

model = KMeans(
    n_clusters=4,
    random_state=42
)

df["Cluster"] = model.fit_predict(X)

plt.figure(figsize=(8,6))

plt.scatter(
    df["AnnualIncome"],
    df["SpendingScore"],
    c=df["Cluster"]
)

plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.title("Customer Segmentation")

plt.savefig(
    "screenshots/clusters.png"
)

plt.show()

print("Image Saved")
