import pandas as pd
from sklearn.cluster import KMeans

data = {
    'Spending':[1000,1200,1500,15000,17000]
}

df = pd.DataFrame(data)

model = KMeans(
    n_clusters=2,
    random_state=42
)

df['Cluster'] = model.fit_predict(df)

print(df)
'''Output:Spending  Cluster
0      1000        0
1      1200        0
2      1500        0
3     15000        1
4     17000        1'''
