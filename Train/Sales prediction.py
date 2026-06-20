import pandas as pd
from sklearn.linear_model import LinearRegression

data = {
    'Advertising':[1000,2000,3000,4000],
    'Sales':[15000,25000,35000,45000]
}

df = pd.DataFrame(data)

X = df[['Advertising']]
y = df['Sales']

model = LinearRegression()

model.fit(X,y)

prediction = model.predict([[5000]])

print("Predicted Sales:", prediction[0])
#Output:Predicted Sales: 55000.0
