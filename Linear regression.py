import pandas as pd
from sklearn.linear_model import LinearRegression

# Dataset
data = {
    'Experience': [1, 2, 3, 4, 5],
    'Salary': [25000, 30000, 35000, 40000, 45000]
}

df = pd.DataFrame(data)

X = df[['Experience']]
y = df['Salary']

# Model
model = LinearRegression()
model.fit(X, y)

# Prediction
prediction = model.predict([[6]])

print("Predicted Salary:", prediction[0])
#Output: Predicted Salary: 50000.0
