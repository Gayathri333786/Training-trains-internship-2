import pandas as pd
from sklearn.linear_model import LinearRegression

data = {
    'Area': [1000, 1500, 2000],
    'Bedrooms': [2, 3, 4],
    'Age': [5, 2, 1],
    'Price': [300000, 450000, 600000]
}

df = pd.DataFrame(data)

X = df[['Area', 'Bedrooms', 'Age']]
y = df['Price']

model = LinearRegression()
model.fit(X, y)

new_house = [[1800, 3, 2]]

prediction = model.predict(new_house)

print("Predicted House Price:", prediction[0])
#Outpur: Predicted House Price: 539999.64000144
