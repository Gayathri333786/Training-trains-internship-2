import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data = {
    'Area':[1000,1500,2000,2500],
    'Bedrooms':[2,3,4,4],
    'Price':[2500000,4500000,6000000,7500000]
}

df = pd.DataFrame(data)

X = df[['Area','Bedrooms']]
y = df['Price']

X_train,X_test,y_train,y_test = train_test_split(
    X,y,test_size=0.2,random_state=42
)

model = LinearRegression()
model.fit(X_train,y_train)

price = model.predict([[1800,3]])

print("Predicted Price:",price[0])
