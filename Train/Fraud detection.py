import pandas as pd

from sklearn.linear_model import LogisticRegression

data = {
    'Amount':[100,200,300,50000,60000],
    'Fraud':[0,0,0,1,1]
}

df = pd.DataFrame(data)

X = df[['Amount']]
y = df['Fraud']

model = LogisticRegression()

model.fit(X,y)

prediction = model.predict([[55000]])

print("Fraud:", prediction[0])
#Fraud: 1
