import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

X = np.array([1, 2, 3, 4, 5]).reshape(-1,1)
y = np.array([10, 25, 50, 80, 95])

poly = PolynomialFeatures(degree=2)

X_poly = poly.fit_transform(X)

model = LinearRegression()
model.fit(X_poly, y)

prediction = model.predict(poly.transform([[6]]))

print("Predicted Score:", prediction[0])

# Visualization
plt.scatter(X, y)

x_range = np.linspace(1,6,100).reshape(-1,1)

plt.plot(
    x_range,
    model.predict(poly.transform(x_range))
)

plt.xlabel("Hours Studied")
plt.ylabel("Score")
plt.title("Polynomial Regression")
plt.show()
