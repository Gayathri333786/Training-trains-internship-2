# Simple Gradient Descent

size = 1000
actual_price = 50

weight = 0.01      # Initial guess
learning_rate = 0.000001

for i in range(10):
    predicted_price = weight * size

    error = predicted_price - actual_price

    gradient = error * size

    weight = weight - learning_rate * gradient

    print(f"Step {i+1}: Weight={weight:.4f}, Error={error:.4f}")
