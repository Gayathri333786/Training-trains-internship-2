actual_marks = 80

predictions = [60, 68, 74, 78, 80]

for p in predictions:
    loss = abs(actual_marks - p)
    print(f"Prediction: {p}, Loss: {loss}")
