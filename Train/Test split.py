from sklearn.model_selection import train_test_split

X = df[['Hours']]
y = df['Result']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

print("Training Samples:", len(X_train))
print("Testing Samples:", len(X_test))
'''Output: Training Samples: 4
Testing Samples: 1'''
