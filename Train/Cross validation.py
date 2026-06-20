from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

iris = load_iris()

X = iris.data
y = iris.target

model = DecisionTreeClassifier()

scores = cross_val_score(
    model,
    X,
    y,
    cv=5
)

print("Scores:", scores)
print("Average Accuracy:", scores.mean())
'''Output: Scores: [0.96666667 0.96666667 0.9        1.         1.        ]
Average Accuracy: 0.9666666666666668'''
