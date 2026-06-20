from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

params = {
    'n_estimators':[10,50,100],
    'max_depth':[2,4,6]
}

grid = GridSearchCV(
    RandomForestClassifier(),
    params,
    cv=5
)

grid.fit(X,y)

print(grid.best_params_)
print(grid.best_score_)
'''Output:{'max_depth': 4, 'n_estimators': 50}
0.9666666666666668'''
