from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

pipe = Pipeline([
    ('scaler',StandardScaler()),
    ('model',SVC())
])

pipe.fit(X_train,y_train)

print(pipe.score(X_test,y_test))
#Output:1.0
