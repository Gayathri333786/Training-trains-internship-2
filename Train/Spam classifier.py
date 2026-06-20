from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

messages = [
    "Win money now",
    "Claim your prize",
    "Project meeting today",
    "Let's study ML"
]

labels = [1,1,0,0]

vectorizer = CountVectorizer()

X = vectorizer.fit_transform(messages)

model = MultinomialNB()

model.fit(X, labels)

test = vectorizer.transform(
    ["Win free prize"]
)

print(model.predict(test))
#Output:[1]
