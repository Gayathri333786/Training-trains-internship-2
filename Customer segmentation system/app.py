from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

cluster_names = {
    0:"Budget Customers",
    1:"Premium Customers",
    2:"Luxury Shoppers",
    3:"Regular Customers"
}

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    age = int(request.form['age'])
    income = float(request.form['income'])
    score = float(request.form['score'])

    data = pd.DataFrame({
        'Age':[age],
        'AnnualIncome':[income],
        'SpendingScore':[score]
    })

    scaled = scaler.transform(data)

    cluster = model.predict(scaled)[0]

    segment = cluster_names.get(cluster,"Unknown")

    return render_template(
        "result.html",
        cluster=cluster,
        segment=segment,
        age=age,
        income=income,
        score=score
    )

if __name__ == "__main__":
    app.run(debug=True)
