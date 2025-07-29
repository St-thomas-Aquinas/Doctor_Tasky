from flask import Flask, request, jsonify
import joblib

# Load everything
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")
mlb = joblib.load("label_binarizer.pkl")

app = Flask(__name__)

@app.route("/")
def home():
    return "Multilabel Task Suggestion API is live!"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data.get("text", "")
    
    # Transform
    X = vectorizer.transform([text])
    pred = model.predict(X)
    tasks = mlb.inverse_transform(pred)[0]
    
    # (Optional) Add dummy schedule info
    result = [{"task": task, "when": "as soon as possible", "after_minutes": 15} for task in tasks]
    
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)
