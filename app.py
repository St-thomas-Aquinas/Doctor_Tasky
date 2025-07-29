from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib

# Initialize app and enable CORS
app = Flask(__name__)
CORS(app)

# Load model and preprocessing tools
model = joblib.load("mlp_task_suggester_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")
mlb = joblib.load("label_binarizer.pkl")

@app.route("/")
def home():
    return "Multilabel Task Suggestion API is live!"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data.get("text", "")
    
    # Transform text and predict
    X = vectorizer.transform([text])
    pred = model.predict(X)
    tasks = mlb.inverse_transform(pred)[0]
    
    # Return with formatted titles and incremental delay
    result = [
        {
            "Title": f"Task {i + 1}",
            "Description": task,
            "after_minutes": 15 + i * 2
        }
        for i, task in enumerate(tasks)
    ]
    
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)
