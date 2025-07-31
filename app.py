from flask import Flask, request, jsonify
from flask_cors import CORS
from sqlalchemy import create_engine, Column, String, Boolean, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import joblib
import uuid
from datetime import datetime

# ------------------ DB SETUP ------------------ #
DATABASE_URL = "postgresql://task_database_pb63_user:tUG9ChwXYE3aWTgwzSIOIjQRAfgbJXg8@dpg-d1v2qser433s73f9iol0-a.oregon-postgres.render.com/task_database_pb63"
engine = create_engine(DATABASE_URL)
Base = declarative_base()
SessionLocal = sessionmaker(bind=engine)

# ------------------ DB MODEL ------------------ #
class Task(Base):
    __tablename__ = "tasktable"
    id = Column(String, primary_key=True, index=True)
    Title = Column(String, nullable=False)
    Description = Column(String, nullable=False)
    DateCreated = Column(DateTime, default=datetime.utcnow)
    LastUpdate = Column(DateTime, default=datetime.utcnow)
    isDeleted = Column(Boolean, default=False)
    isCompleted = Column(Boolean, default=False)
    UserName = Column(String, nullable=False, default="null")

# ------------------ APP SETUP ------------------ #
app = Flask(__name__)
CORS(app)

# Load ML model components
model = joblib.load("mlp_task_suggester_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")
mlb = joblib.load("label_binarizer.pkl")

# Label to expanded description
TASK_DESCRIPTIONS = {
    "rest": "Take a break and lie down in a quiet place.",
    "drink water": "Hydrate with a full glass of clean water.",
    "check blood sugar": "Use your glucometer to monitor sugar levels.",
    "eat snack": "Have a healthy snack to stabilize your blood sugar.",
    "log sugar": "Record your sugar levels in your tracking app or journal.",
    "meditate": "Spend 5-10 minutes doing deep breathing or meditation.",
    "take insulin": "Administer your prescribed insulin dosage.",
    "call doctor": "Contact your healthcare provider for advice.",
    "walk": "Take a 10-minute walk to regulate your blood sugar.",
    "relax": "Find a comfortable space and listen to calming music."
}

@app.route("/")
def home():
    return "✅ Task Suggester API with DB is Live"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    print("📥 Received request:", data)

    text = data.get("text", "")
    username = data.get("username", "")

    if not text or not username:
        return jsonify({"error": "Both 'text' and 'username' are required"}), 400

    # Predict
    X = vectorizer.transform([text])
    probs = model.predict_proba(X)
    threshold = 0.3
    pred_binary = (probs >= threshold).astype(int)
    predicted_labels = mlb.inverse_transform(pred_binary)[0]

    # Save to DB
    session = SessionLocal()
    task_objects = []
    for label in predicted_labels:
        new_task = Task(
            id=str(uuid.uuid4()),
            Title=label,
            Description=TASK_DESCRIPTIONS.get(label, label),
            LastUpdate=datetime.utcnow(),
            UserName=username
        )
        session.add(new_task)
        task_objects.append(new_task)

    session.commit()

    # Prepare JSON result
    result = [
        {
            "Title": t.Title,
            "Description": t.Description,
            "UserName": t.UserName
        } for t in task_objects
    ]

    return jsonify(result), 201

if __name__ == "__main__":
    app.run(debug=True)
