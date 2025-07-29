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

# Load ML model
model = joblib.load("mlp_task_suggester_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")
mlb = joblib.load("label_binarizer.pkl")

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

    # Run model prediction
    X = vectorizer.transform([text])
    pred = model.predict(X)
    tasks = mlb.inverse_transform(pred)[0]

    # Format and save to DB
    session = SessionLocal()
    task_objects = []
    for i, task in enumerate(tasks):
        new_task = Task(
            id=str(uuid.uuid4()),
            Title=f"Task {i + 1}",
            Description=task,
            LastUpdate=datetime.utcnow(),
            UserName=username
        )
        session.add(new_task)
        task_objects.append(new_task)

    session.commit()

    # Query back the tasks created by this user (optional log)
    user_tasks = session.query(Task).filter_by(UserName=username).all()
    print(f"🧾 Saved Tasks for {username}:")
    for t in user_tasks:
        print(f"- {t.Title}: {t.Description}")

    # Response for client
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
