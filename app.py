from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import declarative_base, sessionmaker

# Load model and vectorizers
model = joblib.load("mlp_task_suggester_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")
mlb = joblib.load("label_binarizer.pkl")

# Set up database
DATABASE_URL = "postgresql://task_database_pb63_user:tUG9ChwXYE3aWTgwzSIOIjQRAfgbJXg8@dpg-d1v2qser433s73f9iol0-a.oregon-postgres.render.com/task_database_pb63"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Task model
class Task(Base):
    __tablename__ = "Task"
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String)
    description = Column(String)
    userName = Column(String)

Base.metadata.create_all(bind=engine)

# Set up Flask app
app = Flask(__name__)
CORS(app)

@app.route("/")
def home():
    return "Multilabel Task Suggestion API is live!"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    # 🔍 Log the raw incoming request
    print("📥 Received request:", data)

    text = data.get("text")
    user = data.get("userName")

    if not text or not user:
        return jsonify({"error": "Both 'text' and 'userName' are required"}), 400

    # Make prediction
    X = vectorizer.transform([text])
    pred = model.predict(X)
    tasks = mlb.inverse_transform(pred)[0]

    session = SessionLocal()
    result = []

    for i, task in enumerate(tasks):
        task_entry = Task(
            title=f"Task {i+1}",
            description=task,
            userName=user
        )
        session.add(task_entry)
        result.append({
            "Title": task_entry.title,
            "Description": task_entry.description,
            "userName": user
        })

    session.commit()
    session.close()

    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)
