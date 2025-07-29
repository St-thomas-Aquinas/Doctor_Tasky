from flask import Flask, request, jsonify
from flask_cors import CORS
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import declarative_base, sessionmaker
import joblib

# Initialize Flask app and CORS
app = Flask(__name__)
CORS(app)

# Load ML model & tools
model = joblib.load("mlp_task_suggester_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")
mlb = joblib.load("label_binarizer.pkl")

# PostgreSQL DB setup
DATABASE_URL = "postgresql://task_database_pb63_user:tUG9ChwXYE3aWTgwzSIOIjQRAfgbJXg8@dpg-d1v2qser433s73f9iol0-a.oregon-postgres.render.com/task_database_pb63"
engine = create_engine(DATABASE_URL)
Base = declarative_base()
SessionLocal = sessionmaker(bind=engine)

# Define Task model
class Task(Base):
    __tablename__ = "tasks"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String)
    description = Column(String)
    user_name = Column(String)

# Create table if not exists
Base.metadata.create_all(bind=engine)

@app.route("/")
def home():
    return "Multilabel Task Suggestion API is live!"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data.get("text", "")
    user_name = data.get("userName", "")

    if not text or not user_name:
        return jsonify({"error": "Both 'text' and 'userName' are required"}), 400

    # Transform text and predict
    X = vectorizer.transform([text])
    pred = model.predict(X)
    tasks = mlb.inverse_transform(pred)[0]

    # Save tasks to DB and build response
    db = SessionLocal()
    results = []
    for i, task in enumerate(tasks):
        title = f"Task {i + 1}"
        description = task

        new_task = Task(title=title, description=description, user_name=user_name)
        db.add(new_task)
        db.commit()
        db.refresh(new_task)

        results.append({
            "Title": new_task.title,
            "Description": new_task.description,
            "UserName": new_task.user_name
        })

    db.close()
    return jsonify(results)

if __name__ == "__main__":
    app.run(debug=True)
