# ✅ Required packages:
# pip install flask flask-cors sqlalchemy psycopg2-binary joblib

from flask import Flask, request, jsonify
from flask_cors import CORS
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import joblib

# 📦 Load model and preprocessing tools
model = joblib.load("mlp_task_suggester_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")
mlb = joblib.load("label_binarizer.pkl")

# 🧠 Flask setup
app = Flask(__name__)
CORS(app)

# 🗄 Database setup
DATABASE_URL = "postgresql://task_database_pb63_user:tUG9ChwXYE3aWTgwzSIOIjQRAfgbJXg8@dpg-d1v2qser433s73f9iol0-a.oregon-postgres.render.com/task_database_pb63"
engine = create_engine(DATABASE_URL)
Base = declarative_base()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# 🧱 Task model
class Task(Base):
    __tablename__ = "tasks"
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String)
    description = Column(String)
    username = Column(String)

# Create tables if not already present
Base.metadata.create_all(bind=engine)

@app.route("/")
def home():
    return "Multilabel Task Suggestion API with DB storage is live!"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data.get("text", "")
    username = data.get("username")

    if not username:
        return jsonify({"error": "username is required"}), 400

    X = vectorizer.transform([text])
    pred = model.predict(X)
    tasks = mlb.inverse_transform(pred)[0]

    db = SessionLocal()
    response = []

    for i, task in enumerate(tasks):
        title = f"Task {i + 1}"
        description = task
        task_obj = Task(title=title, description=description, username=username)
        db.add(task_obj)
        response.append({
            "Title": title,
            "Description": description,
            "UserName": username
        })

    db.commit()
    db.close()

    return jsonify(response)

if __name__ == "__main__":
    app.run(debug=True)
