from flask import Flask, request, jsonify
from flask_cors import CORS
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import joblib

# ------------------ Configuration ------------------
DATABASE_URL = "postgresql://task_database_pb63_user:tUG9ChwXYE3aWTgwzSIOIjQRAfgbJXg8@dpg-d1v2qser433s73f9iol0-a.oregon-postgres.render.com/task_database_pb63"

app = Flask(__name__)
CORS(app)

# SQLAlchemy setup
engine = create_engine(DATABASE_URL)
Base = declarative_base()
SessionLocal = sessionmaker(bind=engine)

class Task(Base):
    __tablename__ = "tasks"
    id = Column(Integer, primary_key=True, index=True)
    Title = Column(String)
    Description = Column(String)
    username = Column(String)

Base.metadata.create_all(bind=engine)

# Load ML model and tools
model = joblib.load("mlp_task_suggester_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")
mlb = joblib.load("label_binarizer.pkl")

# ------------------ Routes ------------------

@app.route("/")
def home():
    return "Multilabel Task Suggestion API is live!"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    print(f"📥 Received request: {data}")

    text = data.get("text", "")
    user_name = data.get("username", "")

    if not text or not user_name:
        return jsonify({"error": "Both 'text' and 'username' are required"}), 400

    # Predict tasks
    X = vectorizer.transform([text])
    pred = model.predict(X)
    tasks = mlb.inverse_transform(pred)[0]

    # Format tasks
    result = [
        {"Title": f"Task {i + 1}", "Description": task, "username": user_name}
        for i, task in enumerate(tasks)
    ]

    # Save to database
    db = SessionLocal()
    try:
        for r in result:
            db_task = Task(**r)
            db.add(db_task)
        db.commit()

        # 🔍 Query back all tasks for this user
        user_tasks = db.query(Task).filter(Task.username == user_name).all()
        print(f"✅ Tasks for {user_name}:")
        for t in user_tasks:
            print(f"- {t.Title}: {t.Description}")

    except Exception as e:
        db.rollback()
        print(f"❌ Error saving tasks: {e}")
        return jsonify({"error": "Database error"}), 500
    finally:
        db.close()

    return jsonify({
        "message": f"{len(result)} tasks saved successfully for user {user_name}",
        "tasks": result
    }), 200

# ------------------ Main ------------------
if __name__ == "__main__":
    app.run(debug=True)
