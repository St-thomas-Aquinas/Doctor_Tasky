from datetime import datetime, timedelta
import joblib

# Load once on import
model = joblib.load("mlp_task_suggester_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")
label_binarizer = joblib.load("label_binarizer.pkl")

# Task timing (in minutes)
task_timing_rules = {
    "check blood sugar": 0,
    "drink water": 15,
    "eat healthy meal": 30,
    "exercise": 60,
    "log sugar": 0,
    "rest": 10,
    "take medication": 0
}

def format_time(minutes):
    now = datetime.now()
    future = now + timedelta(minutes=minutes)
    return future.strftime("%H:%M")

def predict_task_with_timing(text):
    X_vec = vectorizer.transform([text])
    pred_probs = model.predict_proba(X_vec)[0]
    threshold = 0.4

    results = []
    for i, prob in enumerate(pred_probs):
        if prob >= threshold:
            task = label_binarizer.classes_[i]
            delay = task_timing_rules.get(task, 0)
            when = (
                "immediately" if delay == 0 else
                f"in {delay} minutes" if delay < 60 else
                f"after {delay // 60} hour(s)"
            )
            results.append({
                "task": task,
                "when": when,
                "at": format_time(delay)
            })
    return results
