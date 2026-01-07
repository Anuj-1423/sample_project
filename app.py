from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# =========================
# Load Model and Scaler
# =========================
import joblib

rf_model = joblib.load("model/random_forest_model.pkl")
scaler = joblib.load("model/scaler.pkl")


# Continuous columns used during training (DO NOT CHANGE ORDER)
continuous_cols = [
    "Age",
    "Chest pain type",
    "BP",
    "Cholesterol",
    "EKG results",
    "Exercise angina",
    "ST depression",
    "Slope of ST",
    "Number of vessels fluro",
    "Thallium"
]

# =========================
# Routes
# =========================
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # =========================
        # Get Raw User Input
        # =========================
        user_input = {
            "Age": float(request.form["Age"]),
            "Sex": int(request.form["Sex"]),
            "Chest pain type": int(request.form["ChestPain"]),
            "BP": float(request.form["BP"]),
            "Cholesterol": float(request.form["Cholesterol"]),
            "EKG results": int(request.form["EKG"]),
            "Exercise angina": int(request.form["ExerciseAngina"]),
            "ST depression": float(request.form["STDepression"]),
            "Slope of ST": int(request.form["Slope"]),
            "Number of vessels fluro": int(request.form["Vessels"]),
            "Thallium": int(request.form["Thallium"])
        }

        input_df = pd.DataFrame([user_input])

        # =========================
        # Apply Scaling (ONLY continuous columns)
        # =========================
        input_df_scaled = input_df.copy()
        input_df_scaled[continuous_cols] = scaler.transform(
            input_df[continuous_cols]
        )

        # =========================
        # Prediction
        # =========================
        prediction = rf_model.predict(input_df_scaled)[0]
        probability = rf_model.predict_proba(input_df_scaled)[0][1]

        # =========================
        # Result Interpretation
        # =========================
        if prediction == 1:
            result = "Heart Disease Detected"
        else:
            result = "No Heart Disease Detected"

        return render_template(
            "result.html",
            prediction=result,
            probability=round(probability * 100, 2)
        )

    except Exception as e:
        return render_template(
            "result.html",
            prediction="Error occurred during prediction",
            probability=0
        )


# =========================
# Run App
# =========================
if __name__ == "__main__":
    app.run(debug=True)
