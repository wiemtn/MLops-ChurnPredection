from flask import Flask, render_template, request
import requests

app = Flask(__name__)

# URL de l'API FastAPI
FASTAPI_URL = "http://localhost:8000/predict/"

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        input_data = {
            "State": int(request.form["State"]),
            "Account_length": int(request.form["Account_length"]),
            "International_plan": int(request.form["International_plan"]),
            "Number_vmail_messages": int(request.form["Number_vmail_messages"]),
            "Total_day_charge": float(request.form["Total_day_charge"]),
            "Total_eve_charge": float(request.form["Total_eve_charge"]),
            "Total_night_charge": float(request.form["Total_night_charge"]),
            "Total_intl_charge": float(request.form["Total_intl_charge"]),
            "Customer_service_calls": int(request.form["Customer_service_calls"]),
            "Day_minutes_per_call": float(request.form["Day_minutes_per_call"]),
            "Eve_minutes_per_call": float(request.form["Eve_minutes_per_call"]),
            "Night_minutes_per_call": float(request.form["Night_minutes_per_call"]),
            "Intl_minutes_per_call": float(request.form["Intl_minutes_per_call"]),
        }

        # Envoi des données à FastAPI pour obtenir la prédiction
        response = requests.post(FASTAPI_URL, json=input_data)
        prediction = response.json().get("prediction", "Erreur dans la prédiction")

        return render_template("index.html", prediction=prediction)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
