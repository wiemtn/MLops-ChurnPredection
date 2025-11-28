import numpy as np
import joblib
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from typing import List
import pandas as pd
from fastapi.responses import JSONResponse
from model_pipeline import send_email, prepare_data, train_model, evaluate_model, load_model, save_model
from sklearn.tree import DecisionTreeClassifier
from fastapi.middleware.cors import CORSMiddleware

# Charger le modèle
MODEL_PATH = "model.pkl"  # Assurez-vous que le chemin est correct
model = joblib.load(MODEL_PATH)

# Définir l'application FastAPI
app = FastAPI()


# Autoriser les requêtes de tous les domaines (modifiez en fonction de votre cas)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Vous pouvez spécifier un domaine particulier, comme ["http://localhost:8000"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Définir un modèle de données pour la requête avec des noms de colonnes explicites
class PredictionInput(BaseModel):
    State: int
    Account_length: int
    International_plan: int
    Number_vmail_messages: int
    Total_day_charge: float
    Total_eve_charge: float
    Total_night_charge: float
    Total_intl_charge: float
    Customer_service_calls: int
    Day_minutes_per_call: float
    Eve_minutes_per_call: float
    Night_minutes_per_call: float
    Intl_minutes_per_call: float

@app.post("/predict/")
def predict(data: PredictionInput):
    try:
        # Convertir les données en DataFrame pour la prédiction
        input_data_dict = data.dict()  # Convertir le modèle en dictionnaire
        input_data_as_numpy_array = np.asarray(list(input_data_dict.values()))
        input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
        
        # Créer un DataFrame avec les noms de colonnes appropriés
        feature_names = [
            'State', 'Account length', 'International plan', 'Number vmail messages',
            'Total day charge', 'Total eve charge', 'Total night charge', 'Total intl charge',
            'Customer service calls', 'Day minutes per call',
            'Eve minutes per call', 'Night minutes per call', 'Intl minutes per call'
        ]
        
        input_data_df = pd.DataFrame(input_data_reshaped, columns=feature_names)
        
        # Effectuer la prédiction
        prediction = model.predict(input_data_df)

        # Déterminer si le client est susceptible de se désabonner
        result = "Le client est fidèle (non churn)" if prediction[0] == 0 else "Le client est susceptible de se désabonner (churn)"

        # Retourner la prédiction
        return {"prediction": result}
    except Exception as e:
        return {"error": str(e)}

@app.post("/predict/")
def predict(input_data: PredictionInput):
    # Traitement de la prédiction ici
    return {"prediction": "Churn"}


class ModelParams(BaseModel):
    """Modèle de validation des hyperparamètres"""
    criterion: str = 'gini'
    splitter: str = 'best'
    max_depth: int = None
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    random_state: int = 42

@app.post("/retrain/")
def retrain(params: ModelParams):
    try:
        print("\n=== Début du réentraînement ===")
        print(f"Paramètres reçus : {params.dict()}")

        # Préparation des données
        X_train, X_test, y_train, y_test, scaler = prepare_data()  # Charger les données d'entraînement

        # Ajustement des paramètres spéciaux
        adjusted_params = params.dict()

        # Création et entraînement du modèle avec les paramètres fournis
        print("\nCréation du nouveau modèle...")
        model = DecisionTreeClassifier(**adjusted_params)
        model.fit(X_train, y_train)

        # Sauvegarder le modèle mis à jour
        save_model(model)

        # Évaluation et retour des performances
        accuracy = evaluate_model(model, X_test, y_test)

        return {"message": "Modèle réentrainé avec succès!", "performance": accuracy}

    except Exception as e:
        return {"error": str(e)}

@app.websocket("/ws/predict")
async def websocket_predict(websocket: WebSocket):
    """Écoute des messages et renvoie des prédictions"""
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()

            # Supposons que les données reçues soient des paramètres ou des données de prédiction
            # Effectuer la prédiction avec le modèle actuel
            # Ex : envoi de texte à un modèle de classification de texte
            X_train, X_test, y_train, y_test, scaler = prepare_data()
            prediction = model.predict(X_test)  # Prédiction avec les données de test

            # Renvoi de la prédiction au client
            await websocket.send_text(f"Prediction: {prediction[0]}")

    except WebSocketDisconnect:
        print("Client déconnecté")
