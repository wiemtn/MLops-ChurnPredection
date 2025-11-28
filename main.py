import argparse
from model_pipeline import prepare_data, train_model, evaluate_model, load_model, save_model  # Added save_model
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
import numpy as np


mlflow.set_tracking_uri("http://localhost:7001") 

mlflow.set_experiment("Prediction Churn")

def main():
    # Création du parser d'arguments
    parser = argparse.ArgumentParser(description="ML Pipeline CLI")
    parser.add_argument("--prepare", action="store_true", help="Préparer les données")
    parser.add_argument("--train", action="store_true", help="Entraîner le modèle")
    parser.add_argument("--evaluate", action="store_true", help="Évaluer le modèle")

    args = parser.parse_args()

    # Initialisation des variables
    X_train, X_test, y_train, y_test, scaler, model = None, None, None, None, None, None

    # Préparation des données si nécessaire
    if args.prepare or args.train or args.evaluate:
        print("Exécution de prepare_data()...")
        X_train, X_test, y_train, y_test, scaler = prepare_data()
        print("Données préparées avec succès !")

    # Entraînement du modèle si nécessaire
    if args.train or args.evaluate:
        print("Exécution de train_model()...")

        with mlflow.start_run():
            model = train_model(X_train, y_train)
            mlflow.log_param("model_type", type(model).__name__)
        # Enregistrer les hyperparamètres
            mlflow.log_param("random_state", 42)

            input_example = X_train[:5]  # Prenez les 5 premières lignes de X_train comme exemple
            signature = infer_signature(X_train, model.predict(X_train))
        # Sauvegarder le modèle dans MLflow (log_model)
            mlflow.sklearn.log_model(model, "model", signature=signature, input_example=input_example)

        # Enregistrer le modèle dans le Model Registry
            model_uri = mlflow.get_artifact_uri("model")
            model_name = "decision_tree_model"
            mlflow.register_model(model_uri, model_name)
            print(f"Le modèle {model_name} a été enregistré avec succès dans le Model Registry.")

        save_model(model)  # Make sure to save the model after training
        print("Modèle entraîné avec succès !")

    # Évaluation du modèle si nécessaire
    if args.evaluate:
        print("Évaluation du modèle...")
        with mlflow.start_run():  # Correction d'indentation ici
            accuracy, precision, recall, f1 = evaluate_model(model, X_test, y_test)

            # Enregistrement des métriques dans MLflow
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)

            print("Métriques enregistrées avec MLflow !")
        print("Évaluation terminée !")

if __name__ == "__main__":
    main()
