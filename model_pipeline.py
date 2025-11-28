import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import warnings
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix
def prepare_data():
    """Load, preprocess the dataset, split into train/test."""
    print("Starting data preparation...")

    data = pd.read_csv("merged_churn1.csv")
    print("Dataset loaded successfully")
    categorical_cols = data.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

    # Apply Label Encoding to all categorical variables
    label_encoder = LabelEncoder()
    for col in categorical_cols:
        data[col] = label_encoder.fit_transform(data[col])
    data = data.drop(columns="Area code")
    data = data.drop(columns="Voice mail plan")
    # Add minutes per call features
    for col in ["Total day minutes", "Total day calls", "Total eve minutes",
                "Total eve calls", "Total night minutes", "Total night calls",
                "Total intl minutes", "Total intl calls"]:
        data[col] = pd.to_numeric(data[col], errors='coerce')

    data["Day minutes per call"] = data["Total day minutes"] / data["Total day calls"]
    data["Eve minutes per call"] = data["Total eve minutes"] / data["Total eve calls"]
    data["Night minutes per call"] = data["Total night minutes"] / data["Total night calls"]
    data["Intl minutes per call"] = data["Total intl minutes"] / data["Total intl calls"]

    # Drop original columns that are no longer needed
    data = data.drop(columns=[
        "Total day minutes", "Total day calls", "Total eve minutes", "Total eve calls",
        "Total night minutes", "Total night calls", "Total intl minutes", "Total intl calls"
    ], errors='ignore')

    X = data.drop('Churn', axis=1)
    y = data['Churn']
    print("Target variable separated")
    
    scaler = StandardScaler()
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print("Data preprocessing completed")
    return X_train, X_test, y_train, y_test, scaler

def train_model(X_train, y_train):
    """Train a Decision Tree model."""
    print("Training the Decision Tree model...")
    #model = DecisionTreeClassifier(criterion="gini", random_state=100, max_depth=6, min_samples_leaf=8)
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    print("Model training completed")
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate the model's performance."""
    print("Evaluating the model...")
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    send_email(accuracy);
    return accuracy , precision ,recall ,f1

def save_model(model):
    """Save model and preprocessing artifacts."""
    print("Saving model...")
    joblib.dump(model, "model.pkl")
    print("Model and scaler saved successfully")

def load_model():
    """Load model and preprocessing artifacts if they exist."""
    try:
        model = joblib.load("model.pkl")
        print("Model and preprocessing loaded successfully")
        return model
    except FileNotFoundError:
        print("Model or scaler file not found.")
        return None


import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from sklearn.metrics import classification_report

# Fonction pour envoyer un email avec le rapport
def send_email(accuracy):
    sender_email = "benromdhanekhalil932@gmail.com"  # Remplace par ton email
    receiver_email = "aminesleh148@gmail.com"  # Remplace par l'email du destinataire
    password = "justinbieber1994"  # Utilise un mot de passe d'application si c'est Gmail

 # Création du message
    msg = MIMEMultipart()
    msg["From"] = sender_email
    msg["To"] = receiver_email
    msg["Subject"] = "Rapport d'évaluation du modèle"

    # Texte du mail
    message = f"""
    Bonjour,

    L'exécution du modèle est terminée.

    ✅ Accuracy du modèle : {accuracy:.2%}

    Cordialement,
    L'équipe MLOps
    """
    
    msg.attach(MIMEText(message, "plain"))

    # Envoi de l'email
    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(sender_email, password)
        server.sendmail(sender_email, receiver_email, msg.as_string())
        server.quit()
        print("✅ Email envoyé avec succès !")
    except Exception as e:
        print("❌ Erreur lors de l'envoi de l'email :", e)
