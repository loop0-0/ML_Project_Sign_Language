"""
retrain_model.py

Ce module permet de :
1. Charger les nouvelles données de production (prod_data.csv) et de référence (ref_data.csv).
2. Évaluer le modèle actuel XGBoost sur les données de production.
3. Réentraîner le modèle avec la concaténation des données de référence et de production.
4. Comparer les performances et remplacer l'ancien modèle si les performances s'améliorent.

Important:
 - ref_data.csv : colonnes 0..99 + 'label'
 - prod_data.csv : colonnes 0..99 + 'label' + 'prediction'
"""

import os
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb

def retrain_with_prod_data(
    ref_data_csv,
    prod_data_csv,
    model_path,
    save_new_model_path=None
):
    """
    Lit les données de ref_data.csv et prod_data.csv, évalue le modèle actuel,
    réentraîne le modèle avec les données combinées, et remplace l'ancien modèle
    si les performances s'améliorent.

    Paramètres :
        ref_data_csv : chemin du CSV de référence (ref_data.csv)
        prod_data_csv : chemin du CSV de production (prod_data.csv)
        model_path : chemin du modèle XGBoost actuel (model_xgb.json)
        save_new_model_path : chemin où sauvegarder le nouveau modèle
                              (si None, on remplace directement model_path)

    Retourne :
        (old_accuracy, new_accuracy) : tuple de précision avant/après réentraînement
    """

    # 1) Charger le modèle XGBoost existant
    model_xgb = xgb.XGBClassifier()
    model_xgb.load_model(model_path)  # Charger le modèle au format JSON

    # 2) Lire les données
    df_ref = pd.read_csv(ref_data_csv)
    df_prod = pd.read_csv(prod_data_csv)

    # Vérifier la présence des colonnes attendues
    expected_ref_cols = [str(i) for i in range(100)] + ["label"]
    expected_prod_cols = [str(i) for i in range(100)] + ["label", "prediction"]

    if not all(col in df_ref.columns for col in expected_ref_cols):
        raise ValueError("Les colonnes de ref_data.csv ne correspondent pas au format attendu.")
    if not all(col in df_prod.columns for col in expected_prod_cols):
        raise ValueError("Les colonnes de prod_data.csv ne correspondent pas au format attendu.")

    # 3) Séparer les features et le label
    X_ref = df_ref[[str(i) for i in range(100)]].values
    y_ref = df_ref["label"].values

    X_prod = df_prod[[str(i) for i in range(100)]].values
    y_prod = df_prod["label"].values

    # 4) Évaluer les performances de l'ancien modèle sur la prod
    y_pred_old = model_xgb.predict(X_prod)
    old_accuracy = accuracy_score(y_prod, y_pred_old)
    print(f"\n[INFO] Ancien modèle - Accuracy sur prod_data.csv : {old_accuracy*100:.2f}%")

    # 5) Concaténer ref_data et prod_data pour le nouvel entraînement
    X_combined = np.vstack((X_ref, X_prod))
    y_combined = np.concatenate((y_ref, y_prod))

    # 6) Créer un nouveau modèle basé sur les mêmes hyperparamètres sans le paramètre 'device'
    new_model_xgb = xgb.XGBClassifier(**model_xgb.get_params())
    
    # Supprimer le paramètre 'device' s'il existe pour éviter l'erreur
    if 'device' in new_model_xgb.get_params():
        new_model_xgb.set_params(device='cpu')  # ou 'gpu' si vous utilisez le GPU

    # 7) Entraîner le nouveau modèle
    print("[INFO] Entraînement du nouveau modèle sur données combinées...")
    new_model_xgb.fit(X_combined, y_combined)

    # 8) Évaluer le nouveau modèle sur les mêmes données de production
    y_pred_new = new_model_xgb.predict(X_prod)
    new_accuracy = accuracy_score(y_prod, y_pred_new)
    print(f"[INFO] Nouveau modèle - Accuracy sur prod_data.csv : {new_accuracy*100:.2f}%")
    print("[INFO] Nouveau modèle - Rapport de classification :")
    print(classification_report(y_prod, y_pred_new))

    # 9) Comparer et remplacer si nécessaire
    if new_accuracy > old_accuracy:
        print("[INFO] Les performances se sont améliorées, on remplace l'ancien modèle.")
        final_path = save_new_model_path if save_new_model_path else model_path
        new_model_xgb.save_model(final_path)  # Sauvegarder au format JSON
    else:
        print("[INFO] Les performances n'ont pas augmenté, l'ancien modèle est conservé.")

    return old_accuracy, new_accuracy

# -------------------------------------------------------------------
# Exemple d'exécution directe du script
# -------------------------------------------------------------------
if __name__ == "__main__":
    base_dir = "/content"
    monitoring_dir = os.path.join(base_dir, "monitoring")
    training_dir = os.path.join(base_dir, "training")

    # Assurez-vous que les dossiers existent
    os.makedirs(monitoring_dir, exist_ok=True)
    os.makedirs(training_dir, exist_ok=True)

    ref_data_path = os.path.join(monitoring_dir, "ref_data.csv")
    prod_data_path = os.path.join(monitoring_dir, "prod_data.csv")
    model_path = os.path.join(training_dir, "model_xgb.json")  # Changer l'extension

    # Si vous avez un modèle au format .pkl, convertissez-le en .json
    # Cela doit être fait une seule fois
    # Exemple de conversion :
    # with open("/content/model_xgb.pkl", 'rb') as f:
    #     model_xgb = pickle.load(f)
    # model_xgb.save_model(model_path)

    old_acc, new_acc = retrain_with_prod_data(
        ref_data_csv=ref_data_path,
        prod_data_csv=prod_data_path,
        model_path=model_path,
        save_new_model_path=None
    )
    print(f"\n[RESULT] Ancien modèle = {old_acc*100:.2f}%, Nouveau modèle = {new_acc*100:.2f}%\n")
