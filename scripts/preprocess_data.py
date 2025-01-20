"""
preprocess_data.py

Ce module contient toutes les fonctions nécessaires pour :
1. Charger et configurer le modèle de feature extraction (ResNet50).
2. Charger les artefacts (scaler et PCA).
3. Appliquer la même pipeline de transformation (prétraitement, extraction de features, normalisation, réduction de dimension).
4. Fournir une fonction pour transformer une image unique en un vecteur prêt pour la prédiction.
"""

import os
import numpy as np
import cv2
import pickle
from keras._tf_keras.keras.applications import ResNet50
from keras._tf_keras.keras.applications.resnet import preprocess_input
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# -------------------------------------------------------------------
# 1) Chargement du modèle pré-entraîné (ResNet50) pour l'extraction de features
# -------------------------------------------------------------------
def load_feature_extractor():
    """
    Charge le modèle ResNet50 (sans la couche fully-connected de sortie)
    pour extraire les features des images.
    Retourne le modèle Keras.
    """
    base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
    return base_model

# -------------------------------------------------------------------
# 2) Chargement des artefacts (scaler, pca)
# -------------------------------------------------------------------
def load_artifacts(scaler_path, pca_path):
    """
    Charge les objets scaler et pca depuis des fichiers pickle.
    Retourne (scaler, pca).
    """
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    with open(pca_path, 'rb') as f:
        pca = pickle.load(f)
    return scaler, pca

# -------------------------------------------------------------------
# 3) Fonction de prétraitement d'une image BGR -> RGB -> preprocess_input
# -------------------------------------------------------------------
def preprocess_image_bgr(image_bgr, target_size=(64, 64)):
    """
    Reçoit une image en format BGR (tel que retourné par cv2.imread).
    La convertit en RGB, la redimensionne et applique la fonction
    de prétraitement 'preprocess_input' de ResNet.
    
    Paramètres :
        image_bgr : np.array (H, W, 3) en BGR
        target_size : tuple (width, height) par défaut (64, 64)

    Retourne :
        image_preprocessed : np.array (1, H', W', 3)
    """
    # Convertir en RGB
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    # Redimensionner
    image_rgb = cv2.resize(image_rgb, target_size)
    # Appliquer la normalisation ResNet
    image_rgb = preprocess_input(image_rgb)
    # Ajouter la dimension batch
    return np.expand_dims(image_rgb, axis=0)

# -------------------------------------------------------------------
# 4) Extraction des features depuis ResNet50 + Scaling + PCA
# -------------------------------------------------------------------
def extract_features(base_model, image_array, scaler, pca):
    """
    Extrait les features depuis ResNet50 (base_model),
    puis applique scaler et pca.

    Paramètres :
        base_model : modèle Keras (ResNet50 sans top)
        image_array : np.array (N, 64, 64, 3) après preprocess_input
        scaler : StandardScaler
        pca : PCA

    Retourne :
        features_pca : np.array de forme (N, n_components_pca)
    """
    # 1) Extraction de features via ResNet50
    features = base_model.predict(image_array, verbose=0)
    # 2) Application du scaler
    features_scaled = scaler.transform(features)
    # 3) Application du PCA
    features_pca = pca.transform(features_scaled)
    return features_pca

# -------------------------------------------------------------------
# 5) Fonction principale pour transformer une image brute en vecteur
# -------------------------------------------------------------------
def transform_single_image(image, base_model, scaler, pca, target_size=(64, 64)):
    """
    Prend le chemin d'une image, charge l'image, applique toutes les étapes
    de prétraitement et renvoie le vecteur final (1, n_components_pca) prêt
    pour la prédiction par le modèle XGBoost.

    Paramètres :
        image_path : str, chemin vers l'image
        base_model : modèle Keras (ResNet50)
        scaler : StandardScaler chargé
        pca : PCA chargé
        target_size : taille de redimensionnement (64, 64) par défaut

    Retourne :
        features_vector : np.array, shape (1, n_components_pca)
    """
    # # Charger l'image en BGR avec OpenCV
    # image_bgr = cv2.imread(image_path)
    # if image_bgr is None:
    #     raise ValueError(f"Impossible de charger l'image : {image_path}")

    # Prétraiter l'image (BGR -> RGB, resize, preprocess_input, etc.)
    image_preprocessed = preprocess_image_bgr(image, target_size=target_size)

    # Extraire les features (ResNet50 + scaling + PCA)
    features_vector = extract_features(base_model, image_preprocessed, scaler, pca)
    return features_vector
