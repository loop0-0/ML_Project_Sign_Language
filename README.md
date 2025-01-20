# **ML_Project_Sign_Language**

## **Description du projet**

Ce projet vise à développer une application de reconnaissance du langage des signes en utilisant des modèles de machine learning. L'objectif est de transformer les données d'entrée en prédictions du langage des signes, rendant la communication plus accessible pour les personnes malentendantes.

---

## **Structure du projet**

### **Racine du projet**
- **artifacts/** : Contient les modèles entraînés et leurs fichiers associés :
  - `model_xgb.pkl` : Modèle XGBoost entraîné.
  - `pca.pkl` et `scaler.pkl` : Objets de réduction de dimension et de normalisation des données.
- **data/** : Données utilisées pour entraîner et tester le modèle :
  - `prod_data.csv` : Données de production.
  - `ref_data.csv` et `test_data.csv` : Données de référence et de test.
- **reporting/** : Contient les fichiers liés à la génération de rapports ou à la gestion des conteneurs.
  - `docker-compose.yml` et `Dockerfile` : Configurations Docker.
- **scripts/** : Scripts Python pour le prétraitement des données et l'entraînement des modèles.
  - `preprocess_data.py` : Script de prétraitement des données.
  - `retrain_model.py` : Script pour réentraîner le modèle.
  - `train_model_ML1.ipynb` : Notebook Jupyter pour l'entraînement du modèle.
  - `transform_data_ML1.ipynb` : Notebook pour la transformation des données.
- **serving/** : Contient les fichiers pour déployer le modèle en production.
  - `api.py` : API pour interagir avec le modèle.
  - `docker-compose.yml` et `Dockerfile` : Configurations Docker pour le déploiement.
- **webapp/** : Interface utilisateur ou backend pour accéder aux fonctionnalités du projet.

---

## **Technologies utilisées**
- **Langage de programmation** : Python.
- **Frameworks et bibliothèques** :
  - Scikit-learn, XGBoost : Pour le développement et l'entraînement des modèles.
  - FastAPI : Pour l'API backend.
  - Streamlite : Pour l'interface web.
  - Evidently : Pour générer un rapport sur l’état de santé de modèle.
- **Outils et plateformes** :
  - Docker : Pour la conteneurisation des applications.
  - Jupyter Notebook : Pour le développement de modèle , l’expérimentation et l’analyse.
  - Git : Pour la gestion de version.
  - VSCode 

---

## **Installation**

1. **Cloner le projet :**
   ```bash
   git clone <URL_DU_DEPOT>
   cd ML_Project_Sign_Language
   ```

2. **Installer les dépendances :**
   Assurez-vous que `pip` est installé, puis exécutez :
   ```bash
   pip install -r requirements.txt
   ```

3. **Configurer Docker :**
   Si vous utilisez Docker pour déployer l'application, construisez l'image :
   ```bash
   docker-compose build
   ```

---

## **Utilisation**

1. **Démarrer l'API :**
   Si vous utilisez Docker, démarrez l'application avec :
   ```bash
   cd serving
   docker-compose up
   ```
   L’API sera accessible via `http://localhost:8080`.

2. **Démarrer l'API :**
   Si vous utilisez Docker, démarrez l'application avec :
   ```bash
   cd webapp
   docker-compose up
   ```
   L’API sera accessible via `http://localhost:8081`.

2. **Démarrer le reporting :**
   Si vous utilisez Docker, démarrez l'application avec :
   ```bash
   cd reporting
   docker-compose up
   ```
   L’API sera accessible via `http://localhost:8082`.
---

## **Contributeurs**

- Djawad Mecheri.
- Lotfi Mayouf.
- Abdennour Slimani.
- Akram Mekbal.
- Massinissa Maouche.
- Ishak Bouchlagheme.

---

## **Licence**

Ce projet est sous licence ..
