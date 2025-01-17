# Utilise une image de base Python 3.11
FROM python:3.11-slim

# Définit le répertoire de travail dans le conteneur
WORKDIR /app

# Copie les fichiers nécessaires
COPY api.py .
COPY requirements.txt .

# Installe les dépendances
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt

# Commande pour démarrer l'interface web
CMD ["streamlit", "run", "api.py", "--server.port=8081", "--server.address=0.0.0.0"]