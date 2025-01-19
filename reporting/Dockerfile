FROM python:3.10-slim

WORKDIR /app

# 1) Install system packages needed: git (to clone the Evidently repo)
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# 2) Copy and install your pinned dependencies
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# 3) Now you can install Evidently from GitHub
RUN pip install git+https://github.com/evidentlyai/evidently.git@main

# 4) Copy your code and define CMD
COPY project.py .

CMD python project.py && evidently ui --host 0.0.0.0 --port 8082
