from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
import numpy as np
import pickle
import cv2
import csv
import os

from keras._tf_keras.keras.applications.resnet import preprocess_input
from keras._tf_keras.keras.applications.resnet50 import ResNet50



classes = ['W', 'K', 'B', 'M', 'R', 'I', 'G', 'P', 'O', 'Z', 'U', 'A', 
           'nothing', 'V', 'del', 'Y', 'L', 'X', 'space', 'F', 'J', 
           'C', 'H', 'Q', 'T', 'D', 'E', 'S', 'N']

# Create a mapping dictionary with indices starting from 1
index_to_label = {i + 1: label for i, label in enumerate(classes)}




app = FastAPI()

# Load pre-trained ResNet50 for feature extraction
resnet_model = ResNet50(weights="imagenet", include_top=False, pooling="avg")


# Path for production data CSV
# PROD_DATA_PATH = "../data/prod_data.csv"

# Load the scaler, PCA, and ML model
with open("../artifacts/scaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

with open("../artifacts/pca.pkl", "rb") as pca_file:
    pca = pickle.load(pca_file)

with open("../artifacts/model_xgb.pkl", "rb") as model_file:
    model = pickle.load(model_file)

@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    try:
        # Validate the file type
        if not image.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Uploaded file is not an image")

        # Read the image
        contents = await image.read()
        np_image = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(np_image, cv2.IMREAD_COLOR)

        # Preprocess the image for ResNet50
        img_resized = cv2.resize(img, (224, 224))  # ResNet50 expects 224x224 images
        img_array = np.expand_dims(img_resized, axis=0)
        img_preprocessed = preprocess_input(img_array)  # Preprocessing for ResNet50

        # Extract features using ResNet50
        features = resnet_model.predict(img_preprocessed)

        # Apply scaler and PCA to the features
        features_scaled = scaler.transform(features)
        features_pca = pca.transform(features_scaled)

        # Make prediction using the ML model
        prediction = model.predict(features_pca)
      
        prediction = prediction.tolist()[0]

        prediction = index_to_label[prediction]

        return JSONResponse(content={"prediction": prediction})

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
    

# Path for the production data CSV
PROD_DATA_PATH = "../data/prod_data.csv"

    
@app.post("/feedback")
async def feedback(image: UploadFile = File(...), target: str = Form(...)):
    try:
        print("Feedback post endpoint")
        
        # Validate the file type
        if not image.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Uploaded file is not an image")

        # Read the image
        contents = await image.read()
        np_image = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(np_image, cv2.IMREAD_COLOR)

        # Preprocess the image for ResNet50
        img_resized = cv2.resize(img, (224, 224))  # ResNet50 expects 224x224 images
        img_array = np.expand_dims(img_resized, axis=0)
        img_preprocessed = preprocess_input(img_array)  # Preprocessing for ResNet50

        # Extract features using ResNet50
        features = resnet_model.predict(img_preprocessed)

        # Apply scaler and PCA to the features
        features_scaled = scaler.transform(features)
        features_pca = pca.transform(features_scaled)

        # Make prediction using the ML model
        prediction = model.predict(features_pca)
        prediction = prediction.tolist()[0]
        prediction_label = index_to_label[prediction]

        # Convert PCA-transformed features to a list for CSV
        pca_features = features_pca[0].tolist()

        print(f"target: {target}")
        print(f"predicted: {prediction_label}")

        # Check if the file exists
        file_exists = os.path.exists(PROD_DATA_PATH)

        # Open the CSV file for appending
        with open(PROD_DATA_PATH, mode="a", newline="") as file:
            writer = csv.writer(file)

            # Write header if the file doesn't exist
            if not file_exists:
                header = [f"PCA_{i+1}" for i in range(len(pca_features))] + ["target", "prediction"]
                writer.writerow(header)

            # Append the data row (PCA features + target + prediction)
            target = target.lower()
            prediction_label = prediction_label.lower()
            row = pca_features + [target, prediction_label]
            writer.writerow(row)

        return {"message": "Feedback saved successfully"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def index():
    return JSONResponse(content={"message" : "The api is working"})
