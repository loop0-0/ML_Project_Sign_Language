from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
import numpy as np
import pickle
import cv2
import csv
import os
import sys
import warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Only show errors

warnings.filterwarnings("ignore", module="sklearn")
warnings.filterwarnings("ignore", module="xgboost")


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct absolute paths for artifacts
scaler_path = os.path.join(current_dir, "../artifacts/scaler.pkl")
pca_path = os.path.join(current_dir, "../artifacts/pca.pkl")
model_path = os.path.join(current_dir, "../artifacts/model_xgb.pkl")


from keras._tf_keras.keras.applications.resnet import preprocess_input
from keras._tf_keras.keras.applications.resnet50 import ResNet50

from scripts.preprocess_data import transform_single_image



classes = [
    'A',        # 0
    'B',        # 1
    'C',        # 2
    'D',        # 3
    'E',        # 4
    'F',        # 5
    'G',        # 6
    'H',        # 7
    'I',        # 8
    'J',        # 9
    'K',        # 10
    'L',        # 11
    'M',        # 12
    'N',        # 13
    'O',        # 14
    'P',        # 15
    'Q',        # 16
    'R',        # 17
    'S',        # 18
    'T',        # 19
    'U',        # 20
    'V',        # 21
    'W',        # 22
    'X',        # 23
    'Y',        # 24
    'Z',        # 25
    'del',      # 26
    'nothing',  # 27
    'space'     # 28
]

# Create a mapping dictionary with indices starting from 1
index_to_label = {i + 1: label for i, label in enumerate(classes)}




app = FastAPI()

# Load pre-trained ResNet50 for feature extraction
base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')


# Path for production data CSV
# PROD_DATA_PATH = "../data/prod_data.csv"

# Load the scaler, PCA, and ML model
with open(scaler_path, "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

with open(pca_path, "rb") as pca_file:
    pca = pickle.load(pca_file)

with open(model_path, "rb") as model_file:
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


        features_pca = transform_single_image(
            image=img,
            base_model=base_model,
            scaler=scaler,
            pca=pca,
            target_size=(64, 64)
        )

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



        # Extract features using ResNet50
        features_pca = transform_single_image(
            image=img,
            base_model=base_model,
            scaler=scaler,
            pca=pca,
            target_size=(64, 64)
        )

        # Make prediction using the ML model
        prediction = model.predict(features_pca)
        prediction = prediction.tolist()[0]
        prediction_label = index_to_label[prediction]

        target_class = classes.index(target.upper() if len(target) == 1 else target.lower())

        # Convert PCA-transformed features to a list for CSV
        pca_features = features_pca

        print(f"target: {target}")
        print(f"predicted: {prediction_label}")

        # Check if the file exists
        file_exists = os.path.exists(PROD_DATA_PATH)

        # Open the CSV file for appending
        with open(PROD_DATA_PATH, mode="a", newline="") as file:
            writer = csv.writer(file)

            # Write header if the file doesn't exist
            if not file_exists:
                header = [f"PCA_{i+1}" for i in range(pca_features.shape[1])] + ["target", "prediction"]
                writer.writerow(header)

            # Append the data row (PCA features + target + prediction)
            
            prediction_label = prediction_label.lower()
            row = np.append(pca_features.flatten(), [target_class, prediction])
            writer.writerow(row)

        return {"message": "Feedback saved successfully"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def index():
    return JSONResponse(content={"message" : "The api is working"})
