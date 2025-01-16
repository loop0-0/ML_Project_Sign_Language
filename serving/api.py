from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import numpy as np
import pickle
import cv2


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

# Load the scaler, PCA, and ML model
with open("/artifacts/scaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

with open("/artifacts/pca.pkl", "rb") as pca_file:
    pca = pickle.load(pca_file)

with open("/artifacts/model_xgb.pkl", "rb") as model_file:
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
      

        return JSONResponse(content={"prediction": prediction.tolist()[0]})

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
    



@app.get("/")
async def index():
    return JSONResponse(content={"message" : "The api is working"})
