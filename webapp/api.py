import streamlit as st
import requests
from PIL import Image
import io

# Define development API URL
PREDICTION_API_URL_DEV = "http://localhost:8000/predict"

# Streamlit app
st.title("Sign Language AI Interpreter")

# Upload image
uploaded_file = st.file_uploader("Upload a sign language image to interpret", type=["png", "jpg", "jpeg"])

if uploaded_file:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Here is the uploaded image", use_container_width=True)

    # Predict button
    if st.button("Predict"):
        # Convert image to bytes
        image_bytes = io.BytesIO()
        image.save(image_bytes, format="PNG")
        image_bytes = image_bytes.getvalue()

        # Send image to prediction API (using localhost dev URL)
        try:
            response = requests.post(
                PREDICTION_API_URL_DEV, 
                files={"image": ("image.png", image_bytes, "image/png")}
            )
            response.raise_for_status()
            data = response.json()

            # Extract prediction and display results
            prediction = data.get("prediction", "Unknown")
            st.success(f"Predicted Letter: {prediction}")
            
            # Display the processed image (if available from API)
            processed_image_data = data.get("processed_image")
            if processed_image_data:
                processed_image = Image.open(io.BytesIO(bytes(processed_image_data)))
                st.image(processed_image, caption="Processed Image", use_container_width=True)

        except requests.exceptions.RequestException as e:
            st.error(f"Error: {e}")
