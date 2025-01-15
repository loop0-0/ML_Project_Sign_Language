import streamlit as st
import requests
from PIL import Image
import io
# Define API URLs
PREDICTION_API_URL = "http://serving-api:8080/predict"
FEEDBACK_API_URL = "http://serving-api:8080/feedback"

# Streamlit app
st.title("Sign Language AI Interpreter")

# Upload image
uploaded_file = st.file_uploader("Upload a sign language image to interpret", type=["png", "jpg", "jpeg"])

if uploaded_file:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="here is the uploaded Image", use_container_width=True)

    # Predict button
    if st.button("Predict"):
        # Convert image to bytes
        image_bytes = io.BytesIO()
        image.save(image_bytes, format="PNG")
        image_bytes = image_bytes.getvalue()

        # Send image to prediction API
        try:
            response = requests.post(
                PREDICTION_API_URL, 
                files={"file": ("image.png", image_bytes, "image/png")}
            )
            response.raise_for_status()
            prediction = response.json().get("prediction", "Unknown")

            # Display prediction
            st.success(f"Predicted Letter: {prediction}")
            
            # Feedback section
            correct_label = st.text_input("If the prediction is incorrect, enter the correct label:")
            
            if st.button("Submit Feedback"):
                feedback_data = {
                    "predicted": prediction,
                    "true_label": correct_label or prediction
                }
                # Send feedback to API
                feedback_response = requests.post(
                    FEEDBACK_API_URL,
                    json=feedback_data
                )
                feedback_response.raise_for_status()
                st.success("Feedback submitted successfully!")

        except requests.exceptions.RequestException as e:
            st.error(f"Error: {e}")