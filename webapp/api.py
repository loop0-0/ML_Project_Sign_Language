import streamlit as st
import requests
from PIL import Image
import io

# Define development API URLs
PREDICTION_API_URL_DEV = "http://serving-api:8080/predict"
FEEDBACK_API_URL_DEV = "http://serving-api:8080/feedback"
# Streamlit app
st.title("Sign Language AI Interpreter")

# Create tabs
tab1, tab2 = st.tabs(["Prediction", "Feedback"])

# Tab 1: Prediction
with tab1:
    st.header("Prediction")

    # Upload image
    uploaded_file = st.file_uploader("Upload a sign language image to interpret", type=["png", "jpg", "jpeg"])

    if uploaded_file:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Here is the uploaded image", use_container_width=True)

        # Predict button
        if st.button("Predict", key="predict"):
            # Convert image to bytes
            image_bytes = io.BytesIO()
            image.save(image_bytes, format="PNG")
            image_bytes = image_bytes.getvalue()

            # Send image to prediction API
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

            except requests.exceptions.RequestException as e:
                st.error(f"Error: {e}")

# Tab 2: Feedback
with tab2:
    st.header("Feedback")

    # Upload image for feedback
    feedback_image = st.file_uploader("Upload an image for feedback", type=["png", "jpg", "jpeg"], key="feedback")

    # Input for true label
    true_label = st.text_input("Enter the correct label (a-z, 'nothing', or 'stop'):")

    # List of valid labels
    valid_labels = [chr(i) for i in range(ord('a'), ord('z') + 1)] + ["nothing", "stop"]

    if feedback_image:
        # Display uploaded image
        image = Image.open(feedback_image)
        st.image(image, caption="Here is the uploaded image for feedback", use_container_width=True)

        # Send feedback button
        if st.button("Send Feedback") and true_label:
            # Validate the true_label input
            if true_label.lower() not in valid_labels:
                st.error("Invalid label! Please enter a letter (a-z), 'nothing', or 'stop'.")
            else:
                # Convert image to bytes
                image_bytes = io.BytesIO()
                image.save(image_bytes, format="PNG")
                image_bytes = image_bytes.getvalue()

                # Send feedback to the API
                try:
                    response = requests.post(
                        FEEDBACK_API_URL_DEV, 
                        files={"image": ("image.png", image_bytes, "image/png")},
                        data={"target": true_label.lower()}  # Ensure the label is lowercase
                    )
                    response.raise_for_status()
                    st.success("Feedback submitted successfully!")

                except requests.exceptions.RequestException as e:
                    st.error(f"Error submitting feedback: {e}")