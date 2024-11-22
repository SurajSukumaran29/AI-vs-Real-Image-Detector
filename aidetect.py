import streamlit as st
from PIL import Image
from skimage.io import imread
from skimage.transform import resize
import numpy as np
from keras.models import load_model

# Load the trained model
model = load_model('ai vs real.h5')

# Define the categories
categories = ['Camera_images', 'Ai_Images']

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Select a page:", ["Home", "Image Prediction"])

# Home Page
if page == "Home":
    st.title("AI vs Real Image Classifier")
    st.write("""
    This app allows you to classify images as either **AI-generated** or **real**.

    - **Home Page**: This page provides a brief description of the app.
    - **Image Prediction Page**: Upload an image and let the model predict whether it's AI-generated or captured by a camera.
    """)

# Image Prediction Page
elif page == "Image Prediction":
    st.title("Predict AI or Real Image")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Convert the image to numpy array
        img_arr = np.array(image)

        # Resize the image
        img_resized = resize(img_arr, (150, 150, 3))
        img_resized = img_resized.reshape(1, 150, 150, 3)

        # Make a prediction
        y_new = model.predict(img_resized)

        # Get the predicted category
        ind = y_new.argmax(axis=1)
        predicted_category = categories[ind.item()]

        st.write(f"The image is predicted as: **{predicted_category}**")
