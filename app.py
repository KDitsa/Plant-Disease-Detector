import streamlit as st
import tensorflow as tf
import json
import numpy as np
from PIL import Image

model = tf.keras.models.load_model('./models/trained_model.keras')
with open("./models/classnames.json","r") as f:
    class_names = json.load(f)

st.set_page_config(page_title="Plant Disease Detector", layout="centered")
st.markdown("""
    <style>
        .main {
            background-color: #f4f4f9;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
        }
        .stFileUploader {
            padding: 10px;
        }
        .stButton>button {
            background-color: #0072C6;  /* Base color (blue) */
            color: white;
            border: none;
            padding: 0.6em 1.2em;
            border-radius: 6px;
            transition: background-color 0.3s ease;
        }
        .stButton>button:hover {
            background-color: #005A9E;  /* Darker blue on hover */
            color: white;
    }
    .stButton>button:active {
        background-color: #004C87;
        color: white !important;
    }
    .stButton>button:focus {
        color: white !important;
        box-shadow: none;
        outline: none;
    </style>
""", unsafe_allow_html=True)
st.title("🌿 Plant Disease Detector")
st.write("Upload an image and the CNN model will predict the disease class.")

uploaded_file = st.file_uploader("📷 Choose an image (JPG, PNG)...", type=["jpg", "jpeg", "png"])

def predict_class(image_path):
    img = tf.keras.preprocessing.image.load_img(image_path,target_size=(128,128))
    input_arr = tf.keras.preprocessing.image.img_to_array(img)
    input_arr = np.array([input_arr])
    prediction = model.predict(input_arr)
    predicted_class_index = np.argmax(prediction)
    predicted_class = class_names[predicted_class_index]
    return predicted_class

if uploaded_file is not None:
    try:
        img = Image.open(uploaded_file).convert("RGB")
    
        # Show image button
        if st.button("Show Image"):
            st.image(img, caption="Uploaded Image", use_container_width=True)
            
        # Predict button
        if st.button("Predict Disease"):
            with st.spinner("Predicting..."):
                predicted_class = predict_class(uploaded_file)
            st.success(f"Predicted Class: **{predicted_class}**")
    except Exception as e:
        st.error(f"⚠️ Error: {e}")
            