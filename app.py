import os
import numpy as np
import cv2
import tensorflow as tf
import streamlit as st
from tensorflow.keras.models import load_model
from streamlit_drawable_canvas import st_canvas

# Define model path (Ensure it's in .keras format for Keras 3)
MODEL_PATH = "model.keras"

# Check if the model exists
if not os.path.exists(MODEL_PATH):
    st.error("❌ Model file not found! Please train your model by running `python train.py` first.")
    st.stop()

# Load the trained model
try:
    model = load_model("model.keras", compile=False)
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

# Streamlit UI
st.title('🖊️ My Digit Recognizer')
st.markdown("**Try to draw a digit and let the model predict it!**")

SIZE = 192  # Canvas size
mode = st.checkbox("Draw (or Delete)?", True)

# Create a drawing canvas
canvas_result = st_canvas(
    fill_color='#000000',  # Black background
    stroke_width=15,  # Adjust pen thickness
    stroke_color='#FFFFFF',  # White pen for drawing
    background_color='#000000',
    width=SIZE,
    height=SIZE,
    drawing_mode="freedraw" if mode else "transform",
    key='canvas'
)

if canvas_result.image_data is not None:
    # Convert canvas image to 28x28 pixels (for MNIST model)
    img = cv2.resize(canvas_result.image_data.astype('uint8'), (28, 28))

    # Convert RGBA (4 channels) to Grayscale (1 channel)
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)

    # Normalize pixel values (0-1) for the model
    img_gray = img_gray / 255.0

    # Display the processed image
    st.write('📌 **Model Input** (Rescaled to 28x28)')
    st.image(cv2.resize(img_gray, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST), clamp=True)

if st.button('🔍 Predict'):
    try:
        # Reshape the image to match model input shape (1, 28, 28, 1)
        img_reshaped = img_gray.reshape(1, 28, 28, 1)

        # Make a prediction
        predictions = model.predict(img_reshaped)
        predicted_digit = np.argmax(predictions[0])

        # Display prediction results
        st.success(f"✅ **Predicted Digit:** {predicted_digit}")
        st.bar_chart(predictions[0])  # Show confidence scores

    except Exception as e:
        st.error(f"❌ Prediction Error: {e}")
