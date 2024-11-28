import streamlit as st
import torch
import numpy as np
from PIL import Image

# Load YOLOv5 model from Ultralytics
@st.cache_resource  # Cache the model to avoid reloading
def load_model():
    return torch.hub.load('ultralytics/yolov5', 'yolov5s')  # Small model (you can switch to yolov5m or yolov5l)

model = load_model()

# Streamlit UI
st.title("🚗 Autonomous Vehicle Perception System")
st.write("Detect and classify objects, pedestrians, and road signs in real-time.")

# Upload an image file
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Open image using PIL and display it
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_container_width=True)  # Updated parameter
    
    # Convert PIL image to NumPy array
    img_cv = np.array(image)
    
    # Perform object detection
    results = model(img_cv)
    
    # Render and display the detection results
    detected_img = np.squeeze(results.render())
    st.image(detected_img, caption='Detected Objects', use_container_width=True)  # Updated parameter

    # Display detected objects and classes
    st.write("### Detection Results:")
    for det in results.pandas().xyxy[0].itertuples():
        st.write(f"- **{det.name}** detected with confidence {det.confidence:.2f}")
