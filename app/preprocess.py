# app/preprocess.py
import streamlit as st
from PIL import Image, ImageFilter
import io

def preprocess_image(image, blur_radius):
    """Apply blur filter to the image."""
    if blur_radius > 0:
        return image.filter(ImageFilter.GaussianBlur(blur_radius))
    return image

def preprocess_page():
    st.subheader("Preprocess Image")

    # Slider for blur radius
    blur_radius = st.slider("Blur Radius", min_value=0, max_value=10, value=0, step=1)

    if 'uploaded_image' in st.session_state:
        # Load the image from session state
        image = Image.open(io.BytesIO(st.session_state['uploaded_image']))

        # Preprocess the image
        processed_image = preprocess_image(image, blur_radius)

        # Display the processed image
        st.image(processed_image, caption="Processed Image", use_column_width=True)
    else:
        st.info("No image uploaded yet. Please upload an image on the Upload page.")
