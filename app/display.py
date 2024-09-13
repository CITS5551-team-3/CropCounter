# app/display.py
import streamlit as st
from PIL import Image
import io

def display_image():
    st.subheader("Display Image")

    if 'uploaded_image' in st.session_state:
        # Load the image from session state
        image = Image.open(io.BytesIO(st.session_state['uploaded_image']))
        
        st.image(image, caption="Uploaded Image", use_column_width=True)
    else:
        st.info("No image uploaded yet. Please upload an image above.")
