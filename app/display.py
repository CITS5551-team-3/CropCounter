# app/display.py
import streamlit as st
from PIL import Image
import io
from utils import preconditons, display_image

def display():
    if not preconditons(): return

    image = Image.open(io.BytesIO(st.session_state['uploaded_image']))
    display_image("Uploaded Image", image, "Uploaded Image")
