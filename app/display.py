# app/display.py
import io

import streamlit as st
from PIL import Image
from utils import display_image, preconditons


def display():
    if not preconditons(): return
    return
    image = Image.open(io.BytesIO(st.session_state['uploaded_image']))
    display_image("Uploaded Image", image, "Uploaded Image")


def display_final():
    if not preconditons(): return

    image = Image.open(io.BytesIO(st.session_state['counted_image']))
    crop_count = st.session_state['crop_count']
    
    st.success("Counted successfully")
    display_image("Crop count", image, "Processed Image")
    st.info(f"Count: {crop_count}")
