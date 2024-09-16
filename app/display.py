# app/display.py
import io

import streamlit as st
from PIL import Image
from utils import display_image, preconditons


def display():
    if not preconditons(): return

    image = Image.open(io.BytesIO(st.session_state['uploaded_image']))
    display_image("Uploaded Image", image, "Uploaded Image")
