# app/app.py
import streamlit as st
from download import download
from upload import upload
from display import display
from count import count
from params import PARAMS
from utils import preconditons, html

def main():
    # Custom CSS for full-page sections
    html("""
        <style>
        body {
            margin: 0;
            font-family: sans-serif;
        }
        
        </style>
        """)

    st.title("Crop Annotator")

    # Upload section
    upload()

    if not preconditons(display_error=False): return

    # Display section
    display()

    PARAMS.display_params()

    # Preprocess section
    # if (st.button("Count", type="primary")):
    count()

    download()

if __name__ == "__main__":
    main()
