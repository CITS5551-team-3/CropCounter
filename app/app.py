# app/app.py
import streamlit as st
from count_ml import count
from display import display, display_final
from download_files import download
from params import Params
from upload import upload
from utils import html, preconditons


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

    st.title("Crop Counter")

    # Upload section
    upload()

    if not preconditons(display_error=False): return
    PARAMS = Params()
    # Display section
    display()

    PARAMS.display_params()

    # Preprocess section
    # if (st.button("Count", type="primary")):

    count(PARAMS)

    display_final()

    download()

if __name__ == "__main__":
    main()
