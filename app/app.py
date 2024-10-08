# app/app.py
import streamlit as st
from display import display, display_final
from download_files import download
from params import Params
from upload import upload
from utils import html, preconditons
import os
from crop import Crop


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


    for file in st.session_state['uploaded_files']:
        st.write(st.session_state.keys())
        # if '0I8A0493.JPG_ssf' in st.session_state.keys():
        #     st.info("Below this")
        #     st.write(st.session_state['0I8A0493.JPG_ssf'])
        st.subheader(file.name)
        filename = file.name
        # PARAMS = Params()
        # Display section
        crop: Crop = st.session_state[filename]
    
        crop.get_params()
        crop.display_counted_image()

        # PARAMS.display_params()

        # # Preprocess section
        # # if (st.button("Count", type="primary")):

        # count(PARAMS)

        # display_final()

        # download()

if __name__ == "__main__":
    main()
