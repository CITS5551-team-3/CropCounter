# app/upload.py
import time
import streamlit as st
import os
from crop import Crop
from params import Params

def upload_crop(uploaded_file):
    filename = None
    if uploaded_file is not None:
        filename = uploaded_file.name
    else: return

    if filename in st.session_state.keys():
        return
    # if filename in st.session_state:
    #     st.warning("Image already uploaded. You can view it below.")
    #     return

    # Save the uploaded image to session state
    
    st.session_state[filename] = Crop(filename, uploaded_file.getvalue())
    st.success("Image uploaded successfully.")


def upload():
    st.subheader("Upload Image")
    uploaded_files = st.file_uploader("Choose image(s)...", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

    valid_uploaded_files_set = set()
    valid_uploaded_files = []
    for file in uploaded_files:
        if file.name not in valid_uploaded_files_set:
            valid_uploaded_files_set.add(file.name)
            valid_uploaded_files.append(file)
    
    uploaded_files = valid_uploaded_files

    if uploaded_files:

        st.session_state['uploaded_files'] = uploaded_files

        for uploaded_file in uploaded_files:
            upload_crop(uploaded_file)

        # Clean up files
        filenames = [uploaded_file.name for uploaded_file in uploaded_files]
        for key in st.session_state.keys():
            if key == "uploaded_files": continue

            if key not in filenames:
                del st.session_state[key]







