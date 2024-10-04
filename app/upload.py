# app/upload.py
import time
import streamlit as st
import os
from crop import Crop

def upload_crop(uploaded_file):
    filename = None
    if uploaded_file is not None:
        filename, _ = os.path.splitext(uploaded_file.name)
    else: return


    # if filename in st.session_state:
    #     st.warning("Image already uploaded. You can view it below.")
    #     return

    # Save the uploaded image to session state
    st.session_state[filename] = Crop(filename, uploaded_file.getvalue())
    st.success("Image uploaded successfully.")


def upload():
    st.subheader("Upload Image")
    uploaded_files = st.file_uploader("Choose image(s)...", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

    if uploaded_files:
        # Delete all the items in Session state
        if 'uploaded_files' in st.session_state:
            del st.session_state['uploaded_files']
        
        st.write(len(st.session_state.keys()))
        st.session_state['uploaded_files'] = uploaded_files

        for uploaded_file in uploaded_files:
            upload_crop(uploaded_file)

        # Clean up files
        filenames = [os.path.splitext(uploaded_file.name)[0] for uploaded_file in uploaded_files]
        for key in st.session_state.keys():
            if key == "uploaded_files": continue

            if key not in filenames:
                del st.session_state[key]

        st.write(st.session_state.keys())






