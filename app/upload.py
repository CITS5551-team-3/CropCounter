# app/upload.py
import streamlit as st


def upload():
    st.subheader("Upload Image")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        # Save the uploaded image to session state
        st.session_state['uploaded_image'] = uploaded_file.getvalue()
        st.success("Image uploaded successfully.")
    elif 'uploaded_image' in st.session_state:
        st.warning("Image already uploaded. You can view it below.")


