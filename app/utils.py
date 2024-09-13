import streamlit as st

def display_image(subheader, image, caption):
    st.subheader(subheader)
    st.image(image, caption=caption, use_column_width=True)

def preconditons(display_error=True):
    if 'uploaded_image' not in st.session_state:
        if display_error: st.error("Error: Please upload a valid image")
        return False
    return True

def html(str):
    st.html(str)