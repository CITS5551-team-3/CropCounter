# app/app.py
import streamlit as st
from upload import upload_image
from display import display_image
from preprocess import preprocess_page

def main():
    # Custom CSS for full-page sections
    st.markdown("""
        <style>
        body {
            margin: 0;
            font-family: sans-serif;
        }
        .full-page {
            # min-height: 50vh; /* Full viewport height */
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            padding: 20px;
            box-sizing: border-box;
            overflow: auto; /* Ensure content can scroll if needed */
        }
        </style>
        """, unsafe_allow_html=True)

    st.title("Image Processing App")

    # Upload section
    upload_image()

    # Display section
    st.markdown('<div class="full-page">', unsafe_allow_html=True)
    display_image()
    st.markdown('</div>', unsafe_allow_html=True)

    # Preprocess section
    st.markdown('<div class="full-page">', unsafe_allow_html=True)
    preprocess_page()
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
