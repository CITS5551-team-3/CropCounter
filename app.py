import streamlit as st
from PIL import Image
import io

def main():
    st.title("Image Upload and Display")

    # File uploader widget
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        # Display filename
        filename = uploaded_file.name
        st.text(f"Filename: {filename}")

        # Read and display the image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

if __name__ == "__main__":
    main()
