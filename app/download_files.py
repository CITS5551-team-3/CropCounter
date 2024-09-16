import io
import json

import streamlit as st


def download():
    # Check if bounding boxes and image are in session state
    if 'bounding_boxes' not in st.session_state or 'counted_image' not in st.session_state:
        st.warning("No data available to download")
        return
    
    bounding_boxes = st.session_state['bounding_boxes']
    counted_image_bytes = st.session_state['counted_image']
    
    # Create JSON data for bounding boxes
    json_data = json.dumps(bounding_boxes, indent=4)
    
    # Create a file-like object for the JSON data
    json_file = io.BytesIO(json_data.encode('utf-8'))
    json_file.name = 'bounding_boxes.json'
    
    # Create an image file-like object from the counted image
    image_file = io.BytesIO(counted_image_bytes)
    image_file.name = 'counted_image.png'
    
    # Create columns for side-by-side buttons
    col1, col2, _ = st.columns(3)
    
    # Download button for JSON
    with col1:
        st.download_button(
            label="Download Bounding Boxes",
            data=json_file,
            file_name=json_file.name,
            mime='application/json'
        )
    
    # Download button for image
    with col2:
        st.download_button(
            label="Download Counted Image",
            data=image_file,
            file_name=image_file.name,
            mime='image/png'
        )
