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


import cv2

def draw_centered_bbox(image, w, h):
    # Get the image dimensions
    img_h, img_w = image.shape[:2]
    
    # Calculate the coordinates for the center of the image
    center_x = img_w // 2
    center_y = img_h // 2

    # Calculate the top-left corner for the bounding box
    top_left_x = center_x - (w // 2)
    top_left_y = center_y - (h // 2)

    # Calculate the bottom-right corner for the bounding box
    bottom_right_x = top_left_x + w
    bottom_right_y = top_left_y + h

    # Draw the blue rectangle (bounding box)
    cv2.rectangle(image, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), (255, 0, 0), 8)  # Blue color in BGR

    return image
