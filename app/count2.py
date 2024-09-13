import cv2
import numpy as np

import math
import time
from typing import cast, Optional

import cv2
import numpy as np
import rasterio

import streamlit as st
from PIL import Image
import io


def count_image(cv2_bgr_image, image_bits, red_channel, green_channel, blue_channel):
    # Ensure image is a valid BGR image
    if not isinstance(cv2_bgr_image, np.ndarray):
        raise TypeError("Image should be a NumPy array")
    if cv2_bgr_image.ndim != 3 or cv2_bgr_image.shape[2] != 3:
        raise ValueError("Image should have 3 channels (BGR)")
    if cv2_bgr_image.dtype != np.uint8:
        raise TypeError("Image should be of type np.uint8")

    # Convert to grayscale
    gray = cv2.cvtColor(cv2_bgr_image, cv2.COLOR_BGR2GRAY)

    # Find contours
    contours, _ = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours on the original image
    img_with_contours = cv2.drawContours(cv2_bgr_image.copy(), contours, -1, (0, 255, 0), 1)

    # Draw rectangles around contours
    for contour in contours:
        if cv2.contourArea(contour) > 10:  # Filter out small contours
            x, y, w, h = cv2.boundingRect(contour)
            img_with_contours = cv2.rectangle(img_with_contours, (x, y), (x + w, y + h), (0, 0, 255), 2)
    
    # Count valid contours
    def contours_filter(contour):
        perimeter = cv2.arcLength(contour, True)
        return perimeter >= 10

    valid_contours = [contour for contour in contours if contours_filter(contour)]
    crop_count = len(valid_contours)
    
    return img_with_contours, crop_count

def pil_to_cv2_bgr(pil_image):
    # Convert PIL image to RGB (PIL image is typically in RGB mode)
    rgb_image = np.array(pil_image.convert('RGB'))
    
    # Convert RGB to BGR
    bgr_image = rgb_image[:, :, ::-1]
    
    return bgr_image

def count():
    np.seterr(divide='ignore', invalid='ignore')

    IMAGE_BITS = 8
    RED_CHANNEL = 1
    GREEN_CHANNEL = 2
    BLUE_CHANNEL = 3

    if 'uploaded_image' in st.session_state:
        pil_image = Image.open(io.BytesIO(st.session_state['uploaded_image']))
        image = pil_to_cv2_bgr(pil_image)
        
        image, crop_count = count_image(
            image,
            image_bits=IMAGE_BITS,
            red_channel=RED_CHANNEL,
            green_channel=GREEN_CHANNEL,
            blue_channel=BLUE_CHANNEL
        )
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Processed Image", use_column_width=True)
        st.info(f"Count: {crop_count}")
    