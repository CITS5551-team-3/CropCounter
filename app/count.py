import math
import time
from typing import cast, Optional

import cv2
import numpy as np
import rasterio

import streamlit as st
from PIL import Image
import io


def count_image(image: any, *, image_bits: int,
                red_channel = math.inf, green_channel = math.inf, blue_channel = math.inf,
                nir_channel = math.inf, re_channel = math.inf) -> int:
    """expects `np.seterr(divide='ignore', invalid='ignore')`"""

    # Extract color channels from the image
    red_raw = image[:, :, 2]  # Red channel
    green_raw = image[:, :, 1]  # Green channel
    blue_raw = image[:, :, 0]  # Blue channel

    # convert from ints to 0-1 floats

    red: Optional[np.ndarray]
    green: Optional[np.ndarray]
    blue: Optional[np.ndarray]
    # nir: Optional[np.ndarray]
    # re: Optional[np.ndarray]

    image_max_value = 2 ** image_bits - 1

    if red_raw is not None:
        red = red_raw.astype(float) / image_max_value

    if green_raw is not None:
        green = green_raw.astype(float) / image_max_value

    if blue_raw is not None:
        blue = blue_raw.astype(float) / image_max_value


    if red is None or green is None or blue is None:
        raise ValueError("not all rgb channels available")

    img = np.multiply(cv2.merge([blue, green, red]), 255).astype(np.uint8)

    # generate mask

    NGRDI = np.subtract(green, red) / np.add(green, red)
    NGRDI_mask = cv2.threshold(NGRDI, 0, 255, cv2.THRESH_BINARY)[1].astype(np.uint8)


    # mask and process the image

    img = cv2.bitwise_or(img, img, mask=NGRDI_mask)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, img = cv2.threshold(img, 10, 255, cv2.THRESH_BINARY)

    # image cleaning with erosion/dilation

    img = cv2.erode(img, np.ones((2, 2), np.uint8), iterations = 8)
    img = cv2.dilate(img, np.ones((3, 3), np.uint8), iterations = 9)

    st.subheader("Plant contours")
    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Plant contours after preprocessing", use_column_width=True)

    contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    def contours_filter(contour: cv2.typing.MatLike) -> bool:
        """True => keep"""

        perimeter = cv2.arcLength(contour, True)
        if perimeter < 10: return False

        return True

    contours = [contour for contour in contours if contours_filter(contour)]
    # Get image dimensions
    image_height, image_width, _ = image.shape

    # Define the thickness as a fraction of the image's width (e.g., 1% of the width)
    thickness = max(1, int(image_width * 0.001))  # Ensuring thickness is at least 1

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), thickness)

    crop_count = len(contours)
    return image, crop_count

def pil_to_cv2_bgr(pil_image):
    # Convert PIL image to RGB (PIL image is typically in RGB mode)
    pil_image = np.array(pil_image)
    
    # Convert RGB to BGR
    bgr_image = cv2.cvtColor(pil_image, cv2.COLOR_RGB2BGR)
    
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
        
        st.subheader("Plant count")
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Processed Image", use_column_width=True)
        st.info(f"Count: {crop_count}")
    