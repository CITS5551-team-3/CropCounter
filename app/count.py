import io
import math
from typing import Optional

import cv2
import numpy as np
import streamlit as st
from params import PARAMS
from PIL import Image
from utils import *
from contours import *


def save_bounding_boxes(image, bounding_boxes, output_json_file):
    # Create a list of bounding box dictionaries
    boxes_data = []
    for (x, y, w, h) in bounding_boxes:
        boxes_data.append({
            "x": int(x),
            "y": int(y),
            "width": int(w),
            "height": int(h)
        })

    st.session_state['bounding_boxes'] = boxes_data


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
    NGRDI_mask = cv2.threshold(NGRDI, PARAMS.mask_threshold, 255, cv2.THRESH_BINARY)[1].astype(np.uint8)


    # mask and process the image

    img = cv2.bitwise_or(img, img, mask=NGRDI_mask)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    display_image("NGRDI_mask", cv2.cvtColor(img, cv2.COLOR_BGR2RGB), "After applying NGRDI_mask")

    _, img = cv2.threshold(img, PARAMS.threshold, 255, cv2.THRESH_BINARY)

    display_image("Thresholding", cv2.cvtColor(img, cv2.COLOR_BGR2RGB), "After applying Thresholding")

    # image cleaning with erosion/dilation

    img = cv2.erode(img, PARAMS.erosion_kernel, iterations = 6)

    display_image("Erode", cv2.cvtColor(img, cv2.COLOR_BGR2RGB), "After applying Erosion")

    img = cv2.dilate(img, PARAMS.dilation_kernel, iterations = 8)

    display_image("Dilate", cv2.cvtColor(img, cv2.COLOR_BGR2RGB), "After applying Dilation")

    contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)


    contours = get_filtered_contours(img, contours)

    # Get image dimensions
    image_height, image_width, _ = image.shape

    # Define the thickness as a fraction of the image's width (e.g., 1% of the width)
    thickness = max(1, int(image_width * 0.001))  # Ensuring thickness is at least 1

    bounding_boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        bounding_boxes.append((x, y, w, h))
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), thickness)

    save_bounding_boxes(image, bounding_boxes, 'bounding_boxes.json')
    crop_count = len(contours)
    return image, crop_count

# Convert OpenCV image to PIL Image
def cv2_to_pil(cv2_image):
    # Convert BGR to RGB
    rgb_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
    # Convert to PIL Image
    pil_image = Image.fromarray(rgb_image)
    return pil_image

def pil_to_cv2(pil_image):
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
        image = pil_to_cv2(pil_image)
        
        image, crop_count = count_image(
            image,
            image_bits=IMAGE_BITS,
            red_channel=RED_CHANNEL,
            green_channel=GREEN_CHANNEL,
            blue_channel=BLUE_CHANNEL
        )


        image = draw_centered_bbox(image, 50, 50)
        image = cv2_to_pil(image)

        display_image("Crop count", image, "Processed Image")
        st.info(f"Count: {crop_count}")
        
        # Convert image to bytes with file format
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        st.session_state['counted_image'] = buffer.getvalue()
        st.session_state['crop_count'] = crop_count

    