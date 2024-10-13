import io
import math
from typing import Optional

import cv2
import numpy as np
import streamlit as st
from params import Params
from PIL import Image
from utils import *
from contours import *

import time

import warnings
# from crop import Crop

# Suppress all warnings
warnings.filterwarnings("ignore")

PARAMS: Params

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

    return boxes_data
    # st.session_state['bounding_boxes'] = boxes_data


def count_image(image: cv2.typing.MatLike, PARAMS: Params, image_bits: int, headless=False) -> int:
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
    start = time.time()
    NGRDI = np.subtract(green, red) / np.add(green, red)
    NGRDI_mask = cv2.threshold(NGRDI, 0, 255, cv2.THRESH_BINARY)[1].astype(np.uint8)
    # print(time.time() - start)


    # mask and process the image

    img = cv2.bitwise_or(img, img, mask=NGRDI_mask)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, img = cv2.threshold(img, 10, 255, cv2.THRESH_BINARY)

    # # image cleaning with erosion/dilation
    # # start = time.time()
    img = cv2.erode(img, np.ones((2, 2), np.uint8), iterations = PARAMS.erosion_iterations)
    # # print(time.time() - start)

    if not headless: display_image("Erode", cv2.cvtColor(img, cv2.COLOR_BGR2RGB), "After applying Erosion")
    # # start = time.time()
    img = cv2.dilate(img, np.ones((3, 3), np.uint8), iterations = PARAMS.dilation_iterations)
    # print(time.time() - start)
    # display_image("initial", img, "Hull")
    # img = cv2.morphologyEx(img, cv2.MORPH_OPEN, np.ones((2, 2)), iterations=6)
    # display_image("open", img, "Hull")
    # img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, np.ones((3, 3)), iterations=3)
    if not headless: display_image("Dilate", cv2.cvtColor(img, cv2.COLOR_BGR2RGB), "After applying Dilation")
    # display_image("close", img, "Hull")
    contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # contour_image = image.copy()
    # cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 8)
    # display_image("contour image", img, "Hull")

    # start = time.time()
    contours = get_filtered_contours(img, contours, PARAMS, headless)
    # print("filtering...")
    # print(time.time() - start)

    # Get image dimensions
    image_height, image_width, _ = image.shape

    # Define the thickness as a fraction of the image's width (e.g., 1% of the width)
    thickness = max(1, int(image_width * 0.001))  # Ensuring thickness is at least 1

    # start = time.time()
    # print(time.time() - start)
    bounding_boxes = []
    contour_image = image.copy()
    # cv2.drawContours(contour_image, contours, -1, (0, 0, 255), thickness=8)  # You can adjust the thickness
    # display_image("Contours", contour_image, "contours" )
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        bounding_boxes.append((x, y, w, h))
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), thickness)

    boxes_data = save_bounding_boxes(image, bounding_boxes, 'bounding_boxes.json')

    # print(time.time() - start)
    crop_count = len(contours)
    return image, crop_count, boxes_data

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

def count_from_image(original_image, params, headless=True):
    IMAGE_BITS = 8

    cv2_image = pil_to_cv2(original_image)

    counted_image, crop_count, bbox = count_image(
        cv2_image,
        params,
        IMAGE_BITS,
        headless
    )

    # image = draw_centered_bbox(image, 50, 50)
    pil_image = cv2_to_pil(counted_image)
    
    # Convert image to bytes with file format
    buffer = io.BytesIO()
    pil_image.save(buffer, format="PNG")

    return buffer.getvalue(), crop_count, bbox

    # crop.update_data(buffer.getvalue(), crop_count, bbox)

    # st.session_state[crop.filename] = crop




# def count_from_crop(crop: Crop,  headless=False):
#     np.seterr(divide='ignore', invalid='ignore')

#     IMAGE_BITS = 8
#     RED_CHANNEL = 1
#     GREEN_CHANNEL = 2
#     BLUE_CHANNEL = 3


#     image = pil_to_cv2(crop.original_image)
#     counted_image, crop_count, bbox = count_image(
#         image,
#         image_bits=IMAGE_BITS,
#         red_channel=RED_CHANNEL,
#         green_channel=GREEN_CHANNEL,
#         blue_channel=BLUE_CHANNEL,
#     )




# if __name__ == "__main__":
#     # start = time.time()
#     PARAMS = Params()
#     count(PARAMS, headless=True)
#     # print(time.time() - start)
