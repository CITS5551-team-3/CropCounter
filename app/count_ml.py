import io
import math
from typing import Optional

import cv2
import numpy as np
import streamlit as st
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

from reassembler import Reassembler, Rect
from params import Params
from PIL import Image
from utils import *
from contours import *

import time

import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")

PARAMS = None

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
                nir_channel = math.inf, re_channel = math.inf, headless=False) -> int:
    """expects `np.seterr(divide='ignore', invalid='ignore')`"""
    useROI = bool(PARAMS.use_fast_mode)
    if useROI:
        #if not headless: display_image("Uploaded Image", cv2.cvtColor(image, cv2.COLOR_BGR2RGB), "Original Image")
        st = time.time()
        image0 = image
        reassembler = Reassembler()
        image = reassembler.reassemble(image, autosize=True, margin=10, border=12)
        print("Reassemble ", time.time() - st)
        #if not headless: display_image("Cropped Image", cv2.cvtColor(image, cv2.COLOR_BGR2RGB), "Image with ROIs")

    st = time.time()
    # Extract color channels from the image

    detection_model = AutoDetectionModel.from_pretrained(
        model_type="yolov8",
        model_path="app/best.pt",
        confidence_threshold=float(PARAMS.confidence_threshold)
    )
    result = get_sliced_prediction(
        image,
        detection_model,
        slice_height=int(PARAMS.slice_size),
        slice_width=int(PARAMS.slice_size),
        overlap_height_ratio=float(PARAMS.overlap_ratio),
        overlap_width_ratio=float(PARAMS.overlap_ratio)
    )
    print(len(result.object_prediction_list))


    print("Detection ", time.time() - st)
    if useROI:
        image = image0

    bounding_boxes = []
    image_height, image_width, _ = image.shape
    thickness = max(1, int(image_width * 0.001))  # Ensuring thickness is at least 1

    st = time.time()
    for obj in result.object_prediction_list:
        bbox = obj.bbox
        x, y, w, h = int(bbox.minx), int(bbox.miny), int(bbox.maxx - bbox.minx), int(bbox.maxy - bbox.miny)
        if useROI:
            rect_o = reassembler.reverse_mapping([Rect(x, y, w, h)])[0].dst
            x, y, w, h = rect_o.x, rect_o.y, rect_o.w, rect_o.h
        bounding_boxes.append((x, y, w, h))
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), thickness)
    print("Bounding Boxes ", time.time() - st)

    save_bounding_boxes(image, bounding_boxes, 'bounding_boxes.json')

    # print(time.time() - start)
    crop_count = len(bounding_boxes)
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


def count(params: Params, headless=False):
    global PARAMS
    PARAMS = params

    np.seterr(divide='ignore', invalid='ignore')

    IMAGE_BITS = 8
    RED_CHANNEL = 1
    GREEN_CHANNEL = 2
    BLUE_CHANNEL = 3

    if headless:
        filename = "../original images/0I8A0573.JPG"
        image = cv2.imread(filename)
        if image is None:
            raise Exception
        image, crop_count = count_image(
            image,
            image_bits=IMAGE_BITS,
            red_channel=RED_CHANNEL,
            green_channel=GREEN_CHANNEL,
            blue_channel=BLUE_CHANNEL,
            headless=True
        )
        # print(crop_count)
        return crop_count

    if 'uploaded_image' in st.session_state:
        pil_image = Image.open(io.BytesIO(st.session_state['uploaded_image']))
        image = pil_to_cv2(pil_image)
        
        image, crop_count = count_image(
            image,
            image_bits=IMAGE_BITS,
            red_channel=RED_CHANNEL,
            green_channel=GREEN_CHANNEL,
            blue_channel=BLUE_CHANNEL,
        )


        # image = draw_centered_bbox(image, 50, 50)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Convert image to bytes with file format
        st.session_state['counted_image'] = image
        st.session_state['crop_count'] = crop_count

    



if __name__ == "__main__":
    # start = time.time()
    PARAMS = Params()
    count(PARAMS, headless=True)
    # print(time.time() - start)
