import cv2
import numpy as np
from scipy.stats import norm
import streamlit as st
import matplotlib.pyplot as plt
from params import Params
import time
from concurrent.futures import ThreadPoolExecutor

PARAMS: Params
counter: dict[str, float] = {
    'mask': 0,
    'findContours': 0
}

def split_contour(img, contour, max_width):
    """
    Split a contour into smaller bounding boxes if it exceeds max_width.
    """
    global counter
    x, y, w, h = cv2.boundingRect(contour)

    if w > max_width:
        # Split the bounding box into smaller regions
        num_splits = int(w / max_width) + 1  # Number of segments to split into
        new_contours = []

        for i in range(num_splits):
            # Define a region to mask (this is one smaller bounding box)
            start = time.time()
            # Loop over the split regions
            x_offset = x + i * max_width

            mask = np.zeros_like(img)
            counter['mask'] += time.time() - start
            cv2.rectangle(mask, (x_offset, y), (min(x_offset + max_width, x + w), y + h), 255, -1)

            # Find contours within this region
            start = time.time()
            split_contours, _ = cv2.findContours(cv2.bitwise_and(mask, img), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            counter['findContours'] += time.time() - start
            
            # # Add the split contours to the list
            new_contours.extend(split_contours)
            
            # new_x = x_offset
            # new_w = min(max_width, x + w - new_x)
            # new_contour = np.array([
            #     [new_x, y],
            #     [new_x + new_w, y],
            #     [new_x + new_w, y + h],
            #     [new_x, y + h]
            # ])
            # new_contours.append(new_contour)


        return new_contours
    else:
        return [contour]  # If width is within the limit, keep it as is


# Variable to define the bottom percentage to remove
bottom_percent = 0.8  # For example, remove the bottom 20%

def filter_by_size(contours):
    # Predefined threshold variables (example values)
    height_threshold = 50  # Example value, adjust as needed
    width_threshold = PARAMS.minimum_width_threshold  # Example value, adjust as needed
    area_threshold = height_threshold * width_threshold  # Example value, adjust as needed
    perimeter_threshold = 2 * (height_threshold + width_threshold)  # Example value, adjust as needed

    filtered_contours = []

    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        x, y, w, h = cv2.boundingRect(contour)

        # Calculate width and height from bounding box
        width = w
        height = h

        # Filter based on global threshold variables
        if (area >= area_threshold and 
            perimeter >= perimeter_threshold and 
            width >= width_threshold and 
            height >= height_threshold):
            filtered_contours.append(contour)

    return filtered_contours


def plot_graph(data, label):
        # Sort areas
    sorted_data = sorted(data)

    # Create bins for grouping (optional)
    bins = np.linspace(min(sorted_data), max(sorted_data), 200)  # 10 bins

    # Plot histogram with bins
    plt.figure(figsize=(10, 5))
    plt.hist(sorted_data, bins=bins, color='blue', edgecolor='black')
    plt.xlabel(label)
    plt.ylabel('Frequency')
    plt.title(f'Contour {label} Distribution')
    plt.grid(axis='y')
    plt.tight_layout()

    # Display the plot in Streamlit
    st.pyplot(plt) # type: ignore

def remove_overlapping_bboxes(contours, overlap_threshold=0.4):
    """
    Remove smaller bounding boxes that overlap with larger ones by at least a certain percentage.
    
    Parameters:
    - contours: List of contours to analyze
    - overlap_threshold: Minimum overlap percentage to consider (default 50%)
    
    Returns:
    - List of non-overlapping contours
    """
    boxes = [cv2.boundingRect(c) for c in contours]
    n = len(boxes)
    to_remove = set()

    for i in range(n):
        x1, y1, w1, h1 = boxes[i]
        area1 = w1 * h1
        for j in range(i + 1, n):
            x2, y2, w2, h2 = boxes[j]
            area2 = w2 * h2

            # Calculate intersection coordinates
            x_overlap = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
            y_overlap = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
            intersection_area = x_overlap * y_overlap
            
            # Calculate overlap percentage
            if intersection_area > 0:
                overlap1 = intersection_area / area1
                overlap2 = intersection_area / area2

                if overlap1 >= overlap_threshold or overlap2 >= overlap_threshold:
                    # Mark the smaller bbox for removal
                    if area1 > area2:
                        to_remove.add(j)  # Remove j
                    else:
                        to_remove.add(i)  # Remove i
                        
    # Filter out the contours based on the removal list
    return [contours[i] for i in range(n) if i not in to_remove]


def get_filtered_contours(img, contours, params: Params):
    import time
    global PARAMS
    PARAMS = params

    split_contours = []
    max_width = int(PARAMS.split_scale_factor * np.mean([cv2.boundingRect(c)[2] for c in contours])) if contours else 200

    # Parallelize contour splitting using ThreadPoolExecutor
    start = time.time()
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(lambda contour: split_contour(img, contour, max_width), contours))

    for result in results:
        split_contours.extend(result)
    # print("splitting")
    # print(time.time() - start)
    
    split_contours = filter_by_size(split_contours)
    split_contours = remove_overlapping_bboxes(split_contours)

    # print(counter)

    return split_contours
