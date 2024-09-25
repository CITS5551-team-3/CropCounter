import cv2
import numpy as np
from scipy.stats import norm
import streamlit as st
import matplotlib.pyplot as plt

def split_contour(img, contour, max_width):
    """
    Split a contour into smaller bounding boxes if it exceeds max_width.
    """
    x, y, w, h = cv2.boundingRect(contour)

    if w > max_width:
        # Split the bounding box into smaller regions
        num_splits = int(w / max_width) + 1  # Number of segments to split into
        new_contours = []

        # Loop over the split regions
        for i in range(num_splits):
            # Define a region to mask (this is one smaller bounding box)
            x_offset = x + i * max_width
            mask = np.zeros_like(img)
            cv2.rectangle(mask, (x_offset, y), (min(x_offset + max_width, x + w), y + h), 255, -1)

            # Find contours within this region
            split_contours, _ = cv2.findContours(cv2.bitwise_and(mask, img), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            # Add the split contours to the list
            new_contours.extend(split_contours)

        return new_contours
    else:
        return [contour]  # If width is within the limit, keep it as is


# Variable to define the bottom percentage to remove
bottom_percent = 0.8  # For example, remove the bottom 20%

def filter_by_size(contours):
    # Predefined threshold variables (example values)
    area_threshold = 2500  # Example value, adjust as needed
    perimeter_threshold = 200  # Example value, adjust as needed
    width_threshold = 50  # Example value, adjust as needed
    height_threshold = 50  # Example value, adjust as needed

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
    st.pyplot(plt)


def get_filtered_contours(img, contours):
    # Split contours first
    split_contours = []
    
    max_width = int(1.5 * np.mean([cv2.boundingRect(c)[2] for c in contours])) if contours else 200

    st.info(max_width)

    for contour in contours:
        split_contours.extend(split_contour(img, contour, max_width))  # Split the contour if necessary
    
    split_contours = filter_by_size(split_contours)
    
    # Assuming you already have contours calculated
    # Example: contours = [...] (your contours list)
    plot_graph([cv2.contourArea(c) for c in split_contours], label="Area")  # Calculate perimeters
    plot_graph([cv2.arcLength(c, True) for c in contours], label="Perimeter")  # Calculate perimeters
    plot_graph([cv2.boundingRect(c)[2] for c in contours], label="Width")  # Calculate perimeters
    plot_graph([cv2.boundingRect(c)[3] for c in contours], label="Height")  # Calculate perimeters

    # Calculate average area, perimeter, width, and height
    areas = [cv2.contourArea(c) for c in contours]
    perimeters = [cv2.arcLength(c, True) for c in contours]
    widths = [cv2.boundingRect(c)[2] for c in contours]  # Widths of each bounding box
    heights = [cv2.boundingRect(c)[3] for c in contours]  # Heights of each bounding box

    avg_area = np.mean(areas) if areas else 0
    avg_perimeter = np.mean(perimeters) if perimeters else 0
    avg_width = np.mean(widths) if widths else 0
    avg_height = np.mean(heights) if heights else 0

    std_area = np.std(areas) if areas else 0
    std_perimeter = np.std(perimeters) if perimeters else 0
    std_width = np.std(widths) if widths else 0
    std_height = np.std(heights) if heights else 0

    z_value = norm.ppf(bottom_percent)
    st.info(z_value)

    threshold_area = avg_area + (std_area * z_value)
    threshold_perimeter = avg_perimeter + (std_perimeter * z_value)
    threshold_width = avg_width + (std_width * z_value)
    threshold_height = avg_height + (std_height * z_value)

    st.info(f"area: {threshold_area}, {avg_area}, {std_area}")
    st.info(f"perimeter: {threshold_perimeter}, {avg_perimeter}, {std_perimeter}")
    st.info(f"width: {threshold_width}, {avg_width}, {std_width}")
    st.info(f"height: {threshold_height}, {avg_height}, {std_height}")

    def contours_filter(contour: cv2.typing.MatLike) -> bool:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        bbox = cv2.boundingRect(contour)
        width = bbox[2]
        height = bbox[3]

        # Filtering conditions based on area, perimeter, width, and height using standard deviation
        if area < threshold_area: return False
        if perimeter < threshold_perimeter: return False
        if width < threshold_width: return False
        if height < threshold_height: return False

        return True

    filtered_contours = []

    # Apply the filter after splitting the contours
    for contour in split_contours:
        if True or contours_filter(contour):
            filtered_contours.append(contour)
    
    return filtered_contours

