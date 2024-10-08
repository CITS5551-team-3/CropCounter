import streamlit as st
import io
import json
from PIL import Image
from params import Params
from count import count_from_image
from utils import display_image, preconditons
import time
import os

class Crop():
    def __init__(self, filename: str, image):
        self.filename = filename
        self.original_image = Image.open(io.BytesIO(image))
        
        self.params = Params(filename)
        self.crop_count = None
        self.bbox = None
        self.counted_image = None

    def get_params(self):
        self.params.display_params()
        self.crop_count = None
        self.bbox = None
        self.counted_image = None
        self.count_crops()

    def count_crops(self):        
        # Unpack params into individual arguments
        headless = not st.checkbox(f"Display intermediate images for {self.filename}", value=False)

        cached_result = self.cached_count_crops(self.original_image, headless, **vars(self.params))
        self.update_data(*cached_result)

    @st.cache_data(show_spinner=False)
    def cached_count_crops(_self, _original_image, headless, filename="", erosion_iterations=6, dilation_iterations=8, split_scale_factor=1.4, minimum_width_threshold=40):
        # Recreate Params object inside the cached function
        params = Params(filename, erosion_iterations, dilation_iterations, split_scale_factor, minimum_width_threshold)
        return count_from_image(_original_image, params, headless)

    def update_data(self, counted_image, crop_count, bbox):
        self.counted_image = counted_image
        self.crop_count = crop_count
        self.bbox = bbox

        st.session_state[self.filename] = self

    def counted_image_file(self):
        if self.crop_count is None or self.counted_image is None:
            st.warning("No data available to download")
            return
        
        # Create an image file-like object from the counted image
        filename_with_count = f"{self.filename} - {self.crop_count}.png"
        image_file = io.BytesIO(self.counted_image)
        image_file.name = filename_with_count

        return image_file

    def bbox_file(self):
        if self.bbox is None:
            st.warning("No data available to download")
            return
        
        json_data = json.dumps(self.bbox, indent=4)
        json_file = io.BytesIO(json_data.encode('utf-8'))
        filename, _ = os.path.splitext(self.filename)
        json_file.name = f'{filename}_{self.crop_count}.json'

        return json_file
    
    def download_button(self):
        # Create columns for side-by-side buttons
        col1, col2 = st.columns(2)
        json_file = self.bbox_file()
        image_file = self.counted_image_file()
        filename, _ = os.path.splitext(self.filename)
        image_file.name = f'{filename}_{self.crop_count}.png'

        # Download button for image
        with col1:
            st.download_button(
                label=f"Download {image_file.name}",
                data=image_file,
                file_name=image_file.name,
                mime='image/png'
            )

        # Download button for JSON
        with col2:
            st.download_button(
                label=f"Download {json_file.name}",
                data=json_file,
                file_name=json_file.name,
                mime='application/json'
            )
        


    def display_counted_image(self):
        image = Image.open(io.BytesIO(self.counted_image))
        display_image(self.filename, image, self.filename)
        self.download_button()
        st.info(f"Crop count for {self.filename} is {self.crop_count}")

    def __repr__(self):
        return self.filename