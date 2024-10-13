import base64

import streamlit as st
import io
import json
from PIL import Image
from streamlit_javascript import st_javascript

from params import Params
from count import count_from_image
from utils import display_image, preconditons
import time
import os


def download_text(image, filename):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    b64 = base64.b64encode(buffer.getbuffer()).decode()
    js_function = f"""(function() {{
        var link = document.createElement('a');
        link.href = 'data:image/png;base64,{b64}';
        link.download = '{filename}';
        link.click();
    }})();"""
    st_javascript(js_function)

def trigger_download():
    st.session_state["trigger_download"] = True


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
        image_file = self.counted_image
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

    @st.fragment
    def download_button(self):
        # Create columns for side-by-side buttons
        col1, col2 = st.columns(2)
        json_file = self.bbox_file()
        image_file = self.counted_image_file()
        filename, _ = os.path.splitext(self.filename)
        image_file.name = f'{filename}_{self.crop_count}.png'

        counted_image_pil = image_file

        # Download button for image
        # with col1:
        #     st.download_button(
        #         label=f"Download {image_file.name}",
        #         data=image_file,
        #         file_name=image_file.name,
        #         mime='image/png'
        #     )

        with col1:
            st.button(f"Download {image_file.name}", on_click=trigger_download, key=image_file.name)
            if st.session_state.get("trigger_download", False):
                # st.write("Exporting, please wait...")
                download_text(counted_image_pil, image_file.name)
                st.session_state["trigger_download"] = False

        # Download button for JSON
        with col2:
            st.download_button(
                label=f"Download {json_file.name}",
                data=json_file,
                file_name=json_file.name,
                mime='application/json'
            )
        


    def display_counted_image(self):
        display_image(self.filename, self.counted_image, self.filename)
        self.download_button()
        st.info(f"Crop count for {self.filename} is {self.crop_count} or {(self.crop_count / st.session_state['fov_area']):.3f} per m²")


    def __repr__(self):
        return self.filename