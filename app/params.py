import numpy as np
import streamlit as st


class Params:
    def __init__(self):
        # Default values for the parameters
        self.mask_threshold = 0
        self.threshold = 10
        self.erosion_kernel = np.ones((3, 3), np.uint8)
        self.dilation_kernel = np.ones((3, 3), np.uint8)

    def display_params(self):
        # Display and update parameters in Streamlit
        st.sidebar.header("Image Processing Parameters")

        self.mask_threshold = st.sidebar.slider(
            "Mask Threshold", 0, 255, self.mask_threshold
        )
        self.threshold = st.sidebar.slider(
            "Threshold", 0, 255, self.threshold
        )

        erosion_size = st.sidebar.slider(
            "Erosion Kernel Size", 1, 10, self.erosion_kernel.shape[0]
        )
        dilation_size = st.sidebar.slider(
            "Dilation Kernel Size", 1, 10, self.dilation_kernel.shape[0]
        )

        self.erosion_kernel = np.ones((erosion_size, erosion_size), np.uint8)
        self.dilation_kernel = np.ones((dilation_size, dilation_size), np.uint8)


PARAMS = Params()
