import numpy as np
import streamlit as st


class Params:
    def __init__(self, ct=0.3, ss=1200, o=0.2):
        # Default values for the parameters
        self.confidence_threshold=ct
        self.slice_size=ss
        self.overlap_ratio=0.2

    def display_params(self):
        # Display and update parameters in Streamlit
        st.sidebar.header("Image Processing Parameters")

        confidence_threshold = st.sidebar.slider(
            "Confidence Threshold", 0.1, 0.9, self.confidence_threshold, step=0.05
        )

        slice_size = st.sidebar.slider(
            "Slice Size", 640, 2500, self.slice_size
        )

        overlap_ratio = st.sidebar.slider(
            "Overlap Ratio", 0.05, 0.5, self.overlap_ratio
        )

        use_fast_mode = st.sidebar.checkbox(
            "Fast Mode", value=False
        )

        self.slice_size = slice_size
        self.confidence_threshold = confidence_threshold
        self.overlap_ratio = overlap_ratio
        self.use_fast_mode = use_fast_mode


PARAMS = Params()
