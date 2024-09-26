import numpy as np
import streamlit as st


class Params:
    def __init__(self, ei=6, di=8, ssf=1.4, mwt=40):
        # Default values for the parameters
        self.erosion_iterations = ei
        self.dilation_iterations = di
        self.split_scale_factor = ssf
        self.minimum_width_threshold = mwt

    def display_params(self):
        # Display and update parameters in Streamlit
        st.sidebar.header("Image Processing Parameters")

        erosion_iterations = st.sidebar.slider(
            "Erosion Iterations", 1, 10, self.erosion_iterations
        )
        dilation_iterations = st.sidebar.slider(
            "Dilation Iterations", 1, 10, self.dilation_iterations
        )
        split_scale_factor = st.sidebar.slider(
            "Split Scale Factor", 1.0, 3.0, self.split_scale_factor, step=0.1
        )
        minimum_width_threshold = st.sidebar.slider(
            "Minimum Width Threshold", 10, 100, self.minimum_width_threshold, step=10
        )


        self.erosion_iterations = erosion_iterations
        self.dilation_iterations = dilation_iterations
        self.split_scale_factor = split_scale_factor
        self.minimum_width_threshold = minimum_width_threshold


PARAMS = Params()
