import numpy as np
import streamlit as st
import time


class Params:
    def __init__(self, filename="", ei=6, di=8, ssf=1.4, mwt=40):
        # Default values for the parameters
        self.filename = filename
        self.erosion_iterations = ei
        self.dilation_iterations = di
        self.split_scale_factor = ssf
        self.minimum_width_threshold = mwt

    def display_params(self):
        # Display and update parameters in Streamlit
        st.sidebar.header(self.filename)
        erosion_iterations = st.sidebar.slider(
            "Erosion Iterations", 1, 10, st.session_state[f"{self.filename}_ei"], key=f"{self.filename}_ei"
        )
        dilation_iterations = st.sidebar.slider(
            "Dilation Iterations", 1, 10, key=f"{self.filename}_di"
        )
        split_scale_factor = st.sidebar.slider(
            "Split Scale Factor", 1.0, 3.0, step=0.1, key=f"{self.filename}_ssf"
        )
        minimum_width_threshold = st.sidebar.slider(
            "Minimum Width Threshold", 10, 100, step=10, key=f"{self.filename}_mwt"
        )


        self.erosion_iterations = erosion_iterations
        self.dilation_iterations = dilation_iterations
        self.split_scale_factor = split_scale_factor
        self.minimum_width_threshold = minimum_width_threshold

    def __eq__(self, other):
            if not isinstance(other, Params):
                return NotImplemented
            return vars(self) == vars(other)
    
    def __hash__(self):
        return hash(tuple(vars(self).values()))

