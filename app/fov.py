import streamlit as st
import math

# Define FOV class
class FOV:
    def __init__(self, height=2000, sensor_width=36.0, sensor_height=24.0, focal_length=24.0):
        """
        Initialize the FOV class with default values for sensor dimensions, focal length, and height.
        
        :param sensor_width: Width of the sensor in mm (default 36.0)
        :param sensor_height: Height of the sensor in mm (default 24.0)
        :param focal_length: Focal length of the camera in mm (default 24.0)
        :param height: Height from the camera to the object in mm (default 1000)
        """
        self.sensor_width = sensor_width
        self.sensor_height = sensor_height
        self.focal_length = focal_length
        self.height = height
        
    def calculate(self):
        """
        Calculate the field of view dimensions (width, height) and area 
        given the current attributes, using height in meters and returning area in m².
        
        :return: tuple (FOV width in meters, FOV height in meters, FOV area in m²)
        """
        # Convert height from meters to millimeters for calculations
        height_mm = self.height * 1000  # height is in meters, convert to mm

        # Calculate horizontal and vertical angle of view
        theta_width = 2 * math.degrees(math.atan((self.sensor_width / 2) / self.focal_length))
        theta_height = 2 * math.degrees(math.atan((self.sensor_height / 2) / self.focal_length))

        # Calculate the field of view (width and height) in mm at the given height
        fov_width_mm = 2 * height_mm * math.tan(math.radians(theta_width / 2))
        fov_height_mm = 2 * height_mm * math.tan(math.radians(theta_height / 2))

        # Convert the field of view dimensions to meters
        fov_width_m = fov_width_mm / 1000
        fov_height_m = fov_height_mm / 1000

        # Calculate the area of the field of view in square meters
        fov_area_m2 = fov_width_m * fov_height_m

        return fov_width_m, fov_height_m, fov_area_m2


    def display(self):
        st.subheader("Field of View (FOV) Calculator")
        
        # Streamlit input fields for FOV parameters
        sensor_width = st.number_input("Sensor Width (mm)", value=36.0, step=1.0)
        sensor_height = st.number_input("Sensor Height (mm)", value=24.0, step=1.0)
        focal_length = st.number_input("Focal Length (mm)", value=24.0, step=1.0)
        height = st.number_input("Height (m)", value=1.0, step=0.1)

        # Initialize the FOV object with inputs
        fov = FOV(sensor_width=sensor_width, sensor_height=sensor_height, focal_length=focal_length, height=height)

        fov_width, fov_height, fov_area = fov.calculate()

        # Store the FOV area in session state
        st.session_state['fov_area'] = fov_area
        st.info(f"FOV Area: {fov_area:.3f} m²")