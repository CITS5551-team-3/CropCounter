# Use the official Python 3.10 image as the base image
FROM python:3.10.11

# Set the working directory in the container
WORKDIR /app

# Copy the requirements.txt file into the container
COPY requirements.txt .

RUN apt-get update

RUN apt-get install ffmpeg libsm6 libxext6 -y

# Install any required packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire app directory into the container
COPY . .

# Expose port 8501 for Streamlit
EXPOSE 8501

# Command to run the Streamlit app
CMD ["streamlit", "run", "app/app.py"]
