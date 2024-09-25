@echo off
set IMAGE_NAME=cropcounter

:: Navigate to the directory where the script is located
cd /d %~dp0

:: Check if the image exists
docker images -q %IMAGE_NAME% >nul 2>&1
if errorlevel 1 (
    echo Image not found. Building the Docker image...
    docker build -t %IMAGE_NAME% .
) else (
    echo Image already exists. Skipping build.
)

:: Run the Docker container
docker run -p 8501:8501 %IMAGE_NAME%
