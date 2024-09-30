@echo off

:: Check if Docker is installed
docker --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Docker is not installed. Please install it from: https://docs.docker.com/get-docker/
    pause
    exit /b 1
)

set IMAGE_NAME=cropcounter

:: Navigate to the directory where the script is located
cd /d "%~dp0.."

:: Check if the image exists
FOR /F "tokens=*" %%i IN ('docker images -q %IMAGE_NAME%') DO SET IMAGE_EXISTS=%%i

if "%IMAGE_EXISTS%"=="" (
    echo Image not found. Building the Docker image...
    docker build -t %IMAGE_NAME% .
) else (
    echo Image already exists. Skipping build.
)

:: Run the Docker container
docker run -p 8501:8501 %IMAGE_NAME%