@echo off

:: Check if Docker is installed
docker --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Docker is not installed. Please install it from: https://docs.docker.com/get-docker/
    pause
    exit /b 1
)

:: Check if Docker is running
docker info >nul 2>&1
if %errorlevel% neq 0 (
    echo Docker is not running. Starting Docker Desktop...
    start "" "C:\Program Files\Docker\Docker\Docker Desktop.exe"  :: Adjust path if necessary
    echo Please wait for Docker to start...

    :: Wait until Docker is running
    :checkDocker
    timeout /t 2 >nul
    docker info >nul 2>&1
    if %errorlevel% neq 0 (
        goto checkDocker
    )
    echo Docker is now running.
)

:: Navigate to the directory where the script is located
cd /d "%~dp0.."

set IMAGE_NAME=cropcounter

:: Check if the image exists
FOR /F "tokens=*" %%i IN ('docker images -q %IMAGE_NAME%') DO SET IMAGE_EXISTS=%%i

if "%IMAGE_EXISTS%"=="" (
    echo Image not found. Building the Docker image...
    docker build -t %IMAGE_NAME% .
) else (
    echo Image already exists. Skipping build.
)

:: Run the Docker container
docker run -d -p 8501:8501 --name cropcounter_container %IMAGE_NAME%

:: Open the URL in the default web browser
start "" "http://localhost:8501"

:: Wait for user input
pause

:: Stop the Docker container without removing it
docker stop cropcounter_container
docker rm cropcounter_container