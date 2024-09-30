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

if defined IMAGE_EXISTS (
    echo Image already exists. Removing the old image...
    docker rmi %IMAGE_NAME% -f
)

echo Building the Docker image...
docker build -t %IMAGE_NAME% .

exit /b 0