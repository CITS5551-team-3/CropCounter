@REM @echo off
@REM set IMAGE_NAME=cropcounter

@REM :: Navigate to the directory where the script is located
@REM cd /d %~dp0

@REM :: Check if the image exists
@REM docker images -q %IMAGE_NAME% >nul 2>&1
@REM if errorlevel 1 (
@REM     echo Image not found. Building the Docker image...
@REM     docker build -t %IMAGE_NAME% .
@REM ) else (
@REM     echo Image already exists. Skipping build.
@REM )

@REM :: Run the Docker container
@REM docker run -p 8501:8501 %IMAGE_NAME%

@echo off
set IMAGE_NAME=cropcounter

:: Navigate to the directory where the script is located
cd /d %~dp0

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