@echo off

:: Navigate to the directory where the script is located
cd /d "%~dp0.."

:: Create a virtual environment using Python 3
python -m venv venv

:: Activate the virtual environment
call venv\Scripts\activate

:: Install dependencies from requirements.txt
pip install -r requirements.txt
