@echo off

:: Navigate to the directory where the script is located
cd /d "%~dp0.."

:: Activate the virtual environment
call venv\Scripts\activate.bat

:: Specify the host and port
set HOST=localhost
set PORT=8501

:: Run the Streamlit app and get the PID
start /B streamlit run app\app.py --server.address %HOST% --server.port %PORT%

set STREAMLIT_PID=%ERRORLEVEL%

:: Wait for user input
set /p dummy="Press any key to exit once program has started... Program Starting..."


:: Kill the Streamlit process using the stored PID
::taskkill /PID %STREAMLIT_PID% /F
taskkill /FI "ImageName eq streamlit*" /T /F

exit /b 0