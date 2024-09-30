<!-- omit in toc -->
## Installation and Running Guide for Windows

There are 3 different ways to install and run the app. You can run the app via Docker, locally install it using a script, or manually install it.

- [1. Docker](#1-docker)
- [2. Local Installation](#2-local-installation)
- [3. Manual Installation](#3-manual-installation)

### 1. Docker

<!-- omit in toc -->
#### Install Docker
Install Docker by following: https://docs.docker.com/get-docker/

<!-- omit in toc -->
#### Install and run the image
Double click on the `install_docker.bat` file, followed by double clicking on the `run_docker.bat` file.
**Note:** To safely exit, ensure you do not force the terminal window, and rather press any key in the terminal as instructed to close and exit the program safely.  
**Note:** In case the app has changed or been updated, the Docker image will need to be rebuilt, in which case run the `install_docker.bat` file again.

<!-- omit in toc -->
#### Troubleshooting
In case of troubleshooting, open Docker Desktop and remove any images or containers that may already exist, and follow the previous step again.

### 2. Local Installation

<!-- omit in toc -->
#### Install Python
Ensure that Python has been installed on the machine already.

<!-- omit in toc -->
#### Install and run the image
Double click on the `install_local.bat` file. This may take some time and will initially look as though nothing is happening. Do not close the window, it will close automaticlly once the installation is complete. After installing, the program can be started by double clicking on the `run_local.bat` file.
**Note:** To safely exit, ensure you do not force the terminal window, and rather press any key in the terminal as instructed to close and exit the program safely.

### 3. Manual Installation

<!-- omit in toc -->
#### Set up virtual environment
1. Open a new Command Prompt window and navigate to the root directory of the app.
2. Create a virtual environment and activate it with the following commands:
    ```
    python -m venv venv 
    .\venv\Scripts\activate
    ```
3. Install the required dependencies by typing:
   ```
   pip install -r requirements.txt
   ```
4. Run the app by typing:
   ```
   streamlit run app\app.py
   ```
5. If the app does not automatically open, then navigate to http://localhost:8501/
6. To close the app, press `Ctrl + C` in the Command Prompt window to exit the app safely.
