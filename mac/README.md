<!-- omit in toc -->
## Installation and Running Guide for MacOS

There are 3 different ways to install and run the app. You can run the app via Docker, locally install it using a script, or manually install it.

- [1. Docker](#1-docker)
- [2. Local Installation](#2-local-installation)
- [3. Manual Installation](#3-manual-installation)

### 1. Docker

<!-- omit in toc -->
#### Install docker
Install docker by following: https://docs.docker.com/get-docker/

<!-- omit in toc -->
#### Install and run the image
Double click on the `install_docker` file, followed by double clicking on the `run_docker` file.
**Note:** To safely exit, ensure you do not force the terminal window, and rather press any key in the terminal as instructed to close and exit the program safely.
**Note:** In case the app has changed or been updated, the docker image will need to rebuild, in which case run the `install_docker` file again.

<!-- omit in toc -->
#### Troubleshooting
In case of trouble shooting, open Docker Desktop and remove any images or containers that may already exist and follow the previou step again.

### 2. Local Installation

<!-- omit in toc -->
#### Install python
Ensure that python has been installed on the machine already, if not then follow the instructions at: https://www.python.org/downloads/.
A python version >= 3.10 is recommended.

<!-- omit in toc -->
#### Install and run the image
Double click on the `install_local` file, followed by double clicking on the `run_local` file.
**Note:** To safely exit, ensure you do not force the terminal window, and rather press any key in the terminal as instructed to close and exit the program safely.

### 3. Manual Installation

<!-- omit in toc -->
#### Install python
Ensure that python has been installed on the machine already, if not then follow the instructions at: https://www.python.org/downloads/.
A python version >= 3.10 is recommended.

<!-- omit in toc -->
#### Set up virtual environment
1. Open a new terminal window and navigate to the root directory of the app.
2. Create a virtual environment and activate it with the following commands:
   ```
   python3 -m venv venv
   source venv/bin/activate
   ```
3. Install the required dependencies by typing:
   ```
   pip3 install -r requirements.txt
   ```
4. Run the app by typing:
   ```
   streamlit run app/app.py
   ```
5. If the app does not automatically open, then navigate to http://localhost:8501/
6. To close the app, press `Control + C` in the terminal window to exit the app safely.



