# Senior Project in Computer Science/Data Science
## Authors: Jack Alberto (jia15), Tommaso Beretta (txb341), William Hugus (wmh42), Maximilian Schulten (mls384), Nathaniel Hahn (nrh51)
[![codecov](https://codecov.io/gh/Max-Schulten/CSDS-395/branch/main/graph/badge.svg)](https://codecov.io/gh/Max-Schulten/CSDS-395)
# Running the application
## With Docker
### Building with Docker
First, `cd` into the project directory. Then, run:
```bash
docker compose build
```
> ***NOTE***: This could take north of 20 minutes due to some relatively heavy dependencies, so be prepared
### Running with Docker
Once the image is baked, running the app is as easy as:
```bash
docker compose up -d
```
We use the -d flag to detach from the shell, but it isn't strictly necessary. The application is now running on `localhost:8000`. It will take a minute to load all models into memory, so don't be surprised if the connection is refused at first.
### Spinning down the container
When done, spin down the container with:
```bash
docker compose down -v
```
## Without Docker
### Install dependencies
In a Python virtual environment or otherwise, using pip:
```bash
pip install -r requirements-dev.txt
```
In a Conda environment the above is also possible, but for Conda's environment solver:
From the base directory:
```bash
conda install --file requirements-dev.txt
```
*Note*: `requirements-dev.txt` includes pytest, this was omitted from `requirements.txt` to slim down the Docker image. For a slightly smaller footprint, consider installing `requirements.txt`.
### Spinning up the app
The easiest start is running:
```bash
python backend/app.py
```
From the base directory. This starts the flask server in debug mode, and after a moment serves the app @ `http://localhost:5000`
# Architecture of the App
<img width="683" height="800" alt="architecture" src="https://github.com/user-attachments/assets/234097a7-a86f-4d19-b3d8-b2f8b68a9b49" />
