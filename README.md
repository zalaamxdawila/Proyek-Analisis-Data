# Bike Sharing Dashboard 🚲

## Setup Environment - Anaconda

conda create --name bike_sharing_dashboard python=3.9
conda activate bike_sharing_dashboard
pip install -r requirements.txt

## Setup Environment - Shell/Terminal

mkdir bike_sharing_dashboard
cd bike_sharing_dashboard
pipenv install
pipenv shell
pip install -r requirements.txt

## Run steamlit app

streamlit run bike_sharing_dashboard.py
