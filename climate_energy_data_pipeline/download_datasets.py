import os
from kaggle.api.kaggle_api_extended import KaggleApi 

def download_kaggle_dataset(dataset_name: str, download_path: str):
	"""
    Downloads a dataset from Kaggle.

    :param dataset_name: Kaggle dataset name (format: 'username/dataset-name')
    :param download_path: path to download the dataset to
	"""

	# Initialize KaggleApi
	api = KaggleApi()

	# For the api to be authenticated, you must have a kaggle.json file in '~/.kaggle/kaggle.json'
	api.authenticate()

	# Ensure download path directory exists
	os.makedirs(download_path, exist_ok=True)

	# Download and unzip dataset
	api.dataset_download_files(dataset_name, path=download_path, unzip=True)
	print(f"Dataset {dataset_name} has been downloaded to {download_path}")


if __name__ == "__main__":

	spain_dataset = "nicholasjhana/energy-consumption-generation-prices-and-weather"
	path_to_download = "data/"

	download_kaggle_dataset(dataset_name=spain_dataset, download_path=path_to_download)