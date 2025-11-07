import requests
import zipfile
import io
import pandas as pd
import os

# Fetches and loads the Last.fm dataset.
class LastFmLoader:

  # URL for the dataset zip file.
  _ZIP_URL = "https://files.grouplens.org/datasets/hetrec2011/hetrec2011-lastfm-2k.zip"
  # Directory to store the extracted data.
  _DATA_DIR = "lastfm-2k"

  def __init__(self):
    self.interactions = None
    self.artists = None
    self._interactions_file = os.path.join(self._DATA_DIR, 'user_artists.dat')
    self._artists_file = os.path.join(self._DATA_DIR, 'artists.dat')

  # Downloads and extracts the dataset if not already present.
  def _download_data(self):
    
    if os.path.exists(self._DATA_DIR):
      print(f"Directory {self._DATA_DIR} already exists. Skipping download.")
      return
    
    os.makedirs(self._DATA_DIR, exist_ok=True)
    
    print(f"Downloading data from {self._ZIP_URL}...")
    try:
      response = requests.get(self._ZIP_URL)
      response.raise_for_status()

      print('Extracting data...')
      with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        z.extractall(self._DATA_DIR)

      print('Download and extraction complete.')

    except requests.exceptions.RequestException as e:
      print(f"Error downloading file: {e}")
      raise
    except zipfile.BadZipFile as e:
      print(f"Error extracting file: {e}")
      raise
    except Exception as e:
      print(f"An error occurred during download/extraction: {e}")

  # Loads the interaction and artist data from the CSV files into pandas DataFrames.
  def load_data(self):

    self._download_data()

    try:
      print('Loading interactions data...')
      self.interactions = pd.read_csv(
          self._interactions_file, 
          sep='	',
          header=0, # Use the first row (row 0) as the header
          encoding='utf-8'
      )

      print('Loading artists data...')
      self.artists = pd.read_csv(
          self._artists_file,
          sep='	',
          header=0, # Use the first row (row 0) as the header
          encoding='utf-8',
          usecols=['id', 'name'] # We still only want these columns
      )
      print('Data loading complete.')
    
    except FileNotFoundError as e:
      print(f"Error loading data: {e}")
      raise
    except Exception as e:
      print(f"An error occurred during data loading: {e}")

# A simple test to run the loader independently.
if __name__ == "__main__":
  loader = DataLoader()
  loader.load_data()
  if loader.interactions is not None:
    print(loader.interactions.head())
  
  if loader.artists is not None:
    print(loader.artists.head())
