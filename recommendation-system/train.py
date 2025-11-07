import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import os

from last_fm_loader import LastFmLoader
from model import MatrixFactorization

# Custom PyTorch Dataset for the Last.fm data.
class LastFmDataset(Dataset):

  def __init__(self, users, artists, weights):
    self.users = torch.LongTensor(users)
    self.artists = torch.LongTensor(artists)
    self.weights = torch.FloatTensor(weights)

  def __len__(self):
    return len(self.weights)
  
  def __getitem__(self, idx):
    return self.users[idx], self.artists[idx], self.weights[idx]
  
# Utility for early stopping to prevent overfitting.
class EarlyStopping:

  def __init__(self, patience=5, verbose=False, delta=0, path='checkpoint.pt'):
    self.patience = patience
    self.verbose = verbose
    self.counter = 0
    self.best_score = None
    self.early_stop = False
    self.val_loss_min = np.inf
    self.delta = delta
    self.path = path

  def __call__(self, val_loss, model):

    score = -val_loss
    if self.best_score is None:
      self.best_score = score
      self.save_checkpoint(val_loss, model)
    elif score < self.best_score + self.delta:
      self.counter += 1
      if self.verbose:
        print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
      if self.counter >= self.patience:
        self.early_stop = True
    else:
      self.best_score = score
      self.save_checkpoint(val_loss, model)
      self.counter = 0
  
  def save_checkpoint(self, val_loss, model):

    if self.verbose:
      print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
    torch.save(model.state_dict(), self.path)
    self.val_loss_min = val_loss

# Creates mappings from original user/artist IDs to sequential integer indices.
def create_id_mapping(df):

  user_id_mapping = {original_id: i for i, original_id in enumerate(df['userID'].unique())}
  artist_id_mapping = {original_id: i for i, original_id in enumerate(df['artistID'].unique())}

  user_inv_map = {i: original_id for original_id, i in user_id_mapping.items()}
  artist_inv_map = {i: original_id for original_id, i in artist_id_mapping.items()}

  return user_id_mapping, artist_id_mapping, user_inv_map, artist_inv_map

# Main function to train the matrix factorization model.
def train_model(epochs=20, batch_size=1024, emb_size=50, learning_rate=0.001, model_save_path="model_store/model.pt"):

  os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

  loader = LastFmLoader()
  loader.load_data()
  df = loader.interactions

  if df is None:
    print("Failed to load data.")
    return
  
  print("Create ID mappings...")
  user_id_mapping, artist_id_mapping, user_inv_map, artist_inv_map = create_id_mapping(df)

  df['userID'] = df['userID'].map(user_id_mapping)
  df['artistID'] = df['artistID'].map(artist_id_mapping)

  df['weight_log'] = np.log1p(df['weight'])

  num_users = len(user_id_mapping)
  num_artists = len(artist_id_mapping)

  print(f"Number of users: {num_users}")
  print(f"Number of artists: {num_artists}")

  train_df, valid_df = train_test_split(df, test_size=0.2, random_state=42)

  train_dataset = LastFmDataset(train_df['userID'].values, train_df['artistID'].values, train_df['weight_log'].values)
  valid_dataset = LastFmDataset(valid_df['userID'].values, valid_df['artistID'].values, valid_df['weight_log'].values)

  train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
  valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

  print("Initializing model...")
  model = MatrixFactorization(num_users, num_artists, embedding_dim=emb_size)

  if torch.cuda.is_available():
    device = torch.device("cuda")
  elif torch.backends.mps.is_available():
      device = torch.device("mps")
  else:
      device = torch.device("cpu")
      
  print(f"Using device: {device}")
  model.to(device)

  loss_fn = nn.MSELoss()
  optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
  early_stopper = EarlyStopping(patience=3, verbose=True, path=model_save_path)

  print("Training model...")
  for epoch in range(epochs):
    model.train()
    total_train_loss = 0.0

    for user, artist, weight in train_loader:
      user, artist, weight = user.to(device), artist.to(device), weight.to(device)

      optimizer.zero_grad()
      prediction = model(user, artist)
      loss = loss_fn(prediction, weight)
      loss.backward()
      optimizer.step()
      total_train_loss += loss.item()

    model.eval()
    total_val_loss = 0.0
    with torch.no_grad():
      for users, artists, weights in valid_loader:
        users, artists, weights = users.to(device), artists.to(device), weights.to(device)
        predictions = model(users, artists)
        val_loss = loss_fn(predictions, weights)
        total_val_loss += val_loss.item()

    avg_train_loss = total_train_loss / len(train_loader)
    avg_val_loss = total_val_loss / len(valid_loader)

    print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}")

    early_stopper(avg_val_loss, model)
    if early_stopper.early_stop:
      print("Early stopping triggered.")
      break

  print(f"\nTraining complete. Model saved to {model_save_path}")

  mapping_path = "model_store/mappings.pth"
  torch.save({
      'user_id_mapping': user_id_mapping,
      'artist_id_mapping': artist_id_mapping,
      'user_inv_map': user_inv_map,
      'artist_inv_map': artist_inv_map
  }, mapping_path)

  print(f"Mappings saved to {mapping_path}")

# Entry point for training the model.
if __name__ == "__main__":
  train_model()