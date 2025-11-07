import streamlit as st
import torch
import pandas as pd
import os
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from model import MatrixFactorization
from last_fm_loader import LastFmLoader
from train import LastFmDataset

MODEL_PATH = os.path.join("model_store", "model.pt")
MAPPINGS_PATH = os.path.join("model_store", "mappings.pth")
SIMULATIONS = 5000

# Caches and loads the trained model, mappings, and artist data.
@st.cache_resource
def load_assets():

  try:

    mappings = torch.load(MAPPINGS_PATH, weights_only=False)
    user_map = mappings['user_id_mapping']
    artist_map = mappings['artist_id_mapping']
    
    num_users = len(user_map)
    num_artists = len(artist_map)

    model = MatrixFactorization(num_users, num_artists, embedding_dim=50)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    loader = LastFmLoader()
    loader.load_data()
    artists_df = loader.artists.set_index('id')

    return model, mappings, artists_df
  
  except FileNotFoundError as e:
    print(f"Error loading assets: {e}")
    st.stop()
  except Exception as e:
    print(f"An error occurred during asset loading: {e}")
    st.stop()

# Generates artist recommendations for a given user.
@st.cache_data(show_spinner="Generating recommendations...")
def get_recommendations(selected_user_id, _model, _mappings, _artists_df, num_recs=10):

  user_idx = _mappings['user_id_mapping'][selected_user_id]
  if user_idx is None:
    st.error(f"User ID {selected_user_id} not found in the mapping.")
    return pd.DataFrame(columns=['Artist', 'Predicted Score'])
  
  user_tensor = torch.LongTensor([user_idx])
  user_vector = _model.user_embedding(user_tensor)

  all_artist_vectors = _model.artist_embedding.weight
  
  with torch.no_grad():
    scores = torch.matmul(user_vector, all_artist_vectors.T). squeeze()

  top_indices = torch.argsort(scores, descending=True)[:num_recs]

  rec_data = []
  for idx in top_indices:
    artist_model_idx = idx.item()
    original_artist_id = _mappings['artist_inv_map'].get(artist_model_idx)
    if original_artist_id:
      artist_name = _artists_df.loc[original_artist_id, 'name']
      rec_data.append((artist_name, scores[idx].item()))
  
  return pd.DataFrame(rec_data, columns=['Artist', 'Score'])

# Simulates new user listening data.
def simulate_new_listen(_mappings, num_simulations=100):
  st.write(f"Simulating {num_simulations} new listens...")
  all_user_indices = list(_mappings['user_inv_map'].keys())
  all_artist_indices = list(_mappings['artist_inv_map'].keys())

  sim_users = np.random.choice(all_user_indices, num_simulations)
  sim_artists = np.random.choice(all_artist_indices, num_simulations)
                                 
  sim_weights = np.random.randint(50, 500, num_simulations)

  sim_df = pd.DataFrame({
      'user_idx': sim_users,
      'artist_idx': sim_artists,
      'weight': sim_weights
  })

  return sim_df

# Retrains the model on new (simulated) data.
def retrain_model(new_data):

  st.sidebar.write("Retraining model...")

  if 'retrained_model' in st.session_state:
    model_to_retrain = st.session_state.retrained_model
    st.sidebar.write("Starting from *previously* retrained model.")
  else:
    model, _, _ = load_assets()
    model_to_retrain = model
    st.sidebar.write("Starting from *original* loaded model.")

  new_data['weight_log'] = np.log1p(new_data['weight'])
  new_dataset = LastFmDataset(new_data['user_idx'].values, new_data['artist_idx'].values, new_data['weight_log'].values)
  new_loader = DataLoader(new_dataset, batch_size=32, shuffle=True)

  optimizer = optim.Adam(model_to_retrain.parameters(), lr=0.001)
  loss_fn = nn.MSELoss()
  
  model_to_retrain.train()

  for users, artists, weights, in new_loader:
    optimizer.zero_grad()
    predictions = model_to_retrain(users, artists)
    loss = loss_fn(predictions, weights)
    loss.backward()
    optimizer.step()

  model_to_retrain.eval()
  st.session_state.retrained_model = model_to_retrain
  st.sidebar.success("Retraining complete!")

#
# Main Streamlit UI
#

st.set_page_config(page_title="Music Recommender", layout="wide")
st.title("Interactive Music Recommender")

model, mappings, artists_df = load_assets()

if 'retrained_model' in st.session_state:
  model_to_use = st.session_state.retrained_model
else:
  model_to_use = model

original_user_ids = list(mappings['user_inv_map'].values())
st.subheader("Select a user to see their recommendations:")
selected_user_id = st.selectbox("Select a user", original_user_ids)


if selected_user_id:
  st.write(f"Top Recommendations for user: **{selected_user_id}**")
  recs_df = get_recommendations(selected_user_id, model_to_use, mappings, artists_df)
  st.table(recs_df.set_index('Artist'))

st.sidebar.title("Retraining Simulation")
st.sidebar.write("Simulate new user activity and retrain.")

if st.sidebar.button(f"Simulate {SIMULATIONS} listens and retrain"):

  new_data = simulate_new_listen(mappings, num_simulations=SIMULATIONS)
  retrain_model(new_data)

  get_recommendations.clear()
  st.rerun()