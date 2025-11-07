import torch
import torch.nn as nn

# Defines the matrix factorization model.
class MatrixFactorization(nn.Module):

  def __init__(self, num_users, num_artists, embedding_dim=500):
    super(MatrixFactorization, self).__init__()

    # Embedding layers for users and artists.
    self.user_embedding = nn.Embedding(num_users, embedding_dim)
    self.artist_embedding = nn.Embedding(num_artists, embedding_dim)

    self.dropout = nn.Dropout(p=0.5)
    
    # Initialize embedding weights.
    self.user_embedding.weight.data.uniform_(0, 0.05)
    self.artist_embedding.weight.data.uniform_(0, 0.05)

  # Defines the forward pass of the model.
  def forward(self, user, artist):

    # Get the embedding vectors for a batch of users and artists.
    user_vector = self.user_embedding(user)
    artist_vector = self.artist_embedding(artist)

    # Calculate the dot product to get the interaction score.
    score = (user_vector * artist_vector).sum(dim=1)

    return score
  
# A simple test to run the model independently.
if __name__ == "__main__":

  print("Testing model.py")

  test_num_users = 100
  test_num_artists = 50
  test_emb_size = 10

  model = MatrixFactorization(test_num_users, test_num_artists, test_emb_size)
  print("Model created.")

  test_user_ids = torch.LongTensor([1, 5, 20, 99])
  test_artist_ids = torch.LongTensor([4, 10, 30, 45])

  predictions = model(test_user_ids, test_artist_ids)
  
  print(f"\nInput user tensor shape: {test_user_ids.shape}")
  print(f"Input artist tensor shape: {test_artist_ids.shape}")
  print(f"Output predictions shape: {predictions.shape}")

  assert predictions.shape == (4,)

  print("\nModel test passed!")
  print("Example predictions (randomly initialized):")
  print(predictions)