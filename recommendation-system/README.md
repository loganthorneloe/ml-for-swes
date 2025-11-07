# Collaborative Filtering Recommendation System

This project aims to help you build your first recommendation system from scratch. It includes both the training code and a UI to visualize the systems in real-time as as it retrains.

This project has **two** accompanying articles. One explains [collaborative filtering in production](https://mlforswes.com/p/spotify-case-study) using Spotify as an example. The other walks you through building the code in this folder (WIP - coming soon!).

![Alt text for the GIF](/assets/recommendations.gif)

## The Project

This project is a collaborative filtering recommendation system for music artists. It is built using PyTorch and trained on the Last.fm 2k dataset.

## File Structure

*   `app.py`: A Flask application to serve the recommendations.
*   `model.py`: Defines the PyTorch model for collaborative filtering.
*   `train.py`: The training script for the model.
*   `last_fm_loader.py`: A data loader for the Last.fm 2k dataset.
*   `requirements.txt`: The Python dependencies for this project.
*   `model_store/`: This directory contains the trained model (`model.pt`) and the user/item mappings (`mappings.pth`).

## Setup

1.  **Create a virtual environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

2.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Training

To train the model, run the `train.py` script:

```bash
python train.py
```

This will create the `model.pt` and `mappings.pth` files in the `model_store` directory.

## Running the Application

To run the Flask application, use the following command:

```bash
streamlit run app.py
```

The application will be available at `http://127.0.0.1:8501`.

You can then make a GET request to `/recommendations?user_id=<user_id>` to get a list of recommended artists for a given user.

## Need Fixes?

Reach out to me on [X](x.com/loganthorneloe) or [Substack](substack.com/@loganthorneloe) **or** submit a PR and I'll take a look.