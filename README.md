# essential-ml-models

This repository collects a short list of machine learning notebooks that every software engineer should understand. Each notebook keeps the math light, focuses on practical intuition, and uses small datasets that run quickly on a laptop.

## Why These Models
- Cover regression, classification, unsupervised learning, generative modeling, and recommendations
- Showcase industry-relevant patterns while staying small enough for a single notebook
- Encourage hands-on experimentation with clear setup and lightweight dependencies

## Model Roadmap
1. **Demand Forecasting – Linear Regression** (`sklearn.datasets.fetch_california_housing`): mirror how Uber and DoorDash forecast rider demand or delivery volume while keeping coefficients interpretable on a lightweight proxy dataset.
2. **Risk Scoring – Gradient-Boosted Trees** (`sklearn.datasets.fetch_openml("credit-g")`): echo the credit-approval pipelines at Capital One or American Express by training a boosted tree model on tabular applicant features.
3. **Anomaly Detection – Isolation Forest** (`sklearn.datasets.fetch_openml("creditcard", version=1)`): follow Stripe’s fraud monitoring playbook by flagging suspicious payment patterns and examining the precision–recall trade-offs.
4. **Image Classification – Convolutional Neural Network** (`torchvision.datasets.FashionMNIST`): shrink Amazon’s product-photo tagging workflow into a compact CNN trained on fashion imagery so engineers can see the end-to-end loop.
5. **Natural Language Processing – Transformer Text Classifier** (`datasets.load_dataset("ag_news")`): mirror Slack’s support-ticket triage by fine-tuning a lightweight transformer that assigns each request to the right resolver queue.
6. **Recommendation Systems – Collaborative Filtering** (`lastfm-dataset-360K`): keep Spotify front and centre as we rebuild the preference–confidence matrix factorization from the case study and test it on hold-out listeners.

## Local Setup
```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Launch Jupyter Lab, VS Code, or another notebook runner from the project root to keep relative paths stable. The `datasets/` folder is generated automatically when a notebook downloads external data.

## Running in Google Colab
1. Upload the notebook you want to explore or open it directly from GitHub via **File → Open notebook → GitHub** in Colab.
2. Execute the optional `!pip install ...` cell to pull dependencies for that session.
3. Run the subsequent cells—the Last.fm notebook will download to Colab’s ephemeral storage under `datasets/` just as it does locally.

## Repository Layout
- Top-level `.ipynb` notebooks: one per model (to be added incrementally)
- `requirements.txt`: Shared dependency list for local development
- `datasets/` (generated): Cached external datasets, git-ignored

## Getting Started
Have ideas for other essential models? Open an issue or PR!
