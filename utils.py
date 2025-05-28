import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
from sklearn.model_selection import train_test_split
import requests
import zipfile
from io import BytesIO

def load_raw_ucihar_data(base_dir="data/UCI HAR Dataset"):
    # Load train and test data
    X_train = np.loadtxt(os.path.join(base_dir, "train", "X_train.txt"))
    y_train = np.loadtxt(os.path.join(base_dir, "train", "y_train.txt")).astype(int)
    
    X_test = np.loadtxt(os.path.join(base_dir, "test", "X_test.txt"))
    y_test = np.loadtxt(os.path.join(base_dir, "test", "y_test.txt")).astype(int)

    # Merge train and test for simplicity
    X = np.vstack((X_train, X_test))
    y = np.concatenate((y_train, y_test))

    return X, y

def normalize_data(X):
    scaler = StandardScaler()
    return scaler.fit_transform(X)

def partition_non_iid(X, y, num_clients=5):
    """Partition data non-IID by activity label."""
    client_data = defaultdict(lambda: {"x": [], "y": []})

    # Create label buckets
    label_buckets = defaultdict(list)
    for xi, yi in zip(X, y):
        label_buckets[yi].append(xi)

    # Assign each client 2 random labels
    label_keys = list(label_buckets.keys())
    for client_id in range(num_clients):
        chosen_labels = np.random.choice(label_keys, size=2, replace=False)
        for label in chosen_labels:
            samples = label_buckets[label][:50]
            label_buckets[label] = label_buckets[label][50:]  # remove assigned
            client_data[client_id]["x"].extend(samples)
            client_data[client_id]["y"].extend([label] * len(samples))

    # Convert to arrays
    for cid in client_data:
        client_data[cid]["x"] = np.array(client_data[cid]["x"])
        client_data[cid]["y"] = np.array(client_data[cid]["y"])

    return client_data

def save_client_data(client_data, path="data/clients.npz"):
    np.savez_compressed(path, **{
        f"client_{cid}_x": client_data[cid]["x"]
        for cid in client_data
    }, **{
        f"client_{cid}_y": client_data[cid]["y"]
        for cid in client_data
    })

def load_client_data(path="data/clients.npz"):
    raw = np.load(path)
    client_data = {}
    for key in raw:
        cid = int(key.split("_")[1])
        if cid not in client_data:
            client_data[cid] = {}
        if key.endswith("_x"):
            client_data[cid]["x"] = raw[key]
        elif key.endswith("_y"):
            client_data[cid]["y"] = raw[key]
    return client_data

def download_uci_har_dataset():
    """Download and extract the UCI HAR dataset."""
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip"
    response = requests.get(url)
    with zipfile.ZipFile(BytesIO(response.content)) as zip_ref:
        zip_ref.extractall("data")

def load_and_preprocess_data():
    """Load and preprocess the UCI HAR dataset."""
    # Load training data
    X_train = pd.read_csv('data/UCI HAR Dataset/train/X_train.txt', sep='\s+', header=None)
    y_train = pd.read_csv('data/UCI HAR Dataset/train/y_train.txt', header=None)
    
    # Load test data
    X_test = pd.read_csv('data/UCI HAR Dataset/test/X_test.txt', sep='\s+', header=None)
    y_test = pd.read_csv('data/UCI HAR Dataset/test/y_test.txt', header=None)
    
    # Normalize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    y_train = y_train.values.ravel() - 1
    y_test = y_test.values.ravel() - 1
    
    return X_train, y_train, X_test, y_test

def create_non_iid_partitions(X, y, n_clients=5, alpha=0.5):
    """
    Create non-IID partitions of the data using Dirichlet distribution.
    
    Args:
        X: Features
        y: Labels
        n_clients: Number of clients
        alpha: Concentration parameter for Dirichlet distribution (lower = more non-IID)
    
    Returns:
        List of (X_client, y_client) tuples for each client
    """
    n_classes = len(np.unique(y))
    client_data = []
    
    # Create label distribution for each client using Dirichlet distribution
    label_distribution = np.random.dirichlet([alpha] * n_classes, n_clients)
    
    # Assign data to clients based on label distribution
    for client_idx in range(n_clients):
        client_indices = []
        for class_idx in range(n_classes):
            # Get indices for current class
            class_indices = np.where(y == class_idx + 1)[0]
            # Sample based on distribution
            n_samples = int(len(class_indices) * label_distribution[client_idx, class_idx])
            selected_indices = np.random.choice(class_indices, n_samples, replace=False)
            client_indices.extend(selected_indices)
        
        # Shuffle indices
        np.random.shuffle(client_indices)
        client_data.append((X[client_indices], y[client_indices]))
    
    return client_data

def prepare_federated_data(n_clients=5, alpha=0.5):
    """
    Prepare federated data by downloading, preprocessing, and partitioning the UCI HAR dataset.
    
    Args:
        n_clients: Number of clients to create
        alpha: Concentration parameter for non-IID distribution
    
    Returns:
        client_data: List of (X_client, y_client) tuples for each client
        X_test: Test features
        y_test: Test labels
    """
    # Download dataset if not exists
    if not os.path.exists('data/UCI HAR Dataset'):
        download_uci_har_dataset()
    
    # Load and preprocess data
    X_train, y_train, X_test, y_test = load_and_preprocess_data()
    
    # Create non-IID partitions
    client_data = create_non_iid_partitions(X_train, y_train, n_clients, alpha)
    
    return client_data, X_test, y_test
