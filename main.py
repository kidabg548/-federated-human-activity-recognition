import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from utils import prepare_federated_data
from client import FederatedClient
from server import FederatedServer
import argparse
import os

def plot_metrics(history, save_dir='plots'):
    """Plot training metrics."""
    os.makedirs(save_dir, exist_ok=True)
    
    # Plot accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(history['accuracy'], label='Training Accuracy')
    plt.title('Federated Learning - Model Accuracy')
    plt.xlabel('Round')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'accuracy.png'))
    plt.close()
    
    # Plot loss
    plt.figure(figsize=(10, 5))
    plt.plot(history['loss'], label='Training Loss')
    plt.title('Federated Learning - Model Loss')
    plt.xlabel('Round')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'loss.png'))
    plt.close()
    
    # Plot participation rate
    plt.figure(figsize=(10, 5))
    plt.plot(history['participation'], label='Client Participation Rate')
    plt.title('Federated Learning - Client Participation')
    plt.xlabel('Round')
    plt.ylabel('Participation Rate')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'participation.png'))
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Federated Learning for HAR')
    parser.add_argument('--n_clients', type=int, default=5, help='Number of clients')
    parser.add_argument('--n_rounds', type=int, default=50, help='Number of training rounds')
    parser.add_argument('--use_dp', action='store_true', help='Use differential privacy')
    parser.add_argument('--dp_noise', type=float, default=0.01, help='Differential privacy noise multiplier')
    parser.add_argument('--alpha', type=float, default=0.5, help='Non-IID distribution parameter')
    args = parser.parse_args()
    
    # Prepare data
    print("Preparing federated data...")
    client_data, X_test, y_test = prepare_federated_data(
        n_clients=args.n_clients,
        alpha=args.alpha
    )
    
    # Create clients
    clients = []
    for i, data in enumerate(client_data):
        is_adversary = (i == args.n_clients - 1)  # Last client is adversary
        is_straggler = (i == args.n_clients - 2)  # Second to last client is straggler
        clients.append(FederatedClient(
            client_id=f"client_{i}",
            data=data,
            is_adversary=is_adversary,
            is_straggler=is_straggler
        ))
    
    # Create server
    server = FederatedServer(
        clients=clients,
        use_dp=args.use_dp,
        dp_noise_multiplier=args.dp_noise
    )
    
    # Initialize model
    server.initialize_model()
    
    # Training history
    history = {
        'loss': [],
        'accuracy': [],
        'participation': []
    }
    
    # Training loop
    print("Starting federated training...")
    for round_num in range(args.n_rounds):
        loss, accuracy = server.train_round(round_num)
        history['loss'].append(loss)
        history['accuracy'].append(accuracy)
        
        # Calculate participation rate
        participating_clients = sum(1 for client in clients if client.train(server.current_weights, round_num) is not None)
        participation_rate = participating_clients / len(clients)
        history['participation'].append(participation_rate)
        
        print(f"Round {round_num + 1}/{args.n_rounds} - "
              f"Loss: {loss:.4f}, Accuracy: {accuracy:.4f}, "
              f"Participation: {participation_rate:.2%}")
    
    # Plot metrics
    plot_metrics(history)
    
    # Save the final model
    server.model.save('models/federated_model.keras')
    
    print("Training completed. Results saved in 'plots' directory.")

if __name__ == '__main__':
    main()
