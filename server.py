import tensorflow as tf
import numpy as np
from typing import List, Tuple
import logging

class FederatedServer:
    def __init__(self, clients: List, use_dp: bool = False, dp_noise_multiplier: float = 0.01):
        self.clients = clients
        self.use_dp = use_dp
        self.dp_noise_multiplier = dp_noise_multiplier
        self.current_weights = None
        self.setup_logging()
        
    def setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    def initialize_model(self):
        """Initialize the model weights."""
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_shape=(561,)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(6, activation='softmax')
        ])
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        self.current_weights = model.get_weights()
        
    def federated_averaging(self, client_weights: List) -> List:
        """Perform federated averaging on client weights."""
        if not client_weights:
            return self.current_weights
            
        # Filter out None values (failed clients)
        valid_weights = [w for w in client_weights if w is not None]
        if not valid_weights:
            return self.current_weights
            
        # Average the weights
        averaged_weights = []
        for weights_list_tuple in zip(*valid_weights):
            averaged_weights.append(
                np.array([np.array(w).mean(axis=0) for w in zip(*weights_list_tuple)])
            )
            
        # Apply differential privacy if enabled
        if self.use_dp:
            averaged_weights = self._apply_differential_privacy(averaged_weights)
            
        return averaged_weights
        
    def _apply_differential_privacy(self, weights: List) -> List:
        """Apply differential privacy by adding Gaussian noise."""
        noisy_weights = []
        for w in weights:
            noise = np.random.normal(0, self.dp_noise_multiplier, size=w.shape)
            noisy_weights.append(w + noise)
        return noisy_weights
        
    def train_round(self, round_num: int) -> Tuple[float, float]:
        """Execute one round of federated training."""
        # Get updates from clients
        client_weights = []
        participating_clients = 0
        
        for client in self.clients:
            weights = client.train(self.current_weights, round_num)
            if weights is not None:
                client_weights.append(weights)
                participating_clients += 1
                
        # Log participation
        self.logger.info(f"Round {round_num}: {participating_clients}/{len(self.clients)} clients participated")
        
        # Update global model
        self.current_weights = self.federated_averaging(client_weights)
        
        # Evaluate on test set
        total_loss = 0
        total_accuracy = 0
        for client in self.clients:
            loss, accuracy = client.evaluate(self.current_weights)
            total_loss += loss
            total_accuracy += accuracy
            
        avg_loss = total_loss / len(self.clients)
        avg_accuracy = total_accuracy / len(self.clients)
        
        self.logger.info(f"Round {round_num} - Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.4f}")
        
        return avg_loss, avg_accuracy
