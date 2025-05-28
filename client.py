import tensorflow as tf
import numpy as np
import time
import random

def create_keras_model():
    """Create a simple MLP model for HAR classification."""
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(561,)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(6, activation='softmax')  # 6 activity classes
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

class FederatedClient:
    def __init__(self, client_id, data, is_adversary=False, is_straggler=False):
        self.client_id = client_id
        self.data = data
        self.is_adversary = is_adversary
        self.is_straggler = is_straggler
        
    def train(self, model_weights, round_num):
        """Train the model on client data."""
        # Simulate straggler behavior
        if self.is_straggler:
            time.sleep(random.uniform(2, 4))
            
        # Simulate client failure
        if random.random() < 0.2:  # 20% chance to fail
            return None
            
        # Create model and set weights
        model = create_keras_model()
        model.set_weights(model_weights)
        
        # Prepare data
        X, y = self.data
        dataset = tf.data.Dataset.from_tensor_slices((X, y)).batch(32)
        
        # Train model
        model.fit(dataset, epochs=1, verbose=0)
        
        # If adversary, return random weights
        if self.is_adversary:
            return [np.random.randn(*w.shape) for w in model.get_weights()]
            
        return model.get_weights()
        
    def evaluate(self, model_weights):
        """Evaluate the model on client data."""
        model = create_keras_model()
        model.set_weights(model_weights)
        
        X, y = self.data
        return model.evaluate(X, y, verbose=0)
