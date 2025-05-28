import tensorflow as tf
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd

def load_and_preprocess_test_data(file_path):
    """Load and preprocess new test data."""
    # Load the data
    data = pd.read_csv(file_path, sep='\s+', header=None)
    
    # Normalize the data
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(data)
    
    return normalized_data

def predict_activity(model, data):
    """Predict activity from sensor data."""
    # Make prediction
    predictions = model.predict(data)
    
    # Get the predicted class
    predicted_class = np.argmax(predictions, axis=1)
    
    # Map class numbers to activity names
    activity_map = {
        0: "WALKING",
        1: "WALKING_UPSTAIRS",
        2: "WALKING_DOWNSTAIRS",
        3: "SITTING",
        4: "STANDING",
        5: "LAYING"
    }
    
    return [activity_map[pred] for pred in predicted_class]

def main():
    # Load the trained model
    try:
        model = tf.keras.models.load_model('final_model.h5')
        print("Model loaded successfully!")
    except:
        print("Error: Could not load model. Make sure 'final_model.h5' exists.")
        return

    # Use the test data path directly
    file_path = 'data/UCI HAR Dataset/test/X_test.txt'
    print(f"Using test data from: {file_path}")
    
    try:
        # Load and preprocess the data
        test_data = load_and_preprocess_test_data(file_path)
        print(f"\nLoaded {len(test_data)} samples for testing")
        
        # Make predictions
        predictions = predict_activity(model, test_data)
        confidence_scores = model.predict(test_data)
        
        # Print detailed results for last 10 samples
        print("\n=== Last 10 Predictions ===")
        print("=" * 50)
        
        for i in range(-10, 0):  # Show last 10 samples
            print(f"\nSample {len(test_data) + i + 1}:")
            print("-" * 30)
            
            # Get the predicted activity and its confidence
            pred_idx = np.argmax(confidence_scores[i])
            pred_activity = predictions[i]
            pred_confidence = confidence_scores[i][pred_idx]
            
            print(f"Predicted Activity: {pred_activity}")
            print(f"Confidence: {pred_confidence:.2%}")
            
            # Show confidence scores for all activities
            print("\nConfidence Scores:")
            for j, score in enumerate(confidence_scores[i]):
                activity = ["WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS", 
                          "SITTING", "STANDING", "LAYING"][j]
                print(f"{activity:20}: {score:.2%}")
            
            print("-" * 30)
        
        # Print overall statistics
        print("\n=== Overall Statistics ===")
        print("=" * 50)
        activity_counts = {}
        for pred in predictions:
            activity_counts[pred] = activity_counts.get(pred, 0) + 1
        
        print("\nActivity Distribution in Last 10 Samples:")
        for activity, count in activity_counts.items():
            print(f"{activity:20}: {count} samples")
            
    except Exception as e:
        print(f"Error processing data: {str(e)}")

if __name__ == "__main__":
    main() 