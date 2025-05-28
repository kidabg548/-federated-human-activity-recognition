import streamlit as st
import os
from datetime import datetime
from PIL import Image

def main():
    st.title("Federated HAR Dashboard")
    st.write("Real-time monitoring of federated learning training process")
    
    # Sidebar
    st.sidebar.title("Configuration")
    n_clients = st.sidebar.slider("Number of Clients", 1, 10, 5)
    n_rounds = st.sidebar.slider("Training Rounds", 10, 100, 50)
    use_dp = st.sidebar.checkbox("Use Differential Privacy")
    dp_noise = st.sidebar.slider("DP Noise", 0.0, 0.1, 0.01) if use_dp else 0.01
    
    # Main content
    st.header("Training Progress")
    
    # Display plots
    if os.path.exists('plots'):
        # Accuracy plot
        if os.path.exists('plots/accuracy.png'):
            st.write("### Accuracy Over Time")
            st.image('plots/accuracy.png')
        
        # Loss plot
        if os.path.exists('plots/loss.png'):
            st.write("### Loss Over Time")
            st.image('plots/loss.png')
        
        # Participation plot
        if os.path.exists('plots/participation.png'):
            st.write("### Client Participation")
            st.image('plots/participation.png')
    else:
        st.warning("No plots found. Please run the training first.")
    
    # Model information
    st.header("Model Information")
    if os.path.exists('final_model.h5'):
        st.success("Model saved successfully!")
        model_size = os.path.getsize('final_model.h5') / 1024  # KB
        st.write(f"Model size: {model_size:.2f} KB")
        st.write(f"Last updated: {datetime.fromtimestamp(os.path.getmtime('final_model.h5'))}")
    else:
        st.warning("No model file found. Please run the training first.")

if __name__ == "__main__":
    main() 