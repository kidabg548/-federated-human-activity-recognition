# Federated Human Activity Recognition (HAR)

A privacy-preserving machine learning project that demonstrates federated learning for human activity recognition using smartphone sensor data.

## Project Overview

This project implements a federated learning system for recognizing human activities (walking, sitting, standing, etc.) using smartphone sensor data. The system preserves user privacy by keeping data on individual devices and only sharing model updates.

### Key Features

- 🔒 Privacy-preserving training
- 📱 Simulated smartphone clients
- 🎯 Activity recognition using sensor data
- 🛡️ Differential privacy support
- 📊 Real-time performance monitoring
- 🎨 Interactive visualizations

## Installation

1. Clone the repository:
```bash
git clone https://github.com/kidabg548/-federated-human-activity-recognition
cd federated-har
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the federated learning simulation:

```bash
python main.py --n_clients 5 --n_rounds 50 --alpha 0.5
```

Optional arguments:
- `--n_clients`: Number of clients (default: 5)
- `--n_rounds`: Number of training rounds (default: 50)
- `--use_dp`: Enable differential privacy
- `--dp_noise`: Differential privacy noise multiplier (default: 0.01)
- `--alpha`: Non-IID distribution parameter (default: 0.5)

## Visualizations

The project generates several plots in the `plots/` directory:
- Training accuracy over time
- Loss progression
- Client participation rates
- Privacy-performance tradeoffs

## Results

After training, you can find:
1. Model performance metrics in the console output
2. Training visualizations in the `plots/` directory
3. The trained model in `models/federated_model.keras`
