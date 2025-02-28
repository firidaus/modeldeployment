from flask import Flask, request, jsonify, render_template
import torch
import joblib
import numpy as np
from scipy.signal import find_peaks
import pandas as pd
import torch.nn as nn
import os

app = Flask(__name__)

# Load the trained models
kmeans = joblib.load('kmeans_model.joblib')

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the Encoder class
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv1d(1, 64, 5)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(64, 128, 3)
        self.relu2 = nn.ReLU()
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(128, 128)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.maxpool1(x)
        x = self.relu2(self.conv2(x))
        x = self.global_avg_pool(x).squeeze(2)
        x = self.fc1(x)
        return x

# Define the ClassificationHead class
class ClassificationHead(nn.Module):
    def __init__(self, num_classes):
        super(ClassificationHead, self).__init__()
        self.encoder = Encoder()
        self.fc2 = nn.Linear(128, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.encoder(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x

# Create an instance of the model
classification_head = ClassificationHead(len(np.unique(kmeans.labels_))).to(device)

# Load the state dictionary into the model
classification_head.load_state_dict(torch.load('classification_head.pth'))
classification_head.eval()

# Define a function to preprocess the input data
def preprocess_data(data):
    df = pd.DataFrame(data)
    peaks, _ = find_peaks(df["MLII"], distance=150)

    heartbeats = []
    for peak in peaks:
        start = max(0, peak - 150)
        end = min(len(df), peak + 150)
        heartbeat = df["MLII"][start:end].values
        if len(heartbeat) == 300:
            heartbeats.append(heartbeat)

    heartbeats = np.array(heartbeats).reshape(-1, 300, 1)
    return torch.tensor(heartbeats, dtype=torch.float32).to(device)

# Map prediction values to labels
def map_predictions(predictions):
    label_map = {0: "Normal", 1: "Abnormal"}
    return [label_map[p] for p in predictions]

# Define the prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    heartbeats = preprocess_data(data)

    with torch.no_grad():
        outputs = classification_head(heartbeats.transpose(1, 2))
        _, predicted = torch.max(outputs, 1)
        predicted_labels = map_predictions(predicted.cpu().numpy().tolist())

    return jsonify({'predictions': predicted_labels})

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        print("Error: No file part in request")
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        print("Error: No file selected")
        return jsonify({'error': 'No selected file'}), 400

    if file and file.filename.endswith('.csv'):
        try:
            data = pd.read_csv(file)
            print("File uploaded successfully. First rows:", data.head())  # Debugging

            if "MLII" not in data.columns:
                print("Error: Missing 'MLII' column")
                return jsonify({'error': "Invalid file format. 'MLII' column is required."}), 400

            heartbeats = preprocess_data(data)

            with torch.no_grad():
                outputs = classification_head(heartbeats.transpose(1, 2))
                _, predicted = torch.max(outputs, 1)
                predicted_labels = map_predictions(predicted.cpu().numpy().tolist())

            return jsonify({'predictions': predicted_labels})
        
        except Exception as e:
            print(f"Error processing file: {str(e)}")  # Log error
            return jsonify({'error': 'Error processing file'}), 500

    print("Error: Invalid file format")
    return jsonify({'error': 'Invalid file format'}), 400

@app.route('/')
def index():
    return render_template('index.html')



if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))  # Default to 5000 if PORT is not set
    app.run(host='0.0.0.0', port=port, debug=True)

