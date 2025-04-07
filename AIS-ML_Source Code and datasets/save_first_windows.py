import pandas as pd
import numpy as np
import pickle
from itertools import chain
from tensorflow.keras.models import load_model
# Import functions
from signal_processing import extract_ffts

# load model
model = load_model('model.keras')
# summarize model.
model.summary()

# Load the scaler
scaler_path = 'scaler.pkl'  # Adjust the path accordingly
with open(scaler_path, 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Read your dataset
df = pd.read_csv('./confusion_matrix/0-1.csv', encoding='latin-1', header=None)

# Windowing parameters
step_size = 512    # Set the step size
window_size = 2048
full_length = 16384

# Define constants
speed_of_sound = 343.2  # in m/s
min_distance = 0.30  # in m
sample_time = 512 * (10 ** -9)  # in s

# Create a list to store the windows
first_windows_list = []

# Loop through each measurement for prediction
for _, measurement in df.iterrows():
    windows = []
    measurement = measurement.values.flatten()  # Convert the row to a 1D array
    # Apply windowing
    for i in range(0, int((full_length - window_size) / step_size) + 1, 1):
        window = measurement[i * step_size:i * step_size + window_size]
        windows.append(window)

    data_window = np.array(windows)

    # Get features
    X = extract_ffts(data_window)
    X_normalized = scaler.transform(X)

    # Prediction
    prediction = model.predict(X_normalized)
    threshold = 0.90
    binary_predictions = (prediction >= threshold).astype(int)

    # Flatten the nested prediction
    flattened_prediction = list(chain.from_iterable(binary_predictions))

    # Get the index of the window containing the first echo
    index_of_window = flattened_prediction.index(1) + 1

    first_window = data_window[index_of_window]
    first_windows_list.append(first_window)

first_windows_array = np.array(first_windows_list)
first_windows_frame = pd.DataFrame(first_windows_array)
first_windows_frame.to_csv('first_windows.csv', index=False, header=False)


