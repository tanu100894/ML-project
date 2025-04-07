import pandas as pd
import numpy as np
import pickle
from itertools import chain
from scipy.signal import find_peaks
from tensorflow.keras.models import load_model
from signal_processing import extract_ffts

# load model
model = load_model('model.keras')
model.summary()

# Load the scaler
scaler_path = 'scaler.pkl'  # Adjust the path accordingly
with open(scaler_path, 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Read your dataset
df = pd.read_csv('./test/Ex6-81cm(pos2).csv', encoding='latin-1', header=None)

# Windowing parameters
step_size = 512    # Set the step size
window_size = 2048
full_length = 16384

# Define constants
speed_of_sound = 343.2  # in m/s
min_distance = 0.30  # in m
sample_time = 512 * (10 ** -9)  # in s

# Open a file for writing
with open('formatted_distances.txt', 'w') as file:
    # Loop through each measurement for prediction
    for _, measurement in df.iterrows():
        windows = []
        measurement = measurement.values.flatten()  # Convert the row to a 1D array
        # Apply windowing
        for i in range(0, int((full_length - window_size) / step_size) + 1, 1):
            window = np.hanning(window_size)
            signal = measurement[i * step_size:i * step_size + window_size]
            window = window * signal
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
        # The next window is chosen instead to make sure that the whole echo is captured
        index_of_window = flattened_prediction.index(1) + 1

        # Find peaks in the first window
        peaks, _ = find_peaks(data_window[index_of_window], height=0)  # Assuming the height threshold is 0

        # Find the index of the maximum peak within the window
        index_of_max_peak_in_window = peaks[np.argmax(data_window[index_of_window][peaks])]

        # Calculate the position of the maximum peak in the measurement
        position_of_max_peak_in_measurement = index_of_window * step_size + index_of_max_peak_in_window

        # Calculate the distance
        distance = ((position_of_max_peak_in_measurement * sample_time * speed_of_sound) / 2) + (min_distance - 0.04)

        # Format the output
        formatted_distance = "{:.2f}".format(distance)  # Limiting to 2 decimal places
        print(f"Predicted distance: {formatted_distance} m")

        # Write the result to the file
        file.write(formatted_distance + '\n')
