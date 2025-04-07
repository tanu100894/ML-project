# load and evaluate a saved model
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from itertools import chain
from scipy.signal import find_peaks


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
df = pd.read_csv('./data/4-2.csv', encoding='latin-1', header=None)

# Windowing parameters
step_size = 512    # Set the step size
window_size = 2048
full_length = 16384

windows = []

denominator = 10

# Take 1 measurement for prediction
measurement = df.sample(n=1).to_numpy().flatten()

# Apply windowing
for i in range(0, int((full_length - window_size) / step_size) + 1, 1):
        window = np.hanning(window_size)
        signal = measurement[i * step_size:i * step_size + window_size]
        window = window * signal
        windows.append(window)

data_window = np.array(windows)
data_frame = pd.DataFrame(data_window)
data_frame.to_csv('test.csv', index=False, header=False)

# Get features
X = extract_ffts(data_window)
X_normalized = scaler.transform(X)

# Prediction
prediction = model.predict(X_normalized)
threshold = 0.90
binary_predictions = (prediction >= threshold).astype(int)
# print(binary_predictions)

# Flatten the nested prediction
flattened_prediction = list(chain.from_iterable(binary_predictions))

# Get the index of the window containing the first echo
index_of_window = flattened_prediction.index(1) + 1

print("Position of the window with the first echo is:", index_of_window)

# Find peaks in the first window
peaks, _ = find_peaks(data_window[index_of_window], height=0)  # Assuming the height threshold is 0

# Find the index of the maximum peak within the window
index_of_max_peak_in_window = peaks[np.argmax(data_window[index_of_window][peaks])]

# Calculate the position of the maximum peak in the measurement
position_of_max_peak_in_measurement = index_of_window * step_size + index_of_max_peak_in_window

print("Position of the maximum peak in the window:", index_of_max_peak_in_window)
print("Position of the maximum peak in the measurement:", position_of_max_peak_in_measurement)
