import pandas as pd
import numpy as np

# Import functions
from signal_processing import extract_ffts_labels

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
import pickle

# Windowing parameters
step_size = 512    # Set the step size
window_size = 2048
full_length = 16384

denominator = 10

labels = []
windows = []

# Load the saved model
saved_model_path = 'model.keras'
loaded_model = load_model(saved_model_path)

# Read your new dataset
new_df = pd.read_csv('./data/4-4.csv', encoding='latin-1', header=None)
new_data = new_df.iloc[:, :]
new_data_length = len(new_data)
new_df_label = pd.read_csv('./data/train_label_4-4.csv', encoding='utf-8', header=None)
new_label = new_df_label[:].to_numpy()

# Apply moving window to raw ADC data line by line
for i in range(0, new_data_length, 1):
    row = new_data.iloc[i, :].to_numpy()
    for j in range(0, int((len(row) - window_size) / step_size) + 1, 1):
        window = np.hanning(window_size)
        signal = row[j * step_size:j * step_size + window_size]
        window = window * signal
        windows.append(window)

# Store data after windowing
new_data_window = np.array(windows)

# Get features for the new dataset
X_new, Y_new = extract_ffts_labels(new_data_window, new_label)

# Normalize the features
# Load the original scaler if available
# Otherwise, fit a new scaler on the training data
scaler_path = 'scaler.pkl'  # Change this to the path where you saved the original scaler
try:
    with open(scaler_path, 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
except FileNotFoundError:
    print(f"Original scaler not found at {scaler_path}. Fitting a new one.")

X_new_normalized = scaler.transform(X_new)

# Split the new dataset
X_new_train, X_new_test, Y_new_train, Y_new_test = train_test_split(X_new_normalized, Y_new, test_size=0.2, random_state=42)

# Continue training the loaded model with the new dataset
early_stopping = EarlyStopping(monitor='val_loss', patience=100, restore_best_weights=True)
loaded_model.fit(X_new_train, Y_new_train, epochs=1000, batch_size=128, validation_split=0.2, callbacks=[early_stopping])

# Evaluate the model on the new test set
_, new_test_accuracy = loaded_model.evaluate(X_new_test, Y_new_test)
print('New Test Accuracy: %.2f' % (new_test_accuracy * 100))

# Save the updated model to disk
loaded_model.save('model.keras')
print("Saved updated model to disk")

# Save the scaler to disk
with open(scaler_path, 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)
print("Saved updated scaler to disk")