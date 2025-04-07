import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from signal_processing import extract_ffts
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import classification_report

# load model
model = load_model('model.keras')
model.summary()

# Load the scaler
scaler_path = 'scaler.pkl'  # Adjust the path accordingly
with open(scaler_path, 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Windowing parameters
step_size = 512    # Set the step size
window_size = 2048
full_length = 16384

# Read your dataset
df = pd.read_csv('./confusion_matrix/6-2.csv', encoding='latin-1', header=None)
new_df_label = pd.read_csv('./confusion_matrix/label_6-2.csv', encoding='utf-8', header=None)
truth = new_df_label[:].to_numpy()
truth = truth.flatten()

# Initialize list of predicted position
predicted_list = []

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

    predicted_list.append(binary_predictions)

# Convert the list to array of predicted positions
predicted = np.array(predicted_list)
predicted = predicted.flatten()

# create and print classification report
report = classification_report(truth, predicted)
print(report)

# define confusion matrix
confusion_matrix = metrics.confusion_matrix(truth, predicted)

# visualize confusion matrix with sklearn metrics
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = ["EMPTY", "FIRST"])

# display matrix
cm_display.plot()
plt.show()
