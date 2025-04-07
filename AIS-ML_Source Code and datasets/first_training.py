import pandas as pd
import numpy as np

# Import functions
from signal_processing import extract_ffts_labels

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pickle

# Read your dataset
df = pd.read_csv('./data/0-1.csv', encoding='latin-1', header=None)
data = df.iloc[:, :]
data_length = len(data)
df_label = pd.read_csv('./data/train_label_0-1.csv', encoding='utf-8', header=None)
label = df_label[:].to_numpy()

# Windowing parameters
step_size = 512    # Set the step size
window_size = 2048
full_length = 16384

labels = []
windows = []

# Apply moving window to raw ADC data line by line
for i in range(0, data_length, 1):
    row = data.iloc[i, :].to_numpy()
    # row = moving_average(row, 5)
    # Apply windowing
    for j in range(0, int((len(row) - window_size) / step_size) + 1, 1):
        window = np.hanning(window_size)
        signal = row[j * step_size:j * step_size + window_size]
        window = window*signal
        windows.append(window)

# Store data after windowing
data_window = np.array(windows)

# Get FFT
X_train, Y_train = extract_ffts_labels(data_window, label)

X_window = np.array(X_train)
# X_data_frame = pd.DataFrame(X_window)
# X_data_frame.to_csv('X_window_data_test.csv', index=False, header=False)

scaler = MinMaxScaler()
X_train_normalized = scaler.fit_transform(X_train)

X_train, X_test, Y_train, Y_test = train_test_split(X_train_normalized, Y_train, test_size=0.2, random_state=42)


# define model
model = Sequential()
# Input layer
model.add(Dense(16, input_dim=42, activation='relu'))
# Hidden layers
model.add(Dense(8, activation='relu', kernel_regularizer=l2(0.01)))
model.add(Dense(4, activation='relu', kernel_regularizer=l2(0.01)))
# Output layers
model.add(Dense(1, activation='sigmoid'))

# Choose a smaller learning rate
optimizer = Adam(learning_rate=0.0001)

# compile model
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# fit model
early_stopping = EarlyStopping(monitor='val_loss', patience=100, restore_best_weights=True)
model.fit(X_train, Y_train, epochs=1000, batch_size=128, validation_split=0.2, callbacks=[early_stopping])

# evaluate the keras model on the test set
_, test_accuracy = model.evaluate(X_test, Y_test)
print('Test Accuracy: %.2f' % (test_accuracy * 100))

# save model and architecture to single file
# model.save('model.keras')
# print("Saved model to disk")

# Save the scaler to disk (optional)
scaler_path = 'scaler.pkl'
with open(scaler_path, 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)