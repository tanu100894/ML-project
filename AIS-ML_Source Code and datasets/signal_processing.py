import pandas as pd
import numpy as np
from scipy.fft import fft
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

# Read your dataset
df = pd.read_csv('./confusion_matrix/6-2.csv', encoding='latin-1', header=None) # path to dataset
data = df.iloc[:, :]
data_length = len(data)

# Windowing parameters
step_size = 512    # Set the step size
window_size = 2048
full_length = 16384

denominator = 10

# min - max frequency
sampling_freq = 1953125 # (125MHz / 64)
min_freq = 30000
max_freq = 50000
freq_factor = sampling_freq / (window_size * 2)

# Initialize lists to store features and labels for classification
fft_magnitudes = []
labels = []
windows = []

def get_fft_values(y_values, window_size):
    f_values = np.linspace(0.0, window_size // 2, window_size // 2)*freq_factor
    fft_values_ = fft(y_values, n=window_size * 2)  # Zero-padding for better frequency resolution
    fft_values = 2.0 / window_size * np.abs(fft_values_[0:window_size // 2])
    return f_values[int(min_freq/freq_factor):int(max_freq/freq_factor)], fft_values[int(min_freq/freq_factor):int(max_freq/freq_factor)]

def extract_ffts_labels(dataset, labels):
    list_of_ffts = []
    list_of_labels = []

    for signal_no in range(len(dataset)):
        signal = dataset[signal_no, :]
        list_of_labels.append(labels[signal_no])
        f_values, fft_values = get_fft_values(signal, window_size)
        list_of_ffts.append(fft_values)
    return np.array(list_of_ffts), np.array(list_of_labels)

def extract_ffts(dataset):
    list_of_ffts = []

    for signal_no in range(len(dataset)):
        signal = dataset[signal_no, :]
        f_values, fft_values = get_fft_values(signal, window_size)
        list_of_ffts.append(fft_values)
    return np.array(list_of_ffts)

def autocorr(x):
    result = np.correlate(x, x, mode='full')
    autocorr_values = result[len(result)//2 :]
    # autocorr_values = np.round(autocorr_values, 3)
    return autocorr_values[10:]

def get_autocorr_values(y_values):
    autocorr_values = autocorr(y_values)
    x_values = np.arange(len(autocorr_values))
    return x_values, autocorr_values

# Apply moving window to raw ADC data line by line
for i in range(0, data_length, 1):
    row = data.iloc[i, :].to_numpy()
    # Apply windowing
    for j in range(0, int((len(row) - window_size) / step_size) + 1, 1):
        window = np.hanning(window_size)
        signal = row[j * step_size:j * step_size + window_size]
        window = window*signal
        windows.append(window)

# Store data after windowing
data_window = np.array(windows)
# data_frame = pd.DataFrame(data_window)
# data_frame.to_csv('window_data_test.csv', index=False, header=False)

# Test: sketch fft of the window containing echo
adc_signal = data_window[16, :]

# f_values, fft_values = get_fft_values(adc_signal, window_size)

# plt.plot(f_values, fft_values, linestyle='-', color='blue')
# plt.xlabel('Frequency', fontsize=8)
# plt.ylabel('Amplitude', fontsize=8)
# plt.title("FFT", fontsize=16)
# plt.show()

# f_values, psd_values = get_psd_values(adc_signal, window_size)

# plt.plot(f_values, psd_values, linestyle='-', color='blue')
# plt.xlabel('Index (Frequency domain)')
# plt.ylabel('Amplitude')
# plt.title("PSD", fontsize=16)
# plt.show()

# t_values, autocorr_values = get_autocorr_values(adc_signal)

# plt.plot(t_values, autocorr_values, linestyle='-', color='blue')
# plt.xlabel('Index (Time domain)')
# plt.ylabel('Autocorrelation amplitude')
# plt.title("ACF", fontsize=16)
# plt.show()