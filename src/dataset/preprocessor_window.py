import numpy as np
from scipy.signal import butter, filtfilt, stft

class Preprocessor_window:
    def __init__(self, fs=128):
        self.fs = fs  # Sampling frequency

    def bandpass_filter(self, data, lowcut=0.5, highcut=47):
        nyquist = 0.5 * self.fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(2, [low, high], btype='band')
        return filtfilt(b, a, data)

    def normalize(self, data):
        return (data - np.mean(data)) / np.std(data)

    def compute_stft(self, data, nperseg=128):
        _, _, Zxx = stft(data, fs=self.fs, nperseg=nperseg)
        return np.abs(Zxx)

    def preprocessing_window(self, window_data, steps=("filter", "normalize")):
        """
        Preprocess an EEG window by applying specified steps.

        Parameters:
            window_data: np.ndarray
                The EEG signal window.
            steps: tuple
                The preprocessing steps to apply (e.g., "filter", "normalize").

        Returns:
            np.ndarray: The preprocessed EEG signal.
        """

        # Apply bandpass filter if specified
        if "filter" in steps:
            window_data = self.bandpass_filter(window_data)

        # Zero-center the signal if normalization is specified
        if "normalize" in steps:
            window_data = window_data - np.mean(window_data)

        return window_data
    

   