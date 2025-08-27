import pandas as pd
import numpy as np
import torch

from tqdm import tqdm
from joblib import Parallel, delayed

from src.dataset.eeg_metrics import EEG_metrics
from src.dataset.preprocessor_window import Preprocessor_window


class EEG_window:
    def __init__(
        self,
        sampling_rate,
        window_size,
        step_size,
        majority_voting,
        for_majority,
        preprocessing,
        inference_mode,
        stft_window_size=32,
        stft_hop=8,
        bis_and_sef=False,
        depth_of_anesthesia=False,
        strategy=None,
    ):
        """
        Initialize the EEG window processor.
        Args:
            sampling_rate (int): Sampling rate of the EEG data.
            window_size (int): Size of the window for processing.
            step_size (int): Step size for moving the window.
            majority_voting (bool): Whether to use majority voting for labels.
            for_majority (int): Threshold for majority voting.
            preprocessing (bool): Whether to apply preprocessing to the EEG data.
            inference_mode (bool): If True, the function does not rely on annotations and does not calculate labels.
            stft_window_size (int): Window size for STFT. Default is 32.
            stft_hop (int): Hop size for STFT. Default is 8.
            bis_and_sef (bool): If True, BIS and SEF features are included. Default is False.
            depth_of_anesthesia (bool): If True, depth of anesthesia features are included. Default is False.
            strategy (str): Strategy for feature extraction. Default is None.
        """
        self.sampling_rate = sampling_rate
        self.bis_and_sef = bis_and_sef
        self.depth_of_anesthesia = depth_of_anesthesia
        self.window_size = window_size
        self.step_size = step_size
        self.majority_voting = majority_voting
        self.for_majority = for_majority
        self.preprocessing = preprocessing
        self.inference_mode = inference_mode
        self.strategy = strategy

        self.stft_window_size = stft_window_size
        self.stft_hop = stft_hop

        self.metrics = EEG_metrics(
            sampling_rate=self.sampling_rate,
            window_length=self.window_size,
            step_size=self.step_size,
        )
        self.preprocess_window = Preprocessor_window(fs=self.sampling_rate)

    def window_metrics(self, total_samples, merged_data):
        """
        Generalized function to calculate metrics for overlapping windows and label them based on the given label type.

        Args:
            total_samples (int): Total number of samples in the data.
            merged_data (pd.DataFrame): Merged dataset containing "EEG Voltage (\u00b5V)", "annotations", and optionally "artefact".
            label_type (str): The label type to evaluate ('Artefact', 'Sleep', or 'Perfect Sleep').
            inference_mode (bool): If True, the function does not rely on annotations and does not calculate labels.

        Returns:
            pd.DataFrame: DataFrame containing windowed metrics (and labels if not in inference mode).
        """
        # Ensure the label_type is valid
        if self.depth_of_anesthesia:
            columns_to_select = [
                "sleep",
                "cr",
                "sspl",
                "burst_suppression",
            ]
        else:
            columns_to_select = ["sleep"]
        # Initialize a list to store the results for each window
        results = []

        # Precompute indices
        indices = range(0, total_samples - self.window_size + 1, self.step_size)

        # Parallel processing
        results = Parallel(n_jobs=-1)(
            delayed(self.process_window)(
                start,
                start + self.window_size,
                merged_data,
                columns_to_select,
            )
            for start in tqdm(indices)
        )

        # Filter out None results
        results = [res for res in results if res is not None]

        # Convert the results to a DataFrame
        windowed_metrics_df = pd.DataFrame(results)
        return windowed_metrics_df

    def calculate_metrics(self, eeg_window):
        """
        Calculate various EEG metrics depending on the chosen strategy for a given window of EEG data.
        Args:
            eeg_window (np.ndarray): Window of EEG data.
        Returns:
            dict: Dictionary containing calculated features.
        """

        if self.strategy == "rawEEG":
            features = {"eeg_window": eeg_window}

        elif self.strategy == "stftEEG":
            window = torch.hann_window(self.stft_window_size)
            stft_result = torch.stft(
                torch.tensor(eeg_window),
                n_fft=self.stft_window_size,
                hop_length=self.stft_hop,
                win_length=self.stft_window_size,
                return_complex=True,
                window=window,
            )
            stft_result = 20 * torch.log10(stft_result.abs() + 1e-6)
            min_freq = 0.5
            max_freq = 47
            freqs = np.fft.rfftfreq(self.stft_window_size, d=1 / self.sampling_rate)
            cropped_stft = stft_result[
                np.argmax(freqs >= min_freq) : np.argmax(freqs > max_freq)
            ]

            features = {"stft_window": cropped_stft.numpy()}

        else:
            features = self.metrics.time_domain_features(eeg_window)
            features.update(self.metrics.entropy_and_complexity_features(eeg_window))
            features.update(self.metrics.ar_coefficients_wang(eeg_window))
            features.update(self.metrics.frequency_domain_features(eeg_window))
            features.update(self.metrics.extract_wavelet_features(eeg_window))
            features.update(self.metrics.catch22(eeg_window))
            features.update(self.metrics.eeg_dfa(eeg_window))

        return features

    def process_window(self, start, end, merged_data, columns_to_select):
        """
        Process a single window of EEG data and calculate metrics and labels.
        Args:
            start (int): Start index of the window.
            end (int): End index of the window.
            merged_data (pd.DataFrame): Merged dataset containing "EEG Voltage (\u00b5V)"
            columns_to_select (list): Columns to select from merged_data for annotation.
            flag (bool): Flag indicating if depth of anesthesia features are included.
        Returns:
            dict: Dictionary containing calculated features and labels.
        """
        # Extract the EEG window
        eeg_window = merged_data["EEG Voltage (\u00b5V)"].iloc[start:end].values

        if not self.inference_mode:
            annotation_window = merged_data[columns_to_select].iloc[start:end].values

        assert len(eeg_window) > 0

        if self.preprocessing:
            eeg_window = self.preprocess_window.preprocessing_window(eeg_window)

        features = self.calculate_metrics(eeg_window)

        if not self.inference_mode:
            sleep = annotation_window[:, 0]
            if self.depth_of_anesthesia:
                cr = annotation_window[:, 1]
                sspl = annotation_window[:, 2]
                burst_suppression = annotation_window[:, 3]

                if self.majority_voting:
                    label_sleep = (np.sum(sleep == 1) > self.for_majority).item()
                    label_cr = (np.sum(cr) > self.for_majority).item()
                    label_sspl = (np.sum(sspl) > self.for_majority).item()

                else:
                    label_cr = (np.any(cr)).item()
                    label_sspl = (np.any(sspl)).item()
                    label_sleep = (np.any(sleep == 1)).item()

                label_burst_supression = (
                    np.sum(burst_suppression) >= 64
                ).item()  # rule based: Burst suppression needs to be at least 0.5s (bursts usually need to be > 0.5s)

                features["cr"] = label_cr
                features["sspl"] = label_sspl
                features["burst_suppression"] = label_burst_supression
            else:
                if self.majority_voting:
                    label_sleep = (np.sum(sleep == 1) > self.for_majority).item()
                else:
                    label_sleep = (np.any(sleep == 1)).item()
            features["sleep"] = label_sleep

        features["Start"] = start
        features["End"] = end - 1
        return features
