import os
import ast
import pandas as pd
from mrmr import mrmr_classif
import torch
import torch.nn as nn
import json
import torch.optim as optim
from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    precision_score,
    recall_score,
    accuracy_score,
    confusion_matrix,
)
import matplotlib as mpl
from matplotlib import font_manager as fm
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
import pickle
import numpy as np
import random
from tqdm import tqdm
import joblib
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader
from sklearn.utils import shuffle
from lime import lime_tabular
from lime.lime_tabular import LimeTabularExplainer
import argparse
from typing import Dict, Optional, Sequence

# Set the random seed
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)


class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset

    Arguments:
        indices: a list of indices
        num_samples: number of samples to draw
        callback_get_label: a callback-like function which takes two arguments - dataset and index
    """

    def __init__(
        self,
        dataset,
    ):
        self.indices = list(range(len(dataset)))
        self.num_samples = len(self.indices)

        df = pd.DataFrame()
        df.index = self.indices
        df["label"] = self._get_labels(dataset)

        df = df.sort_index()

        label_to_count = df["label"].value_counts()

        weights = 1.0 / label_to_count[df["label"]]

        self.weights = torch.DoubleTensor(weights.to_list())

    def _get_labels(self, dataset):
        if isinstance(dataset, torch.utils.data.TensorDataset):
            return dataset.tensors[1].cpu()
        elif isinstance(dataset, torch.utils.data.Subset):
            return dataset.dataset.imgs[:][1]
        elif isinstance(dataset, torch.utils.data.Dataset):
            return dataset.get_labels()
        else:
            raise NotImplementedError

    def __iter__(self):
        return (
            self.indices[i]
            for i in torch.multinomial(self.weights, self.num_samples, replacement=True)
        )

    def __len__(self):
        return self.num_samples


class Utils:
    def __init__(
        self,
        for_majority=None,
        window_size=None,
        step_size=None,
        random_seed=None,
        preprocessing=None,
        sampling_rate=None,
        results_validation_csv_path=None,
        results_test_csv_path=None,
        model_dir=None,
        stft_window_size=None,
        stft_hop=None,
    ):
        """
        Initialize the Utils class with parameters for data preprocessing and model evaluation.
        Args:
            for_majority (bool): Whether to use majority voting.
            window_size (int): Size of the window for data processing.
            step_size (int): Step size for data processing.
            random_seed (int): Random seed for reproducibility.
            preprocessing (str): Preprocessing method name.
            majority_voting (bool): Whether to use majority voting.
            sampling_rate (int): Sampling rate for data processing.

        """
        self.set_seed(random_seed)
        self.window_size = window_size
        self.preprocessing = preprocessing
        self.step_size = step_size
        self.sampling_rate = sampling_rate
        self.for_majority = for_majority
        self.random_seed = random_seed
        self.results_validation_csv_path = results_validation_csv_path
        self.results_test_csv_path = results_test_csv_path
        self.model_dir = model_dir
        self.stft_window_size = stft_window_size
        self.stft_hop = stft_hop

        self.feature_name_map = {
            "relative_beta_ratio_saadeh": "Relative Beta Ratio @Saadeh",
            "spectral_slope": "Spectral Slope",
            "num_zero_crossings": "Number of Zero Crossings",
            "spectral_edge_frequency": "Spectral Edge Frequency",
            "lempel_ziv_complexity": "Lempel Ziv Complexity",
            "permutation_entropy_order_3": "Permutation Entropy (order=3)",
            "permutation_entropy_order_6": "Permutation Entropy (order=6)",
            "beta_ratio_gu": "Beta Ratio @Gu",
            "hjorth_mobility": "Hjorth Mobility",
            "hjorth_activity": "Hjorth Activity",
            "hjorth_complexity": "Hjorth Complexity",
            "petrosian_fractal_dimension": "Petrosian Fractal Dimension",
            "approximate_entropy": "Approximate Entropy",
            "dispersion_entropy": "Dispersion Entropy",
            "alpha_power_ratio": "Alpha Power Ratio",
            "higuchi_fd": "Higuchi Fractal Dimension",
            "ar_coeff_1": "Autoregressive Coefficient 1",
            "ar_coeff_2": "Autoregressive Coefficient 2",
            "ar_coeff_3": "Autoregressive Coefficient 3",
            "ar_coeff_4": "Autoregressive Coefficient 4",
            "ar_coeff_5": "Autoregressive Coefficient 5",
            "sample_entropy": "Sample Entropy",
            "mean_teager_energy": "Mean Teager Energy",
            "wavelet_ratio_abs_avg_A5_to_cD1": "Wavelet Ratio A5/cD1",
            "wavelet_ratio_abs_avg_A5_to_cD2": "Wavelet Ratio A5/cD2",
            "wavelet_ratio_abs_avg_A5_to_cD3": "Wavelet Ratio A5/cD3",
            "wavelet_ratio_abs_avg_A5_to_cD4": "Wavelet Ratio A5/cD4",
            "wavelet_ratio_abs_avg_A5_to_cD5": "Wavelet Ratio A5/cD5",
            "wavelet_ratio_abs_avg_cD2_to_cD1": "Wavelet Ratio cD2/cD1",
            "wavelet_ratio_abs_avg_cD5_to_cD1": "Wavelet Ratio cD5/cD1",
            "wavelet_ratio_abs_avg_cD5_to_cD3": "Wavelet Ratio cD5/cD3",
            "wavelet_ratio_abs_avg_cD5_to_cD4": "Wavelet Ratio cD5/cD4",
            "wavelet_ratio_abs_avg_cD5_to_cD2": "Wavelet Ratio cD5/cD2",
            "wavelet_ratio_abs_avg_cD3_to_cD1": "Wavelet Ratio cD3/cD1",
            "wavelet_ratio_abs_avg_cD3_to_cD2": "Wavelet Ratio cD3/cD2",
            "wavelet_ratio_abs_avg_cD4_to_cD2": "Wavelet Ratio cD4/cD2",
            "wavelet_ratio_abs_avg_cD4_to_cD3": "Wavelet Ratio cD4/cD3",  # Ratio of the mean absolute wavelet coefficients
            "wavelet_ratio_abs_avg_cD4_to_cD1": "Wavelet Ratio cD4/cD1",
            "wavelet_mean_power_cD1": "Wavelet cD1 mean power",
            "wavelet_mean_power_cD2": "Wavelet cD2 mean power",
            "wavelet_mean_power_cD3": "Wavelet cD3 mean power",
            "wavelet_mean_power_cD4": "Wavelet cD4 mean power",
            "wavelet_mean_power_cD5": "Wavelet cD5 mean power",
            "wavelet_std_cD1": "Std of the absolute wavelet cD1",
            "wavelet_std_cD2": "Std of the absolute wavelet cD2",
            "wavelet_std_cD3": "Std of the absolute wavelet cD3",
            "wavelet_std_cD4": "Std of the absolute wavelet cD4",
            "wavelet_std_cD5": "Std of the absolute wavelet cD5",
            "hurst_exponent": "Hurst Exponent",
            "wavelet_A5-mean": "Mean of the absolute wavelet approximation coefficients A5",
            "wavelet_A5-std": "Std of the absolute wavelet approximation coefficients A5",
            "peak_index": "Index of the maximum absolute EEG amplitude within the window",
            "peak_to_peak": "Peak-to-peak amplitude (max − min of the EEG window)",
            "peak": "Maximum absolute amplitude within the EEG window",
            "mean_abs": "Mean of the absolute EEG amplitudes",
            "std_curve_length": "StD of the absolute first derivative (curve length variability)",
            "mean_curve_length": "Mean of the absolute first derivative (average curve length)",
            "max_amplitude": "Max Amplitude",
            "min_amplitude": "Min Amplitude",
            "differential_entropy": "Differential Entropy",
            "gamma_power_ratio": "Gamma Power Ratio",
            "renyi_entropy": "Renyi Entropy",
            "time_variance": "Variance of the EEG amplitudes",
            "time_skewness": "Skewness of the EEG amplitudes",
            "time_kurtosis": "Kurtosis of the EEG amplitudes",
            "time_std": "Standard deviation of the EEG amplitudes",
            "median_frequency": "Median Frequency",
            "mean_frequency": "Mean Frequency",
            "spectral_entropy": "Spectral Entropy",
            "beta_power_ratio": "Beta Power Ratio",
            "delta_power_ratio": "Delta Power Ratio",
            "theta_power_ratio": "Theta Power Ratio",
            "dfa_exponent": "DFA Exponent",
            "time_arithmetic_mean": "Arithmetic mean of the EEG amplitudes",
            "mean_energy": "Mean signal energy (average of squared EEG amplitudes)",
            "time_median": "Median of the EEG amplitudes",
            "waveform_index": "Form Factor",
            "wavelet_mean_abs_cD3": "MAV cD3",  # "Mean Absolute Value of Wavelet Detail Coefficients (cD3)",
            "wavelet_mean_abs_cD1": "MAV cD1",  # "Mean Absolute Value of Wavelet Detail Coefficients (cD1)",
            "wavelet_mean_abs_cD2": "MAV cD2",  # "Mean Absolute Value of Wavelet Detail Coefficients (cD2)",
            "wavelet_mean_abs_cD4": "MAV cD4",  # "Mean Absolute Value of Wavelet Detail Coefficients (cD4)",
            "wavelet_mean_abs_cD5": "MAV cD5",  # "Mean Absolute Value of Wavelet Detail Coefficients (cD5)",
            "margin_index": "Margin Index",
            "rms": "Root mean square (RMS) amplitude of the EEG window",
        }

        if not os.path.exists(os.path.dirname(results_validation_csv_path)):
            os.makedirs(os.path.dirname(results_validation_csv_path))
        if not os.path.exists(os.path.dirname(results_test_csv_path)):
            os.makedirs(os.path.dirname(results_test_csv_path))

        self.results_validation_df = self.load_or_create_results(
            self.results_validation_csv_path
        )
        self.results_test_df = self.load_or_create_results(self.results_test_csv_path)

    # set all random seeds:
    def set_seed(self, seed=42):
        """
        Set the random seed for reproducibility.
        Args:
            seed (int): Random seed value.
        """
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def load_or_create_results(self, csv_path):
        """
        Load existing results from a CSV file or create a new DataFrame if the file does not exist.
        Args:
            csv_path (str): Path to the CSV file.
        Returns:
            pd.DataFrame: DataFrame containing the results.
        """
        # Check if the CSV file exists
        if os.path.exists(csv_path):

            return pd.read_csv(
                csv_path,
                converters={
                    "Confusion Matrix": ast.literal_eval,
                    "Cross-Validation F1 Scores": ast.literal_eval,
                },
            )
        else:
            print("New results DataFrame created.")
            return pd.DataFrame(
                columns=[
                    "Strategy",
                    "classification_type",
                    "Model Name",
                    "Window Size",
                    "Step Size",
                    "Majority Voting",
                    "Model type",
                    "Preprocessing",
                    "Cross-Validation F1 Scores",
                    "Mean CV F1",
                    "F1",
                    "ROC-AUC",
                    "Precision Class 0",
                    "Precision Class 1",
                    "Recall Class 0",
                    "Recall Class 1",
                    "F1 Class 0",
                    "F1 Class 1",
                    "Accuracy",
                    "Confusion Matrix",
                    "Random Seed",
                ]
            )

    def train_and_evaluate_model(
        self,
        name,
        model,
        train_loader_nn=None,
        val_loader_nn=None,
        X=None,
        y=None,
        X_val=None,
        y_val=None,
        skf=None,
        classification_type=None,
        strategy=None,
        number_of_features=None,
        patience=3,
        learning_rate=1e-3,
        device=None,
    ):
        """
        Train and evaluate a machine learning model.

        Args:
            name (str): Name of the model.
            model: The machine learning model to train.
            train_loader_nn (DataLoader): DataLoader for training data (for neural networks).
            val_loader_nn (DataLoader): DataLoader for validation data (for neural networks).
            X (array-like): Training feature set.
            y (array-like): Training labels.
            X_val (array-like): Validation feature set.
            y_val (array-like): Validation labels.
            skf: Stratified K-Fold object for cross-validation.
            classification_type (str): Type of classification.
            strategy (str): Strategy used for classification.
            number_of_features (int, optional): Number of features to use.
            patience (int): Number of epochs with no improvement after which training will be stopped.
            learning_rate (float): Learning rate for the optimizer.
            device (str): Device to use for training (e.g., 'cpu' or 'cuda').
        Returns:
            dict: Dictionary containing evaluation metrics.
            model: The trained model.
        """
        print(f"Training {name}...")
        self.set_seed(self.random_seed)
        if number_of_features is None:
            number_of_features_str = "all"
        else:
            number_of_features_str = str(number_of_features)

        model_filename = (
            self.model_dir
            / f"{strategy}_{classification_type}_ws{self.window_size}_ss{self.step_size}_majority{self.for_majority}_type{model.__class__.__name__}_preproc{self.preprocessing}_randomseed{self.random_seed}_{number_of_features_str}"
        )
        if self.stft_window_size:
            model_filename = f"{model_filename}_{self.stft_window_size}_{self.stft_hop}"
        if isinstance(model, torch.nn.Module):

            optimizer = optim.Adam(
                model.parameters(), lr=learning_rate, weight_decay=1e-3
            )
            criterion = nn.BCEWithLogitsLoss()
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", patience=patience, factor=0.1, verbose=True
            )
            metrics = self.train_model_nn(
                model,
                train_loader=train_loader_nn,
                val_loader=val_loader_nn,
                model_filename=f"{model_filename}.pt",
                device=device,
                patience=patience,
                num_epochs=50,
                criterion=criterion,
                optimizer=optimizer,
                scheduler=scheduler,
            )
        else:
            metrics, _ = self.train_evaluate_ml_model(
                model=model,
                X=np.nan_to_num(X, nan=0),
                y=y,
                X_val=np.nan_to_num(X_val, nan=0),
                y_val=y_val,
                skf=skf,
                model_filename=f"{model_filename}.pkl",
            )

        self.save_results(
            name=name,
            metrics=metrics,
            test=False,
            classification_type=classification_type,
            strategy=strategy,
            number_of_features=number_of_features_str,
        )

    def train_evaluate_ml_model(
        self, model, X, y, X_val, y_val, skf, model_filename=None
    ):
        """
        Train and evaluate a machine learning model using cross-validation.
        Args:
            model: The machine learning model to train.
            X: Training feature set.
            y: Training labels.
            X_val: Validation feature set.
            y_val: Validation labels.
            skf: Stratified K-Fold object for cross-validation.
            model_filename (str): Path to save the trained model.
        Returns:
            dict: Dictionary containing evaluation metrics.
            model: The trained model.
        """
        if skf is None:
            cross_val_scores = None
        else:
            cross_val_scores = cross_val_score(
                model, X, y, cv=skf, scoring="f1", n_jobs=-1
            )
        model.fit(X, y)

        y_validation_pred = model.predict_proba(X_val)[:, 1]
        y_pred_binary = (y_validation_pred > 0.5).astype(int)
        if model_filename is not None:
            with open(model_filename, "wb") as f:
                pickle.dump(model, f)

        return (
            self.calculate_metrics(
                y_ground_truth=y_val,
                y_pred_binary=y_pred_binary,
                y_pred=y_validation_pred,
                cross_val_scores=cross_val_scores,
            ),
            model,
        )

    def calculate_metrics(
        self, y_ground_truth, y_pred_binary, y_pred=None, cross_val_scores=None
    ):
        """
        Calculate evaluation metrics for the model.
        Args:
            y_ground_truth (array-like): True labels.
            y_pred_binary (array-like): Predicted labels (binary).
            y_pred (array-like, optional): Predicted probabilities.
            cross_val_scores (array-like, optional): Cross-validation scores.
        Returns:
            dict: Dictionary containing evaluation metrics."""

        metrics = {
            "F1": f1_score(y_ground_truth, y_pred_binary),
            "ROC-AUC": (
                roc_auc_score(y_ground_truth, y_pred) if y_pred is not None else "N/A"
            ),
            "Precision Class 0": precision_score(
                y_ground_truth, y_pred_binary, pos_label=0
            ),
            "Precision Class 1": precision_score(
                y_ground_truth, y_pred_binary, pos_label=1
            ),
            "Recall Class 0": recall_score(y_ground_truth, y_pred_binary, pos_label=0),
            "Recall Class 1": recall_score(y_ground_truth, y_pred_binary, pos_label=1),
            "F1 Class 0": f1_score(y_ground_truth, y_pred_binary, pos_label=0),
            "F1 Class 1": f1_score(y_ground_truth, y_pred_binary, pos_label=1),
            "Accuracy": accuracy_score(y_ground_truth, y_pred_binary),
            "Confusion Matrix": str(
                confusion_matrix(y_ground_truth, y_pred_binary).tolist()
            ),  # Convert to string
            "Cross-Validation F1 Scores": (
                cross_val_scores.tolist() if cross_val_scores is not None else []
            ),
            "Mean CV F1": (
                cross_val_scores.mean() if cross_val_scores is not None else "N/A"
            ),
        }
        return metrics

    def save_results(
        self,
        name,
        metrics,
        test=False,
        classification_type=None,
        strategy=None,
        number_of_features=None,
    ):
        """
        Save the results of the model evaluation to a CSV file.
        Args:

            name (str): Name of the model.
            metrics (dict): Dictionary containing evaluation metrics.
            test (bool): Whether the results are for the test set.
            classification_type (str): Type of classification.
            strategy (str): Strategy used for classification.
            number_of_features (int): Number of features used in the model."""
        # Prepare the new entry as a DataFrame row
        new_entry = pd.DataFrame(
            [
                {
                    "Strategy": strategy,
                    "classification_type": classification_type.replace(
                        "burst_suppression", "burstsuppression"
                    ),
                    "Model Name": name,
                    "Random Seed": self.random_seed,
                    "Window Size": self.window_size,
                    "Step Size": self.step_size,
                    "STFT Window Size": self.stft_window_size,
                    "STFT Step Size": self.stft_hop,
                    "Majority Voting": self.for_majority,
                    "Preprocessing": self.preprocessing,
                    "Number of Features": number_of_features,
                    **metrics,
                }
            ]
        )
        if test:
            self.results_test_df = pd.concat(
                [self.results_test_df, new_entry], ignore_index=True
            )
            self.results_test_df.to_csv(self.results_test_csv_path, index=False)
            print(f"Test results saved to {self.results_test_csv_path}")
        else:
            self.results_validation_df = pd.concat(
                [self.results_validation_df, new_entry], ignore_index=True
            )
            self.results_validation_df.to_csv(
                self.results_validation_csv_path, index=False
            )
            print(f"Validation results saved to {self.results_validation_csv_path}")

    def train_model_nn(
        self,
        model,
        train_loader,
        val_loader,
        scheduler,
        optimizer,
        criterion,
        patience=3,
        num_epochs=50,
        device=None,
        model_filename=None,
    ):
        """
        Train a PyTorch neural network model with early stopping and learning rate scheduling.
        Args:
            model (torch.nn.Module): The neural network model to train.
            train_loader (DataLoader): DataLoader for training data.
            val_loader (DataLoader): DataLoader for validation data.
            scheduler (torch.optim.lr_scheduler): Learning rate scheduler.
            optimizer (torch.optim.Optimizer): Optimizer for training.
            criterion (torch.nn.Module): Loss function.
            patience (int): Number of epochs with no improvement after which training will be stopped.
            num_epochs (int): Total number of epochs to train the model.
            device (str): Device to use for training
            model_filename (str, optional): Path to save the best model state.
        Returns:
            dict: Dictionary containing evaluation metrics.
        """

        best_val_loss = float("inf")
        early_stop_counter = 0
        best_model_state = None

        # caches for metrics from the best epoch
        best_epoch = -1
        best_y_true = None
        best_y_pred = None
        best_y_bin = None

        for epoch in range(num_epochs):
            model.train()
            train_loss = 0.0

            for X_batch, y_batch in tqdm(
                train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training"
            ):
                X_batch = X_batch.to(device)
                if torch.isnan(X_batch).any():
                    X_batch = torch.nan_to_num(X_batch, nan=0.0)

                y_batch = y_batch.to(device).float()
                logits = model(X_batch)
                if logits.ndim == 2 and logits.size(1) == 1:
                    logits = logits.squeeze(1)
                loss = criterion(logits, y_batch)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += loss.item()

                # -------- Validation --------
            model.eval()
            val_loss = 0.0
            y_val_true, y_val_pred_binary, y_val_pred = [], [], []

            with torch.no_grad():
                for X_batch, y_batch in tqdm(
                    val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Val"
                ):
                    X_batch = X_batch.to(device)
                    if torch.isnan(X_batch).any():
                        X_batch = torch.nan_to_num(X_batch, nan=0.0)

                    y_batch = y_batch.to(device).float()
                    logits = model(X_batch)
                    if logits.ndim == 2 and logits.size(1) == 1:
                        logits = logits.squeeze(1)
                    loss = criterion(logits, y_batch)
                    probs = torch.sigmoid(logits)  
                    preds = (probs >= 0.5).float()
                    y_val_pred.extend(probs.detach().cpu().numpy())

                    val_loss += loss.item()
                    y_val_true.extend(y_batch.detach().cpu().numpy())
                    y_val_pred_binary.extend(preds.detach().cpu().numpy())

            train_loss /= len(train_loader)
            val_loss /= len(val_loader)

            print(
                f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}"
            )

            if scheduler is not None:
                scheduler.step(val_loss)

            # ---- Early stopping + SAVE CACHED METRICS WHEN IMPROVED ----
            if val_loss <= best_val_loss:
                best_val_loss = val_loss
                early_stop_counter = 0
                best_model_state = model.state_dict()
                best_epoch = epoch + 1

                # NEW: store the arrays from THIS validation pass
                best_y_true = np.array(y_val_true)
                best_y_pred = np.array(y_val_pred)
                best_y_bin = np.array(y_val_pred_binary)

                if model_filename:
                    torch.save(best_model_state, model_filename)
            else:
                early_stop_counter += 1
                print(f"No improvement for {early_stop_counter} epoch(s).")

            if early_stop_counter >= patience:
                print(
                    f"Early stopping triggered. Best Validation Loss: {best_val_loss:.4f} (epoch {best_epoch})"
                )
                break

        # Compute metrics directly from cached best-epoch predictions (NO final pass)
        metrics = self.calculate_metrics(
            y_ground_truth=best_y_true,
            y_pred_binary=best_y_bin,
            y_pred=best_y_pred,
        )
        return metrics

    def evaluate_model_on_sliding_test(
        self,
        model_name=None,
        model=None,
        X_test=None,
        y_test=None,
        test_loader=None,
        classification_type=None,
        strategy=None,
        number_of_features=None,
        batch_size=None,
        original_test=None,
        model_directory=None,
        device=None,
    ):
        """
        Evaluates a trained model on test data and returns performance metrics.

        Args:
            model_name (str): Name of the model file to load.
            X_test (ndarray or DataFrame, optional): Test feature set for tree-based models.
            y_test (ndarray or Series, optional): Test labels for tree-based models.
            test_loader (DataLoader, optional): DataLoader for neural networks.

        Returns:
            dict: Dictionary with performance metrics.
        """

        print(f"Evaluating model: {model_name} on test set...")

        if number_of_features is None:
            number_of_features_str = "all"
        else:
            number_of_features_str = str(number_of_features)

        model_filename = (
            model_directory
            / f"{strategy}_{classification_type}_ws{self.window_size}_ss{self.window_size}_majority{self.for_majority}_type{model_name}_preproc{self.preprocessing}_randomseed{self.random_seed}_{number_of_features_str}"
        )
        if self.stft_window_size:
            model_filename = f"{model_filename}_{self.stft_window_size}_{self.stft_hop}"
        print(model_filename)

        y_ground_truth, y_pred, y_pred_binary = [], [], []

        if model_name in [
            "TabularNN",
            "SimpleModel",
            "FCN",
            "UCRResNet",
            "ResNet",
            "LSTMModel",
            "TSTransformerEncoderClassiregressor",
        ]:
            model_path = f"{model_filename}.pt"
            if not os.path.exists(model_path):
                raise FileNotFoundError(
                    f"Model file for {model_name} not found: {model_filename}"
                )

            print(f"{model_name} model detected")

            # Load model
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)
            model.eval()

            with torch.no_grad():
                for X_batch, y_batch in tqdm(test_loader):

                    X_batch = torch.nan_to_num(
                        X_batch, nan=0.0, posinf=1.0, neginf=-1.0
                    ).to(device)
                    y_batch = y_batch.to(device).float()

                    assert torch.all(
                        y_batch == y_batch[0]
                    ), "All y values in the batch are not the same."
                    y_unique = y_batch[0].item()
                    outputs = model(X_batch)  # Raw logits

                    if len(outputs.shape) > 1:
                        y_pred_batch = outputs.squeeze(1)
                    else:
                        y_pred_batch = outputs

                    aggregated_outputs = torch.median(y_pred_batch)
                    pred_binary = (aggregated_outputs > 0.5).float()

                    y_ground_truth.append(y_unique)
                    y_pred_binary.append(pred_binary.cpu().item())
                    y_pred.append(aggregated_outputs.cpu().item())
            print(y_pred_binary)
            print(y_ground_truth)
            print([float(format(float_num, ".1f")) for float_num in y_pred])

        elif model_name in [
            "RandomForestClassifier",
            "CatBoostClassifier",
            "DecisionTreeClassifier",
            "TabICLClassifier",
            "GaussianNB",
        ]:
            model_path = f"{model_filename}.pkl"
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file for {model_name} not found.")

            print("Tree Based Model")
            with open(model_path, "rb") as f:
                model = pickle.load(f)

            y_pred_all = model.predict_proba(X_test)[
                :, 1
            ]  # get probability for the positive class (class 1)
            n_samples = len(y_pred_all)
            # Evaluate without aggregation
            y_pred_binary_test = (y_pred_all > 0.5).astype(int)
            f1 = f1_score(original_test, y_pred_binary_test)
            auc = roc_auc_score(y_test, y_pred_all)

            print("ROC AUC:", auc)
            print("F1 Score:", f1)
            for i in range(0, n_samples, batch_size):
                batch_truths = y_test[i : i + batch_size]
                agg = np.median(y_pred_all[i : i + batch_size])  # or np.mean
                binary = int(agg > 0.5)

                y_pred.append(agg)
                y_pred_binary.append(binary)

                # Assuming all labels in batch are the same
                y_ground_truth.append(batch_truths.iloc[0])

        else:
            raise ValueError(f"{model_name} not found in model list.")

        # Ensure y_pred and y_ground_truth are binary
        y_ground_truth = np.array(y_ground_truth)
        y_pred_binary = np.array(y_pred_binary)

        # Compute evaluation metrics
        metrics = self.calculate_metrics(
            y_ground_truth=y_ground_truth, y_pred_binary=y_pred_binary, y_pred=y_pred
        )

        print("Test Evaluation Results:")
        for key, value in metrics.items():
            print(f"{key}: {value}")

        self.save_results(
            name=model_name,
            metrics=metrics,
            test=True,
            classification_type=classification_type,
            strategy=strategy,
            number_of_features=number_of_features_str,
        )
        print("Training complete!")

    def evaluate_model_on_test(
        self,
        model_name=None,
        model=None,
        X_test=None,
        y_test=None,
        test_loader=None,
        classification_type=None,
        strategy=None,
        number_of_features=None,
        device=None,
    ):
        """
        Evaluates a trained model on test data and returns performance metrics.

        Args:
            model_name (str): Name of the model file to load.
            model: The trained model to evaluate.
            X_test (ndarray or DataFrame, optional): Test feature set for tree-based models.
            y_test (ndarray or Series, optional): Test labels for tree-based models.
            test_loader (DataLoader, optional): DataLoader for neural networks.
            classification_type (str): Type of classification.
            strategy (str): Strategy used for classification.
            number_of_features (int, optional): Number of features to use.
            device (str): Device to use for evaluation (e.g., 'cpu' or 'cuda').

        Returns:
            dict: Dictionary with performance metrics.
        """

        print(f"Evaluating model: {model_name} on test set...")

        if number_of_features is None:
            number_of_features_str = "all"
        else:
            number_of_features_str = str(number_of_features)

        model_filename = (
            self.model_dir
            / f"{strategy}_{classification_type}_ws{self.window_size}_ss{self.step_size}_majority{self.for_majority}_type{model_name}_preproc{self.preprocessing}_randomseed{self.random_seed}_{number_of_features_str}"
        )
        if self.stft_window_size:
            model_filename = f"{model_filename}_{self.stft_window_size}_{self.stft_hop}"
        print(model_filename)
        # Determine if model is a PyTorch neural network or a tree-based model
        if model_name in [
            "TabularNN",
            "MLPNN",
            "SimpleModel",
            "FCN",
            "ResNet",
            "UCRResNet",
            "LSTMModel",
            "TSTransformerEncoderClassiregressor",
        ]:
            model_path = f"{model_filename}.pt"
            if not os.path.exists(model_path):
                raise FileNotFoundError(
                    f"Model file for {model_name} not found: {model_filename}"
                )

            print(f"{model_name} model detected")
            # Load model
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)
            model.eval()

            y_ground_truth, y_pred, y_pred_binary = [], [], []

            with torch.no_grad():
                for X_batch, y_batch in tqdm(test_loader, desc="Testing"):
                    # Clean & move
                    X_batch = torch.nan_to_num(
                        X_batch, nan=0.0, posinf=1.0, neginf=-1.0
                    ).to(device)

                    y_batch = y_batch.to(device).float()
                    logits = model(X_batch)
                    if logits.ndim == 2 and logits.size(1) == 1:
                        logits = logits.squeeze(1)
                    probs = torch.sigmoid(logits)  # convert logits -> probs
                    y_pred_batch = probs
                    # Binary decision: sigmoid >= 0.5
                    pred_binary = (probs >= 0.5).float()

                    y_ground_truth.extend(y_batch.detach().cpu().numpy())
                    y_pred.extend(y_pred_batch.detach().cpu().numpy())
                    y_pred_binary.extend(pred_binary.detach().cpu().numpy())

        elif model_name in [
            "RandomForestClassifier",
            "CatBoostClassifier",
            "DecisionTreeClassifier",
            "TabICLClassifier",
            "GaussianNB",
        ]:
            model_path = f"{model_filename}.pkl"
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file for {model_name} not found.")

            print("Tree Based Model")
            with open(model_path, "rb") as f:
                model = pickle.load(f)

            y_pred = model.predict_proba(X_test)[
                :, 1
            ]  # get probability for the positive class (class 1)
            y_pred_binary = (y_pred > 0.5).astype(int)
            y_ground_truth = y_test

        else:
            raise ValueError(f"{model_name} not found in model list.")

        # Ensure y_pred and y_ground_truth are binary
        y_ground_truth = np.array(y_ground_truth)
        y_pred_binary = np.array(y_pred_binary)

        # Compute evaluation metrics
        metrics = self.calculate_metrics(
            y_ground_truth=y_ground_truth, y_pred_binary=y_pred_binary, y_pred=y_pred
        )

        print("Test Evaluation Results:")
        for key, value in metrics.items():
            print(f"{key}: {value}")

        self.save_results(
            name=model_name,
            metrics=metrics,
            test=True,
            classification_type=classification_type,
            strategy=strategy,
            number_of_features=number_of_features_str,
        )
        print("Testing complete!")

    def perform_feature_selection_mrmr(
        self,
        X,
        y,
        k=25,
        selected_features_filename=None,
    ):
        """
        Perform feature selection using the Minimum Redundancy Maximum Relevance (mRMR) method.

        Parameters:
        X (DataFrame): Feature set
        y (Series): Target variable
        k (int): Number of features to select
        selected_features_filename (str): Path to save the selected features

        Returns:
        list: List of selected feature names
        """

        selected_features = mrmr_classif(X=X, y=y, K=k)

        with open(selected_features_filename, "w") as f:
            json.dump(selected_features, f)
        return selected_features

    def save_and_apply_scaler(self, X, X_val, X_test, scaler_filename=None):
        """
        Create, fit, and apply a StandardScaler, then save it to a file.

        Parameters:
        X (array-like): Training feature set
        X_val (array-like): Validation feature set
        X_test (array-like): Test feature set
        scaler_filename (str): Path to save the scaler

        Returns:
        Tuple: (X_nn, X_val_nn, X_test_nn, X_nn_desc, X_val_nn_desc, X_test_nn_desc)
        1. X_nn: Scaled training features
        2. X_val_nn: Scaled validation features
        3. X_test_nn: Scaled test features
        4. X_nn_desc: DataFrame with scaled training features and original column names
        5. X_val_nn_desc: DataFrame with scaled validation features and original column names
        6. X_test_nn_desc: DataFrame with scaled test features and original column names
        """

        # Extract the directory part of the filename
        scaler_directory = os.path.dirname(scaler_filename)

        # Create the directory if it does not exist
        os.makedirs(scaler_directory, exist_ok=True)

        # Create a StandardScaler instance
        scaler = StandardScaler()

        # Fit and transform the training data for neural networks
        X_nn = scaler.fit_transform(X)

        # Transform the validation and test data
        X_val_nn = scaler.transform(X_val)
        X_test_nn = scaler.transform(X_test)

        X_nn_desc = pd.DataFrame(
            scaler.fit_transform(X), columns=X.columns, index=X.index
        )
        X_val_nn_desc = pd.DataFrame(
            scaler.transform(X_val), columns=X_val.columns, index=X_val.index
        )
        X_test_nn_desc = pd.DataFrame(
            scaler.transform(X_test), columns=X_test.columns, index=X_test.index
        )

        # Save the scaler to a file
        joblib.dump(scaler, scaler_filename)

        # Confirm that the scaler has been saved
        print(f"Scaler saved to {scaler_filename}")

        return (
            X_nn,
            X_val_nn,
            X_test_nn,
            X_nn_desc,
            X_nn_desc,
            X_val_nn_desc,
            X_test_nn_desc,
        )

    def prepare_tensors_and_loaders(
        self,
        X_nn,
        y,
        X_val_nn,
        y_val,
        X_test_nn,
        y_test,
        batch_size=16,
        device=None,
        imbalanced=True,
    ):
        """
        Convert data to PyTorch tensors and create DataLoaders.

        Parameters:
        X_nn (array-like): Processed training features
        y (array-like): Training labels
        X_val_nn (array-like): Processed validation features
        y_val (array-like): Validation labels
        X_test_nn (array-like): Processed test features
        y_test (array-like): Test labels
        batch_size (int, optional): Batch size for DataLoaders (default: 16)
        device (str, optional): Device for tensor storage
        imbalanced (bool, optional): Whether to use an imbalanced dataset sampler (default: True)

        Returns:
        Tuple: (train_loader_nn, val_loader_nn, test_loader_nn, input_size)
        1. train_loader_nn: DataLoader for training data
        2. val_loader_nn: DataLoader for validation data
        3. test_loader_nn: DataLoader for test data
        4. input_size: Number of features in the training data
        """

        y = y.to_numpy(dtype=np.int64)  # Convert Series to NumPy array with int64
        y_val = y_val.to_numpy(dtype=np.int64)
        y_test = y_test.to_numpy(dtype=np.int64)
        # Ensure correct dtype
        X_nn = X_nn.astype(np.float32)  # Convert to float32 for MPS
        X_val_nn = X_val_nn.astype(np.float32)
        X_test_nn = X_test_nn.astype(np.float32)

        # Convert to PyTorch tensors
        X_train_tensor = torch.from_numpy(X_nn).to(device)
        y_train_tensor = torch.from_numpy(y).to(device)

        X_val_tensor = torch.from_numpy(X_val_nn).to(device)
        y_val_tensor = torch.from_numpy(y_val).to(device)

        X_test_tensor = torch.from_numpy(X_test_nn).to(device)
        y_test_tensor = torch.from_numpy(y_test).to(device)

        # Create DataLoaders (no need to call .to(device) here)
        train_dataset_nn = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset_nn = TensorDataset(X_val_tensor, y_val_tensor)
        test_dataset_nn = TensorDataset(X_test_tensor, y_test_tensor)

        if imbalanced:
            train_loader_nn = DataLoader(
                train_dataset_nn,
                batch_size=batch_size,
                sampler=ImbalancedDatasetSampler(train_dataset_nn),
            )
        else:
            train_loader_nn = DataLoader(
                train_dataset_nn,
                batch_size=batch_size,
                shuffle=True,
            )

        val_loader_nn = DataLoader(val_dataset_nn, batch_size=batch_size, shuffle=False)
        test_loader_nn = DataLoader(
            test_dataset_nn, batch_size=batch_size, shuffle=False
        )

        input_size = X_nn.shape[1]  # Number of features
        print("tensor input size", input_size)

        return train_loader_nn, val_loader_nn, test_loader_nn, input_size

    def preprocess_data(
        self,
        X,
        y,
        X_val,
        y_val,
        X_test,
        y_test,
        k=25,
        device=None,
        batch_size=16,
        strategy=None,
        scaling_nn=True,
        classification_type=None,
        imbalanced=True,
        feature_selection=False,
    ):
        """
        Preprocess data for both RandomForest and CNN models.

        Parameters:
        X (DataFrame): Training feature set
        y (Series): Training labels
        X_val (DataFrame): Validation feature set
        y_val (Series): Validation labels
        X_test (DataFrame): Test feature set
        y_test (Series): Test labels
        k (int, optional): Number of features to select (default: 25)
        device (str, optional): Device for tensor storage
        batch_size (int, optional): Batch size for DataLoaders (default: 16)
        strategy (str, optional): Feature selection strategy
        scaling_nn (bool, optional): Whether to scale data for neural networks (default: True)
        classification_type (str, optional): Type of classification
        imbalanced (bool, optional): Whether to use an imbalanced dataset sampler (default: True)
        feature_selection (bool, optional): Whether to perform feature selection
        Returns:
            Tuple: (X, y, X_val, y_val, X_test, y_test, train_loader_nn, val_loader_nn, test_loader_nn, input_size)
            1. X: Processed training features
            2. y: Training labels
            3. X_val: Processed validation features
            4. y_val: Validation labels
            5. X_test: Processed test features
            6. y_test: Test labels
            7. train_loader_nn: DataLoader for training data (neural network)
            8. val_loader_nn: DataLoader for validation data (neural network)
            9. test_loader_nn: DataLoader for test data (neural network)
            10. input_size: Number of features in the training data
            11. selected_feature_names: List of selected feature names (if feature_selection is True)
            12. scaler_filename: Path to the saved scaler (if scaling_nn is True)
            13. X_nn_desc: DataFrame with scaled training features and original column names
            14. X_val_nn_desc: DataFrame with scaled validation features and original column names
            15. X_test_nn_desc: DataFrame with scaled test features and original column names
        """
        if not scaling_nn:
            X_nn = X
            X_val_nn = X_val
            X_test_nn = X_test
            scaler_filename = None

            train_loader_nn, val_loader_nn, test_loader_nn, input_size = (
                self.prepare_tensors_and_loaders(
                    X_nn=X_nn,
                    y=y,
                    X_val_nn=X_val_nn,
                    y_val=y_val,
                    X_test_nn=X_test_nn,
                    y_test=y_test,
                    batch_size=batch_size,
                    device=device,
                    imbalanced=imbalanced,
                )
            )
            return (
                X,
                y,
                X_val,
                y_val,
                X_test,
                y_test,
                train_loader_nn,
                val_loader_nn,
                test_loader_nn,
                input_size,
            )
        selected_features_filename = (
            self.model_dir
            / "selected_features"
            / f"{classification_type}_{strategy}_ws{self.window_size}_ss{self.step_size}_majority{self.for_majority}_preproc{self.preprocessing}_randomseed{self.random_seed}_{k}.json"
        )
        print(selected_features_filename)
        scaler_filename = (
            self.model_dir
            / "scaler"
            / f"{strategy}_{classification_type}_ws{self.window_size}_ss{self.step_size}_majority{self.for_majority}_preproc{self.preprocessing}_randomseed{self.random_seed}_{k}.joblib"
        )
        # Shuffle data
        X, y = shuffle(X, y, random_state=self.random_seed)

        # Perform feature selection if enabled
        selected_feature_names = None
        if feature_selection:
            if strategy == "FeatureBased":
                selected_features = self.perform_feature_selection_mrmr(
                    X=X,
                    y=y,
                    k=k,
                    selected_features_filename=selected_features_filename,
                )
            X = X.loc[:, selected_features]
            X_val = X_val.loc[:, selected_features]
            X_test = X_test.loc[:, selected_features]
        if scaling_nn:
            (
                X_nn,
                X_val_nn,
                X_test_nn,
                X_nn_desc,
                X_nn_desc,
                X_val_nn_desc,
                X_test_nn_desc,
            ) = self.save_and_apply_scaler(
                X=X, X_val=X_val, X_test=X_test, scaler_filename=scaler_filename
            )
        else:
            X_nn = X
            X_val_nn = X_val
            X_test_nn = X_test
            scaler_filename = None

        
        train_loader_nn, val_loader_nn, test_loader_nn, input_size = (
            self.prepare_tensors_and_loaders(
                X_nn=X_nn,
                y=y,
                X_val_nn=X_val_nn,
                y_val=y_val,
                X_test_nn=X_test_nn,
                y_test=y_test,
                batch_size=batch_size,
                device=device,
            )
        )

        return (
            X,
            y,
            X_val,
            y_val,
            X_test,
            y_test,
            train_loader_nn,
            val_loader_nn,
            test_loader_nn,
            input_size,
            selected_feature_names,
            scaler_filename,
            X_nn_desc,
            X_val_nn_desc,
            X_test_nn_desc,
        )

    def shap_summary_TabPFN(self, model_path, X, model_directory, classification_type):
        """
        Compute SHAP values using TabPFN model and save them.

        Args:
            model_path (str): Path to the TabPFN model file.
            X (DataFrame): Input features for which SHAP values are computed.
            model_directory (Path): Directory where the model is stored.
            classification_type (str): Type of classification (e.g., 'binary', 'multiclass').

        Returns:
            None
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file for {model_path} not found.")
        with open(model_path, "rb") as f:
            model = pickle.load(f)

        # Fit SHAP PermutationExplainer
        print("Fitting SHAP PermutationExplainer...")
        explainer = shap.PermutationExplainer(model.predict_proba, X)
        shap_values = explainer(X)
        save_path = (
            model_directory
            / "tabpfn_explanations"
            / f"shap_values_{classification_type}.pkl"
        )
        save_path.parent.mkdir(parents=True, exist_ok=True)
        #  Save SHAP values
        with open(save_path, "wb") as f:
            pickle.dump(shap_values, f)

    def plot_shap_beeswarm(
        self,
        training_data,
        feature_filename,
        scaler_filename,
        model_path=None,
        shap_values_filename=None,
        max_display=7,
    ):
        """
        Plot SHAP values using a beeswarm plot.

        Args:
            training_data (DataFrame): DataFrame containing the training data.
            feature_filename (str): Path to the JSON file with selected features.
            scaler_filename (str): Path to the scaler file.
            model_path (str, optional): Path to the model file. If None, shap_values_filename must be provided.
            shap_values_filename (str, optional): Path to the SHAP values file. If None, model_path must be provided.
            max_display (int, optional): Maximum number of features to display in the plot.

        Returns:
            None
        """
        if (model_path is None) == (shap_values_filename is None):
            raise ValueError(
                "Provide exactly one of model_path or shap_values_filename."
            )

        # --- Load features, scaler, transform ---
        with open(feature_filename, "r") as f:
            selected = json.load(f)
        missing = [c for c in selected if c not in training_data.columns]
        if missing:
            raise KeyError(
                f"Missing features: {missing[:10]}{'...' if len(missing)>10 else ''}"
            )
        scaler = joblib.load(scaler_filename)
        Xraw = training_data[selected]
        if hasattr(scaler, "feature_names_in_") and list(
            scaler.feature_names_in_
        ) != list(Xraw.columns):
            raise ValueError("Scaler feature order mismatch.")
        X = pd.DataFrame(scaler.transform(Xraw), columns=Xraw.columns, index=Xraw.index)

        feat_names = [self.feature_name_map.get(c, c) for c in X.columns]

        # --- Get shap values (CatBoost vs TabPFN) ---
        if shap_values_filename is None:
            print("Get Shap Values for CatBoost")
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
            with open(model_path, "rb") as f:
                model = pickle.load(f)
            sv = shap.TreeExplainer(model).shap_values(X)
            if isinstance(sv, list):  # choose positive class if available
                sv = sv[1] if len(sv) > 1 else sv[0]
            sv = np.asarray(sv)
        else:
            print("Get Shap Values for TabPFN")
            with open(shap_values_filename, "rb") as f:
                stored = pickle.load(f)
            sv = getattr(stored, "values", stored)
            sv = np.asarray(sv)
            if sv.ndim == 3:  # (n, d, n_classes) -> pick class 1 if exists else 0
                sv = sv[:, :, 1 if sv.shape[-1] > 1 else 0]

        if sv.shape != X.shape:
            raise ValueError(f"SHAP/X shape mismatch: {sv.shape} vs {X.shape}")

        print("SHAP shape:", sv.shape)

        # --- Fonts & rc (compact, with fallback if Arial missing) ---
        try:
            fm.findfont("Arial", fallback_to_default=False)
            sans = ["Arial"]
        except Exception:
            print("[warn] Arial not found; using default sans-serif.")
            sans = ["DejaVu Sans", "Liberation Sans"]

        rc = {
            "font.family": "sans-serif",
            "font.sans-serif": sans,
            "font.size": 20,
            "axes.titlesize": 20,
            "axes.labelsize": 20,
            "xtick.labelsize": 18,
            "ytick.labelsize": 20,
            "legend.fontsize": 20,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }

        with plt.rc_context(rc):
            shap.summary_plot(
                sv, X, feature_names=feat_names, show=False, max_display=max_display
            )
            fig, ax = plt.gcf(), plt.gca()

            # Try to load Arial, fallback if missing
            try:
                fm.findfont("Arial", fallback_to_default=False)
                arial_font = fm.FontProperties(family="Arial")
            except Exception:
                print("[warn] Arial not found; using default sans-serif.")
                arial_font = fm.FontProperties(family="sans-serif")

            # Apply to ALL text
            for text in fig.findobj(match=mpl.text.Text):
                text.set_fontsize(20)
                text.set_fontproperties(arial_font)

                vmax = float(np.max(np.abs(sv))) or 1.0

            if shap_values_filename is not None:  # PermutationExplainer branch
                # adaptive limit, but fixed ticks for probability scale
                ax.set_xlim(-max(0.5, vmax), max(0.5, vmax))
                ticks = [-0.5, 0, 0.5]
            else:  # TreeExplainer branch
                # adaptive limit, but fixed ticks for log-odds scale
                ax.set_xlim(-max(1.5, vmax), max(1.5, vmax))
                ticks = [-1.5, 0, 1.5]

            ax.set_xticks(ticks)
            ax.set_xticklabels([f"{t:.1f}" for t in ticks])

            fig.tight_layout()
            plt.show()



