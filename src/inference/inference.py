import pandas as pd
import joblib
import torch
import numpy as np
import random
import os
import json
import pickle
from matplotlib import rcParams
import time
import matplotlib as mpl
from tabpfn import TabPFNClassifier
from sklearn.naive_bayes import GaussianNB
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from tabicl import TabICLClassifier
from scipy.signal import welch, spectrogram
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from torch.utils.data import DataLoader, TensorDataset
from mpl_toolkits.axes_grid1 import make_axes_locatable
from src.models.tabularNN import TabularNN
from src.dataset.eeg_window import EEG_window
from src.dataset.preprocessor_window import Preprocessor_window
from src.models.LSTM import LSTMModel
from src.models.UCR import SimpleModel, FCN, UCRResNet
from matplotlib.collections import LineCollection
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import seaborn as sns

class EEGInference:
    def __init__(
        self,
        inference_mode,
        sampling_rate=None,
        window_size=None,
        step_size=None,
        for_majority=None,
        preprocessing=True,
        majority_voting=True,
        strategy="FeatureBased",
        random_seed=None,
        feature_selection=True,
        scaler_applicable=True,
        device=None,
        depth_of_anesthesia=None,
        visualize_results=True,
        csv_file=None,
        csv_files_groundtruth=None,
        model_filenames=None,
        model_names=None,
        selected_features_filename=None,
        scaler_filenames=None,
        propofol_concentration_blood_plasma=None,
        zoomed_in=False,
    ):

        self.sampling_rate = sampling_rate
        self.window_size = window_size
        self.step_size = step_size
        self.for_majority = for_majority
        self.preprocessing = preprocessing
        self.majority_voting = majority_voting
        self.strategy = strategy
        self.random_seed = random_seed
        self.feature_selection = feature_selection
        self.scaler_applicable = scaler_applicable
        self.device = device
        self.inference_mode = inference_mode
        self.depth_of_anesthesia = depth_of_anesthesia
        self.visualize_results = visualize_results
        self.csv_file = csv_file
        self.csv_files_groundtruth = csv_files_groundtruth
        self.model_filenames = model_filenames
        self.model_names = model_names
        self.selected_features_filename = selected_features_filename
        self.scaler_filenames = scaler_filenames
        self.propofol_concentration_blood_plasma = propofol_concentration_blood_plasma
        self.zoomed_in = zoomed_in
        self.set_seed()

        if self.depth_of_anesthesia:

            self.classification_types = ["burst_suppression", "cr", "sspl", "sleep"]

        else:

            self.classification_types = ["sleep"]

        self.window = EEG_window(
            sampling_rate=self.sampling_rate,
            window_size=self.window_size,
            step_size=self.step_size,
            majority_voting=self.majority_voting,
            for_majority=self.for_majority,
            preprocessing=self.preprocessing,
            inference_mode=self.inference_mode,
            stft_window_size=self.window_size,
            depth_of_anesthesia=self.depth_of_anesthesia,
            strategy=self.strategy,
        )
        self.prediction, self.features_processed = self.run_inference()

    def run_inference(self):
        """Run the inference process.
        Returns:
            predicted_eeg_window (pd.DataFrame): DataFrame containing the predicted EEG window metrics.
            features_processed (dict): Dictionary containing processed features for each classification type.
        """
        if self.inference_mode:
            print("Inference Mode")
            data = pd.read_csv(self.csv_file)
        else:
            data = self.ground_truth()
        # Preprocess the data
        start = time.perf_counter()
        (
            input_sizes,
            test_loaders,
            feature_sets,
            eeg_window_metrics,
            features_processed,
        ) = self.preparing_data(data=data)
        end = time.perf_counter()
        print(f"Finished Data Preprocessing: Execution time: {end - start:.4f} seconds")

        predicted_eeg_window = eeg_window_metrics.copy(deep=True)
        # Prepare the models
        start = time.perf_counter()
        for classification_type in self.classification_types:
            model = self.get_model(
                model_name=self.model_names[classification_type],
                input_size=input_sizes[classification_type],
            )
            if self.scaler_applicable:

                features = features_processed[classification_type]
            else:
                features = feature_sets[classification_type]

            predictions_binary, predictions_probability = self.run_model(
                    test_loader=test_loaders[classification_type],
                    model_filename=self.model_filenames[classification_type],
                    model=model,
                    model_name=self.model_names[classification_type],
                    features=features,
                )

            predicted_eeg_window[f"Probability_predicted_for_{classification_type}"] = (
                predictions_probability
            )
            predicted_eeg_window[f"Predicted_{classification_type}"] = (
                predictions_binary
            )

            data["Processed EEG"] = np.nan
            preprocessor = Preprocessor_window(fs=self.sampling_rate)

            for _, row in predicted_eeg_window.iterrows():
                start_sample = int(row["Start"])
                end_sample = int(row["End"])  # exclusive, as per Python indexing

                # Slice directly by row index (not by time)
                segment = data.iloc[start_sample : end_sample + 1][
                    "EEG Voltage (µV)"
                ].values
                processed = preprocessor.preprocessing_window(segment)
                data.iloc[
                    start_sample : end_sample + 1, data.columns.get_loc("Processed EEG")
                ] = processed
        end = time.perf_counter()
        print(f"Model Predictions: Execution time: {end - start:.4f} seconds")

        if self.visualize_results:
            self.plot_results(
                predicted_data_frame=predicted_eeg_window, original_dataframe=data
            )

        return predicted_eeg_window, features_processed

    def set_seed(self):
        """Set the random seed for reproducibility.
        Args:
            seed (int): The seed value to be set for random number generation.
        """
        torch.manual_seed(self.random_seed)
        torch.cuda.manual_seed(self.random_seed)
        torch.cuda.manual_seed_all(self.random_seed)
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def preparing_data(self, data, eeg_window_metrics=None):
        """
        Prepare the data for inference.
        :param data: The EEG data to be processed.
        :param eeg_window_metrics: Precomputed EEG window metrics, if available.
        :return: A tuple containing input sizes, test loaders, feature sets, and EEG window metrics.
        """
        if eeg_window_metrics is None:
            eeg_window_metrics = self.window.window_metrics(
                total_samples=len(data),
                merged_data=data,
            )
        feature_sets = {}
        input_sizes = {}
        test_loaders = {}
        features_processed = {}

        for classification_type in self.classification_types:
            if self.feature_selection:
                feature_sets[classification_type] = self.get_selected_features(
                    classification_type=classification_type,
                    eeg_window_metrics=eeg_window_metrics,
                )

            else:
                if self.inference_mode:
                    remove_features = ["Start", "End"]
                else:

                    remove_features = [
                        "Start",
                        "End",
                        "sleep",
                        "cr",
                        "sspl",
                        "burst_suppression",
                    ]

                feature_sets[classification_type] = eeg_window_metrics.drop(
                    columns=remove_features
                )
            if self.strategy == "FeatureBased":
                input_size, test_loader, feature_processed = self.get_scaler(
                    features=feature_sets[classification_type],
                    scaler_filename=self.scaler_filenames[classification_type],
                )
                input_sizes[classification_type] = input_size
                test_loaders[classification_type] = test_loader
                features_processed[classification_type] = feature_processed
            elif self.strategy == "rawEEG" or self.strategy == "stftEEG":
                if self.strategy == "rawEEG":
                    X_test = np.vstack(eeg_window_metrics["eeg_window"].values).astype(
                        np.float32
                    )
                    X_test_tensor = torch.from_numpy(X_test).to(self.device)
                    test_dataset = TensorDataset(X_test_tensor)
                    test_loader = DataLoader(test_dataset, batch_size=16)
                    input_size = 1
                elif self.strategy == "stftEEG":
                    X_test = np.expand_dims(
                        np.array(eeg_window_metrics["stft_window"].tolist()), axis=1
                    ).astype(np.float32)
                    X_test_tensor = torch.from_numpy(X_test).to(self.device)
                    test_dataset = TensorDataset(X_test_tensor)
                    test_loader = DataLoader(test_dataset, batch_size=16)
                    input_size = X_test.shape[2]
                input_sizes[classification_type] = input_size
                test_loaders[classification_type] = test_loader
        return (
            input_sizes,
            test_loaders,
            feature_sets,
            eeg_window_metrics,
            features_processed,
        )

    def get_model(self, model_name, input_size):
        """
        Get the model based on the model name and input size.
        :param model_name: The name of the model to be used.
        :param input_size: The size of the input features.
        :param classification_type: The type of classification
        :return: The model instance.
        """
        if self.strategy == "FeatureBased":
            model_dict = {
                "GaussianNB": GaussianNB(),
                "TabICLClassifier": TabICLClassifier(random_state=self.random_seed),
                "RandomForestClassifier": RandomForestClassifier(
                    random_state=self.random_seed
                ),
                "CatBoostClassifier": CatBoostClassifier(
                    verbose=0, random_state=self.random_seed
                ),
                "TabularNN": TabularNN(
                    input_size=input_size,
                    hidden_sizes=[input_size, 32, 16],
                    dropout_rate=0.4,
                ).to(self.device),
                "TabPFNClassifier": TabPFNClassifier(random_state=self.random_seed),
            }
        else:
            model_dict = {
                "FCN": FCN(input_shape=input_size, num_classes=1).to(self.device),
                "UCRResNet": UCRResNet(
                    input_shape=input_size, n_feature_maps=64, nb_classes=1
                ).to(self.device),
                "LSTMModel": LSTMModel(
                    input_size=input_size,
                    hidden_size=256,
                    num_layers=2,
                    output_size=1,
                    dropout_rnn=0.3,
                    dropout=0.1,
                    bidirectional=False,
                ).to(self.device),
            }

        if model_name not in model_dict:
            raise ValueError(f"Model {model_name} is not supported.")

        return model_dict[model_name]

    def run_model(
        self,
        model,
        test_loader,
        model_filename,
        features,
        model_name,
    ):
        """
        Run the model for inference.
        :param model: The model to be used for inference.
        :param test_loader: The DataLoader for the test data.
        :param model_filename: The filename of the model.
        :param features: The features to be used for inference.
        :param model_name: The name of the model.

        :return: The predicted binary labels and probabilities.
        """
        is_torch_model = model_name in [
            "TabularNN",
            "UCRResNet",
            "LSTMModel",
            "FCN",
        ]
        is_pickle_model = model_name in [
            "RandomForestClassifier",
            "CatBoostClassifier",
            "TabICLClassifier",
            "TabPFNClassifier",
            "GaussianNB",
        ]

        print(f"Model name: {model_name}")
        print(f"Model file: {model_filename}")
        print(is_torch_model)
        print(is_pickle_model)
        y_pred_binary, y_pred = [], []

        if is_torch_model:
            if not model_filename.suffix == ".pt":
                raise ValueError(
                    f"Expected a .pt file for torch model, got: {model_filename}"
                )

            print(f"{model_name} (PyTorch model) detected")

            model.load_state_dict(torch.load(model_filename, map_location=self.device))
            model.to(self.device)
            model.eval()

            activation_fn = lambda x: (x.squeeze() > 0.5).float()

            with torch.no_grad():
                for (X_batch,) in tqdm(test_loader):
                    outputs = model(X_batch.to(self.device))
                    preds = activation_fn(outputs)
                    preds_np = preds.cpu().numpy()
                    outputs_np = outputs.squeeze().cpu().numpy()
                    if preds_np.ndim == 0:
                        y_pred_binary.append(preds_np.item())
                        y_pred.append(outputs_np.item())
                    else:
                        y_pred_binary.extend(preds_np.tolist())
                        y_pred.extend(outputs_np.tolist())

        elif is_pickle_model:
            print(model_filename)
            print(f"{model_name} (non-Torch model) detected")

            if not os.path.exists(model_filename):
                raise FileNotFoundError(
                    f"Model file for {model_name} not found at {model_filename}"
                )

            with open(model_filename, "rb") as f:
                model = pickle.load(f)

            X_input = np.nan_to_num(features, nan=0)
            start = time.perf_counter()

            if hasattr(model, "predict_proba"):
                y_pred = model.predict_proba(X_input)[:, 1]
                print("predict proba was used")
            elif hasattr(model, "decision_function"):
                y_pred = model.decision_function(X_input)
            else:
                y_pred = model.predict(X_input)  # fallback

            y_pred_binary = (y_pred > 0.5).astype(int)  # Apply thresholding
            end = time.perf_counter()
            print(f"Inference time: {end - start:.4f} seconds")

        else:
            raise ValueError(f"Unknown model type: {model_name}")

        return y_pred_binary, y_pred

    def get_selected_features(self, eeg_window_metrics, classification_type):
        """
        Perform feature selection based on the specified strategy.
        :param window: The EEG data window.
        :param classification_type: The type of classification.
        :return: The selected features.
        """

        with open(self.selected_features_filename[classification_type], "r") as f:
            print(classification_type)
            selected_feature_names = json.load(f)
            print(selected_feature_names)
            features = eeg_window_metrics[selected_feature_names]
        return features

    def get_scaler(self, features, scaler_filename):
        """
        Load the scaler and transform the features.
        :param classification_type: The type of classification.
        :param features: The features to be transformed.
        :return: The input size and the test data loader."""
        # Load the scaler
        scaler = joblib.load(scaler_filename)
        print(f"Scaler loaded from {scaler_filename}")
        # Transform the features using the loaded scaler
        X_test_nn_desc = pd.DataFrame(
            scaler.transform(features), columns=features.columns, index=features.index
        )
        features = scaler.transform(features)

        X_test = features.astype(np.float32)
        X_test_tensor = torch.from_numpy(X_test).to(self.device)
        test_dataset = TensorDataset(X_test_tensor)
        test_loader = DataLoader(test_dataset, batch_size=16)
        input_size = X_test.shape[1]
        return input_size, test_loader, X_test_nn_desc

    def ground_truth(self):
        merged_session = None
        if os.path.exists(self.csv_file) and os.path.exists(
            self.csv_files_groundtruth["sleep"]
        ):
            print("files exist")
            eeg_data = pd.read_csv(self.csv_file).rename(
                columns={"annotations": "sleep"}
            )
            sleep_data = pd.read_csv(self.csv_files_groundtruth["sleep"])

            merged_session = pd.merge(eeg_data, sleep_data, on="Time (s)", how="inner")
            print("merged session shape: ", merged_session.shape)
            if self.depth_of_anesthesia:
                if (
                    os.path.exists(self.csv_files_groundtruth["bs1sec"])
                    and os.path.exists(self.csv_files_groundtruth["bs3sec"])
                    and os.path.exists(self.csv_files_groundtruth["sspl"])
                    and os.path.exists(self.csv_files_groundtruth["cr"])
                ):

                    bs1sec_data = pd.read_csv(
                        self.csv_files_groundtruth["bs1sec"]
                    ).rename(columns={"annotations": "bs1sec"})
                    bs3sec_data = pd.read_csv(
                        self.csv_files_groundtruth["bs3sec"]
                    ).rename(columns={"annotations": "bs3sec"})
                    sspl_data = pd.read_csv(self.csv_files_groundtruth["sspl"]).rename(
                        columns={"annotations": "sspl"}
                    )
                    cr_data = pd.read_csv(self.csv_files_groundtruth["cr"]).rename(
                        columns={"annotations": "cr"}
                    )

                    # Apply transformations
                    bs1sec_data["bs1sec"] = bs1sec_data["bs1sec"].apply(
                        lambda x: 1 if x != 0 else 0
                    )
                    bs3sec_data["bs3sec"] = bs3sec_data["bs3sec"].apply(
                        lambda x: 1 if x != 0 else 0
                    )
                    sspl_data["sspl"] = sspl_data["sspl"].apply(
                        lambda x: 1 if x != 0 else 0
                    )
                    cr_data["cr"] = cr_data["cr"].apply(lambda x: 1 if x != 0 else 0)

                    bssec_merged = pd.merge(
                        bs1sec_data, bs3sec_data, on="Time (s)", how="inner"
                    )
                    bssec_merged["burst_suppression"] = (
                        bssec_merged["bs1sec"] | bssec_merged["bs3sec"]
                    )
                    # Keep only the timestamp and the new combined label
                    bssec_merged = bssec_merged[["Time (s)", "burst_suppression"]]
                    sspl_cr_merged = pd.merge(
                        sspl_data, cr_data, on="Time (s)", how="inner"
                    )

                    sspl_cr_bssec_merged = pd.merge(
                        sspl_cr_merged, bssec_merged, on="Time (s)", how="inner"
                    )
                    merged_session = pd.merge(
                        merged_session,
                        sspl_cr_bssec_merged,
                        on="Time (s)",
                        how="inner",
                    )
        else:
            print("csv files don't exist")
        merged_session = merged_session.rename(columns={"annotations": "sleep"})
        return merged_session

    def plot_results(self, predicted_data_frame, original_dataframe):
        """Plot the results of the inference.
        Args:
            predicted_data_frame (pd.DataFrame): DataFrame containing the predicted data.
            original_dataframe (pd.DataFrame): DataFrame containing the original EEG data.
        """

        predicted_cols = [
            col for col in predicted_data_frame.columns if col.startswith("Predicted_")
        ]
        predicted_prob_cols = [
            col
            for col in predicted_data_frame.columns
            if col.startswith("Probability_predicted_for_")
        ]
        predicted_data_frame["Time (s)"] = (
            predicted_data_frame["Start"] / self.sampling_rate
        )
        windows = [
            (1020, 1028),
            (1800, 1808),
            (8008, 8016),

        ]
        labels =None  # If you want to show the zoomed in windows, put it to labels = ["A", "B", "C"]
        actual_labels = [
                   "Awake",
                   "Moderate Anesthesia",
                   "Deep  Anesthesia",
        ]
        self.create_inference_figure(
            predicted_data_frame,
            original_dataframe,
            predicted_cols,
            predicted_prob_cols,
            windows,
            labels,
            actual_labels,
        )

    def create_spectogram(self, predicted_data_frame, original_dataframe):
        """Create a spectrogram from the EEG data in the predicted DataFrame.
        Args:
            predicted_data_frame (pd.DataFrame): DataFrame containing the predicted data.
            original_dataframe (pd.DataFrame): DataFrame containing the original EEG data.
        Returns:
            t_full (np.ndarray): Time vector for the spectrogram.
            f (np.ndarray): Frequency vector for the spectrogram.
            log_Sxx_full (np.ndarray): Logarithmic power spectral density of the spectrogram.
        """
        fs = self.sampling_rate
        nperseg = self.window_size

        # Storage for all window spectrograms
        Sxx_list = []
        t_list = []

        for _, row in predicted_data_frame.iterrows():
            start_sample = int(row["Start"])
            end_sample = int(row["End"])

            # Extract processed signal segment
            segment = (
                original_dataframe.iloc[start_sample : end_sample + 1]["Processed EEG"]
                .dropna()
                .values
            )

            if len(segment) < nperseg:
                continue  # skip too-short windows

            # Compute spectrogram for the segment
            f, t, Sxx = spectrogram(segment, fs=fs, nperseg=nperseg)

            # Align time axis to real time
            segment_start_time = original_dataframe.iloc[start_sample]["Time (s)"]
            t_aligned = t + segment_start_time

            Sxx_list.append(Sxx)
            t_list.append(t_aligned)

        # Concatenate all results
        Sxx_full = np.hstack(Sxx_list)
        t_full = np.hstack(t_list)
        log_Sxx_full = 10 * np.log10(Sxx_full + 1e-10)
        return t_full, f, log_Sxx_full

    def create_inference_figure(
        self,
        predicted_data_frame,
        original_dataframe,
        predicted_cols,
        predicted_prob_cols,
        windows,
        labels,
        actual_labels,
    ):
        """Create the inference figure with subplots for EEG trace, spectrogram, propofol concentration, and predictions.
        Args:
            predicted_data_frame (pd.DataFrame): DataFrame containing the predicted data.
            original_dataframe (pd.DataFrame): DataFrame containing the original EEG data.
            predicted_cols (list): List of columns in the predicted DataFrame that start with 'Predicted_'.
            predicted_prob_cols (list): List of columns in the predicted DataFrame that start with 'Probability_predicted_for_'.
            windows (list): List of tuples representing the start and end indices of the windows to be highlighted in the plot.
            labels (list): List of labels for the windows (A, B, C).
            actual_labels (list): List of actual labels corresponding to the windows.
        """
        SINGLE_COL_W = 16  
        FIG_H = 11     
        FS_BASE = 17 
        FS_TICK = 15
        FS_LEG  = 15

        serif_stack = ["Times New Roman", "Times", "Nimbus Roman No9 L", "DejaVu Serif"]

        try:
            fm.findfont(serif_stack[0], fallback_to_default=False)
            chosen_serif = serif_stack[0]
        except Exception:
            chosen_serif = "serif"

        sns.set_theme(
            style="whitegrid",
            font="serif",
            rc={
                "font.family": "serif",
                "font.serif": serif_stack,
                "font.size": FS_BASE,
                "axes.titlesize": FS_BASE,
                "axes.labelsize": FS_BASE,
                "xtick.labelsize": FS_TICK,
                "ytick.labelsize": FS_TICK,
                "legend.fontsize": FS_LEG,
                # Embed fonts properly for IEEE PDFs
                "pdf.fonttype": 42,
                "ps.fonttype": 42,
                "text.usetex": False,
                # Thinner spines/lines for print
                "axes.linewidth": 0.8,
            },
        )
        print(predicted_data_frame)
        fig = plt.figure(figsize=(SINGLE_COL_W, FIG_H))
        gs = gridspec.GridSpec(4, 1, height_ratios=[1, 2, 1, 4])
        plt.subplots_adjust(hspace=0.7)

        # --- Subplot 0: EEG Voltage trace ---
        ax0 = plt.subplot(gs[0])
        ax0.plot(
            original_dataframe["Time (s)"],
            original_dataframe["EEG Voltage (µV)"],
            color="#272727",
            linewidth=0.5,
        )

        ax0.set_ylabel("EEG (µV)", fontsize=20)
        ax0.set_title("EEG Voltage Trace", fontsize=20)

        # --- Subplot 1: Compute spectrogram from EEG trace --- ---
        # Parameters
        t_full, f, log_Sxx_full = self.create_spectogram(
            predicted_data_frame, original_dataframe
        )
        # Plot
        extent = [t_full[0], t_full[-1], f[0], f[-1]]
        ax1 = plt.subplot(gs[1], sharex=ax0)
        im = ax1.imshow(
            log_Sxx_full,
            aspect="auto",
            extent=extent,
            origin="lower",
            cmap="jet",
            vmin=-20,
            vmax=20,
        )
        ax1.set_ylabel("Frequency (Hz)", fontsize=20, labelpad=20)
        ax1.set_title("Spectrogram (Window-wise)", fontsize=20)
        divider = make_axes_locatable(ax1)
        cax = divider.append_axes(
            "bottom", size="10%", pad=0.3
        )  # adjust padding as needed
        cbar = plt.colorbar(im, cax=cax, orientation="horizontal")
        cbar.set_label("Power (dB)", fontsize=20)

        # --- Subplot 2: Propfol Concentrations ---
        ax_combined = plt.subplot(gs[2], sharex=ax0)

        # Plot both datasets on the same axis
        if self.propofol_concentration_blood_plasma is not None:
            ax_combined.plot(
                original_dataframe["Time (s)"],
                self.propofol_concentration_blood_plasma["annotations"],
                color="#272727",
            )
            ax_combined.set_ylabel("Concentration (µg/ml)", fontsize=20, labelpad=25)
            ax_combined.set_title("Propofol Concentration", fontsize=20)
            # Add a legend to distinguish the plots
        
        # --- Subplot 3: Discrete Predictions---
        ax3 = plt.subplot(gs[3], sharex=ax0)
        rename_map = {
            "Predicted_burst_suppression": "BuS",
            "Predicted_sleep": "CNLneg",
            "Predicted_sspl": "SSPLneg",
            "Predicted_cr": "CRneg",
            "sleep": "GT: CNLneg",
            "cr": "GT: CRneg",
            "sspl": "GT: SSPLneg",
            "burst_suppression": "GT: BuS",
            "Probability_predicted_for_burst_suppression": "BuS",
            "Probability_predicted_for_sleep": "CNLneg",
            "Probability_predicted_for_sspl": "SSPLneg",
            "Probability_predicted_for_cr": "CRneg",
        }

        colors = ["#D55E00", "#E69F00", "#009E73", "#56B4E9"]
        colors = ["#272727", "#272727", "#272727", "#272727"]

        for i, col in enumerate(predicted_cols):
            offset = i * 1.5

            # Plot model prediction (assumed continuous and time-aligned)
            values = predicted_data_frame[col] + offset

            ax3.plot(
                predicted_data_frame["Time (s)"],
                values,
                label="Classification" if i == 0 else None,  # add to legend only once
                color=colors[i % len(colors)],
            )
            if self.inference_mode:
                pass
            else:
                # Overlay ground truth (assumed discrete mask with same time base as df_preds)
                gt_col = self.classification_types[i]
                mask = predicted_data_frame[gt_col] == 1
                ax3.fill_between(
                    predicted_data_frame["Time (s)"],
                    offset - 0.3,
                    offset + 0.3,
                    where=mask,
                    color=colors[i % len(colors)],
                    alpha=0.3,
                    label="Ground Truth" if i == 0 else None,  # add to legend only once
                )
        # Update y-ticks and labels
        y_labels = [rename_map.get(col, col) for col in predicted_cols]
        yticks = [i * 1.5 for i in range(len(predicted_cols))]
        ax3.set_yticks(yticks)
        ax3.set_yticklabels(y_labels, fontsize=20)

        ax3.set_xlabel("Time (s)", fontsize=20)
        ax3.set_title(
            "Model Binary Predictions vs. Ground Truth", fontsize=20
        )
        ax3.legend(loc="upper right", ncol=2)
        ax3.grid(True)
        ax0.set_ylim(-200, 200)
       # ax0.set_ylim(-70, 25)
        ax0.set_xlim(
            original_dataframe["Time (s)"].min(),
            original_dataframe["Time (s)"].max(),
        )  # Ensure time alignment

        ax1.set_ylim(0, 47)
        ax_combined.set_ylim(0, 6.0)

        plt.draw()
        if self.zoomed_in is True:
            axes = [ax0, ax1, ax_combined, ax3]

            print(windows)
            print(labels)
            if windows is not None and labels is not None:

                self.add_zoomed_in_labels(axes, windows, actual_labels, labels)
            plt.tight_layout()
            plt.show()
        
        new_fig_height = 10 * (3 / 7)

        fig2 = plt.figure(figsize=(SINGLE_COL_W, new_fig_height))

        ax = fig2.add_subplot(111)

        predicted_data_frame = predicted_data_frame.sort_values("Time (s)").reset_index(
            drop=True
        )
        t = predicted_data_frame["Time (s)"].values

        row_gap = 1.5
        band_half = 0.30
        pad = 0.2

        all_max = float("-inf")
        all_min = float("inf")

        for i, col in enumerate(predicted_prob_cols):
            offset = i * row_gap

            values = (
                predicted_data_frame[col].astype(float).clip(0.0, 1.0).values + offset
            )
            ax.plot(
                t,
                values,
                label="Classification" if i == 0 else None,
                color=colors[i % len(colors)],
                linewidth=1.5,
                zorder=3,
            )

            # Track min/max for ylim
            if len(values) > 0:
                all_max = max(all_max, values.max())
                all_min = min(all_min, values.min())

            ax.hlines(
                offset,
                t.min(),
                t.max(),
                color="0.85",
                linestyles="--",
                linewidth=0.8,
                zorder=1,
            )
            ax.hlines(
                offset + 0.5,
                t.min(),
                t.max(),
                color="0.7",
                linestyles=":",
                linewidth=0.8,
                zorder=1,
            )

            if not self.inference_mode:
                gt_col = self.classification_types[i]
                mask = predicted_data_frame[gt_col].astype(int).values == 1
                ax.fill_between(
                    t,
                    offset - band_half,
                    offset + band_half,
                    where=mask,
                    step="post",
                    color=colors[i % len(colors)],
                    alpha=0.3,
                    label="Ground Truth" if i == 0 else None,
                    zorder=2,
                )

        # Left axis
        y_labels = [rename_map.get(col, col) for col in predicted_cols]
        yticks = [i * row_gap for i in range(len(predicted_prob_cols))]
        ax.set_yticks(yticks)
        ax.set_yticklabels(y_labels, fontsize=22)
        ax.set_xlabel("Time (s)", fontsize=22)
        ax.set_title(
            "Model Predicted Probabilities vs. Ground Truth",
            fontsize=15,
        )
        ax.legend(loc="upper right", ncol=2)
        ax.grid(True)

        # Y-limits
        top_needed = (len(predicted_prob_cols) - 1) * row_gap + 1.0
        upper = max(all_max, top_needed) + pad
        lower = min(all_min, -row_gap * 0.25) - pad
        ax.set_ylim(lower, upper)

        # Right axis with prob ticks
        ax_right = ax.twinx()
        rticks = []
        rticklabels = []
        for i in range(len(predicted_prob_cols)):
            offset = i * row_gap
            rticks.extend([offset + 0.0, offset + 0.5, offset + 1.0])
            rticklabels.extend(["0.0", "0.5", "1.0"])

        ax_right.set_ylim(ax.get_ylim())
        ax_right.set_yticks(rticks)
        ax_right.set_yticklabels(rticklabels, fontsize=15)
        ax_right.set_ylabel("Probability", fontsize=22)
        ax.margins(x=0)
        plt.tight_layout()
        plt.show()

    def add_zoomed_in_labels(self, axes, windows, actual_labels, labels):
        """
        Add zoomed-in windows to the figure.
        :param axes: List of axes to add the labels to.
        :param windows: List of tuples representing the start and end of each window.
        :param actual_labels: List of actual labels for each window.
        :param labels: List of labels to display above the windows.

        """
        SINGLE_COL_W = 13  # inches (IEEE single column ≈ 3.5")
        FIG_H = 5.5        # tune as needed to fit your max_display
        FS_BASE = 22         # axis/title 9–10 pt is typical
        FS_TICK = 22
        FS_LEG  = 22

        # Prefer a real serif TTF that embeds cleanly.
        # Times New Roman may not be available as embeddable TTF on all OSes, so fall back sensibly.
        serif_stack = ["Times New Roman", "Times", "Nimbus Roman No9 L", "DejaVu Serif"]

        try:
            # Will raise if first choice isn't found when fallback=False
            fm.findfont(serif_stack[0], fallback_to_default=False)
            chosen_serif = serif_stack[0]
        except Exception:
            # Use a stack so we stay serif even if TNR isn't present
            chosen_serif = "serif"

        sns.set_theme(
            style="whitegrid",
            font="serif",
            rc={
                "font.family": "serif",
                "font.serif": serif_stack,
                "font.size": FS_BASE,
                "axes.titlesize": FS_BASE,
                "axes.labelsize": FS_BASE,
                "xtick.labelsize": FS_TICK,
                "ytick.labelsize": FS_TICK,
                "legend.fontsize": FS_LEG,
                # Embed fonts properly for IEEE PDFs
                "pdf.fonttype": 42,
                "ps.fonttype": 42,
                "text.usetex": False,
                # Thinner spines/lines for print
                "axes.linewidth": 0.8,
            },
        )
        for start, end in windows:
            for ax in axes:
                ax.axvspan(
                    x=start,
                    color="gray",
                    linestyle="--",
                    alpha=0.5,
                    xmin=start,
                    xmax=end,
                )

        for (start, end), label in zip(windows, labels):
            xpos = (start + end) / 2
            axes[0].annotate(
                label,
                xy=(xpos, 1.02),
                xycoords=("data", "axes fraction"),
                ha="center",
                va="bottom",
                fontsize=FS_BASE,
                color="gray",
            )
        total_length_windows = len(windows)
        for i, (start, end) in enumerate(windows):

            fig = plt.figure(figsize=(SINGLE_COL_W, FIG_H))
            gs_zoom = gridspec.GridSpec(2,1)
            actual_label_section = actual_labels[i]

       #     fig.suptitle(
       #         actual_label_section,
       #         fontsize=FS_BASE + 4,
       #         x=0.01,
       #         ha="left",
       #         backgroundcolor="#f0f0f0",
       #     )  # light grey

            ax0_z = plt.subplot(gs_zoom[0])
            ax0_z.set_xlim(start, end)
            signal_zoom = []

            for line in axes[0].get_lines():
                xdata = line.get_xdata()
                ydata = line.get_ydata()
                ax0_z.plot(xdata, ydata, label=line.get_label(), color="#6E6E6E")

                mask = (xdata >= start) & (xdata <= end)
                y_zoomed = ydata[mask]

                signal_zoom.extend(y_zoomed)

            ax0_z.set_ylim(min(signal_zoom), max(signal_zoom))

            ax0_z.set_ylabel("EEG (µV)", fontsize=FS_BASE)
            ax0_z.tick_params(axis="both", labelsize=FS_BASE)
            ax0_z.grid(True)
            ax0_z.set_xlabel("Time (s)", fontsize=FS_BASE)
            ax0_z.set_title("EEG Voltage Trace", fontsize=FS_BASE)
            ax0_z.set_ylim(-130, 130)

           # ax0_z.set_ylim(-70, 25)
            

            ax1_z = plt.subplot(gs_zoom[1])

            if len(signal_zoom) > 0:
                signal_zoom = np.array(signal_zoom)

                # Compute power spectrum with adjusted nperseg
                nperseg = min(len(signal_zoom), self.sampling_rate)
                f_zoom, Pxx_zoom = welch(
                    signal_zoom, fs=self.sampling_rate, nperseg=nperseg
                )
                Pxx_dB_zoom = 10 * np.log10(Pxx_zoom + 1e-12)

                points = np.array([f_zoom, Pxx_dB_zoom]).T.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)

                norm = mcolors.Normalize(vmin=-25, vmax=30)
                color_map_custom = {
                    "1": "#D55E00",  # orange-red
                    "2": "#E69F00",  # golden
                    "3": "#009E73",  # bluish-green
                    "4": "#56B4E9",  # sky blue
                }

                # Pick red and blue
                red_custom = color_map_custom["1"]
                blue_custom = color_map_custom["4"]

                # Create a custom diverging colormap from blue -> white -> red
                mid_color = "#c4c4c4"
                cmap_custom_soft = mcolors.LinearSegmentedColormap.from_list(
                    "CustomBlueGrayRed", [blue_custom, mid_color, red_custom]
                )
                cmap = cm.get_cmap(cmap_custom_soft)

                lc = LineCollection(segments, cmap=cmap, norm=norm)
                #  lc = LineCollection(segments, color="black", linewidth=1.5)
                lc.set_array(Pxx_dB_zoom)
                lc.set_linewidth(3)
                line = ax1_z.add_collection(lc)

                ax1_z.set_xlim(0, 47)
                ax1_z.set_ylim(-25, 30)
                ax1_z.set_xlabel("Frequency (Hz)", fontsize=FS_BASE)
                ax1_z.tick_params(axis="both", labelsize=FS_BASE)
                ax1_z.set_ylabel(
                    "Power (dB)", fontsize=FS_BASE, labelpad=11
                )
                ax1_z.set_title("Power Spectra", fontsize=FS_BASE)


                if i == total_length_windows - 1:
                    divider = make_axes_locatable(ax1_z)
                    cax = divider.append_axes("bottom", size="25%", pad=0.7)
                    cbar = plt.colorbar(line, cax=cax, orientation="horizontal")
                    cbar.set_label("Power (dB)", fontsize=FS_BASE)
                    fig.canvas.draw()
            
            plt.tight_layout()
            plt.show()
