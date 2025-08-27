import os
import pandas as pd
import random
import numpy as np
from collections import defaultdict
from src.dataset.eeg_window import EEG_window

random.seed(42)
np.random.seed(42)

IDS = [1, 2, 3, 4, 5, 7, 8, 9, 10, 11]


class EEGDataset:
    def __init__(
        self,
        data_dir,
        training_ids=[1, 2, 10, 7, 4, 5],
        validation_ids=[9, 11],
        testing_ids=[8, 3],
        window_size=64,
        step_size=64,
        sampling_rate=128,
        majority_voting=True,
        for_majority=64 / 2,
        preprocessing=True,
        inference_mode=False,
        stft_window_size=32,
        stft_hop=8,
        depth_of_anesthesia=True,
        strategy=None,
    ):
        """
        Initializes the EEGDataset class.
        Parameters:
            data_dir (str): Directory containing the EEG data.
            training_ids (list): List of patient/volunteer IDs for training.
            validation_ids (list): List of patient/volunteer IDs for validation.
            testing_ids (list): List of patient/volunteer IDs for testing.
            window_size (int): Size of the window for processing EEG data.
            step_size (int): Step size for moving the window.
            sampling_rate (int): Sampling rate of the EEG data.
            majority_voting (bool): Whether to use majority voting in window processing.
            for_majority (int): Threshold for majority voting.
            preprocessing (bool): Whether to apply preprocessing to the data.
            inference_mode (bool): Whether to run in inference mode.
            stft_window_size (int): Window size for STFT processing.
            stft_hop (int): Hop size for STFT processing.
            depth_of_anesthesia (bool): Whether to include depth of anesthesia features. (and not just CNLneg)
            strategy: Strategy for processing the data.

        """
        # Initialize parameters
        self.training_ids = training_ids
        self.validation_ids = validation_ids
        self.testing_ids = testing_ids
        self.window_size = window_size
        self.step_size = step_size
        self.majority_voting = majority_voting
        self.for_majority = for_majority
        self.preprocessing = preprocessing
        self.inference_mode = inference_mode
        self.stft_window_size = stft_window_size
        self.stft_hop = stft_hop
        self.data_dir = data_dir
        self.depth_of_anesthesia = depth_of_anesthesia
        self.strategy = strategy
        self.sampling_rate = sampling_rate

        self.window = EEG_window(
            sampling_rate=self.sampling_rate,
            window_size=self.window_size,
            step_size=self.step_size,
            majority_voting=self.majority_voting,
            for_majority=self.for_majority,
            preprocessing=self.preprocessing,
            inference_mode=self.inference_mode,
            stft_window_size=self.stft_window_size,
            stft_hop=self.stft_hop,
            depth_of_anesthesia=self.depth_of_anesthesia,
            strategy=self.strategy,
        )
        self.configure_data()

    ### Load data in directories for each patient and merge eeg and annotations
    def get_eeg_annotation_data(self, type_dataset="train", dataset_splits=None):
        """ Loads EEG and annotation data for a given dataset type (train, val, test).
        Args:
            type_dataset (str): Type of dataset to load ('train', 'val', 'test').
            dataset_splits (dict): Dictionary containing dataset splits.
        Returns:
            dict: Merged EEG and annotation data for each patient ID.
        """
        # Dictionaries to store the data
        merged_data = {}
        # Loop over the IDs and load the data
        for case_path in dataset_splits[type_dataset]:

            case_parts = os.path.basename(case_path).split("_")  # Extract all parts
            case_number = f"{case_parts[1]}-{case_parts[2]}"  # Case number includes both parts (e.g., 12_1)
            session_id = case_parts[1]
            eeg_file = os.path.join(
                case_path,
                f"prop_{case_parts[1]}_{case_parts[2]}eeg_Fp1Fp2.csv",
            )
            sleep_file = os.path.join(
                case_path,
                f"prop{case_number}sleep_Fp1Fp2.csv",
            )
            cr_file = os.path.join(
                case_path,
                f"prop{case_number}cr_Fp1Fp2.csv",
            )
            bs1sec_file = os.path.join(
                case_path,
                f"prop{case_number}bs1sec_Fp1Fp2.csv",
            )
            bs3sec_file = os.path.join(
                case_path,
                f"prop{case_number}bs3sec_Fp1Fp2.csv",
            )
            sspl_file = os.path.join(
                case_path,
                f"prop{case_number}sspl_Fp1Fp2.csv",
            )

            if os.path.exists(eeg_file) and os.path.exists(sleep_file):
                eeg_data =  pd.read_csv(eeg_file).rename(
                            columns={"annotations": "sleep"}
                        )
                sleep_data = pd.read_csv(sleep_file)

                merged_session = pd.merge(
                    eeg_data, sleep_data, on="Time (s)", how="inner"
                )
        

                if self.depth_of_anesthesia:
                    if (
                        os.path.exists(bs1sec_file)
                        and os.path.exists(bs3sec_file)
                        and os.path.exists(cr_file)
                        and os.path.exists(sspl_file)
                    ):

                        bs1sec_data = pd.read_csv(bs1sec_file).rename(
                            columns={"annotations": "bs1sec"}
                        )
                        bs3sec_data = pd.read_csv(bs3sec_file).rename(
                            columns={"annotations": "bs3sec"}
                        )
                        sspl_data = pd.read_csv(sspl_file).rename(
                            columns={"annotations": "sspl"}
                        )
                        cr_data = pd.read_csv(cr_file).rename(
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
                        cr_data["cr"] = cr_data["cr"].apply(
                            lambda x: 1 if x != 0 else 0
                        )

                        bssec_merged = pd.merge(
                            bs1sec_data, bs3sec_data, on="Time (s)", how="inner"
                        )
                        bssec_merged["burst_suppression"] = (
                            bssec_merged["bs1sec"] | bssec_merged["bs3sec"]
                        )
                  
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
                        raise FileNotFoundError(eeg_file, sleep_file)
            else:
                raise FileNotFoundError(eeg_file, sleep_file)
            merged_session = merged_session.rename(columns={"annotations": "sleep"})
            # Store results in merged_data, grouping by session_id
            if session_id not in merged_data:
                merged_data[session_id] = []  # Create list if ID not seen before
            merged_session = merged_session.iloc[
                self.sampling_rate * 5 : -self.sampling_rate * 5
            ]  # Omits the first and last 5 seconds from the data to avoid potential artefacts.
            merged_data[session_id].append(merged_session)

        # Combine sessions per ID
        for session_id in merged_data:
            merged_data[session_id] = pd.concat(
                merged_data[session_id], ignore_index=True
            )  # Concatenate all sessions
            # Print column names for each DataFrame

        for key, df in merged_data.items():
            print(f"Columns in {key}: {list(df.columns)}")
        return merged_data

    ### Loop over the data and process windows for each ID
    def get_windows(self, data):
        """
        This function processes the EEG data for each patient ID, applying windowing and other transformations.
        Parameters:
            data (dict): A dictionary where each key is a patient ID, and the value is a pandas DataFrame containing the
                        combined EEG, and annotation data.
        Returns:
            pd.DataFrame: A DataFrame containing the processed windows for all patient IDs.
        """
        # Initialize an empty list to store all windows
        all_windows = []

        for id in data:
            if data[id] is not None:
                merged_data_ID = data[id]

                processed_df = self.window.window_metrics(
                    len(merged_data_ID), merged_data_ID
                )

                all_windows.append(processed_df)

                print(f"Processing complete for ID {id}.")
            else:
                print(f"No data available for ID {id}. Skipping.")

        concatenated_windows = pd.concat(all_windows, ignore_index=True)
        print(f"Total windows concatenated: {len(concatenated_windows)}")
        return concatenated_windows

    def configure_data(self):
        # Initialize data variables
        data_train = data_val = data_test = None

        # Check if IDs are empty
        if not self.training_ids:
            print("No training IDs provided.")
        if not self.validation_ids:
            print("No validation IDs provided.")
        if not self.testing_ids:
            print("No testing IDs provided.")

        # Gather all case occurrences
        patient_dict = self.gather_case_occurrences()

        # Split them into training, validation, and test sets
        dataset_splits = self.split_cases(
            patient_dict=patient_dict,
        )

        data_train = (
            self.get_eeg_annotation_data("train", dataset_splits) if self.training_ids else None
        )

        data_val = (
            self.get_eeg_annotation_data("val", dataset_splits) if self.validation_ids else None
        )

        data_test = (
            self.get_eeg_annotation_data("test", dataset_splits) if self.testing_ids else None
        )

        # Generate windowed data
        self.train_df = self.get_windows(data_train) if data_train else None
        self.val_df = self.get_windows(data_val) if data_val else None
        self.test_df = self.get_windows(data_test) if data_test else None

    def gather_case_occurrences(self):
        """
        Iterates through all sessions and gathers occurrences of each patient.

        Parameters:
        base_path_to_files (str): Root directory containing session folders.

        Returns:
        dict: A dictionary where keys are patient IDs and values are lists of session paths.
        """
        patient_dict = defaultdict(list)

        for session in sorted(os.listdir(self.data_dir)):
            session_path = os.path.join(self.data_dir, session)
            if not os.path.isdir(session_path):  # Skip non-directory files
                continue

            for case_folder in sorted(os.listdir(session_path)):
                if case_folder.startswith("Case_"):
                    parts = case_folder.split("_")
                    if len(parts) < 3:
                        continue  # Skip malformed folder names

                    patient_id = int(parts[1])  # Extract the patient ID (X)
                    patient_dict[patient_id].append(
                        os.path.join(session_path, case_folder)
                    )

        return patient_dict

    def split_cases(
        self, patient_dict
    ):
        """
        Splits patients into training, validation, and test sets, ensuring all sessions of a patient stay in one set.

        Parameters:
        patient_dict (dict): Dictionary mapping patient IDs to session paths.

        Returns:
        dict: Dictionary with keys 'train', 'val', and 'test', containing patient occurrences.
        """
        dataset_splits = {"train": [], "val": [], "test": []}

        for patient_id, paths in patient_dict.items():
            if self.training_ids and patient_id in self.training_ids:
                dataset_splits["train"].extend(paths)
            elif self.validation_ids and patient_id in self.validation_ids:
                dataset_splits["val"].extend(paths)
            elif self.testing_ids and patient_id in self.testing_ids:
                dataset_splits["test"].extend(paths)
            else:
                print(f"Warning: Patient {patient_id} not assigned to any set.")

                # Check for patient overlap across splits
        train_patients = {
            int(os.path.basename(path).split("_")[1])
            for path in dataset_splits["train"]
        }
        val_patients = {
            int(os.path.basename(path).split("_")[1]) for path in dataset_splits["val"]
        }
        test_patients = {
            int(os.path.basename(path).split("_")[1]) for path in dataset_splits["test"]
        }

        print("Train & Val Overlap:", train_patients & val_patients)
        print("Train & Test Overlap:", train_patients & test_patients)
        print("Val & Test Overlap:", val_patients & test_patients)

        return dataset_splits
