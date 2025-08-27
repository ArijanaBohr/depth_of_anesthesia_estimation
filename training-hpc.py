from src.models.tabularNN import TabularNN
from tabpfn import TabPFNClassifier
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import LocalEntryNotFoundError
import os
import argparse
from sklearn.naive_bayes import GaussianNB
import numpy as np
from src.dataset.eeg_dataset import EEGDataset
import torch
from sklearn.ensemble import RandomForestClassifier
import random
from sklearn.model_selection import StratifiedKFold
import pandas as pd
from catboost import CatBoostClassifier
from tabicl import TabICLClassifier
import json
from pathlib import Path
from src.utils import Utils
import os

os.environ["TABPFN_ALLOW_CPU_LARGE_DATASET"] = "1"


class TrainingHPC:
    def __init__(self, window_size, step_size, number_of_features):
        self.window_size = window_size
        self.step_size = step_size
        self.random_seed = 42
        self.strategy = "FeatureBased"
        print(f"Initialized with window_size={window_size},step_size={step_size}")
        self.preprocessing = True
        self.feature_selection = True
        self.depth_of_anesthesia = True
        self.for_majority = int(self.window_size / 2)
        self.sampling_rate = 128
        self.number_of_features = number_of_features
        self.base_path = Path.cwd()
        print(self.base_path)
        self.ml_models_path = Path("ml_models")
        self.data_path = self.base_path / "EEG_data" / "dataset"
        self.get_data()
        self.training_data = pd.read_csv(
            self.data_path / f"training_data_{self.window_size}_{self.step_size}.csv"
        )
        self.test_data = pd.read_csv(
            self.data_path / f"test_data_{self.window_size}_{self.step_size}.csv"
        )
        self.validation_data = pd.read_csv(
            self.data_path / f"validation_data_{self.window_size}_{self.step_size}.csv"
        )
        self.utils = Utils(
            for_majority=self.for_majority,
            window_size=self.window_size,
            step_size=self.step_size,
            random_seed=self.random_seed,
            preprocessing=self.preprocessing,
            sampling_rate=self.sampling_rate,
            results_validation_csv_path=self.base_path
            / "doA_classification"
            / self.ml_models_path
            / f"final_{self.strategy}_validation_results_df.csv",
            results_test_csv_path=self.base_path
            / "doA_classification"
            / self.ml_models_path
            / f"final_{self.strategy}_test_results_df.csv",
            model_dir=self.base_path / "doA_classification" / self.ml_models_path,
        )

    def utils_stuff(self):

        exclude_columns = ["Start", "End", "sleep"]
        labels = ["sleep"]
        if self.depth_of_anesthesia:
            exclude_columns.extend(["cr", "sspl", "burst_suppression"])
            labels_to_process = ["sleep", "cr", "sspl", "burst_suppression"]
            labels.extend(["cr", "sspl", "burst_suppression"])
        else:
            labels_to_process = ["sleep"]

        # Define features (excluding the necessary columns)
        features = self.training_data.drop(
            columns=exclude_columns, errors="ignore"
        ).columns

        # Create a new dictionary to store preprocessed data
        preprocessed_data_dict = {}

        for label in labels_to_process:
            print(f"Processing {label}...")
            # Preprocess data
            (
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
                X_nn,
                X_val_nn,
                X_test_nn,
            ) = self.utils.preprocess_data(
                X=self.training_data[features],
                y=self.training_data[label],
                X_val=self.validation_data[features],
                y_val=self.validation_data[label],
                X_test=self.test_data[features],
                y_test=self.test_data[label],
                k=self.number_of_features,
                batch_size=16,
                device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
                strategy=self.strategy,
                classification_type=label,
                feature_selection=True,
                imbalanced=True,
            )

            preprocessed_data_dict[label] = {
                "X": X,
                "y": y,
                "X_val": X_val,
                "y_val": y_val,
                "X_test": X_test,
                "y_test": y_test,
                "train_loader_nn": train_loader_nn,
                "val_loader_nn": val_loader_nn,
                "test_loader_nn": test_loader_nn,
                "input_size": input_size,
                "selected_feature_names": selected_feature_names,
                "scaler_filename": scaler_filename,
                "scaled X": X_nn,
                "scaled X_val": X_val_nn,
                "scaled X_test": X_test_nn,
            }

        print("Processing completed for all labels.")
        return preprocessed_data_dict

    def get_data(self):
        eeg_data = EEGDataset(
            data_dir=self.base_path / "EEG_data",
            training_ids=[1, 2, 3, 4, 6, 7, 12],
            validation_ids=[11, 10, 9],
            testing_ids=[8, 5],
            window_size=self.window_size,
            step_size=self.step_size,
            sampling_rate=self.sampling_rate,
            majority_voting=True,
            for_majority=self.for_majority,
            preprocessing=self.preprocessing,
            inference_mode=False,
            depth_of_anesthesia=self.depth_of_anesthesia,
            strategy=self.strategy,
        )
        training_data = eeg_data.train_df
        test_data = eeg_data.test_df
        validation_data = eeg_data.val_df
        training_data.to_csv(
            self.data_path / f"training_data_{self.window_size}_{self.step_size}.csv",
            index=False,
        )
        print("Training data was saved to", self.data_path)
        test_data.to_csv(
            self.data_path / f"test_data_{self.window_size}_{self.step_size}.csv",
            index=False,
        )
        print("Test data was saved to", self.data_path)
        validation_data.to_csv(
            self.data_path / f"validation_data_{self.window_size}_{self.step_size}.csv",
            index=False,
        )
        print("Validation data was saved to", self.data_path)

    def set_seed(self, seed=42):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def train_test(self, preprocessed_data_dict):
        self.set_seed(self.random_seed)
        input_size = preprocessed_data_dict["sleep"]["input_size"]
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        models = {
            "TabICLClassifier": TabICLClassifier(random_state=42, device=device),
            "CatBoostClassifier": CatBoostClassifier(
                verbose=0, random_state=self.random_seed
            ),
            "RandomForestClassifier": RandomForestClassifier(
                random_state=self.random_seed
            ),
            "GaussianNB": GaussianNB(),
            "TabPFNClassifier": TabPFNClassifier(
                random_state=42, device=device, ignore_pretraining_limits=True
            ),
            "TabularNN": TabularNN(
                input_size=input_size,
                hidden_sizes=[input_size, 32, 16],
                dropout_rate=0.4,
            ).to(device),
        }
        for label in ["sleep", "cr", "sspl", "burst_suppression"]:
            for model_name, model in models.items():

                self.utils.train_and_evaluate_model(
                    name=model_name,
                    model=model,
                    train_loader_nn=preprocessed_data_dict[label]["train_loader_nn"],
                    val_loader_nn=preprocessed_data_dict[label]["val_loader_nn"],
                    X=preprocessed_data_dict[label]["X"],
                    y=preprocessed_data_dict[label]["y"],
                    X_val=preprocessed_data_dict[label]["X_val"],
                    y_val=preprocessed_data_dict[label]["y_val"],
                    skf=StratifiedKFold(
                        n_splits=3, shuffle=True, random_state=self.random_seed
                    ),
                    classification_type=label,
                    strategy=self.strategy,
                    number_of_features=self.number_of_features,
                    device=device,
                )
                self.utils.evaluate_model_on_test(
                    model_name=model_name,
                    model=model,
                    X_test=preprocessed_data_dict[label]["X_test"],
                    y_test=preprocessed_data_dict[label]["y_test"],
                    test_loader=preprocessed_data_dict[label]["test_loader_nn"],
                    classification_type=label,
                    strategy=self.strategy,
                    number_of_features=self.number_of_features,
                    device=device,
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--window_size", type=int, default=128 * 60, help="Window_size")
    parser.add_argument("--step_size", type=int, default=128 * 60, help="Step_size")
    parser.add_argument("--feature_count", type=int, default=50, help="Feature Count")
    args = parser.parse_args()
    model = TrainingHPC(
        window_size=args.window_size,
        step_size=args.step_size,
        number_of_features=args.feature_count,
    )
    preprocessed_data_dict = model.utils_stuff()
    model.train_test(preprocessed_data_dict=preprocessed_data_dict)
