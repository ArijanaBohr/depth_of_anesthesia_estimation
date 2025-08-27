#  Depth of Anesthesia Classification in Single-Channel EEG: Analysis of Machine Learning Approaches and Future Perspectives by Anonymized Authors

Estimation of the depth of anesthesia with the help of the parameters of 
CNLneg, CRneg, SSPLneg and BS (details can be found in the associated paper).

This projects trains binary classifiers of CNLneg/SSPLneg/CRneg/BS. 

(in our code the variables for CNLneg/SSPlneg/CRneg/BS are sleep/sspl/cr/burst_suppression)
# Step-by Step Tutorial:
## ⚙️ Installation

This project uses [Conda](https://docs.conda.io/) for environment management.  
The environment can be set up in one of two ways:
### Option 1: Using `environment.yml`

conda eeg_env create -f environment.yml
conda activate eeg_env 

### Option 2: Using `conda_requirements.txt`

conda create -n eeg_env python=3.9
conda activate eeg_env
conda install --file conda_requirements.txt

## ▶️ Quick Start

Once the environment is set up, the main results from the paper with the provided scripts can be reproduced.

### 1. Prepare data
Place your data in the `EEG_data/` folder.  
If using a custom path, update the config or script arguments accordingly.

The data should be in following format:
### 📂 Data Format

The data is stored in CSV files with the following columns:

> 📂 **Note on file formats**:  
> The code assumes input data in **CSV** format.  

The EEG file looks like this: 
<p align="center">
<pre>
Time (s),EEG Voltage (µV)
0.0,-32.259675
0.007812505006167599,-29.858954999999998
</pre>
</p>

> ⚠️ **Note on sampling rate**:  
> Our code assumes the data is sampled at **128 Hz** (the sampling rate used in the paper).  
> To avoid potential errors or misalignment, we recommend using the same sampling rate.  


If you have ground truth files for CNLneg, SSPLneg, CRneg and BS, you can add them also, in the following format.:

- **Time (s):** time in seconds (float)
- **annotations:** label for each time point (integer, e.g., `0` or `1`). `1` means that the class is active. 
Example: 
<p align="center">
<pre>
Time (s),annotations
0.0,0
0.007812505006167599,0
0.015625010012335197,1
</pre>
</p>

A file tree for the data would look like this:
<pre>
``` 
├── EEG_data/
│   ├── Session1/
│   │   ├── Case_1_1
│   │   │    ├── prop_1_1eeg_Fp1Fp2.csv
│   │   │    ├── prop1_1sleep_Fp1Fp2.csv
│   │   │    ├── prop1_1cr_Fp1Fp2.csv
│   │   │    ├── prop1_1sspl_Fp1Fp2.csv
│   │   │    ├── prop1_1burst_suppression_Fp1Fp2.csv
│   │   ├── Case_2_1
│   │   │    ├── prop_2_1eeg_Fp1Fp2.csv
│   │   │    ├── ...
│   ├── Session2/
│   │   ├── Case_1_2
│   │   ├── ...

</pre>

If you have just one session, then put all your volunteers/patients into Session1.
If patients went through multiple sessions, you can put them in the respective sessions, as long as you number them for example like 1_i, where i is the respective session the patient/volunteer went through.
This ensures patients/volunteers are exclusively in the training/test or validation set
### 2. Train the model

The training/validation/testing ids signify which patients/volunteers are assigned to which set. Patients/Volunteers should only be exclusively assigend to one set (no overlap!).

### Option 1: Feature-based Approach:
Run the notebook [training.ipynb](training.ipynb) (The notebook demonstrates how to run it step-by-step)

If you prefer to run it on a high-performance computing system, we recommend using the notebook [training-hpc.py](training-hpc.py). To execute the corresponding script, run in the terminal: `python training-hpc.py`
 

> ⚡ **GPU recommended**:  
> We suggest running the code on a CUDA-enabled GPU for best performance.  
> 
> - If you want to use **TabICL** and **TabPFN**, a GPU is required (or very highly suggested).  
> - If you are running on CPU only, we recommend commenting out the TabICL/TabPFN parts in the script ([training-hpc.py](training-hpc.py)).

### Option 2: Training on the STFT transformed Signal:
Run the notebook [training_stft.ipynb](training_stft.ipynb) (The notebook demonstrates how to run it step-by-step)
### Option 3: Deep Learning on the EEG signal:
Run the notebook [training_without_features.ipynb](training_without_features.ipynb) (The notebook demonstrates how to run it step-by-step)

The trained models will be then found in:

<pre>

├── doA_classification/
│   │   ├── scaler/
│   │   │   ├── FeatureBased_sleep_ws7680_ss7680_majority3840_preprocTrue_randomseed42_numfeatures50.joblib
│   │   │   ├──...
│   │   ├── selected_features/
│   │   │   ├── sleep_FeatureBased_ws7680_ss7680_majority3840_preprocTrue_randomseed42_numfeatures50.json
│   │   │   ├──...
│   │   ├── ml_models/
│   │   │   ├── FeatureBased_validation_results_df.csv 
│   │   │   ├── FeatureBased_test_results_df.csv
│   │   │   ├── FeatureBased_sspl_ws7680_ss7680_majority3840_typeCatBoostClassifier_preprocTrue_randomseed42_50.pkl
│   │   │   ├── FeatureBased_burst_suppression_ws7680_ss7680_majority3840_typeCatBoostClassifier_preprocTrue_randomseed42_50.pkl
│   │   │   ├── FeatureBased_sleep_ws7680_ss7680_majority3840_typeCatBoostClassifier_preprocTrue_randomseed42_50.pkl
│   │   │   ├── FeatureBased_cr_ws7680_ss7680_majority3840_typeCatBoostClassifier_preprocTrue_randomseed42_50.pkl
</pre>

### 3. Run inference
For inference, simply open and run the notebook:

[inference.ipynb](inference.ipynb)

The notebook demonstrates how to load the trained models and run predictions on new data step by step.


## File Tree

```
├── ./
│   ├── training.ipynb          # Training notebook based on the computed features
│   ├── training-hpc.py         # Training script using computed features, equivalent to training.ipynb, but structured as a Python class
│   ├── eeg_inference.ipynb     # Notebook to run the inference
│   ├── README.md               # Project documentation (this file)
│   ├── training_stft.ipynb     # Training notebook for models run on STFT
│   ├── cnn_interpretability.ipynb      #GradCam
│   ├── training_without_features.ipynb #Training notebook for the raw EEG 
│   ├── EEG_data/               # Folder containing EEG data and annotations as well as propofol concentrations
│   │   ├── Session1/..
│   │   ├── dataset/            # Stores the features extracted for the feature based approach
│   │   │   ├── validation_data_7680_7680.csv
│   │   │   ├── test_data_7680_7680.csv
│   │   │   ├── training_data_7680_7680.csv
│   │   ├── Session1/
│   │   ├── propofol_infusion/  # Folder for the propofol concentration rates
│   │   │   ├── prop8-2Cp_Fp1Fp2.csv
│   │   │   ├── prop8-2Ce_Fp1Fp2.csv
│   ├── environment/            # How to install the environment.
│   │   ├── conda_requirements.txt
│   │   ├── environment.yml
│   ├── doA_classification/
│   │   ├── ml_models/
│   │   │   ├── FeatureBased_validation_results_df.csv     # csv for the validation results
│   │   │   ├── FeatureBased_test_results_df.csv           # csv for the test results
│   │   │   ├── FeatureBased_SSPLneg_ws7680_ss7680_majority3840_typeCatBoostClassifier_preprocTrue_randomseed42_50.pkl     # trained models
│   │   │   ├── FeatureBased_BS_ws7680_ss7680_majority3840_typeCatBoostClassifier_preprocTrue_randomseed42_50.pkl
│   │   │   ├── FeatureBased_CNLneg_ws7680_ss7680_majority3840_typeCatBoostClassifier_preprocTrue_randomseed42_50.pkl
│   │   │   ├── FeatureBased_CRneg_ws7680_ss7680_majority3840_typeCatBoostClassifier_preprocTrue_randomseed42_50.pkl
│   ├── src/
│   │   ├── utils.py              # Helper Functions to be used in the training.ipynb and plot generations
│   │   ├── analysis/
│   │   │   ├── interpretability.py
│   │   ├── dataset/
│   │   │   ├── eeg_window.py     # code for getting windows from EEG signal
│   │   │   ├── eeg_metrics.py    # includes all the feature calculation to calculate information from the signal  
│   │   │   ├── eeg_dataset.py    # Python class that helps load the data for inference or training
│   │   │   ├── preprocessor_window.py  # Python Class for preprocessing
│   │   ├── models/               # Code for the ML learning models
│   │   │   ├── BPNet.py
│   │   │   ├── UCR.py
│   │   │   ├── Transformer.py
│   │   │   ├── tabularNN.py      # corresponds to the Neural Network (NN) of the paper
│   │   │   ├── LSTM.py
│   │   ├── inference/            # Code for running the inference
│   │   │   ├── inference.py      #Contains all functions needed to run the inference.ipynb notebook

```
## 📑 Citation
If you use this code, please cite our paper: 
Depth of Anesthesia Classification in Single-Channel EEG:
Analysis of Machine Learning Approaches and Future Perspectives by 
Anonymized Authors (Bibtex to follow)


