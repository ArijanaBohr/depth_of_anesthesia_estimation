#  Advancing Depth of Anesthesia Monitoring based on Single-Channel EEG: A Comparative Study of Machine Learning Approaches by Bohr and Salin et al.

Estimation of the depth of anesthesia with the help of the parameters of 
CNLneg, CRneg, SSPLneg and BuS (details can be found in the associated paper).

This projects trains binary classifiers of CNLneg/SSPLneg/CRneg/BuS. 

(in our code the variables for CNLneg/SSPlneg/CRneg/BuS are sleep/sspl/cr/burst_suppression)
# Step-by Step Tutorial:
## ⚙️ Installation

This project uses [Conda](https://docs.conda.io/) for environment management.  
The environment can be set up in one of two ways:
### Option 1: Using `environment.yml`
<pre>
conda eeg_env create -f environment.yml
conda activate eeg_env 
</pre>
### Option 2: Using `conda_requirements.txt`
<pre>
conda create -n eeg_env python=3.9
conda activate eeg_env
conda install --file conda_requirements.txt
</pre>
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
Run the notebook [training-hpc.py](training-hpc.py) 

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

│   ├── training-hpc.py         # Feature-Based Training script
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
│   │   ├── utils.py              # Helper Functions to be used in the training notebooks and plot generations
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
## Implementation details
The approaches were implemented in Python 3.9, using scikit-learn (Pedregosa et al., 2011) and PyTorch.
Feature extraction from EEG windows was performed using the following Python toolboxes:
NumPy (Harris et al., 2020),
PyWavelets (Lee et al., 2019),
SciPy (Virtanen et al., 2020),
pycatch22 (Lubba et al., 2019),
AntroPy (Vallat, 2021),
StatsModels (Seabold & Perktold, 2010),
lempel_ziv_complexity (Besson, 2019),
and EntropyHub (Flood, 2021).

References:

	•	Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., et al. (2011). Scikit-learn: Machine learning in Python. Journal of Machine Learning Research, 12, 2825–2830.
	•	Harris, C. R., Millman, K. J., van der Walt, S. J., Gommers, R., Virtanen, P., et al. (2020). Array programming with NumPy. Nature, 585(7825), 357–362. https://doi.org/10.1038/s41586-020-2649-2
	•	Lee, G. R., Gommers, R., Wasilewski, F., Wohlfahrt, K., & O’Leary, A. (2019). PyWavelets: A Python package for wavelet analysis. Journal of Open Source Software, 4(36), 1237. https://doi.org/10.21105/joss.01237
	•	Virtanen, P., Gommers, R., Oliphant, T. E., Haberland, M., Reddy, T., et al. (2020). SciPy 1.0: Fundamental algorithms for scientific computing in Python. Nature Methods, 17, 261–272. https://doi.org/10.1038/s41592-019-0686-2
	•	Lubba, C. H., Sethi, S. S., Knaute, P., Schultz, S. R., Fulcher, B. D., & Jones, N. S. (2019). catch22: Canonical time-series characteristics selected through highly comparative time-series analysis. Data Mining and Knowledge Discovery, 33(6), 1821–1852.
	•	Vallat, R. (2021). AntroPy: Entropy and complexity of (EEG) time series in Python. https://github.com/raphaelvallat/antropy — version v0.1.9, BSD 3-Clause License.
	•	Seabold, S., & Perktold, J. (2010). StatsModels: Econometric and statistical modeling with Python. In Proceedings of the 9th Python in Science Conference.
	•	Besson, L. N. (2019). lempel_ziv_complexity: Fast implementation of the Lempel–Ziv complexity algorithm. https://pypi.org/project/lempel-ziv-complexity/ — version 0.2.2, MIT License.
	•	Flood, M. W. (2021). EntropyHub: An open-source toolkit for entropic time series analysis. PLoS ONE, 16(11), e0259448. https://doi.org/10.1371/journal.pone.0259448


## Citation
If you use this repository or code in your research, please cite our paper:

Advancing Depth of Anesthesia Monitoring based on Single-Channel EEG: A Comparative Study of Machine Learning Approaches by 
Anonymized Authors (Bibtex to follow)


