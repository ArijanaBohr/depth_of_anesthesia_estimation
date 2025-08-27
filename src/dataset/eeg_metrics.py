import numpy as np
import pywt  # type: ignore
from scipy.stats import skew, kurtosis, differential_entropy
import pycatch22
import antropy as ant
from hurst import compute_Hc
from scipy.signal import welch
from statsmodels.tsa.ar_model import AutoReg
from lempel_ziv_complexity import lempel_ziv_complexity
from EntropyHub import DispEn

class EEG_metrics:
    def __init__(
        self,
        sampling_rate,
        window_length,
        step_size,
    ):
        """
        Initializes the EEG metrics extractor.
        Parameters:
        - sampling_rate: Sampling rate of the EEG data.
        - window_length: Length of the window for feature extraction.
        - step_size: Step size for windowing.
        """
        # Initialize parameters
        self.sampling_rate = sampling_rate
        self.window_length = window_length
        self.nyquist = 1 / sampling_rate
        self.step_size = step_size
        """
        Compiled Sources for the used features:
        - Arslan, A., Şen, B., Çelebi, F.V., Peker, M. and But, A., 2015, September. A comparison of different classification algorithms for determining the depth of anesthesia level on a new set of attributes. In 2015 International Symposium on Innovations in Intelligent SysTems and Applications (INISTA) (pp. 1-7). IEEE.
        - Gu, Y., Liang, Z. and Hagihira, S., 2019. Use of multiple EEG features and artificial neural network to monitor the depth of anesthesia. Sensors, 19(11), p.2499.
        - Saadeh, W., Khan, F.H. and Altaf, M.A.B., 2019. Design and implementation of a machine learning based EEG processor for accurate estimation of depth of anesthesia. IEEE transactions on biomedical circuits and systems, 13(4), pp.658-669.
        - Shalbaf, R., Behnam, H., Sleigh, J.W., Steyn-Ross, A. and Voss, L.J., 2013. Monitoring the depth of anesthesia using entropy features and an artificial neural network. Journal of neuroscience methods, 218(1), pp.17-24.
        - Shalbaf, A., Saffar, M., Sleigh, J.W. and Shalbaf, R., 2017. Monitoring the depth of anesthesia using a new adaptive neurofuzzy system. IEEE journal of biomedical and health informatics, 22(3), pp.671-677.
        - Wang, M., Zhu, F., Hou, C., Huo, D., Lei, Y., Long, Q. and Luo, X., 2024. Depth classification algorithm of anesthesia based on model fusion. Multimedia Tools and Applications, 83(33), pp.79589-79605.
        - Lubba, C.H., Sethi, S.S., Knaute, P., Schultz, S.R., Fulcher, B.D. and Jones, N.S., 2019. catch22: CAnonical Time-series CHaracteristics: Selected through highly comparative time-series analysis. Data mining and knowledge discovery, 33(6), pp.1821-1852.
        - Smith, J.O., 2007. Mathematics of the discrete Fourier transform (DFT): with audio applications. Julius Smith.
        - Phinyomark, A., Phukpattaranont, P. and Limsakul, C., 2012. Feature reduction and selection for EMG signal classification. Expert systems with applications, 39(8), pp.7420-7431.
        - https://de.mathworks.com/help/signal/ref/peak2rms.html and https://fr.wikipedia.org/wiki/Facteur_de_cr%C3%AAte
        - https://theses.hal.science/tel-04065194
        - Schwender, D., Daunderer, M., Mulzer, S., Klasing, S., Finsterer, U. and Peter, K., 1996. Spectral edge frequency of the electroencephalogram to monitor “depth” of anaesthesia with isoflurane or propofol. British journal of anaesthesia, 77(2), pp.179-184.
        - https://www.sciencedirect.com/topics/engineering/median-frequency
        - Fasol, M.C., Escudero, J. and Gonzalez-Sulser, A., 2023, December. Single-channel EEG artifact identification with the spectral slope. In 2023 IEEE International Conference on Bioinformatics and Biomedicine (BIBM) (pp. 2482-2487). IEEE.
        - Şen, B., Peker, M., Çavuşoğlu, A. and Çelebi, F.V., 2014. A comparative study on classification of sleep stage based on EEG signals using feature selection and classification algorithms. Journal of medical systems, 38(3), p.18.
        - Lempel, A., & Ziv, J. (1976). "On the Complexity of Finite Sequences." IEEE Trans. Information Theory, IT-22(1), 75-81.
        - Kaspar, F., & Schuster, H. G. (1987). "Easily calculable measure for the complexity of spatiotemporal patterns." Phys. Rev. A, 36(2), 842–848.
        - https://pypi.org/project/lempel-ziv-complexity/
        - https://en.wikipedia.org/wiki/R%C3%A9nyi_entropy
        - https://pypi.org/project/hurst/ 
        - Virtanen, P., Gommers, R., Oliphant, T.E., Haberland, M., Reddy, T., Cournapeau, D., Burovski, E., Peterson, P., Weckesser, W., Bright, J. and Van Der Walt, S.J., 2020. SciPy 1.0: fundamental algorithms for scientific computing in Python. Nature methods, 17(3), pp.261-272.
        - Harris, C.R., Millman, K.J., Van Der Walt, S.J., Gommers, R., Virtanen, P., Cournapeau, D., Wieser, E., Taylor, J., Berg, S., Smith, N.J. and Kern, R., 2020. Array programming with NumPy. nature, 585(7825), pp.357-362.
        - Lee, G.R., Gommers, R., Wasilewski, F., Wohlfahrt, K., O’Leary, A., 2019. Pywavelets: A python package for wavelet analysis. Journal of Open Source Software 4, 1237. doi:10.21105/joss.01237.
        - Flood, M.W., 2021. Entropyhub: An open-source toolkit for entropic time series analysis. PLoS One 16, e0259448. doi:10.1371/journal.pone.0259448
        - Vallat, R., 2021. Antropy: Entropy-based complexity measures in python. https://github.com/raphaelvallat/antropy
        - Seabold, S., Perktold, J., 2010. statsmodels: Econometric and statistical modeling with python, in: 9th Python in Science Conference.

        """
    def time_domain_features(self, eeg_window):
        """
        Extracts time-domain features from EEG data.
        Includes:
        - Mean, standard deviation, variance, skewness, and kurtosis.
        Parameters:
        - eeg_window: 1D NumPy array of EEG data.
        Returns:
        - Dictionary of time-domain features.
        """
        # Compute time-domain features
        max_amplitude = np.max(eeg_window)
        min_amplitude = np.min(eeg_window)
        peak_to_peak = max_amplitude - min_amplitude
        rms = np.sqrt(np.mean(eeg_window**2)) #Feature reduction and selection for EMG signal classification by Phinyomark
        abs_window = np.abs(eeg_window)
        peak_index = np.argmax(abs_window)
        mean_abs = np.mean(abs_window)
        peak = np.max(abs_window)

        waveform_index = rms / mean_abs if mean_abs != 0 else 0 # also called Form Factor: Tracking and Estimation of Frequency, Amplitude, and Form Factor of a Harmonic Time Series [Lecture Notes] by Ronald M. Aarts; also called: Shape factor: https://de.mathworks.com/help/predmaint/ug/signal-features.html
        margin_index = rms / peak if peak != 0 else 0 #Inverse Crest Factor https://fr.wikipedia.org/wiki/Facteur_de_cr%C3%AAte or Peak-magnitude-to-RMS ratio https://de.mathworks.com/help/signal/ref/peak2rms.html 

        first_deriv = np.diff(eeg_window)

        # Computed following formula of https://theses.hal.science/tel-04065194
        teo = eeg_window[1:-1] ** 2 - eeg_window[:-2] * eeg_window[2:]

        hjorth_mobility, hjorth_complexity = ant.hjorth_params(eeg_window)
        return {
            "hjorth_activity": np.var(eeg_window),
            "hjorth_mobility": hjorth_mobility,
            "hjorth_complexity": hjorth_complexity,
            "std_curve_length": np.std(np.abs(first_deriv)),
            "mean_curve_length": np.mean(np.abs(first_deriv)),
            "max_amplitude": max_amplitude,  # MaxV
            "min_amplitude": min_amplitude,  # MinV
            "mean_teager_energy": np.mean(teo),
            "peak_to_peak": peak_to_peak,
            "rms": rms,
            "peak_index": peak_index,
            "waveform_index": waveform_index,
            "margin_index": margin_index,
            "time_std": np.std(eeg_window),
            "time_variance": np.var(eeg_window),
            "time_skewness": skew(eeg_window),
            "time_kurtosis": kurtosis(eeg_window),
            "time_arithmic_mean": np.mean(eeg_window),  # AM
            "mean_energy": np.mean(np.square(eeg_window)),
            "hurst_exponent": self.safe_hurst_exponent(eeg_window),
            "time_median": np.median(eeg_window),  # MN
            "num_zero_crossings": ant.num_zerocross(eeg_window),
        }

    def compute_band_power(self, psd_density, freqs, fmin, fmax):
        """Compute the band power between fmin and fmax using Welch PSD."""
        mask = (freqs >= fmin) & (freqs <= fmax)
        power = np.trapz(psd_density[mask], freqs[mask])
        return power

    def band_power(self, psd, freqs, low, high):
        """Computes power in a given frequency band."""
        idx = (freqs >= low) & (freqs < high)
        return np.trapz(psd[idx], freqs[idx])

    def calculate_frequencies(self, cumulative_power, freqs, threshold):
        """
        Calculate the frequency at which the cumulative power reaches a specified threshold.

        Parameters:
        - cumulative_power: Array of cumulative power values.
        - freqs: Array of frequency values corresponding to the cumulative power array.
        - threshold: The cumulative power threshold to find (e.g., 0.5 for median frequency, 0.95 for SEF).

        Returns:
        Feature sets  
        """
        index = np.searchsorted(cumulative_power, threshold)
        return freqs[index] if index < len(freqs) else np.nan

    # https://www.bjanaesthesia.org/article/S0007-0912(17)43581-1/pdf for the definition of SEF and median
    def frequency_domain_features(self, eeg_window):
        # Compute Welch Power Spectral Density (PSD) once
        freqs, psd_density = welch(
            eeg_window,
            fs=self.sampling_rate,
            nperseg=min(256, self.window_length),
            scaling="density",
        )

        ## selecting same frequencies as the bandpass
        mask = (freqs >= 0.5) & (freqs <= 47)
        freqs = freqs[mask]
        psd_density = psd_density[mask]

        # Compute total power
        total_power = np.trapz(psd_density, freqs)

        # High frequency power (30-47 Hz)
        P_high_gu = self.compute_band_power(psd_density, freqs, 30, 47)
        P_low_gu = self.compute_band_power(psd_density, freqs, 11, 20)

        beta_ratio_gu = np.log(P_high_gu / (P_low_gu + 1e-10))

        P_high_saadeh = self.compute_band_power(psd_density, freqs, 20, 48)
        P_low_saadeh = self.compute_band_power(psd_density, freqs, 8, 20)

        relative_beta_ratio_saadeh = np.log2(
            (P_high_saadeh + 1e-10) / (P_low_saadeh + 1e-10)
        )

        # Predefined + custom bands
        all_bands = [
            ("delta_power", 0.5, 4),
            ("theta_power", 4, 8),
            ("alpha_power", 8, 12),
            ("beta_power", 12, 25),
            ("gamma_power", 25, 40),
        ]
        band_features = {}
        for name, low, high in all_bands:
            bp = self.band_power(psd_density, freqs, low, high)
            band_features[f"{name}_ratio"] = bp / total_power if total_power > 0 else 0

        # Spectral centroid (Mean Frequency)
        psd_sum = np.sum(psd_density)
        # Cumulative power for SEF and median frequency
        cumulative_power = np.cumsum(psd_density) / (psd_sum + 1e-10)

        # SEF 95
        sef = self.calculate_frequencies(cumulative_power, freqs, 0.95)
        median_freq = self.calculate_frequencies(cumulative_power, freqs, 0.5)

        # Frequency domain moments
        # defintion of mean_freq from https://www.sciencedirect.com/topics/engineering/median-frequency
        mean_freq = np.sum(freqs * psd_density) / (psd_sum + 1e-10)
        # Spectral features
        features = {
            "spectral_edge_frequency": sef,
            "mean_frequency": mean_freq,
            "median_frequency": median_freq,
            "beta_ratio_gu": beta_ratio_gu,
            "relative_beta_ratio_saadeh": relative_beta_ratio_saadeh,
            **band_features,  # Add band power features
        }

        # Spectral slope: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10385840
        # Compute Welch Slope (log-log linear fit of power spectrum)
        valid_indices = (freqs[1:] > 0) & (psd_density[1:] > 0)  # Avoid log(0)
        if np.any(valid_indices):
            features["spectral_slope"] = np.polyfit(
                np.log(freqs[1:][valid_indices]),
                np.log(psd_density[1:][valid_indices]),
                1,
            )[0]
        else:
            features["spectral_slope"] = np.nan

        return features


    def entropy_and_complexity_features(self, eeg_window):
        """
        Extracts entropy and complexity features using AntroPy.

        Parameters:
        - eeg_window: 1D NumPy array
        - sf: Sampling frequency (Hz)

        Returns:
        - Dictionary of features
        """
        samp_en = ant.sample_entropy(eeg_window, order=2, tolerance=0.15)
        # To account for the infinity values that stem from dividing through zero
        dispersion_entropy, _ = DispEn(
            eeg_window, m=2, c=6, tau=1, Logx=np.e, Norm=True
        )
        try:
            petrosian_fractal_dimension = ant.petrosian_fd(eeg_window)
        except (FloatingPointError, ValueError):
            petrosian_fractal_dimension = 0.0
        return {
            "spectral_entropy": ant.spectral_entropy(eeg_window, sf=self.sampling_rate),
            "approximate_entropy": ant.app_entropy(eeg_window),
            "renyi_entropy": self.renyi_entropy(eeg_window),
            "petrosian_fractal_dimension": petrosian_fractal_dimension,
            "lempel_ziv_complexity": self.compute_lempel_ziv_complexity(eeg_window),
            "permutation_entropy_order_6": ant.perm_entropy(eeg_window, order=6),
            "permutation_entropy_order_3": ant.perm_entropy(eeg_window, order=3),
            "sample_entropy": (np.nan if np.isinf(samp_en) else samp_en),
            "dispersion_entropy": dispersion_entropy,
            "higuchi_fd": ant.higuchi_fd(eeg_window),
            "differential_entropy": differential_entropy(eeg_window),
        }

    # Wavelet features inspired from https://link.springer.com/article/10.1007/s10916-014-0018-0
    def extract_wavelet_features(self, eeg_window):
        """
        Extracts wavelet features from EEG data using the Discrete Wavelet Transform (DWT).
        Parameters:
        - eeg_window: 1D NumPy array of EEG data.
        - level: Level of decomposition for the DWT.
        Returns:
        - Dictionary of wavelet features.
        """
        # Perform Discrete Wavelet Transform (DWT)
        level = min(
            pywt.dwt_max_level(
                data_len=self.window_length, filter_len=pywt.Wavelet("db4").dec_len
            ),
            5,
        )

        coeffs = pywt.wavedec(eeg_window, wavelet="db4", level=level)
        # Initialize dictionary to store features
        features = {}
        # Extract approximation and detail coefficients
        cA = coeffs[0]
        cAabs = np.abs(cA)
        features["wavelet_A5-mean"] = np.mean(cAabs)
        features["wavelet_A5-std"] = np.std(cAabs)

        details = coeffs[1:]  # Detail coefficients for each level

        # Feature extraction from Approximation Coefficients (cA)
        features["wavelet_energy_cA"] = (
            np.sum(np.square(cAabs)) if not np.isnan(np.sum(np.square(cAabs))) else 0
        )
        # Calculate mean absolute values for detail coefficients
        mean_abs_details = [np.mean(np.abs(cD)) for cD in details]

        for i, mean_abs in enumerate(mean_abs_details, start=1):
            level = 6 - i  # Maps details[0] -> D5, details[1] -> D4, etc.
            features[f"wavelet_mean_abs_cD{level}"] = mean_abs

            # Calculate ratios with other detail coefficients
            for j in range(i + 1, len(mean_abs_details) + 1):
                ratio_key = f"wavelet_ratio_abs_avg_cD{level}_to_cD{6 - j}"
                features[ratio_key] = mean_abs / mean_abs_details[j - 1]

            # Calculate ratio with A5 mean
            ratio_key = f"wavelet_ratio_abs_avg_A5_to_cD{level}"
            features[ratio_key] = features["wavelet_A5-mean"] / mean_abs

        for i, cD in enumerate(details, start=1):
            cDabs = np.abs(cD)
            features[f"wavelet_std_cD{i}"] = np.std(cDabs)
            features[f"wavelet_mean_power_cD{i}"] = np.mean(np.square(cD))
        return features

    def catch22(self, eeg_window):
        """
        Computes 22 features from the EEG data using the catch22 library.
        Parameters:
        - eeg_window: 1D NumPy array of EEG data.
        Returns:
        - Dictionary of features computed using catch22.
        """
        # Get the results from pycatch22
        results = pycatch22.catch22_all(eeg_window)

        # Convert results to a dictionary format
        feature_dict = {
            name: value for name, value in zip(results["names"], results["values"])
        }

        return feature_dict

    def eeg_dfa(self, eeg_window):
        return {
            "dfa_exponent": ant.detrended_fluctuation(eeg_window),
        }

    def safe_hurst_exponent(self, eeg_window):
        """
        Computes Hurst exponent safely.
        Returns 0.0 if computation fails (e.g., log10 of zero or invalid eeg_window).
        """
        # Check for empty or invalid eeg_window
        if (
            eeg_window is None
            or len(eeg_window) < 20
            or np.any(np.isnan(eeg_window))
            or np.any(eeg_window <= 0)
        ):
            return 0.0

        H, _, _ = compute_Hc(eeg_window, kind="price", simplified=True)
        if not np.isfinite(H) or H <= 0:
            return 0.0
        return H

    def ar_coefficients_wang(self, eeg_window, order=5):
        try:
            model = AutoReg(eeg_window, lags=order, old_names=False).fit()
            coeffs = model.params[1:]  # Skip intercept
        except:
            coeffs = np.zeros(order)  # Fallback if fit fails

        return {f"ar_coeff_{i+1}": coeff for i, coeff in enumerate(coeffs)}

    def compute_lempel_ziv_complexity(self, eeg_window):
        """
        Compute the Lempel-Ziv complexity of a 1D EEG window using the lempel-ziv-complexity package.

        References:
        - Lempel, A., & Ziv, J. (1976). "On the Complexity of Finite Sequences." IEEE Trans. Information Theory, IT-22(1), 75-81.
        - Kaspar, F., & Schuster, H. G. (1987). "Easily calculable measure for the complexity of spatiotemporal patterns." Phys. Rev. A, 36(2), 842–848.
        - https://pypi.org/project/lempel-ziv-complexity/
        """

        # Step 1: Binarize using median
        median = np.median(eeg_window)
        binary_sequence = (eeg_window > median).astype(int)
        binary_string = "".join(binary_sequence.astype(str))

        # Step 2: Call lempel_ziv_complexity from package: https://pypi.org/project/lempel-ziv-complexity/
        lz_complexity = lempel_ziv_complexity(binary_string)

        return lz_complexity

    def renyi_entropy(self, signal, alpha=2, bins=100):
        """
        Computes the Rényi entropy of a 1D signal (https://en.wikipedia.org/wiki/R%C3%A9nyi_entropy)

        Parameters:
        - signal: 1D NumPy array
        - alpha: Order of Rényi entropy (alpha > 0, alpha ≠ 1)
        - bins: Number of bins for the histogram

        Returns:
        - Rényi entropy value
        """
        if alpha <= 0 or alpha == 1:
            raise ValueError("Alpha must be > 0 and ≠ 1")

        hist, _ = np.histogram(signal, bins=bins, density=True)
        probs = hist[hist > 0]  # Filter out zero-probability bins
        return 1 / (1 - alpha) * np.log(np.sum(probs**alpha))

