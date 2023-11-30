import pandas as pd
import heartpy as hp
from heartpy.analysis import calc_fd_measures
from scipy.stats import kurtosis
from scipy.stats import skew
from heartpy.analysis import clean_rr_intervals
from heartpy.analysis import calc_ts_measures
import pyhrv.tools as tools
from pyhrv.hrv import hrv
import neurokit2 as nk
import numpy as np
import statsmodels.api as sm
from scipy.signal.signaltools import wiener
import scipy
import biosppy
from tqdm import tqdm
import pywt



class Extractor:
    def __init__(self, x):
        self.X = x
        
    def extract(self):
        vals = []
        for index in tqdm(range(0, self.X.shape[0])):
            row = self.X.loc[index, :]
            res = self._extract_one(row)
            vals.append(res)

        res = pd.DataFrame.from_records(vals)

        return res
        
    def _extract_one(self, signal):
        cur_row = {}

        # Automatically process the (raw) ECG signal
        cleaned_ecg = nk.ecg_clean(signal, sampling_rate=300, method='biosppy')
        cleaned_ecg = hp.remove_baseline_wander(cleaned_ecg, 300)
        cleaned_ecg = nk.signal_detrend(cleaned_ecg)

        rpeaks = biosppy.signals.ecg.hamilton_segmenter(cleaned_ecg, 300)
        rpeaks = biosppy.signals.ecg.correct_rpeaks(cleaned_ecg, rpeaks[0], 300.0)[0]

        r_vals = [cleaned_ecg[r] for r in rpeaks if not np.isnan(r)]
        for i, val in enumerate(r_vals):
            cur_row[f"r{i}"] = val

        # Delineate the ECG signal
        _, waves_dwt = nk.ecg_delineate(cleaned_ecg, sampling_rate=300, method="dwt")
        p_onsets = waves_dwt["ECG_P_Onsets"]
        p_offsets = waves_dwt["ECG_P_Offsets"]
        p_peaks = waves_dwt["ECG_P_Peaks"]

        s_peaks = waves_dwt["ECG_S_Peaks"]

        q_peaks = waves_dwt["ECG_Q_Peaks"]

        r_onsets = waves_dwt["ECG_R_Onsets"]
        r_offsets = waves_dwt["ECG_R_Offsets"]

        t_onsets = waves_dwt["ECG_T_Onsets"]
        t_offsets = waves_dwt["ECG_T_Offsets"]
        t_peaks = waves_dwt["ECG_T_Peaks"]

        n_heartbeats = np.size(rpeaks)

        #rr_distance
        rr = (rpeaks[1:] - rpeaks[:n_heartbeats - 1])  # rr-rate in seconds
        cur_row["mean_rr"] = np.nanmean(rr)
        # cur_row["median_rr"] = np.nanmedian(rr)
        cur_row["var_rr"] = np.nanvar(rr)
        cur_row["max_rr"] = np.nanmax(rr)
        cur_row["min_rr"] = np.nanmin(rr)
        cur_row["skew_rr"] = skew(rr)
        cur_row["kurtosis_rr"] = kurtosis(rr)

        #r_amplitude
        r_amplitude = cleaned_ecg[rpeaks]

        cur_row["mean_r"] = np.nanmean(r_amplitude)
        cur_row["median_r"] = np.nanmedian(r_amplitude)
        cur_row["var_r"] = np.nanvar(r_amplitude)
        cur_row["max_r"] = np.nanmax(r_amplitude)
        cur_row["min_r"] = np.nanmin(r_amplitude)

        cur_row["skew_r"] = skew(r_amplitude)
        cur_row["kurtosis_r"] = kurtosis(r_amplitude)

        # q_amplitude
        q_amplitude = q_peaks
        cur_row["mean_q"] = np.nanmean(q_amplitude)
        cur_row["median_q"] = np.nanmedian(q_amplitude)
        cur_row["var_q"] = np.nanvar(q_amplitude)
        cur_row["max_q"] = np.nanmax(q_amplitude)
        cur_row["min_q"] = np.nanmin(q_amplitude)
        cur_row["skew_q"] = skew(q_amplitude)
        cur_row["kurtosis_q"] = kurtosis(q_amplitude)

        # t_amplitude
        t_amplitude = t_peaks
        cur_row["mean_q"] = np.nanmean(t_amplitude)
        cur_row["median_q"] = np.nanmedian(t_amplitude)
        cur_row["var_t"] = np.nanvar(t_amplitude)
        cur_row["max_t"] = np.nanmax(t_amplitude)
        cur_row["min_t"] = np.nanmin(t_amplitude)
        cur_row["skew_t"] = skew(t_amplitude)
        cur_row["kurtosis_t"] = kurtosis(t_amplitude)

        # s_amplitude
        s_amplitude = s_peaks
        cur_row["mean_q"] = np.nanmean(s_amplitude)
        cur_row["median_q"] = np.nanmedian(s_amplitude)
        cur_row["var_s"] = np.nanvar(s_amplitude)
        cur_row["max_s"] = np.nanmax(s_amplitude)
        cur_row["min_s"] = np.nanmin(s_amplitude)
        cur_row["skew_s"] = skew(s_amplitude)
        cur_row["kurtosis_s"] = kurtosis(s_amplitude)

        #qrs_duration
        qrs_duration = [(b - a) for a, b in zip(r_onsets, r_offsets)]
        cur_row["mean_qrs"] = np.nanmean(qrs_duration)
        cur_row["median_qrs"] = np.nanmedian(qrs_duration)
        cur_row["var_qrs"] = np.nanvar(qrs_duration)
        cur_row["max_qrs"] = np.nanmax(qrs_duration)
        cur_row["min_qrs"] = np.nanmin(qrs_duration)
        cur_row["skew_qrs"] = skew(qrs_duration)
        cur_row["kurtosis_qrs"] = kurtosis(qrs_duration)

        #hrv metrics
        hrv_time = nk.hrv_time(rpeaks, sampling_rate=300, show=False)
        cur_row["HRV_IQRNN"] = hrv_time["HRV_IQRNN"].iloc[0]
        cur_row["HRV_HTI"] = hrv_time["HRV_HTI"].iloc[0]
        cur_row["HRV_pNN50"] = hrv_time["HRV_pNN50"].iloc[0]

        # cur_row["HRV_SDNN"] = hrv_time["HRV_SDNN"].iloc[0]
        # cur_row["HRV_RMSSD"] = hrv_time["HRV_RMSSD"].iloc[0]
        # cur_row["HRV_SDSD"] = hrv_time["HRV_SDSD"].iloc[0]
        # cur_row["HRV_CVNN"] = hrv_time["HRV_CVNN"].iloc[0]
        # cur_row["HRV_MedianNN"] = hrv_time["HRV_MedianNN"].iloc[0]
        # cur_row["HRV_pNN50"] = hrv_time["HRV_pNN50"].iloc[0]
        # cur_row["HRV_pNN20"] = hrv_time["HRV_pNN20"].iloc[0]
        # cur_row["HRV_TINN"] = hrv_time["HRV_TINN"].iloc[0]

        sig = signal.dropna().to_numpy(dtype='float32')
        
        # FFT
        fft = scipy.fft.fft(cleaned_ecg)
        S = np.abs(fft)#/len(cleaned_ecg) #careful, gives large numbers If small numbers are desired then /len(cleaned_ecg)
        S = S[:int(len(S)/2)]
        cur_row["fft_max"] = np.max(S)
        cur_row["fft_sum"] = np.sum(S)
        cur_row["fft_mean"] = np.mean(S)
        cur_row["fft_var"] = np.var(S)
        cur_row["fft_peak"] = np.max(S)
        cur_row["fft_skew"] = skew(S)
        cur_row["fft_kurtosis"] = kurtosis(S)

        # Wavelet
        scales = range(np.nanmin(rr), np.nanmax(rr)) #change dist coeff to change number of wavelets, 10 runs in 6 min
        waveletname = pywt.wavelist()[15]
        coeff, freq = pywt.cwt(cleaned_ecg, scales, 'morl', 1)
        energies = np.sum(np.square(np.abs(coeff)), axis=-1)
        
        #for i, val in enumerate(energies):
        #    cur_row[f"wvt_energy_{i}"] = val
        
        cur_row["wvt_max"] = np.max(energies)
        cur_row["wvt_sum"] = np.sum(energies)
        cur_row["wvt_mean"] = np.mean(energies)
        cur_row["wvt_var"] = np.var(energies)
        cur_row["wvt_skew"] = skew(energies)
        cur_row["wvt_kurtosis"] = kurtosis(energies)
        
        r_p = biosppy.signals.ecg.engzee_segmenter(signal, 300)['rpeaks']
        bts = biosppy.signals.ecg.extract_heartbeats(sig, r_p, 300)['templates']
        
        mu = np.mean(bts, axis=0) 
        var = np.std(bts, axis=0)
        
        mean_var_down = mu - var
        mean_var_up = mu - var
        
        cur_row["mean_up"] = np.nanmean(mean_var_up)
        cur_row["median_up"] = np.nanmedian(mean_var_up)
        cur_row["var_up"] = np.nanvar(mean_var_up)
        cur_row["max_up"] = np.nanmax(mean_var_up)
        cur_row["min_up"] = np.nanmin(mean_var_up)
        cur_row["skew_up"] = skew(mean_var_up)
        cur_row["kurtosis_up"] = kurtosis(mean_var_up)
        
        cur_row["mean_down"] = np.nanmean(mean_var_down)
        cur_row["median_down"] = np.nanmedian(mean_var_down)
        cur_row["var_down"] = np.nanvar(mean_var_down)
        cur_row["max_down"] = np.nanmax(mean_var_down)
        cur_row["min_down"] = np.nanmin(mean_var_down)
        cur_row["skew_down"] = skew(mean_var_down)
        cur_row["kurtosis_down"] = kurtosis(mean_var_down)
        
    
        return cur_row

