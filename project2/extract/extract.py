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
from scipy.signal.signaltools import wiener
import biosppy
from tqdm import tqdm


class Extractor:
    def __init__(self, x):
        self.X = x

    def remove_starting_period(self):
        self.X = self.X.drop(self.X.columns[range(700)], axis=1)
    
    def check_flipped(self):
        a = (np.max(self.X, axis=1) <= -0.75*np.min(self.X, axis=1))
        idx = np.where(a)[0]
        self.X.loc[idx,:] = - self.X.loc[idx,:]
        
    def extract(self):
        self.remove_starting_period()
        self.check_flipped()

        r_vals = []

        vals = []
        for index in tqdm(range(0, self.X.shape[0])):
            row = self.X.loc[index, :]
            res, r = self._extract_one(row)
            vals.append(res)
            r_vals.append(r)
        
        min_r_len = min([len(v) for v in r_vals])
        print(min_r_len)
        print(r_vals)
        r_vals = [v[:min_r_len] for v in r_vals]
        r_vals = np.array(r_vals)
        print(r_vals)
        r_vals = r_vals[:, ~np.isnan(r_vals).any(axis=0)]
        col_names = [f'r{i}' for i in range(r_vals.shape[1])]
        print(r_vals)

        res_r = pd.DataFrame(r_vals, columns=col_names)
        res_other = pd.DataFrame.from_records(vals)

        res = pd.concat([res_other, res_r])

        return res
    
    def get_clean_signal(self, file_name):
        vals = []
        for index in tqdm(range(0, self.X.shape[0])):
            signal = self.X.loc[index, :]
            cleaned_ecg = nk.ecg_clean(signal, sampling_rate=300, method='biosppy')
            cleaned_ecg = hp.remove_baseline_wander(cleaned_ecg, 300)
            cleaned_ecg = nk.signal_detrend(cleaned_ecg)
            vals.append(cleaned_ecg)

        res = pd.DataFrame.from_records(vals)
        res.to_csv(f"data/cleaned_{file_name}.csv", header=True, index=True)
        
    def _extract_one(self, signal):
        cur_row = {}
        r = []

        # Automatically process the (raw) ECG signal
        cleaned_ecg = nk.ecg_clean(signal, sampling_rate=300, method='biosppy')
        cleaned_ecg = hp.remove_baseline_wander(cleaned_ecg, 300)
        cleaned_ecg = nk.signal_detrend(cleaned_ecg)

        rpeaks = biosppy.signals.ecg.hamilton_segmenter(cleaned_ecg, 300)
        rpeaks = biosppy.signals.ecg.correct_rpeaks(cleaned_ecg, rpeaks[0], 300.0)[0]

        r_vals = [cleaned_ecg[r] for r in rpeaks if not np.isnan(r)]
        for i, val in enumerate(r_vals):
            r.append(val)

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
        cur_row["mean"] = np.nanmean(cleaned_ecg)
        cur_row["var"] = np.nanvar(cleaned_ecg)
        cur_row["skew"] = skew(cleaned_ecg)


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
        cur_row["q95_r"] = np.quantile(np.diff(rpeaks),0.95)
        cur_row["q95_r"] = np.quantile(np.diff(rpeaks),0.05)

        # q_amplitude
        q_amplitude = q_peaks
        cur_row["mean_q"] = np.nanmean(q_amplitude)
        cur_row["median_q"] = np.nanmedian(q_amplitude)
        cur_row["var_q"] = np.nanvar(q_amplitude)
        cur_row["max_q"] = np.nanmax(q_amplitude)
        cur_row["min_q"] = np.nanmin(q_amplitude)
        cur_row["skew_q"] = skew(q_amplitude)
        cur_row["kurtosis_q"] = kurtosis(q_amplitude)
        cur_row["q95_q"] = np.quantile(np.diff(q_amplitude),0.95)
        cur_row["q95_q"] = np.quantile(np.diff(q_amplitude),0.05)

        # q_amplitude
        p_amplitude = p_peaks
        cur_row["mean_p"] = np.nanmean(p_amplitude)
        cur_row["median_p"] = np.nanmedian(p_amplitude)
        cur_row["var_p"] = np.nanvar(p_amplitude)
        cur_row["max_p"] = np.nanmax(p_amplitude)
        cur_row["min_p"] = np.nanmin(p_amplitude)
        cur_row["skew_p"] = skew(p_amplitude)
        cur_row["kurtosis_p"] = kurtosis(p_amplitude)
        cur_row["q95_p"] = np.quantile(np.diff(p_amplitude),0.95)
        cur_row["q95_p"] = np.quantile(np.diff(p_amplitude),0.05)

        # t_amplitude
        t_amplitude = t_peaks
        cur_row["mean_t"] = np.nanmean(t_amplitude)
        cur_row["median_t"] = np.nanmedian(t_amplitude)
        cur_row["var_t"] = np.nanvar(t_amplitude)
        cur_row["max_t"] = np.nanmax(t_amplitude)
        cur_row["min_t"] = np.nanmin(t_amplitude)
        cur_row["skew_t"] = skew(t_amplitude)
        cur_row["kurtosis_t"] = kurtosis(t_amplitude)
        cur_row["q95_t"] = np.quantile(np.diff(t_amplitude),0.95)
        cur_row["q95_t"] = np.quantile(np.diff(t_amplitude),0.05)

        # s_amplitude
        s_amplitude = s_peaks
        cur_row["mean_s"] = np.nanmean(s_amplitude)
        cur_row["median_s"] = np.nanmedian(s_amplitude)
        cur_row["var_s"] = np.nanvar(s_amplitude)
        cur_row["max_s"] = np.nanmax(s_amplitude)
        cur_row["min_s"] = np.nanmin(s_amplitude)
        cur_row["skew_s"] = skew(s_amplitude)
        cur_row["kurtosis_s"] = kurtosis(s_amplitude)
        cur_row["q95_s"] = np.quantile(np.diff(s_amplitude),0.95)
        cur_row["q95_t"] = np.quantile(np.diff(s_amplitude),0.05)

        #qrs_duration
        qrs_duration = [(b - a) for a, b in zip(t_onsets, t_offsets)]
        cur_row["mean_qrs_t"] = np.nanmean(qrs_duration)
        cur_row["median_qrs_t"] = np.nanmedian(qrs_duration)
        cur_row["var_qrs_t"] = np.nanvar(qrs_duration)
        cur_row["max_qrs_t"] = np.nanmax(qrs_duration)
        cur_row["min_qrs_t"] = np.nanmin(qrs_duration)
        cur_row["skew_qrs_t"] = skew(qrs_duration)
        cur_row["kurtosis_qrs_t"] = kurtosis(qrs_duration)

        #qrs_duration
        qrs_duration = [(b - a) for a, b in zip(r_onsets, r_offsets)]
        cur_row["mean_qrs_r"] = np.nanmean(qrs_duration)
        cur_row["median_qrs_r"] = np.nanmedian(qrs_duration)
        cur_row["var_qrs_r"] = np.nanvar(qrs_duration)
        cur_row["max_qrs_r"] = np.nanmax(qrs_duration)
        cur_row["min_qrs_r"] = np.nanmin(qrs_duration)
        cur_row["skew_qrs_r"] = skew(qrs_duration)
        cur_row["kurtosis_qrs_r"] = kurtosis(qrs_duration)

        #qrs_duration
        qrs_duration = [(b - a) for a, b in zip(p_onsets, p_offsets)]
        cur_row["mean_qrs_p"] = np.nanmean(qrs_duration)
        cur_row["median_qrs_p"] = np.nanmedian(qrs_duration)
        cur_row["var_qrs_p"] = np.nanvar(qrs_duration)
        cur_row["max_qrs_p"] = np.nanmax(qrs_duration)
        cur_row["min_qrs_p"] = np.nanmin(qrs_duration)
        cur_row["skew_qrs_p"] = skew(qrs_duration)
        cur_row["kurtosis_qrs_p"] = kurtosis(qrs_duration)

        #hrv metrics
        hrv_time = nk.hrv_time(rpeaks, sampling_rate=300, show=False)
        cur_row["HRV_IQRNN"] = hrv_time["HRV_IQRNN"].iloc[0]
        cur_row["HRV_HTI"] = hrv_time["HRV_HTI"].iloc[0]
        cur_row["HRV_pNN50"] = hrv_time["HRV_pNN50"].iloc[0]

        cur_row["HRV_SDNN"] = hrv_time["HRV_SDNN"].iloc[0]
        cur_row["HRV_RMSSD"] = hrv_time["HRV_RMSSD"].iloc[0]
        cur_row["HRV_SDSD"] = hrv_time["HRV_SDSD"].iloc[0]
        cur_row["HRV_CVNN"] = hrv_time["HRV_CVNN"].iloc[0]
        cur_row["HRV_MedianNN"] = hrv_time["HRV_MedianNN"].iloc[0]
        cur_row["HRV_pNN50"] = hrv_time["HRV_pNN50"].iloc[0]
        cur_row["HRV_pNN20"] = hrv_time["HRV_pNN20"].iloc[0]
        cur_row["HRV_TINN"] = hrv_time["HRV_TINN"].iloc[0]
    
        return cur_row, r

