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
import biosppy
import pickle


class Extractor:
    def __init__(self, x):
        self.X = x
        self.cols = ["skew_rr", "kurtosis_rr", "skew_r", "kurtosis_r", "skew_q", "kurtosis_q",
                     "skew_qrs", "kurtosis_qrs", "HRV_IQRNN", "HRV_HTI", "HRV_pNN50"]

    def extract(self):
        res = pd.DataFrame(index=range(np.shape(self.X)[0]), columns=self.cols)
        def rowFunc(row,res):
            signal = row.iloc[1:]
            # Automatically process the (raw) ECG signal
            cleaned_ecg = nk.ecg_clean(signal, sampling_rate=300, method='biosppy')
            cleaned_ecg = hp.remove_baseline_wander(cleaned_ecg, 300)
            cleaned_ecg = nk.signal_detrend(cleaned_ecg)

            #_, rpeaks = nk.ecg_peaks(cleaned_ecg,sampling_rate=300)

            rpeaks = biosppy.signals.ecg.hamilton_segmenter(cleaned_ecg, 300)
            rpeaks = biosppy.signals.ecg.correct_rpeaks(cleaned_ecg, rpeaks[0], 300.0)[0]

            # templates, rpeaks = biosppy.signals.ecg.extract_heartbeats(cleaned_ecg, rpeaks, sampling_rate=300, before=0.2, after=0.4)
            # # print(templates)
            # pickle.dump(templates, self.file)

            # Delineate the ECG signal
            signal_dwt, waves_dwt = nk.ecg_delineate(cleaned_ecg, rpeaks, sampling_rate=300, method="dwt")

            r_peaks = rpeaks
            p_onsets = waves_dwt["ECG_P_Onsets"]
            p_offsets = waves_dwt["ECG_P_Offsets"]
            q_peaks = waves_dwt["ECG_Q_Peaks"]
            r_onsets = waves_dwt["ECG_R_Onsets"]
            r_offsets = waves_dwt["ECG_R_Offsets"]
            t_onsets = waves_dwt["ECG_T_Onsets"]
            t_offsets = waves_dwt["ECG_T_Offsets"]
            p_peaks = waves_dwt["ECG_P_Peaks"]
            s_peaks = waves_dwt["ECG_S_Peaks"]
            t_peaks = waves_dwt["ECG_T_Peaks"]

            r_vals = [cleaned_ecg[r] for r in r_peaks if not np.isnan(r)]

            n_heartbeats = np.size(r_peaks)
            cur_row = res.iloc[row.name,:]

            #rr_distance
            rr = (r_peaks[1:] - r_peaks[:n_heartbeats - 1])  # rr-rate in seconds
            # cur_row["mean_rr"] = np.nanmean(rr)
            # cur_row["median_rr"] = np.nanmedian(rr)
            # cur_row["var_rr"] = np.nanvar(rr)
            # cur_row["max_rr"] = np.nanmax(rr)
            # cur_row["min_rr"] = np.nanmin(rr)
            cur_row["skew_rr"] = skew(rr)
            cur_row["kurtosis_rr"] = kurtosis(rr)

            #r_amplitude
            r_amplitude = cleaned_ecg[r_peaks]
            # cur_row["mean_r"] = np.nanmean(r_amplitude)
            # cur_row["median_r"] = np.nanmedian(r_amplitude)
            # cur_row["var_r"] = np.nanvar(r_amplitude)
            # cur_row["max_r"] = np.nanmax(r_amplitude)
            # cur_row["min_r"] = np.nanmin(r_amplitude)
            cur_row["skew_r"] = skew(r_amplitude)
            cur_row["kurtosis_r"] = kurtosis(r_amplitude)


            # q_amplitude
            indices = [a for a in q_peaks if (not np.isnan(a))]
            q_amplitude = cleaned_ecg[indices]
            # cur_row["mean_q"] = np.nanmean(q_amplitude)
            # cur_row["median_q"] = np.nanmedian(q_amplitude)
            # cur_row["var_q"] = np.nanvar(q_amplitude)
            # cur_row["max_q"] = np.nanmax(q_amplitude)
            # cur_row["min_q"] = np.nanmin(q_amplitude)
            cur_row["skew_q"] = skew(q_amplitude)
            cur_row["kurtosis_q"] = kurtosis(q_amplitude)


            #qrs_duration
            qrs_duration = [(b - a) for a, b in zip(r_onsets, r_offsets)]
            # cur_row["mean_qrs"] = np.nanmean(qrs_duration)
            # cur_row["median_qrs"] = np.nanmedian(qrs_duration)
            # cur_row["var_qrs"] = np.nanvar(qrs_duration)
            # cur_row["max_qrs"] = np.nanmax(qrs_duration)
            # cur_row["min_qrs"] = np.nanmin(qrs_duration)
            cur_row["skew_qrs"] = skew(qrs_duration)
            cur_row["kurtosis_qrs"] = kurtosis(qrs_duration)


            #hrv metrics
            hrv_time = nk.hrv_time(r_peaks, sampling_rate=300, show=False)
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

            #hrv_freq = nk.hrv_frequency(r_peaks, sampling_rate=300, show=False)
            # try:
            #     wd, m = hp.process(cleaned_ecg, 300)
            #     wd, m = calc_fd_measures(measures=m, working_data=wd)
            #     pd.set_option("display.max_rows", None, "display.max_columns", None)
            #
            #     FREQ_feats = ['bpm', 'sdsd',
            #                   'pnn20', 'pnn50', 'sd2',
            #                   's', 'sd1/sd2']
            #
            #     for key in FREQ_feats:
            #         if key in m.keys():
            #             if (m[key] != "--"):
            #                 cur_row[key] = m[key]
            #             else:
            #                 cur_row[key] = np.nan
            #         else:
            #             cur_row[key] = np.nan
            # except hp.exceptions.BadSignalWarning:
            #     print("Heartpy warning occured!")

            progress = float(row.name)/np.shape(self.X)[0]
            print("Currently calculating features. Progress: " + str(progress))

        self.X.apply(lambda x: rowFunc(x, res), axis=1)
        return res

