#!/usr/bin/env python3

import os
import sys
import yaml
import gc
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from cmcrameri import cm
from numpy.fft import fft, fftfreq, fftshift
from datetime import datetime, timezone
from matplotlib.colors import LogNorm
from scipy.optimize import curve_fit, least_squares
from scipy import constants
from datetime import datetime, timedelta
from scipy.interpolate import interp1d

plt.switch_backend('Agg')
colours = cm.lapaz

LOG_FILE = '/home/doptrackbox/DTB_Recordings/locally_processed/dt_comparison.log'
EXTRACTION_LOC = '/home/doptrackbox/DTB_Recordings/locally_processed/Processed/'
LOCAL_LOC = '/home/doptrackbox/DTB_Recordings/data/'
REMOTE_LOC = '/home/doptrackbox/tudelft_webdrive/staff-umbrella/doptrack/archive/Dylano/DopTrackBox/'
LOCi = '/home/doptrackbox/DTB_Recordings/locally_processed/'
DopTrackLoc = '/home/doptrackbox/tudelft_webdrive/staff-umbrella/doptrack/'
PLOT_LOC = os.path.join(LOCi, 'Plots/')
#RECOMPUTE_TIMESTAMPS = True

def compute_range_rate(frequencies, f0):
    return constants.c*(1-frequencies/f0)

def compute_doppler(dv, f0):
    return f0*(1-(dv/constants.c))

def find_yml(file, name, year):
    yml_filename = file.rsplit('_', 1)[0]
    used_remote = False
    try:
        yml_subdir = f"{name}/{year}/{yml_filename}.yml"
        if os.path.exists(LOCAL_LOC + yml_subdir):
            yml_path = LOCAL_LOC + yml_subdir
        else:                
            yml_path = REMOTE_LOC + yml_subdir

        with open(yml_path, 'r') as f:
            settings = yaml.load(f, Loader=yaml.FullLoader)

    except Exception as e:
        print(f"[ERROR] Failed to retrieve YML file for file {FILE}: {e}")
        sys.exit(1)
    return settings

def db_to_ratio(db):
    return 10**(db/10)

def ratio_to_db(ratio):
    return 10*np.log10(ratio)

def combine_data(freq_power_data, noise_data):
    noise_times = noise_data[:,0]
    noise_powers = noise_data[:,1]

    idx = np.searchsorted(noise_times, freq_power_data[:,0])
    noise_at_signal_entries = noise_powers[idx]
    
    return np.vstack([freq_power_data, noise_at_signal_entries])

def load_DTB_data(file):
    with open(EXTRACTION_LOC + "ExtractedSignal/" + file + ".txt", 'r') as f:
        header = f.readline().strip()
    main_part = header.split("\tPower")[0]
    entries = [item for item in main_part.split("|")[1:] if ":" in item]
    parts = dict(item.split(":", 1) for item in entries)
    mask_width = float(parts["MW"])
    tune = float(parts["TUNE"])

    # Read from file
    data = np.loadtxt(EXTRACTION_LOC + "ExtractedSignal/" + file + ".txt", skiprows=1)
    return data, mask_width

def get_yml_path(sub_var, file, name, year):
    yml_filename = file.rsplit('_', 1)[0]

    if os.path.exists(DopTrackLoc + f"products/{sub_var}/{name}/{year}/{yml_filename}.yml"):
        yml_path = DopTrackLoc + f"products/{sub_var}/{name}/{year}/{yml_filename}.yml"
        return yml_path
    else:
        parts = file.split('_')
        date_part = parts[2]
        date_time = datetime.strptime(date_part, "%Y%m%d%H%M")

        for m in range(-10,10):
            candidate_datetime = date_time + timedelta(minutes=m)
            yml_filename = f"{parts[0]}_{parts[1]}_"+candidate_datetime.strftime("%Y%m%d%H%M")
            if os.path.exists(DopTrackLoc + f"products/{sub_var}/{name}/{year}/{yml_filename}.yml"):
                yml_path = DopTrackLoc + f"products/{sub_var}/{name}/{year}/{yml_filename}.yml"
                return yml_path        
    return None
    

def load_DT_data(file, name, year):
    yml_path_L1B = get_yml_path("L1B", file, name, year)
    #yml_path_L1C = get_yml_path("L1C", file, name, year)

    if yml_path_L1B is not None: #and yml_path_L1C is not None:
        with open(yml_path_L1B, 'r') as f:
            yml_settings = yaml.load(f, Loader=yaml.FullLoader)

        DT_tca = yml_settings['product']['tca']
        DT_fca = yml_settings['product']['fca']
        DT_start_time = yml_settings['recording']['time_start'].replace(tzinfo=timezone.utc)

        DT_freq_datapath = yml_path_L1B.rsplit('.',1)[0] + ".dat"
        #DT_rr_datapath = yml_path_L1C.rsplit('.',1)[0] + ".dat"

        DT_freq = np.loadtxt(DT_freq_datapath, skiprows=1)
        #DT_rr = np.loadtxt(DT_rr_datapath, skiprows=1)

        return DT_freq, DT_tca, DT_fca, DT_start_time
    else:
        return None, None, None, None
 
def plot_raw_freq(DT_time, DT_freq, DTB_time, DTB_freq, DTB_snr, tune, file, interpolation_function):
    fig_1, ax_1 = plt.subplots(figsize=(8,5)) 
    ax_1.scatter(DT_time, DT_freq-tune, s=10, label = "DopTrack Signal", c='black' )
    scat = ax_1.scatter(DTB_time, DTB_freq-tune, s=10, label = "Extracted DopTrackBox Signal", c=ratio_to_db(DTB_snr))
    
    cbar = plt.colorbar(scat)
    cbar.set_label('Signal-to-Noise Ratio [dB]') 
    ax_1.set_xlabel('Time [s]')
    ax_1.set_ylabel(f'Frequency w.r.t. Tuning Frequency [Hz]')
    ax_1.set_title(f'Comparison of Pass {file}')

    #ax_1.set_ylim(0, 40)
    lowest_minimum = min(np.min(DT_time), np.min(DTB_time))
    higest_minimum = max(np.min(DT_time), np.min(DTB_time))
    higest_maximum = max(np.max(DT_time), np.max(DTB_time))
    lowest_maximum = min(np.max(DT_time), np.max(DTB_time))
    
    ax_1.set_xlim(lowest_minimum, higest_maximum)
    #if mask_width != 0:
    #    ax_1.set_ylim(freq_offset-(mask_width/2), freq_offset+(mask_width/2))
    ax_1.grid()
    #fig_1.set_size_inches(19, 10)
    ax_1.legend()
    fig_1.tight_layout()
    if RECOMPUTE_TIMESTAMPS:
        os.makedirs(PLOT_LOC+"DTBvsDT_V2_freq_recomputed_timestamps", exist_ok=True)
        fig_1.savefig(os.path.join(PLOT_LOC, f"DTBvsDT_V2_freq_recomputed_timestamps/V2_freq_recomputed_timestamps_{file}.png"))
    else:       
        os.makedirs(PLOT_LOC+"DTBvsDT_V2_freq", exist_ok=True)
        fig_1.savefig(os.path.join(PLOT_LOC, f"DTBvsDT_V2_freq/V2_freq_{file}.png"))
    plt.close(fig_1)
    gc.collect()

    # Freq difference by interpolating frequency
    mask = (DTB_time >= DT_time[0]) & (DTB_time <= DT_time[-1])
    selected_DTB_time = DTB_time[mask]
    selected_DTB_freq = DTB_freq[mask]

    DT_freq_interp = interpolation_function(selected_DTB_time)
    freq_diff = selected_DTB_freq - DT_freq_interp

    fig_2, ax_2 = plt.subplots(figsize=(8,5)) 
    ax_2.scatter(selected_DTB_time, freq_diff, s=10, c='black')
    #scat = ax_1.scatter(DTB_time+delta_t, DTB_freq-tune+delta_f, s=10, label = f"Shifted DopTrackBox Signal (delta t = {delta_t:.2f} s, delta f = {delta_f:.1f} Hz)", c=ratio_to_db(DTB_snr))
    
    #cbar = plt.colorbar(scat)
    #cbar.set_label('Signal-to-Noise Ratio [dB]') 
    ax_2.set_xlabel('Time [s]')
    ax_2.set_ylabel('Frequency [Hz]')
    ax_2.set_title(f'Difference between DTB freq and interp. DT freq')

    #ax_1.set_ylim(0, 40)
    lowest_minimum = min(np.min(DT_time), np.min(DTB_time))
    higest_minimum = max(np.min(DT_time), np.min(DTB_time))
    higest_maximum = max(np.max(DT_time), np.max(DTB_time))
    lowest_maximum = min(np.max(DT_time), np.max(DTB_time))
    
    #ax_2.set_xlim(lowest_minimum, higest_maximum)
    #if mask_width != 0:
    #    ax_1.set_ylim(freq_offset-(mask_width/2), freq_offset+(mask_width/2))
    ax_2.grid()
    #fig_1.set_size_inches(19, 10)
    #ax_2.legend()
    fig_2.tight_layout()
    if RECOMPUTE_TIMESTAMPS:
        os.makedirs(PLOT_LOC+"DTBvsDT_V2_freq_diff_recomputed_timestamps", exist_ok=True)
        fig_2.savefig(os.path.join(PLOT_LOC, f"DTBvsDT_V2_freq_diff_recomputed_timestamps/V2_freq_recomputed_timestamps_{file}.png"))
    else:       
        os.makedirs(PLOT_LOC+"DTBvsDT_V2_freq_diff", exist_ok=True)
        fig_2.savefig(os.path.join(PLOT_LOC, f"DTBvsDT_V2_freq_diff/V2_freq_{file}.png"))
    plt.close(fig_2)

    # Combined plot
    fig_3, axes = plt.subplots(
        2, 1, figsize=(8, 7), sharex=True,
        gridspec_kw={'height_ratios': [2, 1]}  # ax_3 is twice as tall as ax_4
    )
    ax_3, ax_4 = axes
    #gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1])  # Top plot 3x taller than bottom
    #ax_3 = fig_3.add_subplot(gs[0])
    #ax_4 = fig_3.add_subplot(gs[1], sharex=ax_3)

    ax_3.scatter(DT_time, DT_freq-tune, s=10, label = "DopTrack Signal", c='black' )
    scat = ax_3.scatter(DTB_time, DTB_freq-tune, s=10, label = "Extracted DopTrackBox Signal", c='red')#, c=ratio_to_db(DTB_snr))

    ax_3.set_ylabel(f'Frequency w.r.t. Tuning Frequency [Hz]')
    ax_3.set_title(f'Comparison of Pass {file}')

    ax_3.set_xlim(lowest_minimum, higest_maximum)
    ax_3.grid()
    ax_3.legend()

    ax_4.scatter(selected_DTB_time, freq_diff, s=10, c='black')
    ax_4.set_xlabel('Time [s]')
    ax_4.set_ylabel('Frequency Difference [Hz]')
    ax_4.grid()

    #cbar = fig_3.colorbar(scat, ax=[ax_3, ax_4], orientation='horizontal', pad=0.15, aspect=40)
    #cbar.set_label('Signal-to-Noise Ratio [dB]') 

    if RECOMPUTE_TIMESTAMPS:
        os.makedirs(PLOT_LOC+"DTBvsDT_V2_comp_recomputed_timestamps", exist_ok=True)
        fig_3.savefig(os.path.join(PLOT_LOC, f"DTBvsDT_V2_comp_recomputed_timestamps/V2_comp_recomputed_timestamps_{file}.png"))
    else:       
        os.makedirs(PLOT_LOC+"DTBvsDT_V2_comp", exist_ok=True)
        fig_3.savefig(os.path.join(PLOT_LOC, f"DTBvsDT_V2_comp/V2_comp_{file}.png"))

    plt.close(fig_3)
    gc.collect()      


def plot_fitted_freq(DT_time, DT_freq, DTB_time, DTB_freq, DTB_snr, tune, file, delta_t, delta_f, interpolation_function):
    fig_1, ax_1 = plt.subplots(figsize=(8,5)) 
    ax_1.scatter(DT_time, DT_freq-tune, s=10, label = "DopTrack Signal", c='black' )
    scat = ax_1.scatter(DTB_time-delta_t, DTB_freq-tune-delta_f, s=10, label = rf"Shifted DopTrackBox Signal ($\Delta$t = {delta_t:.2f} s, $\Delta$f = {delta_f:.2f} Hz)", c=ratio_to_db(DTB_snr))
    
    cbar = plt.colorbar(scat)
    cbar.set_label('Signal-to-Noise Ratio [dB]') 
    ax_1.set_xlabel('Time [s]')
    ax_1.set_ylabel(f'Frequency w.r.t. Tuning Frequency [Hz]')
    ax_1.set_title(f'Comparison of Pass {file}')

    #ax_1.set_ylim(0, 40)
    lowest_minimum = min(np.min(DT_time), np.min(DTB_time))
    higest_minimum = max(np.min(DT_time), np.min(DTB_time))
    higest_maximum = max(np.max(DT_time), np.max(DTB_time))
    lowest_maximum = min(np.max(DT_time), np.max(DTB_time))
    
    ax_1.set_xlim(lowest_minimum, higest_maximum)
    #if mask_width != 0:
    #    ax_1.set_ylim(freq_offset-(mask_width/2), freq_offset+(mask_width/2))
    ax_1.grid()
    #fig_1.set_size_inches(19, 10)
    ax_1.legend()
    fig_1.tight_layout()
    if RECOMPUTE_TIMESTAMPS:
        os.makedirs(PLOT_LOC+"DTBvsDT_V2_fitted_freq_recomputed_timestamps", exist_ok=True)
        fig_1.savefig(os.path.join(PLOT_LOC, f"DTBvsDT_V2_fitted_freq_recomputed_timestamps/V2_shifted_freq_recomputed_timestamps_{file}.png"))        
    else:
        os.makedirs(PLOT_LOC+"DTBvsDT_V2_fitted_freq", exist_ok=True)
        fig_1.savefig(os.path.join(PLOT_LOC, f"DTBvsDT_V2_fitted_freq/V2_shifted_freq_{file}.png"))
    plt.close(fig_1)
    gc.collect()

    # Freq difference by interpolating frequency
    shifted_DTB_time = DTB_time - delta_t 
    shifted_DTB_freq = DTB_freq - delta_f

    #mask = (DTB_time >= DT_time.min()+2) & (DTB_time <= DT_time.max()-2)
    mask = (shifted_DTB_time >= DT_time[0]) & (shifted_DTB_time <= DT_time[-1])
    selected_DTB_time = shifted_DTB_time[mask]
    selected_DTB_freq = shifted_DTB_freq[mask]

    DT_freq_interp = interpolation_function(selected_DTB_time)
    freq_diff = selected_DTB_freq - DT_freq_interp

    fig_2, ax_2 = plt.subplots(figsize=(8,5)) 
    ax_2.scatter(selected_DTB_time, freq_diff, s=10, c='black')
    #scat = ax_1.scatter(DTB_time+delta_t, DTB_freq-tune+delta_f, s=10, label = f"Shifted DopTrackBox Signal (delta t = {delta_t:.2f} s, delta f = {delta_f:.1f} Hz)", c=ratio_to_db(DTB_snr))
    
    #cbar = plt.colorbar(scat)
    #cbar.set_label('Signal-to-Noise Ratio [dB]') 
    ax_2.set_xlabel('Time [s]')
    ax_2.set_ylabel('Frequency [Hz]')
    ax_2.set_title(f'Difference between DTB freq and interp. DT freq')

    #ax_1.set_ylim(0, 40)
    
    #ax_2.set_xlim(lowest_minimum, higest_maximum)
    #if mask_width != 0:
    #    ax_1.set_ylim(freq_offset-(mask_width/2), freq_offset+(mask_width/2))
    ax_2.grid()
    #fig_1.set_size_inches(19, 10)
    #ax_2.legend()
    fig_2.tight_layout()
    if RECOMPUTE_TIMESTAMPS:
        os.makedirs(PLOT_LOC+"DTBvsDT_V2_fitted_freq_diff_recomputed_timestamps", exist_ok=True)
        fig_2.savefig(os.path.join(PLOT_LOC, f"DTBvsDT_V2_fitted_freq_diff_recomputed_timestamps/V2_shifted_freq_recomputed_timestamps_{file}.png"))        
    else:
        os.makedirs(PLOT_LOC+"DTBvsDT_V2_fitted_freq_diff", exist_ok=True)
        fig_2.savefig(os.path.join(PLOT_LOC, f"DTBvsDT_V2_fitted_freq_diff/V2_shifted_freq_{file}.png"))
    plt.close(fig_2)


    # Combined plot
    fig_3, axes = plt.subplots(
        2, 1, figsize=(8, 7), sharex=True,
        gridspec_kw={'height_ratios': [2, 1]}  # ax_3 is twice as tall as ax_4
    )
    ax_3, ax_4 = axes
    #gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1])  # Top plot 3x taller than bottom
    #ax_3 = fig_3.add_subplot(gs[0])
    #ax_4 = fig_3.add_subplot(gs[1], sharex=ax_3)

    ax_3.scatter(DT_time, DT_freq-tune, s=10, label = "DopTrack Signal", c='black' )
    scat = ax_3.scatter(DTB_time-delta_t, DTB_freq-tune-delta_f, s=10, c='red', label = rf"Shifted DopTrackBox Signal ($\Delta$t = {delta_t:.2f} s, $\Delta$f = {delta_f:.2f} Hz)")#, c=ratio_to_db(DTB_snr))
    
    ax_3.set_ylabel(f'Frequency w.r.t. Tuning Frequency [Hz]')
    ax_3.set_title(f'Comparison of Pass {file}')

    ax_3.set_xlim(lowest_minimum, higest_maximum)
    ax_3.grid()
    ax_3.legend()

    ax_4.scatter(selected_DTB_time, freq_diff, s=10, c='black')
    ax_4.set_xlabel('Time [s]')
    ax_4.set_ylabel('Frequency Difference [Hz]')
    ax_4.grid()

    #cbar = fig_3.colorbar(scat, ax=[ax_3, ax_4], orientation='horizontal', pad=0.15, aspect=40)
    #cbar.set_label('Signal-to-Noise Ratio [dB]') 

    if RECOMPUTE_TIMESTAMPS:
        os.makedirs(PLOT_LOC+"DTBvsDT_V2_comp_fitted_recomputed_timestamps", exist_ok=True)
        fig_3.savefig(os.path.join(PLOT_LOC, f"DTBvsDT_V2_comp_fitted_recomputed_timestamps/V2_shifted_freq_recomputed_timestamps_{file}.png"))        
    else:
        os.makedirs(PLOT_LOC+"DTBvsDT_V2_comp_fitted", exist_ok=True)
        fig_3.savefig(os.path.join(PLOT_LOC, f"DTBvsDT_V2_comp_fitted/V2_shifted_freq_{file}.png"))
    plt.close(fig_3)
    gc.collect()    

def save_fit_params_yaml(delta_t, delta_f, delta_t_uncertainty, delta_f_uncertainty, file, recomputed):
    fit_params = {'delta_t': float(delta_t), 'delta_f': float(delta_f), 'delta_t_uncertainty': float(delta_t_uncertainty), 'delta_f_uncertainty': float(delta_f_uncertainty)}
    if recomputed:
        outpath = EXTRACTION_LOC + "/FittedBiasRecomputedV2/"
    else:
        outpath = EXTRACTION_LOC + "/FittedBiasV2/"
    os.makedirs(outpath, exist_ok=True)
    with open(outpath+f"bias_fit_params_{file}.yml", 'w') as f:
        yaml.dump(fit_params, f)

def data_shift_residuals(params, DTB_time, DTB_freq, interpolation_function, DT_time, DT_tca):
    delta_t, delta_f = params
    shifted_DTB_time = DTB_time - delta_t 
    shifted_DTB_freq = DTB_freq - delta_f

    #mask = (DTB_time >= DT_time.min()+2) & (DTB_time <= DT_time.max()-2)
    mask = (DTB_time >= DT_tca - 180) & (DTB_time <= DT_tca + 180)

    DT_freq_interpolated = interpolation_function(shifted_DTB_time[mask])
    return shifted_DTB_freq[mask] - DT_freq_interpolated


def update_processed_log(processed_file):
    with open(LOG_FILE, 'a') as f:
        f.write(processed_file + '\n')

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: curvefit.py <base_filepath>")
        sys.exit(1)

    FILE = sys.argv[1]
    try:    
        # First extract relevant info from the filename
        parts = FILE.split('_')
        name = parts[0]  
        date_part = parts[2]
        year = date_part[:4]
        center_frequency = 1000*int(parts[-1].replace("kHz", ""))
        
        # Then find and load the yml file to retrieve tuning freq and supposed rec start time
        yml = find_yml(FILE, name, year)
        TUNE = yml['Sat']['State']['Tuning Frequency']
        INDICATED_REC_START = yml['Sat']['Record']['time1 UTC'].replace(tzinfo=timezone.utc) # TIme is stored in YML as UTC, the filename is Local Time
        REC_END = yml['Sat']['Record']['time2 UTC'].replace(tzinfo=timezone.utc)
        REC_SAMPLES = yml['Sat']['Record']['num_sample']
        REC_SAMPLE_RATE = yml['Sat']['Record']['sample_rate']
        REC_DURATION = REC_SAMPLES/REC_SAMPLE_RATE
        RECOMPUTED_REC_START = REC_END - timedelta(seconds=REC_DURATION)

        for RECOMPUTE_TIMESTAMPS in [True, False]:
            if RECOMPUTE_TIMESTAMPS:
                REC_START = RECOMPUTED_REC_START
            else:
                REC_START = INDICATED_REC_START
            
            freq_offset = center_frequency - TUNE

            # Retrieve data and perform fit
            DTB_data, mask_width  = load_DTB_data(FILE)
            DT_freq_data, DT_tca_wrt_start, DT_fca, DT_starttime = load_DT_data(FILE, name, year)  

            #print(DT_freq_data[:,0])
            if DT_freq_data is not None:
                #print(INDICATED_REC_START, RECOMPUTED_REC_START, DT_starttime)
                #DT_tca_wrt_start = (DT_tca - DT_starttime).total_seconds()
                starttime_diff = (DT_starttime - REC_START).total_seconds()
                #print(starttime_diff)
                DTB_data[:,0] -= starttime_diff # Start times of the recordings is not the same, so shift DTB data to start w.r.t. DTB start time
                interpolation_function = interp1d(DT_freq_data[:,0], DT_freq_data[:,1], kind='linear', fill_value="extrapolate", )
                
                # Only take datapoints around X Hz from the DT data
                DT_freq_interp = interpolation_function(DTB_data[:,0])
                freq_diff = np.abs(DTB_data[:,1] - DT_freq_interp)
                mask = freq_diff < 150
                filtered_DTB_data = DTB_data[mask]

                plot_raw_freq(DT_freq_data[:,0], DT_freq_data[:,1], filtered_DTB_data[:,0], filtered_DTB_data[:,1], filtered_DTB_data[:,4], TUNE, FILE, interpolation_function)

                params0 = [0.0, 0.0]  # [delta_t, delta_f]
                lower = [-6, -50]
                upper = [6, 50]
                #lower = [-6, -0.1]
                #upper = [6, 0.1]
                result = least_squares(data_shift_residuals, params0, bounds=(lower, upper), jac='3-point', args=(filtered_DTB_data[:,0], filtered_DTB_data[:,1], interpolation_function, DT_freq_data[:,0], DT_tca_wrt_start))
                delta_t, delta_f = result.x

                data_res = data_shift_residuals(params0, DTB_data[:,0], DTB_data[:,1], interpolation_function, DT_freq_data[:,0], DT_tca_wrt_start)
                J = result.jac
                residual_variance = np.sum(result.fun**2) / (len(data_res) - len(result.x))
                cov = residual_variance * np.linalg.inv(J.T @ J)
                param_uncertainties = np.sqrt(np.diag(cov))
                delta_t_uncert, delta_f_uncert = param_uncertainties

                print(f"Estimated time shift (delta_t +- uncertainty): {delta_t:.2f} +/- {delta_t_uncert:.2f}")
                print(f"Estimated frequency shift (delta_f +- uncertainty): {delta_f:.2f} +/- {delta_f_uncert:.2f}")

                plot_fitted_freq(DT_freq_data[:,0], DT_freq_data[:,1], filtered_DTB_data[:,0], filtered_DTB_data[:,1], filtered_DTB_data[:,4], TUNE, FILE, delta_t, delta_f, interpolation_function)
                save_fit_params_yaml(delta_t, delta_f, delta_t_uncert, delta_f_uncert, FILE, RECOMPUTE_TIMESTAMPS)

        #update_processed_log(FILE)

    except Exception as e:
        print(f"[ERROR] Failed to process {FILE}: {e}")
        sys.exit(1)
