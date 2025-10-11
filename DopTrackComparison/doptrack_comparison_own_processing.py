#!/usr/bin/env python3

import os
import sys
import yaml
import gc
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from cmcrameri import cm
from numpy.fft import fft, fftfreq, fftshift
from datetime import datetime, timezone, timedelta
from matplotlib.colors import LogNorm
from scipy.optimize import curve_fit
from scipy import constants


plt.switch_backend('Agg')
colours = cm.lapaz

LOG_FILE = '/home/doptrackbox/DTB_Recordings/locally_processed/dt_comparison_own_processing.log'
EXTRACTION_LOC = '/home/doptrackbox/DTB_Recordings/locally_processed/Processed/'
LOCAL_LOC = '/home/doptrackbox/DTB_Recordings/data/'
REMOTE_LOC = '/home/doptrackbox/tudelft_webdrive/staff-umbrella/doptrack/archive/Dylano/DopTrackBox/'
LOCi = '/home/doptrackbox/DTB_Recordings/locally_processed/'
DopTrackLoc = '/home/doptrackbox/tudelft_webdrive/staff-umbrella/doptrack/'
PLOT_LOC = os.path.join(LOCi, 'Plots/DopTrack/')

def compute_range_rate(frequencies, f0):
    return constants.c*(1-frequencies/f0)

def compute_doppler(dv, f0):
    return f0*(1-(dv/constants.c))

def find_yml(file, name, year):
    yml_filename = file.rsplit('_', 1)[0]
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

def filter_SNR(data, treshold_dB):
    treshold_ratio = db_to_ratio(treshold_dB)
    filtered = data[data[:,4] > treshold_ratio]
    return filtered


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
    yml_path_L0 = get_yml_path("L0", file, name, year)

    if yml_path_L0 is not None:
        with open(yml_path_L0, 'r') as f:
            yml_settings = yaml.load(f, Loader=yaml.FullLoader)

        DT_start_time = yml_settings['recording']['time_start'].replace(tzinfo=timezone.utc)


        DT_data_path = LOCi + "Processed/DopTrack/ExtractedSignal/" + yml_path_L0.rsplit('/',1)[-1].rsplit('.',1)[0] + "_" + file.rsplit('_',1)[-1] + ".txt"
       # DT_SNR_datapath = LOCi + "Processed/DopTrack/ExtractedSNR/" + yml_path_L1C.rsplit('/',1)[-1].rsplit('.',1)[0] + "_" + file.rsplit('_',1)[-1] + ".txt"
        DT_data = np.loadtxt(DT_data_path, skiprows=1)
        #DT_SNR = np.loadtxt(DT_SNR_datapath, skiprows=1)

        #DT_data, dt = combine_data(DT_freq, DT_SNR)

        return DT_data, DT_start_time
    else:
        return None, None
 
def plot_2d_freq(DT_data, DTB_data, tune, file, filtered):
    fig_1, ax_1 = plt.subplots(figsize=(10,5)) 

    if filtered:
        DT_data = filter_SNR(DT_data, 7.75)
        DTB_data = filter_SNR(DTB_data, 7.1)

    DT_time = DT_data[:,0]
    DT_freq = DT_data[:,1]
    DT_snr = DT_data[:,4]

    DTB_time = DTB_data[:,0]
    DTB_freq = DTB_data[:,1]
    DTB_snr = DTB_data[:,4]

    ax_1.scatter(DT_time, DT_freq-tune, s=10, label = "Extracted DopTrack Signal", marker='s', c=ratio_to_db(DT_snr))
    scat = ax_1.scatter(DTB_time, DTB_freq-tune, s=10, label = "Extracted DopTrackBox Signal", c=ratio_to_db(DTB_snr))
    
    cbar = plt.colorbar(scat)
    cbar.set_label('Signal-to-Noise Ratio [dB]') 
    ax_1.set_xlabel('Time [s]')
    ax_1.set_ylabel('Frequency [Hz]')
    ax_1.set_title(f'Extracted Signal Frequency w.r.t. tuning frequency (= {tune}Hz)')

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
    if filtered:
        os.makedirs(PLOT_LOC+"DTBvsDT_freq_own_processing_filtered", exist_ok=True)
        fig_1.savefig(os.path.join(PLOT_LOC, f"DTBvsDT_freq_own_processing_filtered/freq_{file}.png"))
    else:
        os.makedirs(PLOT_LOC+"DTBvsDT_freq_own_processing", exist_ok=True)
        fig_1.savefig(os.path.join(PLOT_LOC, f"DTBvsDT_freq_own_processing/freq_{file}.png"))
        
    plt.close(fig_1)
    gc.collect()

def plot_sig_noise_snr(DT_data, DTB_data, tune, file):
    DT_time = DT_data[:,0]
    DT_freq = DT_data[:,1]
    DT_sig = DT_data[:,2]
    DT_noise = DT_data[:,3]
    DT_snr = DT_data[:,4]

    DTB_time = DTB_data[:,0]
    DTB_freq = DTB_data[:,1]
    DTB_sig = DTB_data[:,2] 
    DTB_noise = DTB_data[:,3] 
    DTB_snr = DTB_data[:,4] 

    fig_1, axes = plt.subplots(2,1,figsize=(6,5)) 
    fig_3, ax3 = plt.subplots(1,1,figsize=(6,5)) 
    fig_4, ax4 = plt.subplots(1,1,figsize=(6,5)) 
    ax1, ax2 = axes

    ax1.scatter(DT_time, DT_sig, s=10, label = "DopTrack")
    ax1.scatter(DTB_time, DTB_sig, s=10, label = "DopTrackBox")
    
    #scat = ax1.scatter(DTB_time, DTB_freq-tune, s=10, label = "Extracted DopTrackBox Signal", c=ratio_to_db(DTB_snr))
    ax2.scatter(DT_time, DT_noise, s=10, label = "DopTrack") 
    ax2.scatter(DTB_time, DTB_noise, s=10, label = "DopTrackBox")

    ax3.scatter(DT_time, DT_snr, s=10, label = "DopTrack")
    ax3.scatter(DTB_time, DTB_snr, s=10, label = "DopTrackBox")
    ax4.scatter(DT_time, ratio_to_db(DT_snr), s=10, label = "DopTrack") 
    ax4.scatter(DTB_time, ratio_to_db(DTB_snr), s=10, label = "DopTrackBox")
    
    #cbar = plt.colorbar(scat)
    #cbar.set_label('Signal-to-Noise Ratio [dB]') 
    ax1.set_yscale('log')
    ax2.set_yscale('log')
    ax1.set_ylabel('Signal Power [-]')
    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('Noise Power [-]')
    ax1.set_title(f'Extracted Signal and Noise Power')

    ax3.set_xlabel('Time [s]')
    ax4.set_xlabel('Time [s]')
    ax3.set_ylabel('Signal-to-Noise Ratio [-]')
    ax4.set_ylabel('Signal-to-Noise Ratio [dB]')

    ax3.set_title(f"SNR Comparison of Pass {file}")
    ax4.set_title(f"SNR Comparison of Pass {file}")

    #ax_1.set_ylim(0, 40)
    lowest_minimum = min(np.min(DT_time), np.min(DTB_time))
    higest_minimum = max(np.min(DT_time), np.min(DTB_time))
    higest_maximum = max(np.max(DT_time), np.max(DTB_time))
    lowest_maximum = min(np.max(DT_time), np.max(DTB_time))
    
    ax1.set_xlim(lowest_minimum, higest_maximum)
    ax3.set_xlim(lowest_minimum, higest_maximum)
    ax4.set_xlim(lowest_minimum, higest_maximum)
    #if mask_width != 0:
    #    ax_1.set_ylim(freq_offset-(mask_width/2), freq_offset+(mask_width/2))
    ax1.grid()
    ax2.grid()
    ax3.grid()
    ax4.grid()
    #fig_1.set_size_inches(19, 10)
    ax1.legend()
    ax3.legend()
    ax4.legend()
    
    fig_1.tight_layout()
    fig_3.tight_layout()
    fig_4.tight_layout()
    #if filtered:
    #    os.makedirs(PLOT_LOC+"DTBvsDT_freq_own_processing_filtered", exist_ok=True)
    #    fig_1.savefig(os.path.join(PLOT_LOC, f"DTBvsDT_freq_own_processing_filtered/freq_{file}.png"))
    #else:
    os.makedirs(PLOT_LOC+"DTBvsDT_sig_noise_own_processing", exist_ok=True)
    fig_1.savefig(os.path.join(PLOT_LOC, f"DTBvsDT_sig_noise_own_processing/sig_power_and_noise_comparison_{file}.png"))

    os.makedirs(PLOT_LOC+"DTBvsDT_snr_own_processing", exist_ok=True)
    fig_3.savefig(os.path.join(PLOT_LOC, f"DTBvsDT_snr_own_processing/snr_comparison_{file}.png"))

    os.makedirs(PLOT_LOC+"DTBvsDT_snr_db_own_processing", exist_ok=True)
    fig_4.savefig(os.path.join(PLOT_LOC, f"DTBvsDT_snr_db_own_processing/snr_comparison_db_{file}.png"))
        
    plt.close(fig_1)
    plt.close(fig_3)
    plt.close(fig_4)
    gc.collect()    


def update_processed_log(processed_file):
    with open(LOG_FILE, 'a') as f:
        f.write(processed_file + '\n')

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: curvefit.py <base_filepath>")
        sys.exit(1)

    FILE = sys.argv[1]
    print(FILE)
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
        REC_START = yml['Sat']['Record']['time1 UTC'].replace(tzinfo=timezone.utc) # Time is stored in YML as UTC, the filename is Local Time
        REC_END = yml['Sat']['Record']['time2 UTC'].replace(tzinfo=timezone.utc)
        REC_SAMPLES = yml['Sat']['Record']['num_sample']
        REC_SAMPLE_RATE = yml['Sat']['Record']['sample_rate']
        REC_DURATION = REC_SAMPLES/REC_SAMPLE_RATE
        RECOMPUTED_REC_START = REC_END - timedelta(seconds=REC_DURATION)
        freq_offset = center_frequency - TUNE

        # Retrieve data and perform fit
        DTB_data, mask_width  = load_DTB_data(FILE)
        DT_data, DT_starttime = load_DT_data(FILE, name, year)
        
        #print(DT_freq_data[:,0])
        if DT_data is not None:
            print("Found DT data")
            starttime_diff = (DT_starttime - RECOMPUTED_REC_START).total_seconds()
            DTB_data[:,0] -= starttime_diff # Start times of the recordings is not the same, so shift DT data to start w.r.t. DTB start time
            plot_2d_freq(DT_data, DTB_data, TUNE, FILE, False)
            plot_2d_freq(DT_data, DTB_data, TUNE, FILE, True)
            plot_sig_noise_snr(DT_data, DTB_data, TUNE, FILE)
        else:
            print("NO DT data")

        update_processed_log(FILE)

    except Exception as e:
        print(f"[ERROR] Failed to process {FILE}: {e}")
        sys.exit(1)
