#!/usr/bin/env python3

import os
import sys
import yaml
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timezone, timedelta
import subprocess

plt.switch_backend('Agg')

# Data locations (adjust if needed)
LOC = '/home/doptrackbox/DTB_Recordings/data/'
LOCAL_LOC = '/home/doptrackbox/DTB_Recordings/data/'
REMOTE_LOC = '/home/doptrackbox/tudelft_webdrive/staff-umbrella/doptrack/archive/Dylano/DopTrackBox/'
MNT_DRIVE = '/home/doptrackbox/DTB_Software/GroundControl_DTB_Revamp/Profiles/doptrackbox-delft/Fitlet3.computer/'
LOCi = '/home/doptrackbox/DTB_Recordings/locally_processed/'
DopTrackLoc = '/home/doptrackbox/tudelft_webdrive/staff-umbrella/doptrack/'
PLOT_LOC = "./Plots/"
EXTRACTION_LOC = os.path.join(LOCi, 'Processed/')

MNT_READY = os.path.exists(REMOTE_LOC) # Used to check if webdrive is already mounted. If so, do not unmount at the end because other processes might be using it
if not MNT_READY:
    out = subprocess.run([MNT_DRIVE+"mount_webdrive.sh"])

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
        print(f"[ERROR] Failed to retrieve YML file for file {file}: {e}")
        sys.exit(1)
    return settings

def db_to_ratio(db):
    return 10**(db/10)

def ratio_to_db(ratio):
    return 10*np.log10(ratio)

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
        print(DT_data_path)
        DT_data = np.loadtxt(DT_data_path, skiprows=1)

        return DT_data, DT_start_time
    else:
        return None, None
    
#file_names = [  "FUNcube-1_39444_202509291618_145935kHz",
#                "FUNcube-1_39444_202510011612_145935kHz",
#                "FUNcube-1_39444_202510051600_145935kHz",
#                "FUNcube-1_39444_202510071554_145935kHz"]

file_names = [  "FUNcube-1_39444_202509291618_145935kHz",
                "FUNcube-1_39444_202510011612_145935kHz",
                "FUNcube-1_39444_202510041603_145935kHz",
                "FUNcube-1_39444_202510071554_145935kHz"]

labels = ["Without GPS Clock", "Antenna Exchange", "Relocated DopTrackBox & Antenna", "Box Door Opened"]

# Define plots
rows = 2
columns = 2
fig, axes = plt.subplots(rows,columns, figsize=(10,10), sharey=True)
axes = axes.flatten()

current_row = 1
current_col = 1

for file, label, ax in zip(file_names, labels, axes):

    # First extract relevant info from the filename
    parts = file.split('_')
    name = parts[0]  
    date_part = parts[2]
    year = date_part[:4]
    center_frequency = 1000*int(parts[-1].replace("kHz", ""))
    
    # Then find and load the yml file to retrieve tuning freq and supposed rec start time
    yml = find_yml(file, name, year)
    TUNE = yml['Sat']['State']['Tuning Frequency']
    REC_START = yml['Sat']['Record']['time1 UTC'].replace(tzinfo=timezone.utc) # Time is stored in YML as UTC, the filename is Local Time
    REC_END = yml['Sat']['Record']['time2 UTC'].replace(tzinfo=timezone.utc)
    REC_SAMPLES = yml['Sat']['Record']['num_sample']
    REC_SAMPLE_RATE = yml['Sat']['Record']['sample_rate']
    REC_DURATION = REC_SAMPLES/REC_SAMPLE_RATE
    RECOMPUTED_REC_START = REC_END - timedelta(seconds=REC_DURATION)
    freq_offset = center_frequency - TUNE

    # Retrieve data
    DTB_data, mask_width  = load_DTB_data(file)
    DT_data, DT_starttime = load_DT_data(file, name, year)

    starttime_diff = (DT_starttime - RECOMPUTED_REC_START).total_seconds()
    DTB_data[:,0] -= starttime_diff # Start times of the recordings is not the same, so shift DT data to start w.r.t. DTB start time

    DT_time = DT_data[:,0]
    DT_snr = DT_data[:,4]

    DTB_time = DTB_data[:,0]
    DTB_snr = DTB_data[:,4] 

    ax.scatter(DT_time, ratio_to_db(DT_snr), s=10, label = "DopTrack") 
    ax.scatter(DTB_time, ratio_to_db(DTB_snr), s=10, label = "DopTrackBox")
    
    file_without_freq = file.rsplit('_', 1)[0]
    ax.set_xlim(DT_time[0], DT_time[-1])    
    ax.set_title(label + f"\n ({file_without_freq})")
    if current_row == rows:
        ax.set_xlabel("Time [s]")
    if current_col == 1:
        ax.set_ylabel('Signal-to-Noise Ratio [dB]')
    ax.set_ylabel('Signal-to-Noise Ratio [dB]')
    ax.grid()
    ax.legend()

    current_col += 1
    if current_col > columns:
        current_row += 1
        current_col = 1

fig.tight_layout()
fig.savefig(PLOT_LOC + f"DTBvsDT_outside_changes.png")

if not MNT_READY:
    out = subprocess.run([MNT_DRIVE+"unmount_webdrive.sh"])