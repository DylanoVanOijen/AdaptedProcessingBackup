import os
import sys
import yaml
import numpy as np
from datetime import datetime, timezone, timedelta
import subprocess


np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})

LOG_FILE = '/home/doptrackbox/DTB_Recordings/locally_processed/dt_comparison.log'
EXTRACTION_LOC = '/home/doptrackbox/DTB_Recordings/locally_processed/Processed/'
LOCAL_LOC = '/home/doptrackbox/DTB_Recordings/data/FUNcube-1/2025/'
REMOTE_LOC = '/home/doptrackbox/tudelft_webdrive/staff-umbrella/doptrack/archive/Dylano/DopTrackBox/'
LOCi = '/home/doptrackbox/DTB_Recordings/locally_processed/'
DopTrackLoc = '/home/doptrackbox/tudelft_webdrive/staff-umbrella/doptrack/'
MNT_DRIVE = '/home/doptrackbox/DTB_Software/GroundControl_DTB_Revamp/Profiles/doptrackbox-delft/Fitlet3.computer/'
PLOT_LOC = os.path.join(LOCi, 'Plots/')

MNT_READY = os.path.exists(REMOTE_LOC) # Used to check if webdrive is already mounted. If so, do not unmount at the end because other processes might be using it
if not MNT_READY:
    out = subprocess.run([MNT_DRIVE+"mount_webdrive.sh"])

files = ["09221504",
         "09221638",
         "09230410",
         "09230544",
         "09230720",
         "09231635",
         "09240407",
         "09240542",
         "09240717",
         "09241458",
         "09241633",
         "09250404",
         "09250539",
         "09250714",
         "09251456",
         "09251630",
         "09260402",
         "09260536",
         "09260712",
         "09261627",
         "09270533",
         "09271624",
         "09280530",
         "09281621",
         "09290527",
         "08281604",
         "08300509",
         "08301600",
         "08310507",
         "09010505",
         "09020502",
         "09040458",
         "09050455",
         "09051546",
         "09060453",
         "09070450",
         "09080448",
         "09100443",
         "09111532",
         "09160428",
         "09160603",
         "09161519",
         "09170425",
         "09170600",
         "09171517",
         "09171652",
         "09180423",
         "09180558",
         "09190420",
         "09200418",
         "09210550",
         "09211641",
         "09220547",
         "10021436",
         "10021746",
         "10030515",
         "10030651"]

def get_yml(subdir, file, name="FUNcube-1", year="2025"):
    if os.path.exists(DopTrackLoc + f"{subdir}/{name}/{year}/{file}.yml"):
        yml_path = DopTrackLoc + f"{subdir}/{name}/{year}/{file}.yml"
        return yml_path
    else:
        parts = file.split('_')
        date_part = parts[2]
        date_time = datetime.strptime(date_part, "%Y%m%d%H%M")

        for m in range(-10,10):
            candidate_datetime = date_time + timedelta(minutes=m)
            yml_filename = f"{parts[0]}_{parts[1]}_"+candidate_datetime.strftime("%Y%m%d%H%M")
            if os.path.exists(DopTrackLoc + f"{subdir}/{name}/{year}/{yml_filename}.yml"):
                yml_path = DopTrackLoc + f"{subdir}/{name}/{year}/{yml_filename}.yml"
                return yml_path        
    return None


diffs_1 = []
diffs_2 = []
for file in files:
    full_ident = f"FUNcube-1_39444_2025{file}_145935kHz"
    yml_ident = full_ident.rsplit('_', 1)[0]
    L0_yml = get_yml("L0", yml_ident)
    processed_L0_yml = get_yml("products/L0", yml_ident)
    if L0_yml is not None and processed_L0_yml is not None:
        with open(L0_yml, 'r') as f:
            L0_data = yaml.load(f, Loader=yaml.FullLoader)

        with open(processed_L0_yml, 'r') as f:
            processed_L0_data = yaml.load(f, Loader=yaml.FullLoader)

        t1 = L0_data["Sat"]["Record"]["time1 UTC"].replace(tzinfo=timezone.utc)
        t2 = L0_data["Sat"]["Record"]["time2 UTC"].replace(tzinfo=timezone.utc)

        t_start = processed_L0_data["recording"]["time_start"].replace(tzinfo=timezone.utc)
        t_stop = processed_L0_data["recording"]["time_stop"].replace(tzinfo=timezone.utc)
        duration = processed_L0_data["recording"]["duration"]

        print(t1, t2)
        print(t_start, t_stop)
        diffs_1.append((t2 - t_stop).total_seconds())
        diffs_2.append((t2-t1).total_seconds() - duration)

    else:
        print(f"Missing YML for file: {yml_ident}")    

diffs_1 = np.array(diffs_1)
mean_1 = np.mean(diffs_1)
std_1 = np.std(diffs_1)

print(f"Mean = {mean_1:.5f}")
print(f"STD = {std_1:.5f}")

diffs_2 = np.array(diffs_2)
mean_2 = np.mean(diffs_2)
std_2 = np.std(diffs_2)

print(f"Mean = {mean_2:.5f}")
print(f"STD = {std_2:.5f}")


if not MNT_READY:
    out = subprocess.run([MNT_DRIVE+"unmount_webdrive.sh"])