#!/usr/bin/env python3
import os
import subprocess
from datetime import datetime, timezone
import time

LOC = '/home/doptrackbox/DTB_Recordings/locally_processed/Processed/'
LOG_FILE = '/home/doptrackbox/DTB_Recordings/locally_processed/dt_comparison_own_processing.log'
MNT_DRIVE = '/home/doptrackbox/DTB_Software/GroundControl_DTB_Revamp/Profiles/doptrackbox-delft/Fitlet3.computer/'
LOCAL_LOC = '/home/doptrackbox/DTB_Recordings/data/'
REMOTE_LOC = '/home/doptrackbox/tudelft_webdrive/staff-umbrella/doptrack/archive/Dylano/DopTrackBox/'
MNT_READY = os.path.exists(REMOTE_LOC) # Used to check if webdrive is already mounted. If so, do not unmount at the end because other processes might be using it

print("Running plotting scheduler script")

def load_processed_log():
    if not os.path.exists(LOG_FILE):
        return set()
    with open(LOG_FILE, 'r') as f:
        return set(line.strip() for line in f if line.strip())

def get_unprocessed_recordings():
    processed = load_processed_log()
    all_files = os.listdir(LOC+"ExtractedSignal/")
    now = datetime.now(timezone.utc)

    candidates = []
    for base_file in sorted(all_files):
        base = base_file.rsplit('.', 1)[0]
        if base not in processed and base.startswith("FUNcube"):
            candidates.append(base)
    return candidates

# MAIN: schedule 1 by 1
if not MNT_READY:
    out = subprocess.run([MNT_DRIVE+"mount_webdrive.sh"])
unprocessed_files = get_unprocessed_recordings()
for file_base in unprocessed_files:
    print(f"[INFO] Launching DT vs DTB comparison for: {file_base}")
    result = subprocess.run(["python3", "doptrack_comparison_own_processing.py", file_base])
    if result.returncode != 0:
        print(f"[ERROR] Processing {os.path.basename(file_base)} failed with exit code {result.returncode}")
if not MNT_READY:
    out = subprocess.run([MNT_DRIVE+"unmount_webdrive.sh"])