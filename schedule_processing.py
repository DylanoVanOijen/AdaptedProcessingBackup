#!/usr/bin/env python3
import os
import subprocess
from datetime import datetime, timezone
import time

PROCESS_REMOTE = False

LOCAL_LOC = '/home/doptrackbox/DTB_Recordings/data/'
REMOTE_LOC = '/home/doptrackbox/tudelft_webdrive/staff-umbrella/doptrack/archive/Dylano/DopTrackBox/'
MNT_DRIVE = '/home/doptrackbox/DTB_Software/GroundControl_DTB_Revamp/Profiles/doptrackbox-delft/Fitlet3.computer/'
LOG_FILE = '/home/doptrackbox/DTB_Recordings/locally_processed/processed_files.log'
MNT_READY = os.path.exists(REMOTE_LOC) # Used to check if webdrive is already mounted. If so, do not unmount at the end because other processes might be using it

print("Running processing scheduler script")
skiplist = ["CAS-6", "DIWATA-2", "FOX-1B", "VELOX-PII", "JY1Sat", "RIDU-Sat-1"]

def prepare_data_path():
    if PROCESS_REMOTE:
        if not MNT_READY:
            out = subprocess.run([MNT_DRIVE+"mount_webdrive.sh"])
        return REMOTE_LOC
    else:
        return LOCAL_LOC

def load_processed_log():
    if not os.path.exists(LOG_FILE):
        return set()
    with open(LOG_FILE, 'r') as f:
        return set(line.strip() for line in f if line.strip())

def get_unprocessed_recordings():
    processed = load_processed_log()

    LOC = prepare_data_path()
    all_files = os.listdir(LOC)
    now_local = datetime.now().astimezone()
    now_utc = now_local.astimezone(timezone.utc)
    min_age_minutes = 10

    bases = set()
    for root, dirs, files in os.walk(LOC):
        for fname in files:
            if fname.endswith('.32fc') or fname.endswith('.yml'):
                base = fname.rsplit('.', 1)[0]
                bases.add(os.path.join(root, base))

    candidates = []
    for base in sorted(bases):
        base_name = os.path.basename(base)
        #print(base_name)
        if base_name in processed or base_name.startswith("sensor_data") or base_name.startswith("testsat") or any(base_name.startswith(skip) for skip in skiplist):
            continue
        try:
            # This comparison is not correct, because file timestamp is local time (not UTC as indicated below), and compaing it with UTC
            timestamp_str = base_name.split('_')[-1]
            recording_time = datetime.strptime(timestamp_str, '%Y%m%d%H%M').astimezone()
            if (now_local - recording_time).total_seconds() / 60.0 < min_age_minutes:
                continue
        except:
            continue
        if os.path.exists(base + '.32fc') and os.path.exists(base + '.yml'):
            candidates.append(base)
    return candidates

# MAIN: schedule 1 by 1
unprocessed_files = get_unprocessed_recordings()
for file_base in unprocessed_files:
    print(f"[INFO] Launching processing for: {os.path.basename(file_base)}")
    result = subprocess.run(["python3", "process_single_pass.py", file_base])
    if result.returncode != 0:
        print(f"[ERROR] Processing {os.path.basename(file_base)} failed with exit code {result.returncode}")
    time.sleep(1)

# Only unmount drive if used, while it was not opened already
if REMOTE_LOC and not MNT_READY:
    out = subprocess.run([MNT_DRIVE+"unmount_webdrive.sh"])
