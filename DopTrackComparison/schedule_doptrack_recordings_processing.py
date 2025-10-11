#!/usr/bin/env python3
import os
import subprocess
from datetime import datetime, timezone
import time


REMOTE_LOC = '/home/doptrackbox/tudelft_webdrive/staff-umbrella/doptrack/products/L0/'
MNT_DRIVE = '/home/doptrackbox/DTB_Software/GroundControl_DTB_Revamp/Profiles/doptrackbox-delft/Fitlet3.computer/'
LOG_FILE = '/home/doptrackbox/DTB_Recordings/locally_processed/processed_doptrack_files.log'
MNT_READY = os.path.exists(REMOTE_LOC) # Used to check if webdrive is already mounted. If so, do not unmount at the end because other processes might be using it

print("Running processing scheduler script")
skiplist = ["Delfi-C3", "Delfi-n3Xt", "Nayif-1"]


def load_processed_log():
    if not os.path.exists(LOG_FILE):
        return set()
    with open(LOG_FILE, 'r') as f:
        return set(line.strip() for line in f if line.strip())

def get_unprocessed_recordings():
    processed = load_processed_log()

    LOC = REMOTE_LOC
    all_files = os.listdir(LOC)
    now = datetime.now(timezone.utc)
    min_age_minutes = 20

    bases = set()
    for root, dirs, files in os.walk(LOC):
        for fname in files:
            if fname.endswith('.fc32') or fname.endswith('.yml'):
                base = fname.rsplit('.', 1)[0]
                bases.add(os.path.join(root, base))

    candidates = []
    for base in sorted(bases):
        base_name = os.path.basename(base)
        if base_name in processed or any(base_name.startswith(skip) for skip in skiplist):
            continue
        try:
            # This comparison is not correct, because file timestamp is local time (not UTC as indicated below), and compaing it with UTC
            timestamp_str = base_name.split('_')[-1]
            recording_time = datetime.strptime(timestamp_str, '%Y%m%d%H%M').replace(tzinfo=timezone.utc)
            if (now - recording_time).total_seconds() / 60.0 < min_age_minutes or recording_time < datetime(2025, 7, 25, 0, 0, 0, tzinfo=timezone.utc):
                continue
        except:
            continue
        if os.path.exists(base + '.fc32') and os.path.exists(base + '.yml'):
            candidates.append(base)
    return candidates

if not MNT_READY:
    out = subprocess.run([MNT_DRIVE+"mount_webdrive.sh"])

# MAIN: schedule 1 by 1
unprocessed_files = get_unprocessed_recordings()
for file_base in unprocessed_files:
    print(f"[INFO] Launching processing for: {os.path.basename(file_base)}")
    result = subprocess.run(["python3", "process_doptrack_pass.py", file_base])
    if result.returncode != 0:
        print(f"[ERROR] Processing {os.path.basename(file_base)} failed with exit code {result.returncode}")
    time.sleep(1)

# Only unmount drive if used, while it was not opened already
if REMOTE_LOC and not MNT_READY:
    out = subprocess.run([MNT_DRIVE+"unmount_webdrive.sh"])
