#!/usr/bin/env python3
import os
import subprocess
from datetime import datetime, timezone
import time

LOC = '/home/doptrackbox/DTB_Recordings/locally_processed/Processed/RawSignalsFreq/'
LOG_FILE = '/home/doptrackbox/DTB_Recordings/locally_processed/plotted_files.log'

print("Running plotting scheduler script")

def load_processed_log():
    if not os.path.exists(LOG_FILE):
        return set()
    with open(LOG_FILE, 'r') as f:
        return set(line.strip() for line in f if line.strip())

def get_unprocessed_recordings():
    processed = load_processed_log()
    all_files = os.listdir(LOC)
    now = datetime.now(timezone.utc)

    candidates = []
    for base_file in sorted(all_files):
        base = base_file.rsplit('.', 1)[0]
        if base not in processed:# or base_name.startswith("EMPTY"):
            candidates.append(base)
    return candidates

# MAIN: schedule 1 by 1
unprocessed_files = get_unprocessed_recordings()
for file_base in unprocessed_files:
    print(f"[INFO] Launching plotting for: {file_base}")
    result = subprocess.run(["python3", "make_pass_plots.py", file_base])
    if result.returncode != 0:
        print(f"[ERROR] Processing {os.path.basename(file_base)} failed with exit code {result.returncode}")
