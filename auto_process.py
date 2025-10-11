#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt 
import os
from cmcrameri import cm
colours = cm.lapaz

from numpy.fft import fftshift, fftfreq, fft
from scipy.stats import mode
import yaml
from datetime import datetime, timezone
import gc

plt.switch_backend('Agg')  # <<< prevents GUI memory leaks if backend not headless


# Data locations
CORR = 0
LOC = '/home/doptrackbox/DTB_Recordings/data/'
LOCi = '/home/doptrackbox/DTB_Recordings/locally_processed/'
PLOT_LOC = os.path.join(LOCi, 'Plots/')
PROCESS_LOC = os.path.join(LOCi, 'Processed/')
LOG_FILE = os.path.join(LOCi, 'processed_files.log')

params = {'axes.labelsize': 20,'axes.titlesize': 20, 'xtick.labelsize': 16, 'ytick.labelsize': 16, 'figure.titlesize': 20, 'legend.fontsize': 16}
plt.rcParams.update(params)

def load_processed_log():
    """Load list of previously processed files from log."""
    if not os.path.exists(LOG_FILE):
        return set()
    with open(LOG_FILE, 'r') as f:
        return set(line.strip() for line in f if line.strip())

def update_processed_log(processed_files):
    """Append newly processed files to the log."""
    with open(LOG_FILE, 'a') as f:
        for filename in processed_files:
            f.write(filename + '\n')

def get_unprocessed_recordings():
    """Get list of all available recordings in LOC that:
       - Haven't been processed yet,
       - Are older than 20 minutes,
       - Do not start with 'testsat'.
    """
    processed = load_processed_log()
    all_files = os.listdir(LOC)
    candidates = []

    now = datetime.now(timezone.utc)
    min_age_minutes = 20

    bases = set()
    for fname in all_files:
        if fname.endswith('.32fc') or fname.endswith('.yml'):
            base = fname.rsplit('.', 1)[0]
            bases.add(base)

    for base in sorted(bases):
        if base in processed:
            continue
        if base.startswith("testsat"):
            continue

        # Extract timestamp
        try:
            timestamp_str = base.split('_')[-1]
            recording_time = datetime.strptime(timestamp_str, '%Y%m%d%H%M').replace(tzinfo=timezone.utc)
            age_minutes = (now - recording_time).total_seconds() / 60.0
            if age_minutes < min_age_minutes:
                continue
        except Exception as e:
            print(f"[WARN] Skipping file with invalid timestamp '{base}': {e}")
            continue

        fc_path = os.path.join(LOC, base + '.32fc')
        yml_path = os.path.join(LOC, base + '.yml')
        if os.path.exists(fc_path) and os.path.exists(yml_path):
            candidates.append(base)

    return candidates


def folder_check(folder_name):
    os.makedirs(folder_name, exist_ok=True)

def read_arraydata_from_fc32(path, n_bins, bin_size):
    count = 2 * bin_size
    with open(path, 'rb') as file:
        for _ in range(n_bins):
            array = np.fromfile(file, dtype=np.float32, count=count)
            if array.size != count:
                break  # <<< handle short reads
            complex_array = array[::2] + 1j * array[1::2]
            yield complex_array

def load_data(file, dt, duration, n_samples):
    n_bins = int(duration / dt)
    bin_size = int(n_samples / n_bins)
    yield from read_arraydata_from_fc32(LOC + file + ".32fc", n_bins, bin_size)

def extract_signal(data, dt, sampling_rate, tune_freq, mask_center=None, mask_width=None):
    nfft = int(dt * sampling_rate)  # Use optimal value of nfft
    frequency = fftshift(fftfreq(nfft, 1/sampling_rate)) + tune_freq

    if mask_width:
        lower = theoretical_freq + mask_center - mask_width / 2
        upper = theoretical_freq + mask_center + mask_width / 2
        # Exclude upper limit from mask to give nicer mask length, e.g. 2800 instead of 2801 when including upper limit.
        mask, = np.nonzero((lower <= frequency) & (frequency < upper))
        frequency = frequency[mask]
    else:
        mask = None

    rows = []
    noise = []
    signal_average = []
    signal_maximum = []
    signal_median = []
    signal_freq = []

    for i, chunk in enumerate(data):
        noise_floor = []
        signal_line = []

        row = abs(fftshift(fft(chunk, nfft))) # PSD per chunk
        row = row[mask] if mask is not None else row # Apply mask
        mean = np.average(row)
        deviation = np.std(row, dtype=np.float64)
        signal_limit = mean + 2*deviation

        # Compute noise entries
        for entry in row:
            if entry < mean+deviation:
                noise_floor.append(entry)
            if entry >= signal_limit:
                signal_line.append(entry)
        
        median_noise = np.median(noise_floor)
        noise.append(median_noise)

        # Compute signal
        signal_average.append(np.average(signal_line))
        signal_maximum.append(np.max(signal_line,initial=0))
        signal_median.append(np.median(signal_line))
        signal_freq.append(frequency[np.argmax(row)])
        #print(row)
        #row = row/np.average(noise_floor)
        # noise calculation using only the mean
        #noise.append(np.average(row))
        #row = row/np.average(row)

        row_SNR = row / median_noise
        #row_SNR_dB = np.log10(row_SNR)        
        rows.append(row_SNR)
    image = np.array(rows)
    time = np.arange(image.shape[0]) * dt
    #print(signal_freq)
    return np.array(signal_freq), time, frequency, image

def process_dataset_ADEV(file, tune_freq, freq_offset, dt, sampling_rate, mask_width, duration, n_samples):
    sig_freqs, time, freqs, spec = extract_signal(
        load_data(file, dt, duration, n_samples),
        dt,
        sampling_rate,
        tune_freq,
        freq_offset,
        mask_width
    )
    label = file
    plot_spectogram(spec, time, freqs, tune_freq, label)
    del sig_freqs, time, freqs, spec
    gc.collect() 

def plot_spectogram(spec, time, freqs, tuning_freq, label):
    fig_1, ax_1 = plt.subplots()
    extent = ((freqs[0]-tuning_freq), (freqs[-1]-tuning_freq+(freqs[-1]-freqs[-2])), time[-1]+(time[-1]-time[-2]), time[0])
    img_1 = ax_1.imshow(spec, aspect='auto', cmap=colours, extent=extent, interpolation='None')
    cbar = fig_1.colorbar(img_1, ax=ax_1)

    # Axis labels
    ax_1.set_xlabel(f'Frequency offset w.r.t. tuning frequency ({(tuning_freq/1e6):.6f} MHz) [Hz]')
    ax_1.set_ylabel('Time [s]')
    ax_1.set_title('Spectrogram of ' + label )
    cbar.set_label('Signal to noise ratio [-]')

    # Plotting and automatic saving
    scale = 2.5
    fig_1.set_size_inches(19, 10)
    fig_1.tight_layout()
    fig_1.savefig(PLOT_LOC + f"spec_{label}_spectral.png", dpi=scale*100, bbox_inches = 'tight')
    plt.close(fig_1)
    gc.collect()

### Process datasets
#files = ["EC20S_62500Hz_60s_testrun.32fc"]


unprocessed_files = get_unprocessed_recordings()[:2]
newly_processed = []

for FILE in unprocessed_files:
    try:
        META = LOC + FILE + '.yml'
        with open(META) as f:
            settings = yaml.load(f, Loader=yaml.FullLoader)

        TUNE = settings['Sat']['State']['Tuning Frequency']
        n_samples = settings['Sat']['Record']['num_sample']
        sample_rate = settings['Sat']['Record']['sample_rate']
        duration = n_samples / sample_rate
        dt = 0.5  # Time resolution of the FFT
        theoretical_freq = TUNE
        mask_width = sample_rate if sample_rate <= 100000 else 100000

        process_dataset_ADEV(FILE, TUNE, CORR, dt, sample_rate, mask_width, duration, n_samples)
        newly_processed.append(FILE)

        # Force cleanup
        del settings, TUNE, n_samples, sample_rate, duration, dt, mask_width
        gc.collect()
    except Exception as e:
        print(f"[ERROR] Failed to process {FILE}: {e}")

# Update the log
update_processed_log(newly_processed)
