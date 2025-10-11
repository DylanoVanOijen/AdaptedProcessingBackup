import numpy as np
import matplotlib.pyplot as plt 
import os
from cmcrameri import cm
from numpy.fft import fftshift, fftfreq, fft
from scipy.stats import mode
import yaml

colours = cm.lapaz

### User input
sat = "FUNcube-1"
norad = "39444"

sat = "testsat"
norad = "99999"
#files_dates = ['202507250446']
""" files_dates = ['202507250446', 
               '202507251537',
               '202507251713',
               '202507260444',
               '202507260619',
               '202507261535',
               '202507261711',
               '202507270442',
               '202507270617',
               '202507271534',
               '202507271709',
               '202507280441',
               '202507280616',] """
files_dates = ['202500000000','202500000001'] # test recording of 100.7Mhz

#CORR = 8000	# Frequency correction in Hz w.r.t. tuning freq
CORR = 0

# Data input settings
#LOC = 'C:/Users/Dylan/OneDrive/Bureaublad/TUDelft/Thesis/Analysis/Amber_data/Delfi-N3XT/2022/'
LOC = '/home/doptrackbox/DTB_Recordings/data/'

# Plot and processed data output settings
LOCi = '/home/doptrackbox/DTB_Recordings/locally_processed/'
PLOT_LOC = LOCi + 'Plots/'
PROCESS_LOC = LOCi + 'Processed/'

params = {'axes.labelsize': 20,'axes.titlesize': 20, 'xtick.labelsize': 16, 'ytick.labelsize': 16, 'figure.titlesize': 20, 'legend.fontsize': 16}
plt.rcParams.update(params)

#theoretical_freq = 145.935*10**6
#theoretical_freq = 145.950*10**6 
#theoretical_freq = 145.934500*10**6 
#theoretical_freq = 145.933000*10**6 

def folder_check(folder_name):
    os.makedirs(folder_name, exist_ok=True)

def read_arraydata_from_fc32(path, n_bins: int, bin_size: int):
    count = 2 * bin_size
    with open(path, 'r') as file:
        for i in range(n_bins):
            array = np.fromfile(file, dtype=np.float32, count=count)
            complex_array = np.zeros(int(len(array) / 2), dtype=complex)
            complex_array.real = array[::2]
            complex_array.imag = array[1::2]
            yield complex_array

def load_data(file, dt):
    n_bins = int(duration / dt)
    bin_size = int(n_samples / n_bins)
    yield from read_arraydata_from_fc32(LOC+file+".32fc", n_bins, bin_size)


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

        #print(row)
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
        percentile_noise = np.percentile(noise_floor, 10)
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

        #median_noise = 1
        row_SNR = row / percentile_noise   
        rows.append(row)
    image = np.array(rows)
    time = np.arange(image.shape[0]) * dt
    #print(signal_freq)
    return np.array(signal_freq), time, frequency, image



def plot_spectogram(spec, time, freqs, tuning_freq, label):
    fig_1, ax_1 = plt.subplots()
    extent = ((freqs[0]-tuning_freq), (freqs[-1]-tuning_freq+(freqs[-1]-freqs[-2])), time[-1]+(time[-1]-time[-2]), time[0])
    img_1 = ax_1.imshow(spec, aspect='auto', cmap=colours, extent=extent, interpolation='None')
    cbar = fig_1.colorbar(img_1, ax=ax_1)

    # Axis labels
    ax_1.set_xlabel(f'Frequency offset w.r.t. tuning frequency ({(tuning_freq/1e6):.6f} MHz) [Hz]')
    ax_1.set_ylabel('Time [s]')
    ax_1.set_title('Spectrogram of ' + label )
    cbar.set_label('Signal [-]')

    # Plotting and automatic saving
    scale = 2.5
    fig_1.set_size_inches(19, 10)
    fig_1.tight_layout()
    fig_1.savefig(PLOT_LOC + f"spec_{label}_spectral.png", dpi=scale*100, bbox_inches = 'tight')

### Process datasets
#files = ["EC20S_62500Hz_60s_testrun.32fc"]

FILE = sat + '_' + norad + '_' + files_dates[0]
META = LOC + FILE +'.yml'
DATA = LOC + FILE +'.32fc'

settings = yaml.load(open(META), Loader=yaml.FullLoader)

tune_freq = settings['Sat']['State']['Tuning Frequency']
n_samples = settings['Sat']['Record']['num_sample']
sampling_rate = settings['Sat']['Record']['sample_rate']
duration = n_samples/sampling_rate
dt = 0.05 # Time resolution of the FFT
theoretical_freq = tune_freq
mask_width = sampling_rate

freq_offset = 0 

sig_freqs, time, freqs, spec1 = extract_signal(load_data(sat + '_' + norad + '_' + '202500000000', dt), dt, sampling_rate, tune_freq, freq_offset, mask_width)   
sig_freqs, time, freqs, spec2 = extract_signal(load_data(sat + '_' + norad + '_' + '202500000001', dt), dt, sampling_rate, tune_freq, freq_offset, mask_width)   
print(spec1.shape, spec2.shape)

spec = np.vstack((spec1, spec2))
print(spec.shape)

label = "combined_test"
plot_spectogram(spec, time, freqs, tune_freq, label)