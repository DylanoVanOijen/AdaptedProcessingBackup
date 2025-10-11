from doptrack.recording import Recording
import doptrack.processing as processing
import doptrack.signals as signals
from cmcrameri import cm
from os.path import exists
import matplotlib.pyplot as plt
import numpy as np
import yaml

### User input
sat = "FUNcube-1"
norad = "39444"
files_dates = ['202507250446']
CORR = 8000	# Frequency correction in Hz w.r.t. tuning freq

### Settings
show_plots = False

# Data input settings
#LOC = 'C:/Users/Dylan/OneDrive/Bureaublad/TUDelft/Thesis/Analysis/Amber_data/Delfi-N3XT/2022/'
LOC = '/home/doptrackbox/DTB_Recordings/data/'
FILE = 'FUNcube-1_39444_202507250446'

# Plot and processed data output settings
LOCi = '/home/doptrackbox/DTB_Recordings/locally_processed/'
PLOT_LOC = LOCi + 'Plots/'
PROCESS_LOC = LOCi + 'Processed/'

params = {'axes.labelsize': 20,'axes.titlesize': 20, 'xtick.labelsize': 16, 'ytick.labelsize': 16, 'figure.titlesize': 20, 'legend.fontsize': 16}
plt.rcParams.update(params)

def process_file(date):
	# Noise calculation settings. Set noise_calc to TRUE to enable noise post processing calculations or to FALSE to disable them.
	# Choose the frequency and time points for which you want specific signal strength plots. If set to a negative value, the plot is disabled.
	noise_calc = 0
	frequency_point = -1
	time_point = -1

	colours = cm.lapaz
	colour_auto = 0
	colour_min = -5
	colour_max = 20

	#######################
	### assemble
	#######################
	FILE = sat + '_' + norad + '_' + date
	META = LOC + FILE +'.yml'
	DATA = LOC + FILE +'.32fc'

	settings = yaml.load(open(META), Loader=yaml.FullLoader)
	TUNE = settings['Sat']['State']['Tuning Frequency']
	FREQ = TUNE + CORR

	rec = Recording.load(metafile=META , datafile=DATA)
	spec = processing.create_spectrogram_product(recording=rec, dt=1, center_frequency=FREQ)

	### Create Spectogram plot
	fig_1, ax_1 = plt.subplots()
	extent = ((spec.frequency[0]-TUNE)/1E3, (spec.frequency[-1]-TUNE)/1E3, spec.time[-1], spec.time[0])
	img_1 = ax_1.imshow(10*np.log10(spec.image), aspect='auto', cmap=colours, extent=extent)
	cbar = fig_1.colorbar(img_1, ax=ax_1)

	# Colourbar settings. Set colour_auto to 1 for TRUE and 0 for FALSE. min/max only work if auto is false!
	if colour_auto==1:
		colour_max = 'auto'
	else:
		img_1.set_clim(colour_min,colour_max)

	# Axis labels
	ax_1.set_xlabel(f'Frequency offset w.r.t. tuning frequency ({(TUNE/1e6):.3f} MHz) [kHz]')
	ax_1.set_ylabel('Time [s]')
	ax_1.set_title('Spectrogram of ' + FILE )
	cbar.set_label('SNR [dB]')

	# Plotting and automatic saving
	scale = 2.5
	fig_1.set_size_inches(19, 10)
	fig_1.tight_layout()
	fig_1.savefig(PLOT_LOC + 'Spectrogram ' + FILE + '_cmax='+str(colour_max), dpi=scale*100, bbox_inches = 'tight')


	### Create Noise Plot
	NOISE = FILE +'_NOISE.dat'
	if not exists(PROCESS_LOC + NOISE):
		np.savetxt(PROCESS_LOC + NOISE, spec.noise)

	fig_2, ax_2 = plt.subplots()
	ax_2.plot(spec.noise,spec.time)
	ax_2.axis([0,np.nanmax(spec.noise)*1.1,np.max(spec.time),0])

	ax_2.set_title('Noise floor') # smaller title for quarter size
	ax_2.set_xlabel('Noise strength [-]') #, fontsize=22)
	ax_2.set_ylabel('Time [s]') #, fontsize=22)

	fig_2.set_size_inches(5, 10)  #quarter size
	fig_2.savefig(PLOT_LOC + 'Spectrogram ' + FILE + ' Noise floor', dpi=scale*100, bbox_inches = 'tight')


	### Create SNR Plot
	print ('Extracting signal data...')
	s_avg = np.array(spec.signal_average)
	s_max = np.array(spec.signal_maximum)
	s_med = np.array(spec.signal_median)
	n = np.array(spec.noise)

	SNR_avg = s_avg/n
	SNR_max = s_max/n
	SNR_med = s_med/n

	SNR = FILE +'_SNR.dat'
	if not exists(PROCESS_LOC + SNR):
		print ('Creating SNR data file...')
		SNR_out = np.column_stack((SNR_avg,SNR_max,SNR_med))
		#SNR_out = np.hstack((SNR_avg,SNR_max))
		#SNR_out = np.hstack((SNR_out,SNR_med))
		np.savetxt(PROCESS_LOC + SNR, SNR_out)
		print ('SNR data written to file',SNR)
		print (' ')
	else:
		print ('SNR data already exists and has therefore not been updated')
		print (' ')

	fig_3, ax_3 = plt.subplots(1,2)
	fig_3.suptitle('SNR for ' + FILE)

	ax_3[0].plot(spec.time, SNR_avg, label='Mean SNR')
	ax_3[0].plot(spec.time, SNR_med, label='Median SNR')
	ax_3[0].axis([0,np.max(spec.time),np.nanmin(SNR_avg)*0.75,np.nanmax(SNR_avg)*1.05])
	#axs[0].axis([0,np.max(spec.time),np.min(SNR_avg)*0.75,np.max(SNR_avg)*1.05])
	#axs[0].axis([0,np.max(spec.time),2,7])

	ax_3[1].plot(spec.time, SNR_max,'g',label='Maximum SNR')
	ax_3[1].axis([0,np.max(spec.time),0,np.nanmax(SNR_max)*1.05])
	#axs[1].axis([0,np.max(spec.time),0,40])

	ax_3[0].set_xlabel('Time [s]')
	ax_3[0].set_ylabel('SNR [dB]')
	ax_3[0].legend()
	ax_3[1].set_xlabel('Time [s]')
	ax_3[1].set_ylabel('SNR [dB]')
	ax_3[1].legend()

	fig_3.set_size_inches(19, 10)
	fig_3.savefig(PLOT_LOC + 'Spectrogram ' + FILE + ' SNR', dpi=scale*100, bbox_inches = 'tight')

	PROCESS = FILE +'.dat'
	rewrite = exists(PROCESS_LOC + PROCESS)

	if rewrite==0:
		print ('Creating output data file...')
		np.savetxt(PROCESS_LOC + PROCESS, spec.image)
		print ('Processed spectrogram data written to file',PROCESS)
		print (' ')
	else:
		print ('Processed spectrogram data file already exists and has therefore not been updated')
		print (' ')

	# Noise data plots
	if noise_calc==1:
		print ('Post processing noise data...')
		if frequency_point<0:
			print ('Calculations for a specific frequency through time have been disabled')
		else:
			abs_freq = (FREQ+frequency_point)/1e6
			freq_point = frequency_point - 1
			time_data = spec.image[:,freq_point]
			time_avg = np.average(time_data)
			time_avg_complete = np.ones(time_data.size)*time_avg
			print ('Average noise value at f= ',abs_freq,'MHz:', time_avg)

			plt.plot(time_data)
			plt.plot(time_avg_complete, 'r--')
			plt.axis([0, time_data.size, 0, np.max(time_data)*1.1])

			# Axis labels
			plt.title('Noise over time at ' + str(abs_freq) + ' MHz for ' + FILE + ' ' + INFO)
			plt.xlabel('Time [s]')
			plt.ylabel('Noise strenght [?]')

		if time_point<0:
			print ('Calculations for a specific time over all frequencies are disabled')
		else:
			ti_point = time_point - 1
			freq_data = spec.image[ti_point,:]
			freq_avg = np.average(freq_data)
			freq_avg_complete = np.ones(freq_data.size)*freq_avg
			print ('Average noise value at t= ', time_point, 'seconds:',freq_avg)

			plt.plot(freq_data)
			plt.plot(freq_avg_complete, 'r--')
			plt.axis([0, freq_data.size, 0, np.max(freq_data)*1.1])

			#Axis labels
			plt.title('Signal strength at ' + str(time_point) + ' seconds for ' + FILE + ' ' + INFO)
			plt.xlabel('Frequency offset to ' + str(FREQ/1e6) + 'MHz [Hz]')
			plt.ylabel('Signal strength [?]')
	else:
		print ('Noise data calculations have been set to disabled in the program settings by the user')

	if show_plots:
		plt.show()

for file in files_dates:
	process_file(file)
