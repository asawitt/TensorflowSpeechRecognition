import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
import numpy as np
import FileUtil


# #Reads and graphs a wave file given a datapoint
def graph_wave_file(datapoint, window_size=20, step_size=10,eps=1e-10):
	sample_rate, samples = wavfile.read(datapoint.filename)
	nperseg = int(round(window_size * sample_rate / 1e3))
	noverlap = int(round(step_size * sample_rate / 1e3))
	frequencies, times, spectrogram = signal.spectrogram(samples, 
														fs=sample_rate,
														window='hann',
														nperseg=nperseg,
														noverlap=noverlap,
														detrend=False)

	spectrogram = np.log(spectrogram.T.astype(np.float32) + eps)

	fig = plt.figure(figsize=(14, 8))
	ax1 = fig.add_subplot(211)
	ax1.set_title('Raw wave of ' + datapoint.category + "#" + str(datapoint.index))
	ax1.set_ylabel('Amplitude')
	ticks = np.linspace(0,sample_rate/len(samples), len(samples))
	ax1.plot(ticks, samples)

	ax2 = fig.add_subplot(212)
	ax2.imshow(spectrogram.T, aspect='auto', origin='lower', 
	           extent=[times.min(), times.max(), frequencies.min(), frequencies.max()])
	ax2.set_yticks(frequencies[::16])
	ax2.set_xticks(times[::16])
	ax2.set_title('Spectrogram of ' + datapoint.category + "#" + str(datapoint.index))
	ax2.set_ylabel('frequencies in Hz')
	ax2.set_xlabel('Seconds')

	plt.show()
def main():
	index = 0
	for dp in FileUtil.get_all_datapoints():
		graph_wave_file(dp)
		index += 1
		if index == 10:
			break

if __name__ == '__main__':
	main()