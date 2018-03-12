import os
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import signal
import numpy as np


UNPROCESSED_WAVE_DIRECTORY = '../Datasets/Training/Unprocessed/'
PROCESSED_WAVE_DIRECTORY = '../Datasets/Training/Processed/'

#Fenerates absolute path for all dirs in UNPROCESSED_WAVE_DIRECTORY that don't begin with "_"
def get_all_categories(directory):
    for category in os.listdir(directory):
        if os.path.isdir(os.path.join(directory,category)) and category [0] != "_":
            yield os.path.join(directory,category)

#Generates absolute path for all wave files in the UNPROCESSED_WAVE_DIRECTORY
def get_all_wave_filenames(category = None):
    categories = [category]
    if category is None:
        categories = get_all_categories()
    for category in categories:
        for filename in os.listdir(category):
            if filename.endswith('.wav'):
                yield os.path.join(category,filename)

def get_all_datapoints():
    for category in get_all_categories(UNPROCESSED_WAVE_DIRECTORY):
        for index,filename in enumerate(get_all_wave_filenames(category)):
            yield Datapoint(os.path.basename(os.path.normpath(category)),filename, index) 

#returns list of full pathnames for all processed spectrograms
def get_all_processed_filenames():
    for category in get_all_categories(PROCESSED_WAVE_DIRECTORY):
        for filename in os.listdir(category):
            if filename.endswith('.png'):
                yield os.path.join(category,filename)


#saves a datapoint as an image
def save_dp_as_image(datapoint,directory,filename,window_size=20,step_size=10,eps=1e-10):
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

    ax2 = fig.add_subplot(111)
    ax2.imshow(spectrogram.T, aspect='auto', origin='lower', 
               extent=[times.min(), times.max(), frequencies.min(), frequencies.max()])

    # ax2.set_yticks(frequencies[::16])
    # ax2.set_xticks(times[::16])
    # ax2.set_title('Spectrogram of ' + datapoint.category + "#" + str(datapoint.index))
    # ax2.set_ylabel('frequencies in Hz')
    # ax2.set_xlabel('Seconds')


    fs_split = filename.split("/")
    output_filename = os.path.join(directory,fs_split[-2])
    if not os.path.exists(output_filename):
        os.makedirs(output_filename)

    output_filename += "/" + fs_split[-1]
    output_filename = output_filename[:-4]
    output_filename += ".png"


    plt.axis('off')
    fig.axes[0].get_xaxis().set_visible(False)
    fig.axes[0].get_yaxis().set_visible(False)
    plt.savefig(output_filename,bbox_inches='tight',pad_inches=0)
    plt.close()

class Datapoint:
    def __init__(self, category, filename, index):
        self.filename = filename
        self.category = category
        self.index = index

    def __str__(self):
        return "{}: {}".format(self.category,self.filename)

