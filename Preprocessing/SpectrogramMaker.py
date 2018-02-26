import matplotlib as plt
from scipy import signal
from scipy.io import wavfile
import FileUtil




def main():
	for dp in FileUtil.get_all_datapoints():
		print(dp)


if __name__ == '__main__':
	main()