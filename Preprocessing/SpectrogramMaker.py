import FileUtil

PROCESSED_WAVE_DIRECTORY = '../Datasets/Training/Processed/'

def main():
	index = 0
	for dp in FileUtil.get_all_datapoints():
		FileUtil.save_dp_as_image(dp,PROCESSED_WAVE_DIRECTORY,dp.filename)
		index += 1
		if not index % 10:
			print(100*index/4000,"%% done")

if __name__ == '__main__':
	main()