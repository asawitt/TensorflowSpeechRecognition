import os

UNPROCESSED_WAVE_DIRECTORY = '../Datasets/Training/Unprocessed/'

#Fenerates absolute path for all dirs in UNPROCESSED_WAVE_DIRECTORY that don't begin with "_"
def get_all_categories():
	for category in os.listdir(UNPROCESSED_WAVE_DIRECTORY):
		if os.path.isdir(os.path.join(UNPROCESSED_WAVE_DIRECTORY,category)) and category [0] != "_":
			yield os.path.join(UNPROCESSED_WAVE_DIRECTORY,category)

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
	for category in get_all_categories():
		for index,filename in enumerate(get_all_wave_filenames(category)):
			yield Datapoint(os.path.basename(os.path.normpath(category)),filename, index) 
			if index == 3:
				break

class Datapoint:
	def __init__(self, category, filename, index):
		self.filename = filename
		self.category = category
		self.index = index

	def __str__(self):
		return "{}: {}".format(self.category,self.filename)