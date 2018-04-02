from keras.models import Sequential
from keras.layers import Dense,Activation
from keras import optimizers,losses
import numpy as np
from collections import defaultdict
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from keras.wrappers.scikit_learn import KerasClassifier
from keras.layers.advanced_activations import LeakyReLU,PReLU
from sklearn.preprocessing import normalize
import math
import re
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten
import keras
import os
import PIL

from sklearn.utils import shuffle

num_epochs = 30
batch_size = 20

IMG_WIDTH = 60
IMG_HEIGHT = 65

training_base_directory = "../Datasets/Training/Processed/"
categories = ['cat','dog']
num_categories = len(categories)
cat_to_index = {cat:i for i,cat in enumerate(categories)}

one_hot = [[0]*num_categories for i in range(num_categories)] 
for i in range(num_categories):
	 one_hot[i][i] = 1

def make_model():
	model = Sequential()
	model.add(Conv2D(32, kernel_size=(3, 3),
		activation='relu',
		input_shape=(IMG_WIDTH,IMG_HEIGHT,1)))
	model.add(Conv2D(64, (3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))
	model.add(Flatten())
	model.add(Dense(128, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(num_categories, activation='softmax'))
	loss_function = keras.losses.binary_crossentropy
	model.compile(loss=loss_function,
		optimizer=keras.optimizers.Adadelta(),
		metrics=['accuracy'])

	return model

def make_prediction(x,model,b_s=100):
	prediction = model.predict(x,batch_size=b_s,verbose=0).tolist()
	return prediction

def shape(line):
	print('line:',line)
	return 0
	# return np.array(list(map(lambda x: int(x), line.split(",")))).reshape(-1,)

def normalize(img):
	img.setflags(write=1)
	for r,row in enumerate(img):
		for c,item in enumerate(row):
			img[r][c] = item/255
	return img

	

def main():
	#####READ TRAINING DATA INTO NP ARRAY#######
	train_data = []
	train_labels = []

	index = 0
	for category in categories:
		print("Now working on ", category + "s")
		training_directory = os.path.join(training_base_directory,category)
		label = one_hot[cat_to_index[category]]
		for train_file in os.listdir(training_directory):
			filename = os.path.join(training_directory,train_file)
			img = normalize(np.asarray(PIL.Image.open(filename),dtype=np.float32))
			train_data.append(img)
			train_labels.append(label)
			print(100*index/800,"% done")
			index += 1
			if not index % 400:
				break


	train_data = np.array(train_data).reshape(len(train_data),IMG_WIDTH,IMG_HEIGHT,1)
	train_labels=np.array(train_labels)
	train_data, train_labels = shuffle(train_data,train_labels)

	print("Done reading data, fitting data...")
	model = make_model()
	model.fit(train_data,train_labels,epochs=num_epochs,batch_size=batch_size,validation_split=.2)
	# model.save('trained_digits_vs_letters_200bs_15epochs_scaled.h5')

if __name__=="__main__":
	main()

