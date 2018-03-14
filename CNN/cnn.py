# Imports
#	import numpy as np
#import tensorflow as tf
#from scipy import misc
import os
import sys

sys.path.append('/home/campus/adrien.gaste/Documents/TensorflowSpeechRecognition/Preprocessing')

import FileUtil


sys.argv[0]

TRAINING_DIRECTORY = '../Datasets/Training/Processed'
#~ #EVALUATION_DIRECTORY = '../Datasets/Evaluation/Processed'
#~ BASE_CATEGORIES = FileUtil.get_all_base_categories()

#~ #ONE_HOT = get_one_hot(BASE_CATEGORIES)

#~ def one_hot(categories):
	#~ one_hot = {category:[0]*(len(categories)) for category in categories}
	#~ index = 0
	#~ for key,val in enumerate(categories.items()):
		#~ val[category][index] = 1
		#~ index += 1
	#~ return one_hot
	
#~ #print(ONE_HOT)
	

def cnnClassifier(features,labels,mode):
	
	# Input Layer
	input_layer = tf.reshape(features["x"],[-1,1085,640,3]) # Change 2nd parameter for image width,height,number of color channels
	
	# Convolutional layer
	conv1 = tf.layers.conv2d(
		inputs=input_layer, 
		filters=32,
		kernel_size=[5,5],
		padding="same",
		activation=tf.nn.relu)

	# Pooling Layer #1
	pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=2, strides=2)

	# Convolutional Layer #2 and Pooling Layer #2
	conv2 = tf.layers.conv2d(
	  inputs=pool1,
	  filters=64,
	  kernel_size=[5, 5],
	  padding="same",
	  activation=tf.nn.relu)
	pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=2, strides=2)

	# Dense Layer
	pool2_flat = tf.reshape(pool2, [-1, 1085/4 * 640/4 * 64])
	dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
	dropout = tf.layers.dropout(
	  inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

	# Logits Layer
	logits = tf.layers.dense(inputs=dropout, units=2)

	# Predictions
	predictions = {
	  # Generate predictions (for PREDICT and EVAL mode)
	  "classes": tf.argmax(input=logits, axis=1),
	  # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
	  # `logging_hook`.
	  "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
	}
	if mode == tf.estimator.ModeKeys.PREDICT:
		return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
		
	# One-hot encoding
	onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=2)
	loss = tf.losses.softmax_cross_entropy(
    onehot_labels=onehot_labels, logits=logits)

	

	# Calculate Loss (for both TRAIN and EVAL modes)
	loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

	# Configure the Training Op (for TRAIN mode)
	if mode == tf.estimator.ModeKeys.TRAIN:
		optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
		train_op = optimizer.minimize(
			loss=loss,
			global_step=tf.train.get_global_step())
		return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

	# Add evaluation metrics (for EVAL mode)
	eval_metric_ops = {
	  "accuracy": tf.metrics.accuracy(
		  labels=labels, predictions=predictions["classes"])}
	return tf.estimator.EstimatorSpec(
	  mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def main(unused_argv):
	# Load training data and labels
	train_data = np.array()
	for f in get_all_processed_filenames():
		files = misc.imread()
		train_data.append(files)
	train_labels = np.array()
	
	
	# Estimator class
	scd_classifier = tf.estimator.Estimator(
		model_fn=cnnClassifier, model_dir="../tmp/scd_convnet_model")
	
	# Logging hook
	tensors_to_log = {"probabilities": "softmax_tensor"}
	logging_hook = tf.train.LoggingTensorHook(
	  tensors=tensors_to_log, every_n_iter=50)
	
	# Training the model
	train_input = tf.estimator.inputs.numpy_input_fn(
		x={"x": train_data},
		y=train_labels,
		batch_size=100,
		num_epochs=None,
		shuffle=True)
	scd_classifier.train(
		input_fn=train_input_fn,
		steps=2000,
		hooks=[logging_hook])
		
	
	# Load eval data and labels
	eval_data = np.array()
	for f in get_all_processed_filenames(): # Change method for eval data fetching
		files = misc.imread()
		eval_data.append(files)
	eval_labels = np.array()
	
		
	# Evaluate the model and print results
	eval_input = tf.estimator.inputs.numpy_input_fn(
		x={"x": eval_data},
		y=eval_labels,
		num_epochs=1,
		shuffle=False)
	eval_results = scd_classifier.evaluate(input_fn=eval_input)
	print(eval_results)
