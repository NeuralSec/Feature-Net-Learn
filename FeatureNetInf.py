import tensorflow as tf
import numpy as np
import pickle
import TrainingFeatureNetInf as TFNI
import sklearn.metrics as metrics


def Classifier(layer_num, neuron_num, Input_feature_shape, Input_label_shape, test_dict):
	l_input = len(test_dict[Input_feature_shape][0])
	hidden_layer_shape = TFNI.Hidden_layer_shape(layer_num, neuron_num)
	classes = 2
	#batch_size = 100

	ANN = TFNI.neural_network(Input_feature_shape, l_input, hidden_layer_shape, classes)  # Re-define the nural network shape.
	saver = tf.train.Saver()
	with tf.Session() as sess:
		saver.restore(sess, "FNI.ckpt")  # Restore variables from disk.
		print("Model restored.")
		correct = tf.equal(tf.argmax(ANN , 1), tf.argmax(Input_label_shape, 1)) #tf.equal(Predicted labels, True labels), Returns the truth value of (Predicted labels == True labels) element-wise
		accuracy = tf.reduce_mean(tf.cast(correct, "float"))
		prediction = tf.argmax(ANN,1).eval(test_dict)
		y_true = np.argmax(test_dict[Input_label_shape],1)
		precision = metrics.precision_score(np.array(y_true), np.array(prediction))
		recall = metrics.recall_score(np.array(y_true), np.array(prediction))
		print("predictions:", prediction)
		print("Probabilities:", ANN.eval(test_dict))
		print("Test Accuracy:", accuracy.eval(test_dict))
		print("Precision:", precision)
		print("Recall:", recall)
		print(tf.trainable_variables())
		return prediction, ANN.eval(test_dict), precision, recall


if __name__ == "__main__":
	with open("FacebookFeatures of 2500 node pairs for experiment.pickle", 'rb') as pickle_file:    # input testing data
		_, _, _, test_x, test_y, test_edge_names = pickle.load(pickle_file)
		print("Test data example:", test_x[0], test_y[0])

	#Define Input tensor shape, features as x: height x Width, labels as y: width.
	x_shape = tf.placeholder('float',[None, len(test_x[0])]) #features
	y_shape = tf.placeholder('float') #label

	feed_dict = {x_shape: test_x, y_shape: test_y}
	Predictions, Probabilities, P, R = Classifier(1, 800, x_shape, y_shape, feed_dict) # define layer and neuron number read in classifier
	Probabilities = np.array(Probabilities)
	print("Output probability array:", Probabilities)
	print("Node pairs classified as positive:", Probabilities[Probabilities[:, 0] >= Probabilities[:, 1]])
