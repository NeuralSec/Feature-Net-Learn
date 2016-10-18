import tensorflow as tf
import numpy as np
import pickle

# HiddenLayer_dict[layer] = Number of hidden neurons in this layer, start from 1 !!!
def Hidden_layer_shape(l,n):
	hidden_layer_shape = {}
	# Modify hidden layer number "l" and neuron number "n" in each layer.
	for layer in range(1,l+1):
		hidden_layer_shape[layer]=n
	print("Hidden layer number:", len(hidden_layer_shape.keys()))
	return hidden_layer_shape


# HiddenLayer_dict[layer] = Number of hidden neurons in this layer, start from 1 !!!
def neural_network(Input_feature_shape, n_nodes_input, HiddenLayer_dict, n_classes):
	
	# Define input feature number and number of neurons in hidden layers.
	Input_and_hidden_layer = HiddenLayer_dict.copy()
	Input_and_hidden_layer[0] = n_nodes_input

	# Define shapes of hidden layers.
	Hidden_Layer_structure = {}
	for layer in HiddenLayer_dict.keys(): # Iteratively assign weight and bias to hidden layers.
		Hidden_Layer_structure[layer] = {'weights': tf.Variable(tf.random_normal([Input_and_hidden_layer[(layer-1)] , HiddenLayer_dict[layer]])),
										'biases': tf.Variable(tf.random_normal([HiddenLayer_dict[layer]]))}
	# Neuron number of the last hidden layer as the input number of output layer.
	output_layer = {'weights': tf.Variable(tf.random_normal([HiddenLayer_dict[max(HiddenLayer_dict.keys())], n_classes])),
					'biases': tf.Variable(tf.random_normal([n_classes]))}

	# Generate output from neuron network.
	l_previous = Input_feature_shape
	for layer in HiddenLayer_dict.keys():
		l_current = tf.add(tf.matmul(l_previous, Hidden_Layer_structure[layer]['weights']), Hidden_Layer_structure[layer]['biases'])
		l_current = tf.nn.relu(l_current)   # Use relu as Activation function.
		l_previous = l_current
	output = tf.matmul(l_current, output_layer['weights']) + output_layer['biases']
	return output


def Training(layer_num, neuron_num, input_feature_shape, input_label_shape, train_x, train_y, test_x, test_y):
	n_input = len(train_x[0])
	hidden_layer_shape = Hidden_layer_shape(layer_num, neuron_num) # Initialise hidden layer shape
	classes = 2
	batch_size = 100
	
	prediction = neural_network(input_feature_shape, n_input, hidden_layer_shape, classes)
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction, input_label_shape))
	optimizer = tf.train.AdamOptimizer().minimize(cost)
	epoch_num = 100

	saver = tf.train.Saver()
	with tf.Session() as sess:
		sess.run(tf.initialize_all_variables())
		accuracy_record = 0
		invalid_training_number = 0
		true_count = 0
		for epoch in range(epoch_num):
			epoch_loss = 0
			i = 0
			while i < len(train_x):
				start = i
				end = i + batch_size

				batch_x = np.array(train_x[start:end])
				batch_y = np.array(train_y[start:end])
				
				_, c = sess.run([optimizer, cost], feed_dict = {input_feature_shape: batch_x, input_label_shape: batch_y})
				epoch_loss += c
				i += batch_size

			print("Epoch:", epoch+1, "complete out of", epoch_num, "loss:", epoch_loss)
			correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(input_label_shape, 1))
			accuracy = tf.reduce_mean(tf.cast(correct, "float"))
			ACC = accuracy.eval({input_feature_shape: test_x, input_label_shape: test_y})
			print("Accuracy:", ACC)
			if ACC > accuracy_record:	# Save the best variables!
				accuracy_record = ACC
				invalid_training_number = 0
				save_path = saver.save(sess, "FNI_tuning.ckpt")
				print("Training time %s has improvement. Model saved in file: %s" % (epoch+1, save_path))
			else:
				invalid_training_number += 1
			if invalid_training_number >= 20: # Epoch Window
				break
	return prediction
# End training

if __name__ == "__main__":
	with open("FacebookFeatures of 5000 node pairs for Training.pickle", 'rb') as pickle_file:
		train_x, train_y, train_edge_names, test_x, test_y, test_edge_names = pickle.load(pickle_file)
		print("Feature length:", len(train_x[0]), "Feature example:", train_x[0], "Label: ", train_y[0])
	#Define Input tensor shape, features as x: height x Width, labels as y: width.
	x_shape = tf.placeholder('float',[None, len(train_x[0])]) #features
	y_shape = tf.placeholder('float') #label
	Training(2, 600, x_shape, y_shape, train_x, train_y, test_x, test_y)