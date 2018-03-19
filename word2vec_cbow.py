from __future__ import print_function
from __future__ import division

from tensorflow.python import debug as tf_debug


import tensorflow as tf
import numpy as np
import six 
import random
import collections

#reading, restricting, and encoding the wikipedia corpus
file_path = './data/wiki'
data = open(file_path).read().split(" ")
print('Collected Data.')
vocab_size = 50000
embed_size = 300

#Replace rare words with 'UNK' token.
def restrict_vocab(data, vocab_size):
	vocab = []
	count = [['UNK',-1]]
	freqs = collections.Counter(data)
	#vocab_size - N frequent words, including UNK
	least_common = freqs.most_common(len(data)-1)[(vocab_size-1):]	
	#add N-1 most frequent known words		
	count.extend(freqs.most_common(vocab_size - 1))
	unk_count = 0
	for pair in least_common:
		unk_count+=pair[1]
	count[0][1] = unk_count
	known_words = dict()
	word_id = 0
	for word, i in count:
		vocab.append(word) 
		known_words[word] = word_id
		word_id+=1
	return vocab, known_words

def encode_data(data, word_ids):
	coded_data = []
	for word in data: 
		ind = word_ids.get(word, -1)
		if ind==-1:
			coded_data.append(0) #UNKNOWN WORD
		else:
			coded_data.append(ind)
	return coded_data


vocab, word_ids = restrict_vocab(data, vocab_size)
coded_data = encode_data(data, word_ids)
print('Encoded data.')
del data

data_index = 0
batch_size = 64
window = 4

def get_batch():
	global data_index 
	inputs = np.zeros(shape=(batch_size, 2*window), dtype=np.int32)
	labels = np.zeros(shape=(batch_size, 1), dtype=np.int32)
	for b in range(batch_size):
		#center word
		c = data_index + window 
		if not c>=len(coded_data):
			s = max(0, data_index)
			e = min(c + window, len(coded_data))
			context = [k for k in range(s,e) if k!=c]
			col_idx = 0 
			for i in context:
				inputs[b, col_idx] = coded_data[i]
				col_idx+=1
			#one hot label vector
			labels[b] = coded_data[c]
		data_index = (data_index + 1) % (len(coded_data))
	return inputs, labels 

graph = tf.Graph()

num_sampled = 64

with graph.as_default():
	
	#B * C  input matrix
	x_train = tf.placeholder(tf.int32, shape=(batch_size, 2*window)) 
	# B labels 
	y_train = tf.placeholder(tf.int32, shape=(batch_size, 1))
	
	# V * N embedding table
	all_embeddings = tf.Variable(
		tf.random_uniform([vocab_size, embed_size], -1.0, 1.0), name = 'embed') 
	# B * C * N embeddings
	relevant_embeddings = tf.nn.embedding_lookup(all_embeddings, x_train) 
	# bias for each of N embeddings
	b1 = tf.Variable(tf.random_normal([embed_size]), name = 'b1') 
	
	# take mean over dimension C to get a B * N matrix + bias
	hidden_layer = tf.reduce_mean(relevant_embeddings, axis=1) + b1 
	
	#Noise Contrastive Estimation loss to speed up. 
	nce_w = tf.Variable(tf.truncated_normal([vocab_size, embed_size], stddev=1.0 / tf.sqrt(1.0 * embed_size)), name = 'nce_w') 
	#V matrix, bias for each vocab word
	nce_b = tf.Variable(tf.random_normal([vocab_size]), name = 'nce_b') 

	loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_w, biases=nce_b, labels=y_train, 
		inputs= hidden_layer, num_sampled=num_sampled, num_classes=vocab_size))

	output = tf.matmul(hidden_layer, tf.transpose(nce_w)) + nce_b 
	correct_predicts = tf.equal(tf.argmax(output, 1), tf.cast(y_train, tf.int64))
	accuracy = tf.reduce_mean(tf.cast(correct_predicts, tf.float32))
	
	optimizer = tf.train.GradientDescentOptimizer(.01).minimize(loss)

	init = tf.global_variables_initializer()
	saver = tf.train.Saver()

num_steps = 100000

with tf.Session(graph=graph) as sess:
	print('Session Running')
	init.run()
	print('Global variables initialized.')
	avg_loss = 0
	avg_acc = 0
	#sess = tf_debug.LocalCLIDebugWrapperSession(sess)
	#sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
	for i in xrange(num_steps):
		inputs, labels = get_batch()
		feed_dict = {x_train: inputs, y_train: labels}
		_, loss_val, acc_val = sess.run([optimizer, loss, accuracy], feed_dict=feed_dict)
		avg_loss+=loss_val
		avg_acc+= acc_val
		print('for batch at step', i, 'accuracy value is ', acc_val)
		#every 1000 steps
		if (i%2000) == 0: 
			if i > 0:
				avg_loss/=2000
				avg_acc/=2000
				print('Average loss is ', avg_loss, ' at step ', i)
				print('Average accuracy is', avg_acc, 'at step', i)
				avg_loss = 0	
	#TODO change this to only saving the embedding variable 
	saver.save(sess, './models/model-final') 
	print("Final model saved")
