#import modules
import numpy as np
import json
from numpy.core.arrayprint import set_string_function
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# import tensorflow as tf
import pickle as pkl

# measure time
import time

# import tensorflow 2 and use 1.x version
import tensorflow.compat.v1 as tf
from tensorflow.python.framework import ops

tf.disable_v2_behavior()

class GenerateHands:
	"""
	A class to generate doodle hand images using Generative Adversarial Networks.

	Attributes
	----------
	hyperparamfile: str
		file name of the json config file
	datafile: str
		file name/ location of data in form of string. Data is to be in npy format.

	Methods
	-------

	getData():
		input is nothing, it would take in the data file provided during initialization and use it read the file.
		returns data in form of numpy array.
	
	plotsampleData(data):
		input: data, numpy array
		returns image
	
	model_inputs, processDataforTrain, generator, discriminator :
		these are methods needed to prepare the model

	compileModel(data):
		input: data, numpy array
		compile the model
		returns variables needed for calling methods
	
	trainModel(data):
		input: data, numpy array
		train the model for specific amount of batches and epochs and measure time
		returns None
	"""
	def __init__(self, hyperparamfile, datafile) -> None:
		
		self.hyperparams = json.load(open(hyperparamfile))
		self.r_size = self.hyperparams['r_size']
		self.z_size = self.hyperparams['z_size']
		self.g_units = self.hyperparams['g_units']
		self.d_units = self.hyperparams['d_units']
		self.alpha = self.hyperparams['alpha']
		self.smooth = self.hyperparams['smooth']
		self.lr = self.hyperparams['lr']
		self.batchsize = self.hyperparams['batchsize']
		self.epochs = self.hyperparams['epochs']
		self.datafile = datafile
	
	def getData(self):
		'''get data'''
		data = np.load(self.datafile)
		print("data loaded (shape): ", data.shape)
		return data
	
	def plotsampleData(self, data):
		'''plot sample data'''
		fig=plt.figure(figsize=(10, 10))
		columns = 15
		rows = 10
		for i in range(1, columns*rows):
			img = data[i+44].reshape((28,28))
			fig.add_subplot(rows, columns, i)
			plt.imshow(img)
		plt.show()

	def model_inputs(self, real_dims, z_dims):
		'''prepare data for input for tensorflow'''
	
		inputs_real = tf.placeholder(tf.float32, shape=(None, real_dims), name='input_real')
		inputs_z = tf.placeholder(tf.float32, shape=(None, z_dims), name='input_z')

		return inputs_real, inputs_z
	
	def processDataforTrain(self, data):
		'''process data for train test split'''
		try:
			Y = []
			for i in trange(data.shape[0]):
				Y.append([1,0])
			Y = np.array(Y)

			(x_train, y_train, x_test, y_test) = train_test_split(data, Y)
			x_train = (x_train.astype(np.float32)) / 255
			x_train = x_train.reshape(x_train.shape[0], 784)

			return (x_train, y_train, x_test, y_test)  

		except Exception as e:
			print (e)
	
	def generator(self, z, out_dims, n_units=128, reuse=False, alpha=0.01):
		'''basic generator model'''
		with tf.variable_scope('generator', reuse=reuse):
			#hidden layer
			h1 = tf.layers.dense(z,n_units, activation=None,)
			#leaky relu implementation
			h1 = tf.maximum(alpha*h1, h1)
			#tanh 
			logits = tf.layers.dense(h1, out_dims)
			
			out = tf.tanh(logits)
			
			return out
	
	def discriminator(self, x, n_units=128, reuse=False, alpha=0.01):
		'''basic discriminator model'''
		with tf.variable_scope('discriminator', reuse=reuse):
			#hidden layer
			h1 = tf.layers.dense(x, n_units, activation=None)
			#leaky_relu
			h1 = tf.maximum(alpha*h1, h1)
			
			#sigmoid
			logits = tf.layers.dense(h1, 1, activation=None)
			out = tf.sigmoid(logits)
			
			return out, logits

	def compileModel(self, data):
		'''compile model for train graph'''

		print("Compiling of data and started.")
		x_train, y_train, x_test, y_test = self.processDataforTrain(data)
		ops.reset_default_graph()
		inputs_real, inputs_z = self.model_inputs(self.r_size, self.z_size)
		g_out = self.generator(inputs_z, self.r_size, self.g_units)

		d_out_real, real_logit = self.discriminator(inputs_real,)
		d_out_fake, fake_logits = self.discriminator(g_out, reuse=True)

		d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=real_logit, labels=tf.ones_like(real_logit)*(1-self.smooth)))
		d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_logits, labels=tf.zeros_like(fake_logits)))

		d_loss = d_loss_fake+d_loss_real
		g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_logits, labels=tf.ones_like(fake_logits)))

		tvar = tf.trainable_variables()
		gvar = [var for var in tvar if var.name.startswith('generator')]
		dvar = [var for var in tvar if var.name.startswith('discriminator')]

		d_opt = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(d_loss, var_list=dvar)
		g_opt = tf.train.AdamOptimizer(self.lr).minimize(g_loss,var_list=gvar)

		print("Data Ready, Compilation over.")
		return x_train, gvar, dvar, d_opt, g_opt, inputs_real, inputs_z, g_loss, d_loss

	
	def trainModel(self, data):
		'''train the model and check the time taken'''
		start = time.time()
		samples = []
		losses = []

		x_train, gvar, dvar, d_opt, g_opt, inputs_real, inputs_z, g_loss, d_loss = self.compileModel(data)
		# Only save generator variables
		saver = tf.train.Saver(var_list=gvar)
		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())
			for e in range(self.epochs):
				for ii in range(x_train.shape[0]//self.batchsize):
					batch = x_train[np.random.randint(0, x_train.shape[0], size=self.batchsize)]
					
					batch_images = batch*2 - 1
					
					# Sample random noise for G
					batch_z = np.random.uniform(-1, 1, size=(self.batchsize, self.z_size))
					
					# Run optimizers
					_ = sess.run(d_opt, feed_dict={inputs_real: batch_images, inputs_z: batch_z})
					_ = sess.run(g_opt, feed_dict={inputs_z: batch_z})
				
				# At the end of each epoch, get the losses and print them out
				train_loss_d = sess.run(d_loss, {inputs_z: batch_z, inputs_real: batch_images})
				train_loss_g = g_loss.eval({inputs_z: batch_z})
				if e%10 == 0:    
					print("Epoch {}/{}...".format(e, self.epochs),
							"Discriminator Loss: {:.4f}...".format(train_loss_d),
							"Generator Loss: {:.4f}".format(train_loss_g))    
				# Save losses to view after training
				losses.append((train_loss_d, train_loss_g))
				
				# Sample from generator as we're training for viewing afterwards
				sample_z = np.random.uniform(-1, 1, size=(16, self.z_size))
				gen_samples = sess.run(
							self.generator(inputs_z, self.r_size, n_units=self.g_units, reuse=True, alpha=self.alpha),
							feed_dict={inputs_z: sample_z})
				samples.append(gen_samples)
				saver.save(sess, './checkpoints/generator.ckpt')

		# Save training generator samples
		with open('train_samples.pkl', 'wb') as f:
			pkl.dump(samples, f)

		end = time.time()
		print('Time elapsed : ', (end-start)/60, ' mins.')



if __name__ == "__main__":
	generate = GenerateHands("hyperParams.json", "../Data/hand.npy")
	data = generate.getData()
	generate.trainModel(data)
