import tensorflow as tf 
import pandas as pd 
import testing_tools
# This will be a simplistic three layer feed forward neural network that will move from our 22 fantasy
# football datapoints to 11 to 2 
# the network will be trained via backprop, and will have, as its error function the one specified in 
# the parametric t-SNE paper


test_values_path = '/home/tyler/2018_Development/Fantasy_Stats_ML/FFNN/test0generation0child0/'

# if this is functional, it should form the skeletal basis for either amending our FFNN class or coming
# up with a secondary parametricTSNE class that will use FFNN as a pretrainer

class ptsne(object):

	def __init__(self):

		# variables for our ffnn
		pth = test_values_path + 'test0generation0child0_rbm_1/'
		df = (pd.read_csv(pth + 'W.csv', sep=',' , header=None)).values
		self.layer1_W = tf.Variable(df, name='layer1_W' , dtype=tf.float32)
		df = (pd.read_csv(pth + 'bh_.csv', sep=',' , header=None )).values
		self.layer1_bh_ = tf.Variable(df, name='layer1_bh_' , dtype=tf.float32)
		df = (pd.read_csv(pth + 'bv_.csv', sep=',' , header=None)).values
		self.layer1_bv_ = tf.Variable(df, name='layer1_bv_')

		pth = test_values_path + 'test0generation0child0_rbm_2/'
		df = (pd.read_csv(pth + 'W.csv', sep=',' , header=None)).values
		self.layer2_W = tf.Variable(df, name='layer2_W', dtype=tf.float32)
		df = (pd.read_csv(pth + 'bh_.csv', sep=',' , header=None)).values
		self.layer2_bh_ = tf.Variable(df, name='layer2_bh_', dtype=tf.float32)
		df= (pd.read_csv(pth + 'bv_.csv', sep=',' , header=None)).values
		self.layer2_bv_ = tf.Variable(df, name='layer2_bv_', dtype=tf.float32)
		
		pth = test_values_path + 'test0generation0child0_rbm_3/'
		df = pd.read_csv(pth + 'W.csv', sep=',' , header=None)
		self.layer3_W = tf.Variable(df.values, name='layer3_W', dtype=tf.float32)
		df = pd.read_csv(pth + 'bh_.csv', sep=',' , header=None)
		self.layer3_bh_ = tf.Variable(df.values , name='layer3_bh_', dtype=tf.float32)
		df = pd.read_csv(pth + 'bv_.csv', sep=',' , header=None)
		self.layer3_bv_ = tf.Variable(df.values , name='layer3_bv_', dtype=tf.float32)

		self.X = tf.placeholder(dtype=tf.float32 , name='Dataset')

		#self.testfunc = tf.sigmoid(tf.matmul(self.X, self.layer1_W) + self.layer1_bh_)
		self.layer1 = tf.sigmoid(tf.matmul(self.X, self.layer1_W) + tf.transpose(self.layer1_bh_))
		self.layer2 = tf.sigmoid(tf.matmul(self.layer1 , self.layer2_W) + tf.transpose(self.layer2_bh_))
		self.layer3 = tf.sigmoid(tf.matmul(self.layer2 , self.layer3_W) + tf.transpose(self.layer3_bh_))

		#self.loss = tf.distributions.kl_divergence(self.X , self.layer3)

	def test(self , dataset):
		init_op = tf.global_variables_initializer()

		with tf.Session() as sess:
			sess.run(init_op)
			#sess.run(self.testfunc  , feed_dict={self.X:dataset})
			#print(self.layer1_W)
			#x = (sess.run(self.layer3 , feed_dict={self.X:dataset}).shape)
			x = sess.run(self.layer3 , feed_dict={self.X:dataset})
			
			testing_tools.write_csv(x , 'didthiswork.csv')

			#sess.run(self.layer3_bh_.eval())
		

TRAINING_SET_PATH = "RBMTrainingDataset/training_set.csv"
TEST_SET_PATH = "FormattedFantasyData/2018_data.csv"
NUM_DATAPOINTS = 22

test_set = pd.read_csv(TEST_SET_PATH , sep=',' , header=None)
test_set = test_set.values

x = ptsne()
x.test(test_set)