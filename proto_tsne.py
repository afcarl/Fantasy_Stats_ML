import tensorflow as tf 
import pandas as pd 

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
		df = pd.read_csv(pth + 'W.csv', sep=',' , header=None)
		self.layer1_W = tf.Variable( df.values, name='layer1_W', dtype=tf.float32)
		df = pd.read_csv(pth + 'bh_.csv', sep=',' , header=None )
		self.layer1_bh_ = tf.Variable(df.values, name='layer1_bh_' , dtype=tf.float32)
		df = pd.read_csv(pth + 'bv_.csv', sep=',' , header=None)
		self.layer1_bv_ = tf.Variable(df.values, name='layer1_bv_')

		pth = test_values_path + 'test0generation0child0_rbm_2/'
		df = pd.read_csv(pth + 'W.csv', sep=',' , header=None)
		self.layer2_W = tf.Variable(df.values, name='layer2_W')
		pd.read_csv(pth + 'bh_.csv', sep=',' , header=None)
		self.layer2_bh_ = tf.Variable(df.values, name='layer2_bh_')
		pd.read_csv(pth + 'bv_.csv', sep=',' , header=None)
		self.layer2_bv_ = tf.Variable(df.values, name='layer2_bv_')
		
		pth = test_values_path + 'test0generation0child0_rbm_3/'
		df = pd.read_csv(pth + 'W.csv', sep=',' , header=None)
		self.layer3_W = tf.Variable(df.values, name='layer3_W')
		df = pd.read_csv(pth + 'bh_.csv', sep=',' , header=None)
		self.layer3_bh_ = tf.Variable(df.values , name='layer3_bh_')
		df = pd.read_csv(pth + 'bv_.csv', sep=',' , header=None)
		self.layer3_bv_ = tf.Variable(df.values , name='layer3_bv_')

		self.X = tf.placeholder(dtype=tf.float32 , name='Dataset')

		self.testfunc = tf.sigmoid(tf.matmul(self.X, self.layer1_W) + self.layer1_bh_)

	def test(self , dataset):
		init_op = tf.global_variables_initializer()

		with tf.Session() as sess:
			sess.run(init_op)
			sess.run(self.testfunc  , feed_dict={self.X:dataset})
			#sess.run(self.layer3_bh_.eval())
		

TRAINING_SET_PATH = "RBMTrainingDataset/training_set.csv"
TEST_SET_PATH = "FormattedFantasyData/2018_data.csv"
NUM_DATAPOINTS = 22

test_set = pd.read_csv(TEST_SET_PATH , sep=',' , header=None)
test_set = test_set.values

x = ptsne()
x.test(test_set)