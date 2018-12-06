# our specially designed feed forward neural network
# this FFNN will have multiple RBM which it will pretrain, using output of one as input for the other
# it will then use this stack as a pretrained FFNN that will train using standard backprop

import tensorflow as tf 
import pandas as pd
import numpy as np 
import rbm
import testing_tools

TRAINING_SET_PATH = "RBMTrainingDataset/training_set.csv"
TEST_SET_PATH = "FormattedFantasyData/2018_data.csv"
NUM_DATAPOINTS = 22
TESTING_PATH = 'FFNN_Dev_Testing'

class FFNN(object):
	

	def __init__(self, name):
		self.rbms = []
		self.name = name
		print("hi my name is: " , self.name)

	def testing_run_hardcoded_RBM_pass(self):
		self.rbms.append(rbm.RBM(NUM_DATAPOINTS, 3 , visible_unit_type='gauss', model_name=self.name +"_layer_1" , verbose=1, main_dir='layer_1_test'))
		self.rbms.append(rbm.RBM(3, 2 , visible_unit_type='bin' , model_name=self.name+'_layer_2' , verbose=1 , main_dir='layer_2_test'))

		training_set = pd.read_csv(TRAINING_SET_PATH , sep=',' , header=None)
		training_set = training_set.values

		self.rbms[0].fit(training_set)

		test_set = pd.read_csv(TEST_SET_PATH , sep=',' , header=None)
		test_set = test_set.values


		training_set_transform = self.rbms[0].transform(training_set)

		testing_tools.write_csv(training_set_transform , TESTING_PATH+'/L1_training_set_transformed.csv')

		self.rbms[1].fit(training_set_transform)

		x = self.rbms[1].transform(training_set_transform)
		testing_tools.write_csv(x , TESTING_PATH+'/L2_training_set_transformed_transformed.csv')

	def testing_load_previous_and_transform(self):
		# Note - this doesn't work and i don't know why
		# load up our rbms[2]
		r = rbm.RBM(3 , 2 , visible_unit_type='bin' , model_name=self.name+'_layer_2' , verbose=1 , main_dir='layer_2_test')
		r.load_model([[3,2],[22,3]] , 1, '/home/tyler/2018_Development/Fantasy_Stats_ML/layer_2_test/models/test1_layer_2') 

		test_set = pd.read_csv('FFNN_Dev_Testing/L1_training_set_transformed.csv' , sep=',' , header=None)

		testing_tools.write_csv(r.transform(test_set) , TESTING_PATH+'/Loaded_L2_transform.csv')






