# our specially designed feed forward neural network
# this FFNN will have multiple RBM which it will pretrain, using output of one as input for the other
# it will then use this stack as a pretrained FFNN that will train using standard backprop

import tensorflow as tf 
import pandas as pd
import numpy as np 
import rbm
import testing_tools
import genetic_helpers as genhelp

TRAINING_SET_PATH = "RBMTrainingDataset/training_set.csv"
TEST_SET_PATH = "FormattedFantasyData/2018_data.csv"
NUM_DATAPOINTS = 22
TESTING_PATH = 'FFNN_Dev_Testing'

class FFNN(object):
	

	def __init__(self, name, input_layer_size , output_layer_size, dna):
		self.dna = dna # the genetic code that shapes our FFNN
		self.rbms = []
		self.rbm_shapes = []
		self.name = name
		self.input_layer_size = input_layer_size
		self.output_layer_size = output_layer_size
		# set up RBMs
		self._setup(genhelp.read_blueprint(self.dna))

	# 
	def _setup(self, blueprint):
		self._setup_shapes(blueprint)
		#self._create_RBMs()


	# function that accepts a list of layer sizes (read from the dna bitstring) and populates a list of 2-element lists specifying sizes
	#	for all of our RBMs
	def _setup_shapes(self, blueprint):
		# determine how many RBMs we will have
		# NOTE - this should be the len of the blueprint plus one 
		num_rbms = len(blueprint) + 1

		# special case, there is only one layer
		if num_rbms == 1:
			layer_1_RBM_size = [self.input_layer_size , self.output_layer_size]
		else:
			# the first entry in our blueprint is the hidden layer for our first RBM (the visible layer is input_layer_size for this FFNN object)
			layer_1_RBM_size = [self.input_layer_size , blueprint[0]]

		self.rbm_shapes.append(layer_1_RBM_size)

		# now we get the sizes of the other RBMs
		for vsize , hsize in zip(blueprint , blueprint[1:]):
			dims = [vsize,hsize]
			self.rbm_shapes.append(dims)

		# if our special case wasn't true above, we need to create the 'top' layer using output_layer_size of FFNN object
		if num_rbms == 1:
			pass # we only have one layer, and the size of it was appended via layer_1_RBM_size
		else:
			final_layer_size = [blueprint[-1] , self.output_layer_size]
			self.rbm_shapes.append(final_layer_size)

	# function that uses the list of RBM shapes and creates our stack of RBMs


	# returns the true DNA of this instance of FFNN
	def get_dna(self):
		return self.dna

	# returns a gamete (mutated bistring) for this FFNN
	def get_gamete(self):
		return genhelp.mutate_bitstring(self.dna , 0.1)




	''' ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ TESTING FUNCTIONS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ '''

	def show_RBM_layers(self):
		print(self.rbm_shapes)

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






