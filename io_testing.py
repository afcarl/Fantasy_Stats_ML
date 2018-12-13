import zconfig
import os
import feed_forward_neural_network as ffnn 
import pandas as pd

dummy_simple_blueprint = '0000000010000000000001011000000000000111'

z = '000000000000000000000000000000000000000000000000000000000000000000000000000000000000000010000000000010101000000000000111'


# traning dataset path
TRAINING_SET_PATH = "RBMTrainingDataset/training_set.csv"
TEST_SET_PATH = "FormattedFantasyData/2018_data.csv"
NUM_DATAPOINTS = 22


# import our training dataset into a dataframe, then immediately recast it to numpy array
#	this dataset contains fantasy stats for the past decate (excluding this current season)
#	and is numerical only
# 	the data has 22 points of data for each player
training_set = pd.read_csv(TRAINING_SET_PATH , sep=',' , header=None)
training_set = training_set.values

# get our test set
test_set = pd.read_csv(TEST_SET_PATH , sep=',' , header=None)
test_set = test_set.values

# simple test scrip to test directory setup
x = ffnn.FFNN('testFFNN' , 22 , 2 , z , main_dir='FFNN')
x.train_RBMs(training_set)
x.save_RBM_data()