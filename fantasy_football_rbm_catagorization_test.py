import rbm
import pandas as pd 
import testing_tools
import os 
from pathlib import Path


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

# initialize RBM
r = rbm.RBM(NUM_DATAPOINTS , 3 , visible_unit_type='gauss', model_name="fantasy_position" , verbose=1, main_dir='fantasy_test')

# fit for training set
r.fit(training_set)

# see what this puppy thinks of the test set
testing_tools.write_csv(r.transform(test_set) , 'RBMTestingResults/2018_data_transform_first.csv')

print("finished")
