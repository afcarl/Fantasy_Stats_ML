# script that formats our raw fantasy football data into a format for categorization by our RBM
# this data will split the headers from the raw numbers, and also remove the data
# that lists fantasy output (seeing as we want our analysis to be potentially fantasy
# relevant, it seems inappropriate to include overall fantasy ranking in the dataset)

# necessary imports
#import pandas as pd 
import os
from pathlib import Path

# ability to iterate through every file that begins with a yearin the Raw_Fantasy_Data 
#	folder
dir_path = os.path.dirname(os.path.realpath(__file__))
data_dir_name = 'RawFantasyData'

# get the parent of the current working directory
if dir_path[-1] is not '/' : dir_path = dir_path + '/'
parent_dir = str(Path(dir_path).parent)

# get a handle for the data directory we want to access
if parent_dir[-1] is not '/' : parent_dir = parent_dir + '/'
data_dir = parent_dir + data_dir_name 

# get all the files in our RawFantasyData folder
data_filenames = os.listdir(path=data_dir)

# for each file, split the header away and save to our chosen directoyr, remove the fantasy


# example of how this can be split in pandas


#	data and store the numbers in the same folder under a raw.csv file