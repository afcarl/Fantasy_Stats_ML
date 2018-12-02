# file will open both our transformed data and our labels of that data and produce groupings of players
# according to whatever the trained RBM though 
import os 
from pathlib import Path
import csv


def format_transformed_data(three_point_list):
	# list comprehension to turn every float to int
	try:
		ints = [ int(float(x)) for x in three_point_list ]
		result = ''
		for i in ints:
			result = result + str(i)
		return result
	except ValueError:
		print("issue with three_point_list: " , three_point_list)


### GET HANDLES FOR FILES ###

TRANSFORMED_DATA = 'RBMTestingResults/2018_data_transform_first.csv'
DATA_LABELS = 'FormattedFantasyData/2018_labels.csv'

# get our current dir
current_dir = os.path.dirname(os.path.realpath(__file__))
if current_dir[-1] is not '/' : current_dir = current_dir + '/'
# get a handle for the data directory we want to access
parent_dir = str(Path(current_dir).parent)
if parent_dir[-1] is not '/' : parent_dir = parent_dir + '/'

data_path = parent_dir + TRANSFORMED_DATA
labels_path = parent_dir + DATA_LABELS

numdata = open(data_path , 'r')
labeldata = open(labels_path , 'r')

numdata_reader = csv.reader(numdata, delimiter=',')
labeldata_reader = csv.reader(labeldata, delimiter=',')


players = {}



for datapoint, label in zip(numdata_reader, labeldata_reader):
	try:
		name, team, pos = label
	except ValueError:
		print("error processing: " , label)
	bitstring = format_transformed_data(datapoint)
	playerlabel = name + ',' + team + ',' + pos

	# build our data
	if bitstring not in players:
		players[bitstring] = [playerlabel]
	else:
		players[bitstring].append(playerlabel)


for index in players:
	print(index)
	players_list = players[index]
	pos_count = {}
	for player in players_list:
		try:
			name, team, pos = player.split(',')
		except ValueError:
			print("error processing: " , player)
		if pos not in pos_count:
			pos_count[pos] = 1
		else:
			pos_count[pos] = pos_count[pos] + 1
	group_list = []
	for tup in pos_count.items():
		group_list.append(tup)

	sorted_group_list = sorted(group_list , key=lambda tup: tup[1], reverse=True)
	print(sorted_group_list)








