# testing driver for our FFNN
import feed_forward_neural_network as ffnn 
import genetic_helpers as g 
import random as rand 
# instance my ffnn


''' testing hardcoded rbm strining
network = ffnn.FFNN("test1")
network.testing_run_hardcoded_RBM_pass()
''' 

''' testing loading DOESN'T WORK AT ALL LOLZ
network = ffnn.FFNN('test1')
network.testing_load_previous_and_transform()
'''

def test_breeding():
	# test a generation
	num_kids = 10
	num_eras = 5
	kidz = []
	gen = [] # this is a list of FFNN objects



	for i in range(num_kids):
		# get a random dna string
		dna = g.generate_blueprint()
		network = ffnn.FFNN(str(i)+'th', 22 , 3 , dna)
		gen.append(network)

	for i in gen:
		i.show_RBM_layers()

	# randomly pick two fittest networks
	two_fittest = []
	two_fittest.append(rand.randint(0,num_kids-1))
	next_val = rand.randint(0,num_kids-1)
	while next_val in two_fittest:
		next_val = rand.randint(0,num_kids-1)
	two_fittest.append(next_val)
	a,b = two_fittest # the indicies of our two fittest for testing


	for i in range(num_kids // 2):
		try:
			kid1 , kid2 = g.breed_bitstrings(gen[a].get_gamete() , gen[b].get_gamete())
		except IndexError:
			print("ERROR")
			print(a, ":" , b)
		kidz.append(ffnn.FFNN(str(i)+'st_offspring' , 22, 3, kid1))
		kidz.append(ffnn.FFNN(str(i)+'st_offspring' , 22, 3, kid2))

	print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

	for i in kidz:
		i.show_RBM_layers()



test_breeding()

