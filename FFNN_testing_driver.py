# testing driver for our FFNN
import feed_forward_neural_network as ffnn 

# instance my ffnn


''' testing hardcoded rbm strining
network = ffnn.FFNN("test1")
network.testing_run_hardcoded_RBM_pass()
''' 

''' testing loading '''
network = ffnn.FFNN('test1')
network.testing_load_previous_and_transform()