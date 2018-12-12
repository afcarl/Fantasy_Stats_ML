import zconfig
import os
import feed_forward_neural_network as ffnn 

dummy_simple_blueprint = '0000000010000000000001011000000000000111'

z = '000000000000000000000000000000000000000000000000000000000000000000000000000000000000000010000000000010101000000000000111'


# simple test scrip to test directory setup
x = ffnn.FFNN('testFFNN' , 22 , 3 , z , main_dir='BigMoodTest')

