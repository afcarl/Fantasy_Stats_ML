import rbm
import pandas as pd 
import testing_tools as tt 

# import our data (numerical values only)
TEST_SET_PATH = "FormattedFantasyData/2018_data.csv"
test_set = pd.read_csv(TEST_SET_PATH , sep=',' , header=None)
test_set = test_set.values

r = rbm.RBM(22 , 2 , visible_unit_type='gauss', model_name="test_model" , verbose=1, main_dir='sametest')

r.fit(test_set)

#tt.write_csv(r.get_model_parameters() , 'model_made_data_')
#tt.write_csv(r.transform(df) , 'model_trained_data_transformed_1.csv')

print(r.get_model_parameters())

