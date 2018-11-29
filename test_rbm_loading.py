import rbm
import pandas as pd 
import testing_tools as tt 

# import our data (numerical values only)
df = pd.read_csv("nums_only.csv" , sep=',' , header=None)
df = df.values # just recasts our dataframe as a numpy array without labels or column names etc

r = rbm.RBM(27 , 2 , visible_unit_type='gauss', model_name="test_model" , verbose=1, main_dir='sametest')

r.load_model([27,2] , 1, '/home/tyler/2018_Development/Fantasy_Stats_ML/sametest/models/test_model') 

print(r.get_model_parameters())

tt.write_csv(r.transform(df) , 'model_trained_data_transformed_2.csv')
