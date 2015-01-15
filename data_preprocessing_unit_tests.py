import unittest
import pandas as pd

class PreprocessingTest(unittest.TestCase, data):
	# Check class is only 1 or 0 and never blank
	#def no_nan_class(self):
		# test that the class variable has no nan values
		# achieved by counting the number of missing values
		# based on a pandas data array
		null_count = sum(isnull(row) for row in data['class'])
		self.assertTrue(null_count == 0, 'Test failed.')

	#def no_missing_class(self):
		# test no missing values	

	#def class_binary(self):
		# check that the class variable type is binary
# check types of input vars

	# When subsetting data, check that the number of remaining fields is as 
	# expected.

if __name__ == '__main__':
	TEST_INPUT_DIR = '/Users/tobybalfre/Documents/Data Science and Analytics/Code Library/python/machine learning/python_modeling_unit_testing_branch/data/'
	test_file_name =  'german_credit_car_fraud_testcut.ssv'
	try:
		data = pd.read_csv(INPUT_DIR + file_name,
			sep = ' ', #using this to set what separator is used (e.g. , or | etc)
			header = 0) #this tells the parser if there is a header and on which row
	except IOError:
		print('cannot open file')
	unittest.main(data)

