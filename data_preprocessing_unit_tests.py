import unittest
import pandas as pd
import data_preprocessing_functions as ppfuncs 

class DataProcessingTest(unittest.TestCase):

	def setUp(self):
		try:
			self.fixture = pd.read_csv('data/german_credit_car_fraud_testcut.ssv',
				sep = ' ',
				header = 0) 
		except IOError:
			print 'cannot open file'
		

	def tearDown(self):
		# called after each test
		del self.fixture

	def test1(self):
		self.assertTrue(0==0, 'Test1 failed.')

	def test2(self):
		self.assertEquals(1,1, 'Test2 failed')

	# Test variable categorisation function
	def test_NoCatVars(self):
		# Count number of remaining cat vars

	def test_totVars(self):
		# Check total number of variables after categorical conversion and
		# categorical var removal functions

	# Check the modify class variable function is working properly.
	# Need to be corrected to take account returning y now as well as data from modifyclassvar
	def test_classMaxVal(self):
		test, y = ppfuncs.modifyClassVariable(self.fixture)
		maxVal = test['class'].max()
		self.assertEqual(maxVal,1,'Class Max Value More Than 1')

	def test_classMinVal(self):
		test = ppfuncs.modifyClassVariable(self.fixture)
		minVal = test['class'].min()
		self.assertEqual(minVal,0,'Class Min Value Not 0')

	def test_classMatchVals(self):
		test = ppfuncs.modifyClassVariable(self.fixture)
		classVals = test['class']
		compare = [0, 1, 0, 0, 1, 0, 0, 0, 0, 1]
		self.assertItemsEqual(classVals, compare, 'Class Vals Not As Expected')

	def test_noNanClass(self):
		test = ppfuncs.modifyClassVariable(self.fixture)
		self.assertTrue(test['class'].isnull().sum() == 0, 'NaNs in Class Var')

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
	
	unittest.main()

	
