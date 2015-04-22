import unittest
import pandas as pd
import data_preprocessing_functions as ppfuncs 

class DataProcessingTest(unittest.TestCase):

    def setUp(self):
        try:
            test_data = pd.read_csv('data/dataset.csv',
                sep = ',',
                header = 0)
            test_data['end_date'] = pd.to_datetime(test_data['end_date'])
            test_data['start_date'] = pd.to_datetime(test_data['start_date'])
            self.fixture = test_data
        except IOError:
            print ('cannot open file')


    def tearDown(self):
        # called after each test
        del self.fixture

    def test_classMaxVal(self):
        test = ppfuncs.modifyClassVariable(self.fixture)
        maxVal = test['class_variable'].max()
        self.assertEqual(maxVal,1,'Class Max Value More Than 1')

    def test_classMinVal(self):
        test = ppfuncs.modifyClassVariable(self.fixture)
        minVal = test['class_variable'].min()
        self.assertEqual(minVal,0,'Class Min Value Not 0')

    def test_noNanClass(self):
        test = ppfuncs.modifyClassVariable(self.fixture)
        self.assertTrue(test['class_variable'].isnull().sum() == 0, 'NaNs in Class Var')	


    def test_EndDateMax(self):
        # check that the end date is no later than expected and filtering has worded successfully.
        test = ppfuncs.rowSelection(self.fixture)
        self.assertTrue(test['end_date'].dt.year.max() == 2015, 'Max End Date Not 2015')
        
    def test_EndDateMin(self):
        # check that the end date is no later than expected and filtering has worded successfully.
        test = ppfuncs.rowSelection(self.fixture)
        self.assertTrue(test['end_date'].dt.year.min() == 2013, 'Min End Date Not 2013')
    
    def test_restStartDateMax(self):
        # check that the start date is no later than expected and filtering has worded successfully.
        test = ppfuncs.rowSelection(self.fixture)
        self.assertTrue(test['start_date'].dt.year.max() == 2015, 'Max Start Date Not 2015')
    
    def test_restStartDateMin(self):
        # check that the start date is no later than expected and filtering has worded successfully.
        test = ppfuncs.rowSelection(self.fixture)
        self.assertTrue(test['start_date'].dt.year.min() == 2013, 'Min Start Date Not 2013')
    
    def test_numberRows(self):
        # check that the number of rows post filtering is as expected
        test = ppfuncs.rowSelection(self.fixture)
        self.assertTrue(len(test['end_date']) == 4, 'Incorrect number of rows being returned by RowSelector')
    
        
    def test_removeOutliers(self):
        # Check  that outlier removal returns the expected number of rows
        var_names = {'outliertest1','outliertest2'}
        test = ppfuncs.removeOutliers(self.fixture, var_names)
        print(len(test))
        self.assertTrue(len(test) == 6, 'Number of rows returned from outlier removal not as expected')
    
    def test_removeOutliersMaxVal(self):
        # Second check on outlier removal
        var_names = {'outliertest1','outliertest2'}
        test = ppfuncs.removeOutliers(self.fixture, var_names)
        result = test['outliertest1'].max()
        self.assertTrue(result == 2, 'Expected max value for outliertest1 after outlier removal is not as expected.')
        
    def test_removeOutliersMinVal(self):
        # Second check on outlier removal
        var_names = {'outliertest1','outliertest2'}
        test = ppfuncs.removeOutliers(self.fixture, var_names)
        result = test['outliertest1'].min()
        self.assertTrue(result == 1, 'Expected min value for outliertest1 after outlier removal is not as expected.')
    
if __name__ == '__main__':

    

    unittest.main()
