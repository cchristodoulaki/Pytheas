import unittest
import pandas as pd
import pytheas.table_classifier_utilities as table_util
import pprint
pp = pprint.PrettyPrinter(indent=4)

class TestTableClassifierUtilities(unittest.TestCase):
    def setUp(self):
        """ Your setUp """
        # test_file_path =  '/home/christina/OPEN_DATA_CRAWL_2018/resource-csv-only/open.canada.ca_data/06797992-acbb-4410-b6f1-f08b7ff2b6aa/f86563e2-aaec-4ed3-b61b-a559502c6ca1'
        test_file_path =  '/home/christina/OPEN_DATA_CRAWL_2018/resource-csv-only/open.canada.ca_data/45eb6514-0e38-48da-9a12-233d16813f4b/668a682f-80b0-4db9-b0ec-a848ca7ab949/2017-11/Table_1.csv'
        try:
            data = pd.read_csv(test_file_path, sep = ',', header = None)

        except IOError:
            print('cannot open file')

        table_dataframe = data.loc[0:8,:]
        null_columns = data.columns[table_dataframe.isna().all()].tolist()  

        self.filedf = data
        self.table_dataframe = table_dataframe.drop(null_columns, axis=1)
        self.header_df = self.table_dataframe.loc[[0,1]]
        self.header_df = self.table_dataframe.loc[[2,3,4]]

    def test_combo_row(self):
        combo_row = table_util.combo_row(self.header_df)
        print(combo_row)
        column_names = table_util.name_table_columns(self.header_df)
        pp.pprint(column_names)

        for csv_column, column in column_names.items():
            print(f'table_column_{column["table_column"]}={(" ".join([cell["value"] for cell in column["column_header"]])).strip()}')

