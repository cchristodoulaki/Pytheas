import unittest
import pandas as pd
import pprint
from pytheas import pytheas
pp = pprint.PrettyPrinter(indent=4)

class TestPytheas(unittest.TestCase):

    def setUp(self):
        trained_rules = "../pytheas/trained_rules.json"
        self.Pytheas = pytheas.API()
        self.Pytheas.load_weights(trained_rules)

    def test_infer_annotations(self):
        filepath = "/home/christina/OPEN_DATA_CRAWL_2018/resource-csv-only/open.canada.ca_data/7537cd0c-fbcb-4d15-8c8f-ecb1d4849205/3f7ff461-32ff-4f16-b537-dac32461c295"
        max_lines = 100
        annotations = self.Pytheas.infer_annotations(filepath, max_lines)
        pp.pprint(annotations)