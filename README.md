# Pytheas: Pattern-based Table Discovery in CSV Files
Code repository for [Pytheas: Pattern-based Table Discovery in CSV Files](http://www.vldb.org/pvldb/vol13/p2075-christodoulakis.pdf), presented at VLDB 2020.
[![VLDB2020 Pytheas YouTube Presentation](http://img.youtube.com/vi/PHc-tGeZeD0/0.jpg)](http://www.youtube.com/watch?v=PHc-tGeZeD0 "VLDB2020 Pytheas YouTube Presentation")


#### Abstract

CSV is a popular Open Data format widely used in a variety of domains for its simplicity and effectiveness in storing and disseminating data. Unfortunately, data published in this format often does not conform to strict specifications, making automated data extraction from CSV files a painful task. While table discovery from HTML pages or spreadsheets has been studied extensively, extracting tables from CSV files still poses a considerable challenge due to their loosely defined format and limited embedded metadata.  
In this work we lay out the challenges of discovering tables in CSV files, and propose Pytheas: a principled method for automatically classifying lines in a CSV file and discovering tables within it based on the intuition that tables maintain a coherency of values in each column. We evaluate our methods over two manually annotated data sets: 2000 CSV files sampled from four Canadian Open Data portals, and 2500 additional files sampled from Canadian, US, UK and Australian portals. Our comparison to state-of-the-art approaches shows that Pytheas is able to successfully discover tables with precision and recall of over 95.9% and 95.7% respectively, while current approaches achieve around 89.6% precision and 81.3% recall. Furthermore, Pytheas’s accuracy for correctly classifying all lines per CSV file is 95.6%, versus a maximum of 86.9% for compared approaches. Pytheas generalizes well to new data, with a table discovery Fmeasure above 95% even when trained on Canadian data and applied to data from different countries. Finally, we introduce a confidence measure for table discovery and demonstrate its value for accurately identifying potential errors. 

#### Cite Pytheas :newspaper:

Christina Christodoulakis, Eric B. Munson, Moshe Gabel, Angela Demke Brown, and Renée J. Miller. Pytheas: Pattern-based Table Discovery in CSV Files. PVLDB, 13(11): 2075-2089, 2020. DOI: https://doi.org/10.14778/3407790.3407810

```bib
@article{DBLP:journals/pvldb/Christodoulakis20,
  author    = {Christina Christodoulakis and
               Eric Munson and
               Moshe Gabel and
               Angela Demke Brown and
               Ren{\'{e}}e J. Miller},
  title     = {Pytheas: Pattern-based Table Discovery in {CSV} Files},
  journal   = {Proc. {VLDB} Endow.},
  volume    = {13},
  number    = {11},
  pages     = {2075--2089},
  year      = {2020},
  url       = {http://www.vldb.org/pvldb/vol13/p2075-christodoulakis.pdf}
}
```

### Pytheas rule-set description
[HTML rendering of Pytheas rules](https://cchristodoulaki.github.io/Pytheas/) listed in [https://github.com/cchristodoulaki/Pytheas/tree/master/pytheas/rules](https://github.com/cchristodoulaki/Pytheas/tree/master/pytheas/rules)

## Installation instructions

The following instructions have been tested on a newly created Ubuntu 18.04 LTS with Python3.7 on a virtual environment.
Create your virtual environment by running `python3 -m venv pytheas-venv`, and activate it by running `source pytheas-venv/bin/activate`.
Clone the repo to your machine using git `git clone https://github.com/cchristodoulaki/Pytheas`.

### Setup

Package the Pytheas project into a [wheel](https://realpython.com/python-wheels/), and install it using pip:
```
cd src
python setup.py sdist bdist_wheel
pip install  --upgrade --force-reinstall dist/pytheas-0.0.1-py3-none-any.whl
```

You may need to download nltk stopwords:
```
python -m nltk.downloader stopwords
```

### Pytheas Python API

#### Load trained weights
We have pretrained Pytheas rules using a set of 2000 Open Data CSV files from Canadian CKAN portals. We expect to release those files and their annotations  in the future. 

```python
from pytheas import pytheas
import json
from pprint import pprint
Pytheas = pytheas.API()
Pytheas.load_weights('pytheas/trained_rules.json')
```
We now have a trained instance of Pytheas. 

#### Train from dataset

If you want, you can also train Pytheas using your own files and annotations. 
Training from annotated data requires two folders, one with CSV files and one with JSON files containing user annotations over the CSV files. A file with annotations for a CSV file must have the same filename as the CSV file (and extension `.json`).

```
from pytheas import pytheas
Pytheas = pytheas.API()
Pytheas.learn_and_save_weights('../data/Canada/csv_files','../data/Canada/csv_annotations')
```

#### Apply Pytheas to CSV file
To apply Pytheas to a CSV file, we must either train the model on an annotated dataset, or load pretrained rules.

In the example below, we load rule weights from a file, and apply demo.csv:

```python

from pytheas import pytheas
from pprint import pprint
Pytheas = pytheas.API()

#load pretrained rule weights
Pytheas.load_weights('pytheas/trained_rules.json')

filepath = '../data/examples/demo.csv'     
file_annotations = Pytheas.infer_annotations(filepath)
pprint(file_annotations) 

```

### Pytheas CLI

#### Train
```
pytheas train --files files --annotations annotations
```

E.g.:

```
python pytheas.py train -c ../data/Canada/csv_files -a ../data/Canada/csv_annotations -o train_output.json
```

#### Infer
```
pytheas infer --weights weightfile  --filepath filepath
```

E.g.:
```
python pytheas.py infer -w trained_rules.json -f ../../data/examples/demo.csv -o inferred_annotation.json
```
