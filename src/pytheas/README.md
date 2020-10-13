# Pytheas Python API
## Train from dataset
Training from annotated data requires two folders, one with CSV files and one with JSON files containing manual annotations over the CSV files. A file with annotations for a CSV file must have the same filename as the CSV file (and extension `.json`).

```
import pytheas
pytheas = pytheas.API()
pytheas.learn_and_save_weights('../../data/Canada/csv_files','../../data/Canada/csv_annotations')
```
```python
f1 = 'trained_rules.json'
f2 = 'train_output.json'
with open(f1) as f: 
    a = json.load(f)

with open(f2) as f: 
    b = json.load(f)['fuzzy_rules']    

a==b
```    
## Load trained weights
```python
import pytheas
import json
from pprint import pprint
Pytheas = pytheas.API()
Pytheas.load_weights('trained_rules.json')
```

# Apply Pytheas to CSV file
To apply Pytheas to a CSV file, we must either train the model on an annotated dataset, or load pretrained rules.

In the example below, we load rule weights from a file, and apply to three separate CSV files downloaded from open.canada.ca/data:

```python

import pytheas
from pprint import pprint
Pytheas = pytheas.API()

#load pretrained rule weights
Pytheas.load_weights('trained_rules.json')


# https://open.canada.ca/data/en/dataset/3f718801-099d-4037-bb0a-1d41ba8aca8b  
# Canada Small Business Financing Program (CSBFP)
filepath = '../../data/examples/open.canada.ca/3f718801-099d-4037-bb0a-1d41ba8aca8b/200d710b-17ac-4c1a-a624-fdcc5fc62af9'     
file_annotations = Pytheas.infer_annotations(filepath)
pprint(file_annotations) 


# https://open.canada.ca/data/en/dataset/a3ac17d5-2181-4360-accf-6a85fd3abc29
# Terminal elevator tariff summaries, Western
# A list of maximum tariffs (fees) that licensed grain companies charge for elevating, cleaning, drying and storing grain.
filepath = '../../data/examples/open.canada.ca/a3ac17d5-2181-4360-accf-6a85fd3abc29/1a250950-c78b-4e5c-be71-f84bf62737b2'     
file_annotations = Pytheas.infer_annotations(filepath)
pprint(file_annotations) 


# https://open.canada.ca/data/en/dataset/6609320b-ac9e-4737-8e9c-304e6e843c17
# Facts and Figures 2016: Immigration Overview - Temporary Residents – Annual IRCC Updates
filepath = '../../data/examples/open.canada.ca/6609320b-ac9e-4737-8e9c-304e6e843c17/IRCC_FF_TR_2016_01_CSV.csv'
file_annotations = Pytheas.infer_annotations(filepath)
pprint(file_annotations)
```

## set_params(params)
[TODO]

# Pytheas CLI

### learn_and_save_weights(files, annotations, output_path, parameters=None)
learns weights and saves weights and parameters to output_path

add a main function to the api?
so there can be a command line way of using pytheas

## Train
```
pytheas train --files files --annotations annotations
```

E.g.:

```
python pytheas.py train -c ../../data/Canada/csv_files -a ../../data/Canada/csv_annotations -o train_output.json
```

## Infer
```
pytheas infer --weights weightfile  --filepath filepath
```

E.g.:
```
python pytheas.py infer -w trained_rules.json -f ../../data/examples/open.canada.ca/3f718801-099d-4037-bb0a-1d41ba8aca8b/200d710b-17ac-4c1a-a624-fdcc5fc62af9 -o inferred_annotation.json
```

