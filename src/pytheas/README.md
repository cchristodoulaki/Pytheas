# Setup

```
python -m nltk.downloader stopwords
```
# Pytheas Python API

## Train from dataset
Training from annotated data requires two folders, one with CSV files and one with JSON files containing manual annotations over the CSV files. A file with annotations for a CSV file must have the same filename as the CSV file (and extension `.json`).

```
from pytheas import pytheas
pytheas = pytheas.API()
pytheas.learn_and_save_weights('../../data/Canada/csv_files','../../data/Canada/csv_annotations')
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

In the example below, we load rule weights from a file, and apply demo.csv:

```python

import pytheas
from pprint import pprint
Pytheas = pytheas.API()

#load pretrained rule weights
Pytheas.load_weights('trained_rules.json')

filepath = '../../data/examples/demo.csv'     
file_annotations = Pytheas.infer_annotations(filepath)
pprint(file_annotations) 

```

# Pytheas CLI

### learn_and_save_weights(files, annotations, output_path, parameters=None)
learns weights and saves weights and parameters to output_path

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
python pytheas.py infer -w trained_rules.json -f ../../data/examples/demo.csv -o inferred_annotation.json
```

# Install with pip:
```
pip install --upgrade --force-reinstall pytheas-0.0.1-py3-none-any.whl
```

To run unit tests:
```
python -m unittest
```
