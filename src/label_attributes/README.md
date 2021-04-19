This project takes as input a file and a table boundary annotation (i.e., first and last line of the table body, and lines participating in the table header).

The table is retrieved from the file, attributes of the file are identified, and the header section is processed to produce a name for each attribute. 


Apply label_attributes to tables discovered from `open.canada.ca`:
```
python label_attributes.py --input_portals open.canada.ca_data --crawl_directory /home/christina/OPEN_DATA_CRAWL_2018 --NPROC 64 --db_cred_file ../database_credentials.json
```