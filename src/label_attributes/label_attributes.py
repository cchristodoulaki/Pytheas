import os, sys, argparse, traceback
from sqlalchemy import MetaData, Table, Column, Integer, String, create_engine
from dotmap import DotMap
import json
import pandas as pd
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm
import pprint as pp
sys.path.append('../pytheas')
from table_classifier_utilities import predict_header_indexes, combo_row


def save_table_attributes(datatable_key, attributes):
    metadata = MetaData()
    metadata.reflect(opendata_engine, only=['datacolumn'])
    datacolumn_table = Table('datacolumn', metadata, autoload=True, autoload_with=opendata_engine)

    with opendata_engine.begin() as conn:
        # delete any saved attributes that might exist for this table
        conn.execute("DELETE FROM datacolumn where datatable=%s", (datatable_key, ))

        # insert attributes
        insert_values = list()
        for attribute_idx in attributes.keys():
            values = {
                "datatable":datatable_key, 
                "attribute_idx":attribute_idx, 
                "csv_idx":attributes[attribute_idx]["csv_idx"], 
                "attribute_name":attributes[attribute_idx]["attribute_name"], 
                "merged_name":attributes[attribute_idx]["merged_name"],
                "pd_dtype":attributes[attribute_idx]["pd_dtype"]
                }
            insert_values.append(values)
        opendata_engine.execute(datacolumn_table.insert(),insert_values)


def label_attributes_in_file(task):  
    with opendata_engine.begin() as  conn:
            conn.execute("""DELETE FROM label_attributes WHERE datafile = %s""", (task.datafile_key))  
    try:
        # low_memory=False -> whole columns will be read in first, then  proper types determined.
        file_dataframe = pd.read_csv(task.path, delimiter=task.delimiter, encoding=task.encoding, header=None, low_memory=False, engine='c')

        # print(f"\nFILE:\n\n{file_dataframe}")
        # pp.pprint(task)
        header_upper_bound = 0
        tables = sorted(task.annotations["tables"], key = lambda table: table["table_counter"])
        
        for table in tables:

            discovered_body_dataframe = file_dataframe.loc[table["data_start"]:table["data_end"]]

            discovered_body_dataframe = discovered_body_dataframe.convert_dtypes(infer_objects=True, convert_string=True, convert_integer=True, convert_boolean=True)  
            for c in discovered_body_dataframe.columns:
                discovered_body_dataframe[c]=pd.to_numeric(discovered_body_dataframe[c], errors='ignore')
            discovered_header_dataframe = pd.DataFrame()

            if header_upper_bound != table["data_start"]:
                candidate_header_dataframe = file_dataframe.loc[header_upper_bound:table["data_start"]-1]                
                discovered_header_idxs, _ = predict_header_indexes(pd.concat([candidate_header_dataframe, discovered_body_dataframe], ignore_index=True), len(candidate_header_dataframe), table['table_counter'])
                discovered_header_idxs = [i+header_upper_bound for i in discovered_header_idxs]
                if len(discovered_header_idxs)>0:
                    discovered_header_dataframe = file_dataframe.loc[discovered_header_idxs]

            # print(f"\n\n\n-- Table {table['table_counter']}:\n\n")
            # print(f"Table Header (candidate): \n\n {candidate_header_dataframe}\n\n")  
            # print(f"Table Header (discovered): \n\n {discovered_header_dataframe}\n\n")            
            # print(f"Table Body (discovered): \n\n{discovered_body_dataframe}\n\n") 

            # update header_upper_bound for next table
            header_upper_bound = table["data_end"]+1

            # stack discovered header with discovered body
            try:
                table_dataframe = pd.concat([discovered_header_dataframe, discovered_body_dataframe])
            except:
                table_dataframe=discovered_body_dataframe

            # remove all empty columns
            table_dataframe.dropna(axis='columns', how='all', inplace=True)
            try:
                attribute_names = combo_row(table_dataframe.loc[discovered_header_idxs])
            except:
                attribute_names = [None for c in table_dataframe.columns]

            table_attributes = {}
            for attribute_idx, csv_idx in enumerate(table_dataframe.columns):
                merged_name = True #TODO implement flag for merged attribute name
                table_attributes[attribute_idx] = {
                    "csv_idx":csv_idx,
                    "attribute_name": None if pd.isnull(attribute_names[attribute_idx]) else attribute_names[attribute_idx],
                    "merged_name": merged_name,
                    "pd_dtype": pd.api.types.infer_dtype(discovered_body_dataframe[csv_idx], skipna=True)
                }

            # pp.pprint(table_attributes)
            save_table_attributes(table["datatable_key"], table_attributes)
            
    except Exception as e:

        exc_type, exc_value, exc_traceback = sys.exc_info()
        # print("*** print_tb:")
        # traceback.print_tb(exc_traceback, limit=1, file=sys.stdout)

        # print("*** print_exception:")
        # exc_type below is ignored on 3.5 and later
        # traceback.print_exception(exc_type, exc_value, exc_traceback,
        #                         limit=2, file=sys.stdout)
        with opendata_engine.begin() as  conn:
            conn.execute("""INSERT INTO  label_attributes (datafile, exc_type, exc_value, exception) VALUES (%s, %s, %s, %s)""",(task.datafile_key, str(exc_type), str(exc_value), type(e).__name__))

        # print("*** print_exc:")
        # traceback.print_exc(limit=2, file=sys.stdout)
        # print("*** format_exc, first and last line:")
        # formatted_lines = traceback.format_exc().splitlines()
        # print(formatted_lines[0])
        # print(formatted_lines[-1])
        # print("*** format_exception:")
        # # exc_type below is ignored on 3.5 and later
        # print(repr(traceback.format_exception(exc_type, exc_value,
        #                                     exc_traceback)))
        # print("*** extract_tb:")
        # print(repr(traceback.extract_tb(exc_traceback)))
        # print("*** format_tb:")
        # print(repr(traceback.format_tb(exc_traceback)))
        # print("*** tb_lineno:", exc_traceback.tb_lineno)


def task_generator(files):
    for datafile in files:
        
        task = DotMap()                
        task.path = os.path.join(args.crawl_directory, datafile["path"])
        task.datafile_key = datafile["datafile_key"]
        task.delimiter = datafile["delimiter"]
        task.encoding = datafile["encoding"]
        task.annotations = {"tables":[]}        

        with opendata_engine.connect() as conn:
            result = conn.execute("""SELECT datatable_key, table_index, from_index, to_index
                                        FROM datatable
                                        WHERE datafile = %s
                                        ORDER BY table_index""", (task.datafile_key,)) 
            for table in result:
                task.annotations["tables"].append({
                    "datatable_key":table["datatable_key"],
                    "table_counter":table["table_index"],
                    "data_start":table["from_index"],
                    "data_end": table["to_index"]
                })                                                

        yield(task)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()    
    parser.add_argument("-i", "--input_portals", nargs="*", default=['open.canada.ca_data'])
    parser.add_argument("-d", "--crawl_directory", default = "/home/christina/OPEN_DATA_CRAWL_2018")
    parser.add_argument("-p", "--NPROC", type = int, default = 4)
    parser.add_argument("-c", "--db_cred_file", default="../database_credentials.json")

    args = parser.parse_args(sys.argv[1:])

    with open(args.db_cred_file) as f:
        credentials = json.load(f)

    # Database connection credentials 
    db_cred = DotMap() 
    db_cred.user = credentials["user"]
    db_cred.password = credentials["password"]
    db_cred.database = credentials["ground_truth_db"]
    db_cred.opendata_database = credentials["profile_db"]
    db_cred.port = credentials["port"]

    opendata_engine = create_engine(f'postgresql+psycopg2://{db_cred.user}:{db_cred.password}@localhost:{db_cred.port}/{db_cred.opendata_database}')


    if not opendata_engine.dialect.has_table(opendata_engine, 'label_attributes'):  # If table don't exist, Create.
        metadata = MetaData(opendata_engine)
        # Create a table with the appropriate Columns
        Table('label_attributes', metadata,
            Column('datafile', Integer, primary_key=True, nullable=False), 
            Column('exc_type', String), 
            Column('exc_value', String),
            Column('exception', String))
        # Implement the creation
        metadata.create_all()
            

    for portal in args.input_portals:
        with opendata_engine.connect() as conn: 
            result = conn.execute("""SELECT count(1) as count 
                                     FROM datafile 
                                     WHERE endpoint_dbname = %s
                                            AND datafile_key in 
                                                (SELECT datafile from datatable)""", (portal, ))

            NINPUTS = result.fetchone()[0]
            print(f'NINPUTS={NINPUTS}')

            files = conn.execute("""SELECT datafile_key, path, encoding, delimiter
                            FROM datafile 
                            WHERE endpoint_dbname = %s
                                AND datafile_key in  (SELECT datafile from datatable)"""
                                , (portal, )) 

            with Pool(processes=args.NPROC) as pool:
                with tqdm(total=NINPUTS) as pbar:
                    for r in pool.imap_unordered(label_attributes_in_file, task_generator(files)):
                        pbar.update(1)