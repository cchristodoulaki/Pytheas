import os, sys, argparse
from sqlalchemy import create_engine
from dotmap import DotMap
import json
import pandas as pd
from multiprocessing import Pool
from tqdm import tqdm
import pprint as pp
sys.path.append('../pytheas')
from table_classifier_utilities import predict_header_indexes, combo_row

def label_attributes_in_file(task):
    with opendata_engine.connect() as conn:
        result = conn.execute("""SELECT table_index, from_index, to_index
                                    FROM datatable
                                    WHERE datafile = %s
                                    ORDER BY table_index""", (task.datafile_key,)) 

        file_dataframe = pd.read_csv(task.path, delimiter=task.delimiter, encoding=task.encoding, header=None)

        print(f"FILE:\n\n{file_dataframe}")
        header_upper_bound = 0
        for table in result:
                
            discovered_body_dataframe = file_dataframe.loc[table["from_index"]:table["to_index"]]           
            if header_upper_bound!=table["from_index"]:
                candidate_header_dataframe = file_dataframe.loc[header_upper_bound:table["from_index"]-1]                
                discovered_header_idxs, _ = predict_header_indexes(pd.concat([candidate_header_dataframe,discovered_body_dataframe], ignore_index=True), len(candidate_header_dataframe), table['table_index'])
                discovered_header_idxs = [i+header_upper_bound for i in discovered_header_idxs]
                discovered_header_dataframe = file_dataframe.loc[discovered_header_idxs]
            print(f"\n\n\n-- Table {table['table_index']}:\n\n")
            print(f"Table Header (candidate): \n\n {candidate_header_dataframe}\n\n")  
            print(f"Table Header (discovered): \n\n {discovered_header_dataframe}\n\n")            
            input(f"Table Body (discovered): \n\n{discovered_body_dataframe}\n\n")
            # update header_upper_bound for next table
            header_upper_bound = table["to_index"]

            # stack discovered header with discovered body
            table_dataframe = pd.concat([discovered_header_dataframe,discovered_body_dataframe])
            # remove all empty columns
            table_dataframe.dropna(axis='columns', how='all', inplace=True)
            attribute_names = combo_row(table_dataframe.loc[discovered_header_idxs])
            attributes = {}
            for attribute_idx, csv_idx in enumerate(table_dataframe.columns):
                attributes[attribute_idx] = {
                    "csv_idx":csv_idx,
                    "attribute_name":attribute_names[attribute_idx]
                }
            pp.pprint(attributes)
                


            

if __name__ == "__main__":

    parser = argparse.ArgumentParser()    
    parser.add_argument("-i", "--input_portals", nargs="*", default=['open.canada.ca_data'])
    parser.add_argument("-d", "--crawl_directory", default = "/home/christina/OPEN_DATA_CRAWL_2018")
    parser.add_argument("-p", "--NPROC", type = int, default = 4)
    parser.add_argument("-c", "--db_cred_file", default="../database_credentials.json")

    args = parser.parse_args(sys.argv[1:])

    with open(args.db_cred_file) as f:
        credentials = json.load(f)

    args = parser.parse_args(sys.argv[1:])
    # Database connection credentials 
    db_cred = DotMap() 
    db_cred.user = credentials["user"]
    db_cred.password = credentials["password"]
    db_cred.database = credentials["ground_truth_db"]
    db_cred.opendata_database = credentials["profile_db"]
    db_cred.port = credentials["port"]

    opendata_engine = create_engine(f'postgresql+psycopg2://{db_cred.user}:{db_cred.password}@localhost:{db_cred.port}/{db_cred.opendata_database}')

    for portal in args.input_portals:
        with opendata_engine.connect() as conn: 
            result = conn.execute("""SELECT count(1) as count 
                                     FROM datafile 
                                     WHERE endpoint_dbname = %s
                                            AND datafile_key in 
                                                (SELECT datafile from datatable)""", (portal, ))

            NINPUTS = result.fetchone()[0]
            input(f'NINPUTS={NINPUTS}')

            files = conn.execute("""SELECT datafile_key, path, encoding, delimiter
                            FROM datafile 
                            WHERE endpoint_dbname = %s
                                AND datafile_key in 
                                    (SELECT datafile from datatable)""", (portal, )) 

            # with Pool(processes=args.NPROC) as pool:
            #     with tqdm(total=NINPUTS) as pbar:
            #         for r in pool.imap_unordered(label_attributes_in_file, files):
            for datafile in files:
                
                task = DotMap()                
                task.path = os.path.join(args.crawl_directory, datafile["path"])
                task.datafile_key = datafile["datafile_key"]
                task.delimiter = datafile["delimiter"]
                task.encoding = datafile["encoding"]
                input(f'Enter to process task {task}')
                label_attributes_in_file(task)



