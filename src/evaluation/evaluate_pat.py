import os, argparse, sys
from getpass import getpass
from multiprocessing import Pool
from tqdm import tqdm
import psycopg2 
import io
from sqlalchemy import create_engine
from timeit import default_timer as timer
from datetime import timedelta
import random
import pandas as pd
import numpy as np
from dotmap import DotMap

from psycopg2 import connect
from psycopg2.extras import Json
from psycopg2.extras import execute_values

import json
import pickle
import string
sys.path.append('../pytheas')
import pytheas as pat
import file_utilities as file_utilities
import evaluation_utilities
from evaluation_utilities import assign_class

import pprint
pp = pprint.PrettyPrinter(indent=4)

def eval_pat_line_kfold(pat_classifier, 
                        k_folds, 
                        db_cred, 
                        top_level_dir,
                        inject_percent_null,
                        inject_percent_outlier,
                        num_processors,
                        max_lines=100, 
                        assume_multi_tables=True):

    

    labels = ['BLANK', 'OTHER', 'HEADER','DATA', 'CONTEXT', 'FOOTNOTE', 'SUBHEADER']
    con=connect(dbname=db_cred.database, user=db_cred.user, host = 'localhost', password=db_cred.password, port = 5532)
    cur=con.cursor()

    cur.execute(f"""DROP TABLE IF EXISTS "{k_folds}fold_cross_validation_pat_model" """)
    con.commit()

    cur.execute(f"""CREATE TABLE "{k_folds}fold_cross_validation_pat_model"(
        fold_id integer,
        fuzzy_rules json
    ) """)
    con.commit()

    cur.execute(f"""DROP TABLE IF EXISTS "{k_folds}fold_cross_validation_pat_cells" """)
    con.commit()

    cur.execute(f"""CREATE TABLE "{k_folds}fold_cross_validation_pat_cells"(
        fold_id integer,
        measure text
    ) """)
    con.commit()

    for label in labels:
        cur.execute(f"""ALTER TABLE "{k_folds}fold_cross_validation_pat_cells" add column {label} real""")
    con.commit()
    cur.execute("DROP TABLE IF EXISTS pat_cell_predictions")
    con.commit()
    cur.execute("""CREATE TABLE pat_cell_predictions(
                    fold_id integer,
                    crawl_datafile_key integer, 
                    line_index integer, column_index integer, 
                    ground_truth_class text, pat_predicted_class text
                    )""")
    con.commit()

    cur.execute("DROP TABLE IF EXISTS pat_line_predictions")
    con.commit()
    
    cur.execute("""CREATE TABLE pat_line_predictions(
                    fold_id integer,
                    crawl_datafile_key integer, line_index integer,
                    gt_table_counter integer,pat_table_counter integer,
                    ground_truth_class text, pat_predicted_class text, 
                    gt_fdl text, pat_fdl text, 
                    gt_ldl text, pat_ldl text
                    )""")
    con.commit()

    cur.execute(f"""DROP TABLE IF EXISTS "{k_folds}fold_cross_validation_pat_lines" """)
    con.commit()

    cur.execute(f"""DROP TABLE IF EXISTS pat_table_confidences""")
    con.commit()

    cur.execute(f"""CREATE TABLE "{k_folds}fold_cross_validation_pat_lines"(
        fold_id integer,
        measure text
    ) """)
    con.commit()

    for label in labels:
        cur.execute(f"""ALTER TABLE "{k_folds}fold_cross_validation_pat_lines" add column {label} real""")
    con.commit()

    print('\nCollecting experiment setup...')  
    folds=dict()
    cur.execute(f"""SELECT fold_id, training, validation  FROM "{k_folds}fold_cross_validation" """)

    for result in cur:
        folds[result[0]]= dict()
        folds[result[0]]['train']= sorted(result[1])
        folds[result[0]]['test']= sorted(result[2])    
    cur.close()
    con.close()

    print('Experiments collected!\n') 

    results = dict()

    for fold_id in folds.keys():    
        results[fold_id] = dict()
        print('\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print(f'\n~~~~~~~~~~~~~~~~~~~~~ Cross Validation Fold {fold_id}:  ~~~~~~~~~~~~~~~~~~')
        print('\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        
        training_keys = folds[fold_id]['train']
        print('Unique files in train: ', len(set(training_keys)))
        test_keys = folds[fold_id]['test']
        print('Total files in test: ', len(test_keys))
            
        print('\nPreparing Training Data')
        start = timer()

        con=connect(dbname=db_cred.database, user=db_cred.user, host = 'localhost', password=db_cred.password, port = db_cred.port)
        undersampled_cell_data = pd.read_sql_query(
                sql = f"SELECT * FROM pat_data_cell_rules WHERE undersample=True and crawl_datafile_key in {tuple(training_keys)}", con=con)
        con.close()
        # print(f'undersampled_cell_data={undersampled_cell_data.head()}')

        con=connect(dbname=db_cred.database, user=db_cred.user, host = 'localhost', password=db_cred.password, port = db_cred.port)
        undersampled_line_data = pd.read_sql_query(
                sql = f"SELECT * FROM pat_data_line_rules WHERE undersample=True and crawl_datafile_key in {tuple(training_keys)}", con=con)
        con.close()
        # print(f'undersampled_line_data={undersampled_line_data.head()}')

        con=connect(dbname=db_cred.database, user=db_cred.user, host = 'localhost', password=db_cred.password, port = db_cred.port)
        undersampled_cell_not_data = pd.read_sql_query(
                sql = f"SELECT * FROM pat_not_data_cell_rules WHERE undersample=True and crawl_datafile_key in {tuple(training_keys)}", con=con)
        con.close()
        # print(f'undersampled_cell_not_data={undersampled_cell_not_data.head()}')

        con=connect(dbname=db_cred.database, user=db_cred.user, host = 'localhost', password=db_cred.password, port = db_cred.port)
        undersampled_line_not_data = pd.read_sql_query(
                sql = f"SELECT * FROM pat_not_data_line_rules WHERE undersample=True and crawl_datafile_key in {tuple(training_keys)}", con=con)
        con.close()
        # print(f'undersampled_line_not_data={undersampled_line_not_data.head()}')
        # input()

        pat_classifier.train_rules(undersampled_cell_data, undersampled_cell_not_data, undersampled_line_data, undersampled_line_not_data)
        end = timer() 

        con=connect(dbname=db_cred.database, user=db_cred.user, host = 'localhost', password=db_cred.password, port = db_cred.port)    
        # pp.pprint(pat_classifier.fuzzy_rules)
        cur = con.cursor()
        cur.execute(f""" INSERT INTO "{k_folds}fold_cross_validation_pat_model" (fold_id, fuzzy_rules) VALUES (%s, %s)""",
        (fold_id, Json(pat_classifier.fuzzy_rules)))
        con.commit()
        cur.close()
        con.close()

        print(f'\n-training data generated in {timedelta(seconds=end - start)}')
        print(f'Available CPUS:{pat.available_cpu_count()}')
        NINPUTS = len(test_keys)
        NPROC = min(num_processors,pat.available_cpu_count())

        print(f'NINPUTS={NINPUTS}')
        print(f'NPROC={NPROC}')
        # Process files
        processed_files=[]
        with Pool(processes=NPROC) as pool:
            with tqdm(total=NINPUTS) as pbar:
                for r in pool.imap_unordered(process_csv_worker,
                                             generate_process_csv_tasks(db_cred,
                                                                        top_level_dir,
                                                                        pat_classifier, 
                                                                        fold_id,
                                                                        test_keys, 
                                                                        max_lines, 
                                                                        assume_multi_tables,
                                                                        inject_percent_null,
                                                                        inject_percent_outlier
                                                                        )):
                    processed_files.append(r)
                    pbar.update(1) 



        print(f'\n-processed_files in fold {fold_id} in {timedelta(seconds=end - start)}')

        #### SAVE Ytest,Ypred ###
        start = timer()
        line_prediction_list  = [worker_output[0] for worker_output in processed_files]
        line_prediction = pd.concat(line_prediction_list)         
        cell_prediction_list  = [worker_output[1] for worker_output in processed_files]
        cell_prediction = pd.concat(cell_prediction_list)

        
        pat_line_classification_DATA = list(line_prediction.itertuples(index=False, name=None))
        pat_cell_classification_DATA = list(cell_prediction.itertuples(index=False, name=None))
        con = connect(dbname=db_cred.database, 
                        user=db_cred.user, 
                        host = 'localhost', 
                        password=db_cred.password, 
                        port = db_cred.port) 
        cur = con.cursor()             
        execute_values(cur,"""INSERT INTO pat_line_predictions 
                        (fold_id, crawl_datafile_key, line_index, gt_table_counter, pat_table_counter,
                        ground_truth_class, pat_predicted_class, 
                        gt_fdl, pat_fdl, 
                        gt_ldl, pat_ldl
                        ) VALUES %s""",
                        pat_line_classification_DATA) 
        execute_values(cur,"""INSERT INTO pat_cell_predictions 
                        (fold_id, crawl_datafile_key, line_index, 
                        column_index, ground_truth_class, pat_predicted_class
                        ) VALUES %s""",
                        pat_cell_classification_DATA) 
        con.commit()
        cur.close()
        con.close()
         
        #### PERFORMANCE ###       
        line_performance = evaluation_utilities.predict_performance(labels, line_prediction["annotated_label"], line_prediction["predicted_label"])
        print(f'\n\nline_performance=\n\n{line_performance}')
        results[fold_id]["line"] = line_performance
        evaluation_utilities.save_performance(db_cred, line_performance, f'"{k_folds}fold_cross_validation_pat_lines"', fold_id)
        
        boundary_performance = evaluation_utilities.predict_performance(["DATA_START"], 
                                                                        line_prediction["annotated_fdl"], 
                                                                        line_prediction["predicted_fdl"])

        data_end_performance = evaluation_utilities.predict_performance(["DATA_END"], 
                                                                        line_prediction["annotated_ldl"], 
                                                                        line_prediction["predicted_ldl"])

        boundary_performance = boundary_performance.join(data_end_performance, how='outer')
        evaluation_utilities.save_performance(db_cred, boundary_performance, f'"{k_folds}fold_cross_validation_pat_boundary"', fold_id)
        print(f'\n\nboundary_performance=\n\n{boundary_performance}')
        results[fold_id]["boundary_performance"] = boundary_performance

        first_table_line_performance = evaluation_utilities.predict_performance(labels, 
                                                                                line_prediction.query('predicted_table_counter<=1')["annotated_label"], 
                                                                                line_prediction.query('predicted_table_counter<=1')["predicted_label"])
                                                                                
        results[fold_id]["first_table_line"] = first_table_line_performance
        print(f'\n\nfirst_table_line_performance=\n\n{first_table_line_performance}')
        evaluation_utilities.save_performance(db_cred, first_table_line_performance, f'"{k_folds}fold_cross_validation_pat_line_first"', fold_id)

        first_table_boundary_performance = evaluation_utilities.predict_performance(["DATA_START"],
                                                                                    line_prediction.query('predicted_table_counter<=1')["annotated_fdl"], 
                                                                                    line_prediction.query('predicted_table_counter<=1')["predicted_fdl"])

        data_end_performance = evaluation_utilities.predict_performance(["DATA_END"], 
                                                                        line_prediction.query('predicted_table_counter<=1')["annotated_ldl"], 
                                                                        line_prediction.query('predicted_table_counter<=1')["predicted_ldl"])

        first_table_boundary_performance=first_table_boundary_performance.join(data_end_performance, how='outer')
        results[fold_id]["first_table_boundary_performance"] = first_table_boundary_performance
        evaluation_utilities.save_performance(db_cred, first_table_boundary_performance, f'"{k_folds}fold_cross_validation_pat_boundary_first"', fold_id)

        print(f'\n\nfirst_table_boundary_performance=\n\n{first_table_boundary_performance}')

 
        cell_performance = evaluation_utilities.predict_performance(labels, 
                                                                    cell_prediction["annotated_label"], 
                                                                    cell_prediction["predicted_label"])
        print(f'\ncell_performance=\n\n{cell_performance}')
        results[fold_id]["cell"] = cell_performance
        evaluation_utilities.save_performance(db_cred, cell_performance, f'"{k_folds}fold_cross_validation_pat_cell"', fold_id)

        table_confusion_matrices = [worker_output[3] for worker_output in processed_files]
        
        table_confusion_matrix = [[t.real_positive, t.true_positive_table, t.true_positive_data, t.predicted_positive_table, t.predicted_positive_data] for t in table_confusion_matrices]
        real_positive, true_positive_table, true_positive_data, predicted_positive_table, predicted_positive_data = map(sum, zip(*table_confusion_matrix))
        
        section_performance = pd.DataFrame(columns = ["Data", "Data_Header"]).astype(np.float)
        section_performance.loc["precision","Data"] = evaluation_utilities.recall(true_positive_data, predicted_positive_data)
        section_performance.loc["recall","Data"] = evaluation_utilities.recall(true_positive_data, real_positive)
        section_performance.loc["fmeasure","Data"] = evaluation_utilities.fmeasure(section_performance.loc["precision","Data"], section_performance.loc["recall","Data"])
        section_performance.loc["precision","Data_Header"]  = evaluation_utilities.recall(true_positive_table, predicted_positive_table)
        section_performance.loc["recall","Data_Header"] = evaluation_utilities.recall(true_positive_table, real_positive)
        section_performance.loc["fmeasure","Data_Header"] = evaluation_utilities.fmeasure(section_performance.loc["precision","Data_Header"], section_performance.loc["recall","Data_Header"])
        evaluation_utilities.save_performance(db_cred, section_performance, f'"{k_folds}fold_cross_validation_pat_relation"', fold_id)
        results[fold_id]["table"] = section_performance

        print(f'\n\nsection_performance=\n\n{section_performance}')

        file_prediction_list  = [worker_output[2] for worker_output in processed_files]
        file_accuracy = file_prediction_list.count(True)/len(file_prediction_list) 
        print(f'\n\nfile_accuracy={file_accuracy}\n')        
        results[fold_id]["file"] = file_accuracy

        file_data_jaccard = evaluation_utilities.jaccard_similarity_coefficient(['DATA', 'SUBHEADER'], 
                                                                                cell_prediction["annotated_label"], 
                                                                                cell_prediction["predicted_label"])
        print(f'file_data_cell_jaccard={file_data_jaccard}')
        results[fold_id]["file_jaccard"] = file_data_jaccard    
        end = timer()
        print(f'>>> predict_performance calculated in {timedelta(seconds=end - start)}')


        table_confidences_list = [worker_output[4] for worker_output in processed_files]
        table_confidences = pd.concat(table_confidences_list)
        table_confidences['avg_confidence']=table_confidences[['top_confidence','bottom_confidence']].mean(axis=1)
        table_confidences['fold_id'] = fold_id

        engine = create_engine(f'postgresql+psycopg2://{db_cred.user}:{db_cred.password}@localhost:{db_cred.port}/{db_cred.database}')
        table_confidences.head(0).to_sql('pat_table_confidences', engine, if_exists='append',index=False) 
        conn = engine.raw_connection()
        cur = conn.cursor()
        output = io.StringIO()
        table_confidences.to_csv(output, sep='\t', header=False, index=False)
        output.seek(0)
        contents = output.getvalue()
        cur.copy_from(output, 'pat_table_confidences', null="") # null values become ''
        conn.commit()        

    file_to_write=f"evaluate_pat_{k_folds}cross_val"
    if inject_percent_null!=None:
        file_to_write=os.path.join('add_nulls',file_to_write)
        file_to_write=file_to_write+f"-{inject_percent_null}"
    elif inject_percent_outlier!=None:
        file_to_write=os.path.join('add_noise',file_to_write)
        file_to_write=file_to_write+f"-{inject_percent_outlier}"    
    file_to_write=file_to_write+".pkl"

    with open(file_to_write, "wb")as f:
        pickle.dump(results, f)
    
    average_results = evaluation_utilities.average_performance(file_to_write)
    return average_results

def generate_noise():
    noise=''
    letter = random.choice(string.ascii_letters)
    digit = random.choice(string.digits)
    noise = letter+digit if random.choice([True,False]) else digit+letter
    return noise    
    

def inject_noise(dataframe, annotations, inject_percent):

    random.seed( len(dataframe) )
    df = dataframe.copy()
    if 'tables' in annotations:
        annotated_tables = annotations['tables']
        for table in annotated_tables:
            if 'data_start' in table.keys():
                datastart=table['data_start']
                dataend=table['data_end']
                labels = [(row, col) for row in range(datastart, dataend+1) for col in df.columns]
                for row, col in random.sample(labels, int(round(inject_percent/100*len(labels)))):
                    if df.loc[row, col] not in [None, np.nan]:
                        noise = generate_noise()
                        df.loc[row, col] = str(df.loc[row, col])+noise if random.choice([True,False]) else noise+str(df.loc[row, col])
                        

    # ix = [(row, col) for row in range(df.shape[0]) for col in range(df.shape[1])]
    # for row, col in random.sample(ix, int(round(inject_percent_null/100*len(ix)))):
    #     df.iat[row, col] = np.nan
    return df

def inject_nulls(dataframe, annotations, inject_percent_null):

    random.seed( len(dataframe) )
    df = dataframe.copy()
    if 'tables' in annotations:
        annotated_tables = annotations['tables']
        for table in annotated_tables:
            if 'data_start' in table.keys():
                datastart=table['data_start']
                dataend=table['data_end']
                labels = [(row, col) for row in range(datastart, dataend+1) for col in df.columns]
                for row, col in random.sample(labels, int(round(inject_percent_null/100*len(labels)))):
                    df.loc[row, col] = np.nan

    # ix = [(row, col) for row in range(df.shape[0]) for col in range(df.shape[1])]
    # for row, col in random.sample(ix, int(round(inject_percent_null/100*len(ix)))):
    #     df.iat[row, col] = np.nan
    return df


def generate_process_csv_tasks(db_cred,top_level_dir,
                                pat_classifier, fold_id,test_keys, max_lines, assume_multi_tables, 
                                inject_percent_null, inject_percent_outlier, approach='PAT'):
    con = connect(dbname=db_cred.database, 
                    user=db_cred.user, 
                    host = 'localhost', 
                    password=db_cred.password, 
                    port = db_cred.port
                    )

    cur = con.cursor()
    cur.execute("""SELECT crawl_datafile_key, groundtruth_key, annotations, original_path, failure
                    FROM ground_truth_2k_canada 
                    WHERE annotations is not null 
                    AND endpoint_dbname in %s
                    AND crawl_datafile_key in %s
                    ORDER BY crawl_datafile_key""",
                    (('open.canada.ca_data', 
                      'data.surrey.ca', 
                      'donnees.ville.montreal.qc.ca', 
                      'data.ottawa.ca'), 
                    tuple(test_keys)))
    test_files = cur.fetchall()
    cur.close()
    con.close()

    file_counter = 0
    for file_object in test_files:
        crawl_datafile_key = file_object[0]
        original_path = file_object[3]
        annotations = file_object[2]
        failure = file_object[4]

        if  failure is not None or annotations is None:
            print('skip')
        else:
            file_counter += 1
            task = (db_cred, file_counter, file_object, pat_classifier, fold_id, top_level_dir, max_lines, assume_multi_tables, approach, inject_percent_null, inject_percent_outlier)
            yield task

def process_csv_worker(task):
    line_predictions = None
    cell_predictions = None
    annotated_tables = dict()
    db_cred, file_counter, file_object, pat_classifier, fold_id, top_level_dir, max_lines, assume_multi_tables, approach, inject_percent_null, inject_percent_outlier = task

    crawl_datafile_key = file_object[0]
    annotations = file_object[2]
    original_path = file_object[3]
    
    filepath=os.path.join(top_level_dir, original_path)
    file_dataframe = file_utilities.get_dataframe(filepath, max_lines)
    bottom_boundary = file_dataframe.shape[0]-1 #initialize

    if 'tables' in annotations.keys():
        annotated_tables = annotations['tables']

    if assume_multi_tables==False:
        if 'tables' in annotations.keys():
            for table in annotations['tables']:
                if 'data_start' in table.keys():
                    bottom_boundary = table['bottom_boundary']
                break
        file_dataframe = file_dataframe.loc[:bottom_boundary]

    blank_lines=[]
    blank_lines=list(file_dataframe[file_dataframe.isnull().all(axis=1)].index) 
   
    file_dataframe_trimmed= file_dataframe.copy()
    # print(file_dataframe.head(15))
    if pat_classifier.parameters.max_attributes!= None:
        max_attributes=pat_classifier.parameters.max_attributes
        if pat_classifier.parameters.ignore_left!=None:
            max_attributes = pat_classifier.parameters.max_attributes+pat_classifier.parameters.ignore_left
        slice_idx = min(max_attributes,file_dataframe.shape[1])+1
        file_dataframe_trimmed = file_dataframe.iloc[:,:slice_idx] 


    if inject_percent_null!=None:    
        file_dataframe_trimmed=inject_nulls(file_dataframe_trimmed, 
                                            annotations, 
                                            inject_percent_null)
    if inject_percent_outlier!=None:
        file_dataframe_trimmed=inject_noise(file_dataframe_trimmed, 
                                            annotations, 
                                            inject_percent_outlier)    

    
    discovered_tables = pat_classifier.extract_tables(file_dataframe_trimmed, blank_lines)
    line_predictions, cell_predictions = assign_class(db_cred, file_dataframe_trimmed, discovered_tables, blank_lines, crawl_datafile_key, annotations, fold_id)


    
    table_confusion_matrix,table_confidences = evaluation_utilities.evaluate_relation_extraction(annotated_tables, discovered_tables) 
    table_confidences['crawl_datafile_key'] = crawl_datafile_key

    file_parsed_correctly=False
    if line_predictions["annotated_label"].equals(line_predictions["predicted_label"]):
        file_parsed_correctly=True

    return line_predictions, cell_predictions, file_parsed_correctly, table_confusion_matrix, table_confidences

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-d", "--database", default="ground_truth_2k_canada", help="database for experimentation with ground truth")
    parser.add_argument("-u", "--user", default="christina", help="user for the database connection")
    parser.add_argument("-p", "--port", default=5532, help="port that postgresql database listens to")

    parser.add_argument("-n", "--num_processors", default = 64, type=int, help="number of processors to be used")
    parser.add_argument("-t", "--top_level_dir", default="/home/christina/OPEN_DATA_CRAWL_2018", help="path to Open Data Crawl")
    parser.add_argument("-e", "--evaluation_method", default="cross", help="one of ['bootstrap', 'cross']")
    parser.add_argument("-k", "--k_folds", default=10, type=int, help="number of folds in cross-validation")
    parser.add_argument("-i", "--inject_percent_null", default='None')
    parser.add_argument("-o", "--inject_percent_outlier", default='None')

    args = parser.parse_args(sys.argv[1:])
    num_processors=min(args.num_processors,pat.available_cpu_count())
    top_level_dir=args.top_level_dir
    evaluation_method = args.evaluation_method
    
    inject_percent_null=eval(args.inject_percent_null)
    inject_percent_outlier = eval(args.inject_percent_outlier)
    
    if not os.path.exists('add_nulls'):
        os.makedirs('add_nulls')
    if not os.path.exists('add_noise'):
        os.makedirs('add_noise')

    print(f'inject_percent_null={inject_percent_null}')    
    print(f'inject_percent_outlier={inject_percent_outlier}')
    print(f'num_processors={num_processors}')
    


    k_folds = args.k_folds 
    db_cred = DotMap()
    db_cred.user = args.user
    db_cred.database = args.database
    db_cred.port = args.port
    db_cred.password = getpass(prompt=f'Please enter password for user {db_cred.user} on database {db_cred.database}:') 
    max_lines = 100


    pat_classifier = pat.PYTHEAS()
    average_results = eval_pat_line_kfold(pat_classifier,
                        k_folds,
                        db_cred,
                        top_level_dir,
                        inject_percent_null,
                        inject_percent_outlier,
                        num_processors, 
                        max_lines
                        )
