import pandas as pd
import numpy as np
from timeit import default_timer as timer
from datetime import timedelta

from psycopg2 import connect
from psycopg2.extras import execute_values

import pytheas.file_utilities
import pickle
from dotmap import DotMap

def evaluate_relation_extraction(annotated_tables, discovered_tables):
    real_positive = len(annotated_tables)

    predicted_positive_data = 0
    predicted_positive_table = 0
    true_positive_data = 0
    true_positive_table = 0

    confusion_matrix= DotMap()
    
    table_confidences_tuples = []
    table_counter = 0
    for dt in discovered_tables.values():
        table_counter+=1
        predicted_positive_data+=1
        predicted_positive_table+=1
        body_correct = False
        table_correct = False
        top_confidence = 0
        bottom_confidence = 0
        for at in annotated_tables:
            if dt['data_start']==at['data_start'] and dt['data_end'] == at['data_end']:
                true_positive_data+=1
                body_correct = True
                if dt['header']==at['header']:
                    true_positive_table+=1
                    table_correct = True

        if 'fdl_confidence' in dt.keys():            
            top_confidence = dt['fdl_confidence']["avg_majority_confidence"]
        if 'data_end_confidence' in dt.keys():            
            bottom_confidence = dt['data_end_confidence']  

        table_confidences_tuples.append([table_counter,
                                        top_confidence,
                                        bottom_confidence,
                                        body_correct,
                                        table_correct])

    table_confidences = pd.DataFrame(table_confidences_tuples, 
                                    columns=['table_counter', 
                                            'top_confidence', 
                                            'bottom_confidence',
                                            'body_correct',
                                            'table_correct'])

    confusion_matrix.real_positive = real_positive
    confusion_matrix.predicted_positive_data = predicted_positive_data
    confusion_matrix.predicted_positive_table = predicted_positive_table
    confusion_matrix.true_positive_data = true_positive_data
    confusion_matrix.true_positive_table = true_positive_table

    return confusion_matrix, table_confidences


def avg_and_confidence(frame, label, num_folds):
    average_performance = (frame.mean() * 100).reindex(['precision', 'recall', 'fmeasure'])
    performance_interval = (frame.sem() * 100).reindex(['precision', 'recall', 'fmeasure'])
    average_performance.rename(columns=lambda c: c.title(), inplace=True)
    performance_interval.rename(columns=lambda c: c.title(), inplace=True)
    print(f'\n{label} (average over {num_folds} folds)=\n{average_performance.transpose()}')
    print(f'\nwith 95% confidence interval=\n{performance_interval.transpose()}')
    return average_performance, performance_interval


def average_performance(pkl_filepath):
    # read python dict back from the file
    with open(pkl_filepath, 'rb') as pkl_file:
        results = pickle.load(pkl_file)
    average_results = DotMap()
    num_folds = len(results)
    
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    
    print(f'\n\n\nReporting average over {num_folds} folds:')
    
    print('ALL TABLES-----------------------------------')
    
    if "line" in results[1].keys():
        avg, interval = avg_and_confidence(pd.concat([results[fold_id]["line"] for fold_id in results.keys()]).groupby(level=0), 'average_line_performance', num_folds)
        average_results.line = avg
        average_results.line_inteval = interval

    if "boundary_performance" in results[1].keys():
        avg, interval = avg_and_confidence(pd.concat([results[fold_id]["boundary_performance"] for fold_id in results.keys()]).groupby(level=0), 'average_boundary_performance', num_folds)
        average_results.boundary = avg
        average_results.boundary_inteval = interval

    if "cell" in results[1].keys():
        avg, interval = avg_and_confidence(pd.concat([results[fold_id]["cell"] for fold_id in results.keys()]).groupby(level=0), 'average_cell_performance', num_folds)
        average_results.cell = avg
        average_results.cell_inteval = interval

    print('- - - - - - - - - - - - - - - - - - - - - - -')

    print('FIRST TABLE---------------------------------')

    if "first_table_line" in results[1].keys():
        avg_and_confidence(pd.concat([results[fold_id]["first_table_line"] for fold_id in results.keys()]).groupby(level=0), 'average_first_table_line_performance', num_folds)
    
    if "first_table_boundary_performance" in results[1].keys():
        avg_and_confidence(pd.concat([results[fold_id]["first_table_boundary_performance"] for fold_id in results.keys()]).groupby(level=0), 'average_first_table_boundary_performance', num_folds)
    
    print('- - - - - - - - - - - - - - - - - - - - - - -')
    
    print('RELATION EXTRACTION------------------------------')
    
    if "table" in results[1].keys():
        avg, interval = avg_and_confidence(pd.concat([results[fold_id]["table"] for fold_id in results.keys()]).groupby(level=0), 'average_relation_performance', num_folds)
        average_results.relation = avg
        average_results.relation_inteval = interval
    
    print('- - - - - - - - - - - - - - - - - - - - - - -')
    
    print('CSV PARSING ---------------------------------')
    
    if "file" in results[1].keys():
        file_accuracies = [results[fold_id]["file"] for fold_id in results.keys()]
        avg_file_accuracy = sum(file_accuracies)/len(file_accuracies)
        print(f'\navg_file_accuracy (over {num_folds} folds)={avg_file_accuracy}\n\n\n')
        average_results.file = avg_file_accuracy
    
    if "file_jaccard" in results[1].keys():       
        jaccard_cell_accuracies = [results[fold_id]["file_jaccard"] for fold_id in results.keys()]
        avg_jaccard_cell_accuracy = sum(jaccard_cell_accuracies)/len(jaccard_cell_accuracies)
        print(f'avg jaccard data section cell accuracy (over {num_folds} folds)={avg_jaccard_cell_accuracy}')
        average_results.jaccard_cell = avg_jaccard_cell_accuracy
    
    print('- - - - - - - - - - - - - - - - - - - - - - -')
    
    return average_results


def save_performance(db_cred, performance_df, relation_name, fold_id):
    con=connect(dbname=db_cred.database, 
                user=db_cred.user, 
                host = 'localhost', 
                password=db_cred.password, 
                port = db_cred.port)
    cur=con.cursor() 
    if fold_id == 1:
        cur.execute(f"DROP TABLE  IF EXISTS {relation_name}")
        con.commit()

        cur.execute(f"""CREATE TABLE {relation_name} (
            fold_id integer, 
            measure text
            ) """)
        con.commit()
        for column in performance_df.columns:
            cur.execute(f"""ALTER TABLE {relation_name} ADD COLUMN {column} real""")
            con.commit()

    insert_tuples = []
    for measure, row in performance_df.iterrows():
        insert_tuples.append(tuple([fold_id, measure]+row.tolist()))

   
    execute_values(cur,f"""INSERT INTO  {relation_name}
                        (fold_id, measure, {','.join(performance_df.columns)}) VALUES %s""",
                        insert_tuples) 
    con.commit()
    cur.close()
    con.close()

def predict_performance(labels, y_test, y_pred):
    
    performance = pd.DataFrame(columns=labels).astype(np.float)
    for label in labels:
        df = pd.DataFrame({ 'y_test': y_test, 'y_pred': y_pred })
        
        true_positive = sum(df.isin([label]).all(axis='columns'))
        predicted_positive = sum(y_pred.isin([label]))
        real_positive = sum(y_test.isin([label]))
        performance.loc['precision', label] = precision(true_positive, predicted_positive)
        performance.loc['recall', label]= recall(true_positive, real_positive)
        performance.loc['fmeasure', label]= fmeasure(performance.loc['precision', label], performance.loc['recall', label])

    return performance


def precision(true_positive, predicted_positive):
    precision = 1
    if predicted_positive!=0:
        precision = true_positive/predicted_positive
    return precision   

def recall(true_positive, real_positive):
    recall = 1
    if real_positive!=0:
        recall = true_positive/real_positive
    return recall  


def fmeasure(precision, recall):
    fmeasure = 0
    if (precision+recall) > 0:
        fmeasure = (2*precision*recall)/(precision+recall)
    return fmeasure


def jaccard_similarity_coefficient(labels, y_test, y_pred):
    intersection = y_test.isin(labels) & y_pred.isin(labels)
    union = y_test.isin(labels) | y_pred.isin(labels)
    jaccard_similarity_coefficient = intersection.sum()/union.sum()
    return jaccard_similarity_coefficient


def assign_class(csv_file, discovered_tables, blank_lines, crawl_datafile_key, annotations, fold_id=None, method='pat'):
    line_predictions = None
    cell_predictions = None

    headers = {}
    data = {}
    data_start = {}
    data_end = {}
    subheaders = {}
    footnotes = {}
    top_boundaries = {}
    for table_counter in iter(discovered_tables):
        table = discovered_tables[table_counter] 
        if table!=None:
            top_boundaries[table['top_boundary']] = table_counter
            headers[table_counter] = table['header']
            data_section = list(range(table['data_start'], table['data_end']+1))
            if 'subheader_scope' not in table.keys():
                table['subheader_scope'] = dict()
            subheader_scope = table['subheader_scope']
            data[table_counter] = sorted(list(set(data_section)-set(blank_lines)-set(subheader_scope.keys())))
            data_start[table_counter] = table['data_start']
            data_end[table_counter] = table['data_end']
            subheaders[table_counter] = list(subheader_scope.keys())
            if 'footnotes' not in table.keys():
                table['footnotes'] = list()
            footnotes[table_counter] = table['footnotes']

    gt_header_lines = {}
    gt_data = {}
    gt_data_start = {}
    gt_data_end = {}
    top_boundaries= {}
    gt_subheaders= {}
    gt_footnotes= {}

    if 'tables' in annotations.keys():
        for table in annotations['tables']:            
            # pp.pprint(table)
            gt_table_counter = table['table_counter']
            gt_top_boundary = table['top_boundary']
            # gt_bottom_boundary = table['bottom_boundary']
            top_boundaries[gt_top_boundary] = gt_table_counter
            gt_header_lines[gt_table_counter] = table['header']
            gt_data_start[gt_table_counter] = table['data_start']
            gt_data_end[gt_table_counter] = table['data_end'] 
            gt_data[gt_table_counter] = sorted(list(set(list(range(table['data_start'], table['data_end']+1)) )-set(table['subheaders'])))
            gt_subheaders[gt_table_counter] = table['subheaders']
            gt_footnotes[gt_table_counter] = table['footnotes']   

    labels = {}
    line_classification_DATA= []
    cell_classification_DATA = []
    gt_table_counter = 0
    table_counter=0

    for index, _ in csv_file.iterrows():
        fdl = 'OTHER'
        ldl = 'OTHER'
        gt_fdl = 'OTHER'
        gt_ldl = 'OTHER'
        labels[index] = dict()
        labels[index]['crawl_datafile_key']=crawl_datafile_key
        labels[index]['line_index']=index

        if index in top_boundaries.keys():
            gt_table_counter = top_boundaries[index]

        if 'tables' in annotations.keys() and len(annotations['tables'])==0:
            labels[index]['gt'] = 'OTHER'
        else:
            labels[index]['gt'] = 'CONTEXT'

        if index in blank_lines:
            labels[index]['gt'] = 'BLANK'
        elif index in file_utilities.flatten(gt_header_lines):
            labels[index]['gt'] = 'HEADER'
        elif index in file_utilities.flatten(gt_footnotes):
            labels[index]['gt'] = 'FOOTNOTE'
        elif index in file_utilities.flatten(gt_subheaders):
            labels[index]['gt'] = 'SUBHEADER'
        elif index in file_utilities.flatten(gt_data):
            labels[index]['gt'] = 'DATA'

        if index in gt_data_start.values():
            gt_fdl = 'DATA_START'
        if index in gt_data_end.values():   
            gt_ldl = 'DATA_END'        


        if index in top_boundaries.keys():
            table_counter = top_boundaries[index]

        if len(discovered_tables)==0:
            labels[index][method] = 'OTHER'
        else:
            labels[index][method] = 'CONTEXT'
        if index in blank_lines :
            labels[index][method] = 'BLANK'
        elif index in file_utilities.flatten(headers):
            labels[index][method] = 'HEADER'
        elif index in file_utilities.flatten(footnotes):
            labels[index][method] = 'FOOTNOTE'
        elif index in file_utilities.flatten(subheaders):
            labels[index][method] = 'SUBHEADER'
        elif index in file_utilities.flatten(data):
            labels[index][method] = 'DATA'

        if index in data_start.values():   
            fdl = 'DATA_START'
        if index in data_end.values():   
            ldl = 'DATA_END'

        line_classification_DATA.append(
            (   
                fold_id,
                crawl_datafile_key,index,
                gt_table_counter, table_counter,
                labels[index]['gt'], labels[index][method], 
                gt_fdl, fdl,
                gt_ldl, ldl
            )
        ) 
        for column_index in csv_file.columns:
            gt_cell_label  = labels[index]['gt']
            cell_label = labels[index][method]
            if str(csv_file.loc[index,column_index]).lower() in ['' or 'nan']:
                cell_label = 'BLANK'
                gt_cell_label = 'BLANK'
 
            cell_classification_DATA.append(
                (
                    fold_id, crawl_datafile_key,
                    index, column_index,
                    gt_cell_label, cell_label
                )
            ) 

    predicted_label = "predicted_label"
    table_counter_label = "predicted_table_counter"
    fdl_label = "predicted_fdl"
    ldl_label = "predicted_ldl"

    line_predictions = pd.DataFrame(
        line_classification_DATA, 
        columns = ["fold_id", "crawl_datafile_key", "line_index", 
                    "annotated_table_counter", table_counter_label,
                    "annotated_label", predicted_label, 
                    "annotated_fdl", fdl_label, 
                    "annotated_ldl", ldl_label])

    cell_predictions = pd.DataFrame(
        cell_classification_DATA, 
        columns = ["fold_id", "crawl_datafile_key", "line_index", 
                    "column_index", 
                    "annotated_label", predicted_label]
                    )
    return line_predictions, cell_predictions         
