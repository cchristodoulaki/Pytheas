from psycopg2 import connect
from psycopg2.extras import Json
import argparse, sys,os
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from pathlib import Path
import codecs
import csv
import pandas as pd
import numpy as np
import string
import ast
sys.path.append('../pytheas')
import file_utilities
import pprint
import json
pp = pprint.PrettyPrinter(indent=4)
pd.options.display.max_rows = 100

def annotateCSV(filepath, max_lines):
    Json_file_annotations = {}
    all_csv_tuples= None
    failure= None
    all_csv_tuples, discovered_delimiter, discovered_encoding, encoding_language, encoding_confidence, failure, blanklines, google_detected_lang = file_utilities.sample_file(filepath,max_lines)
    
    print(f'discovered_delimiter={discovered_delimiter}')
    print(f'discovered_encoding={discovered_encoding}')
    print(f'failure={failure}')
    print(f'blanklines={blanklines}\n')

    if failure==None:
        dataframe = file_utilities.get_dataframe(filepath, max_lines)
        if len(dataframe.index)==0:
            print(dataframe)
            print('dataframe is empty.')
            return Json_file_annotations

        offset = dataframe.index[0]
        limit = 50
        while offset<dataframe.shape[0]:
            input(dataframe.loc[offset:min(dataframe.shape[0]+1, offset+limit)])
            offset+= limit  


        file_annotations = {}
        file_annotations["blanklines"] = blanklines
        file_annotations["tables"] = []
        context_end = -1
        table_counter = 0
        has_tables = input('\n\nDoes file contain any tables? [y/n]:')

        if has_tables == 'n':
            file_annotations["contains_data_tables"] = False
        else:

            multiple_tables = input('\n\nDoes file contain multiple tables? [y/n]:')
            while int(context_end)+1<=dataframe.index[-1]:    
                table_counter+=1
                table_context = {}
                table_context["table_counter"] = table_counter
                
                if multiple_tables=='n':
                    context_start = 0
                    context_end = dataframe.index[-1]
                else:
                    print(dataframe.loc[int(context_end)+1:])
                    context_start = input('\n\nIndicate TOP_BOUNDARY of table context starting at row index '+str(int(context_end)+1)+': \n')
                    context_end = input('\nIndicate BOTTOM_BOUNDARY of table context after row index '+str(context_start)+': \n')
                    print('Table context is indicated between row index '+str(context_start)+ ' and row index '+str(context_end)+'. \n')
                    print(dataframe.loc[int(context_start):int(context_end)])

                table_context["top_boundary"]= int(context_start)
                table_context["bottom_boundary"]= int(context_end)

                data_start = input('\n\nIndicate index of START of TABLE BODY section within rows '+ str(context_start) +' and '+ str(context_end) +':\n')
                data_end = input('\nIndicate index of END of TABLE BODY section within rows '+ str(data_start) +' and '+ str(context_end) +':\n')
                table_context["data_start"]= int(data_start)
                data_end = min(int(data_end), int(context_end))
                table_context["data_end"]= int(data_end)            

                header_indexes = []
                pre_context = []
                title = []
                headnotes = []

                if int(data_start)>int(context_start):
                    headnotes = list(range(int(context_start), int(data_start)))
                    print(dataframe.loc[int(context_start):int(data_start)])
                    user_input = input('\n\nIndicate indexes of HEADER for table before row index '+str(data_start)+': \n')
                    if len(user_input)>0:
                        header_indexes = user_input.split(' ')
                        header_indexes = [int(i) for i in header_indexes]
                        print(header_indexes)
            
                # if len(header_indexes)< int(data_start)- int(context_start):
                #     user_input = input('\nIndicate table TITLE row indexes between '+str(context_start)+ ' and '+str(data_start)+': \n')
                #     if len(user_input)>0:
                #         title =   user_input.split(' ')
                #         title =   [int(i) for i in title]               
                
                # table_context["title"] = title
            
                table_context["headnotes"] = headnotes
                table_context["header"] = header_indexes
                subheader_indexes = []
                user_input = input('\nDoes data contain subheaders? [y/n]\n')
                if user_input=='y':
                    user_input = input('\nIndicate indexes of SUB_HEADERS for table between row index '+str(data_start)+ ' and row index '+str(data_end)+': \n')
                    if len(user_input)>0:
                        subheader_indexes = user_input.split(' ')
                        subheader_indexes = [int(i) for i in subheader_indexes]
                        print(subheader_indexes)
                table_context["subheaders"] = subheader_indexes
                
                data_indexes = sorted(list(set(range(int(data_start),int(data_end)+1)) - set(subheader_indexes) - set(blanklines)))
                print(data_indexes)
                table_context["data_indexes"] = data_indexes
                user_input = input('\nIs '+str(data_indexes[0])+ ' the first row of data? [y/n]\n')
                if user_input == 'y':
                    table_context["first_data_line"] = data_indexes[0]
                else:
                    input('Please start again..')
                table_context["not_data"] = sorted(list(set(range(int(context_start), int(context_end)+1)) - set(data_indexes)))
                footnotes = []
                if int(data_end)<int(context_end):
                    footnotes = sorted(list(set(range(int(data_end)+1,int(context_end)+1)) - set(blanklines)))
                table_context["footnotes"] = footnotes

                file_annotations["tables"].append(table_context)
                
                print('\nREVIEW TABLE '+str(table_counter)+' ANNOTATIONS:')
                for key in table_context:
                    print(f'{key}:{table_context[key]}')
                print('~~~  ~~~  ~~~  ~~~  ~~~  ~~~  ~~~  ~~~  ~~~  \n')    
        pp.pprint(file_annotations)


    return file_annotations

def annotateCSVfolder(csv_dir, annotations_dir, max_lines):
    for fileindex in sorted([int(f.strip('.csv')) for f in os.listdir(csv_dir) if os.path.isfile(os.path.join(csv_dir, f))]):
        filename = f'{fileindex}.csv'
        print(f'annotations_dir={annotations_dir}')
        print(f'filename={filename}')

        filepath = Path(os.path.join(csv_dir,filename))
        annotation_path = Path(os.path.join(annotations_dir,filename)).with_suffix('.json')
        print(f'annotation_path={annotation_path}')
        if not os.path.exists(annotation_path):
            file_annotations = annotateCSV(filepath, 
                    max_lines)
            if annotation_path is not None:
                with open(annotation_path, 'w') as outfile:
                    json.dump(file_annotations, outfile)
    return True


if __name__ == "__main__":
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
 
    # there are two context, each with its own arguments: "file" and "folder"
    subparsers = parser.add_subparsers(title='context', dest='context')
    subparsers.required = True # python bug workaround

    # figures sub-parser
    file_parser = subparsers.add_parser('file', help='annotate CSV file')
    file_parser.add_argument('-f', '--filepath',
                               help='path to CSV file for user annotation')
    file_parser.add_argument("-o", "--output_file", 
                            default = None)

    file_parser.add_argument("-m", "--max_lines", 
                            type = int, default=None, 
                            help="maximum lines to display to user for annotation, default is None which will display entire file")

    # folder sub-parser
    folder_parser = subparsers.add_parser('folder', help='annotate CSV files in folder')
    folder_parser.add_argument('-c', '--csv_directory' ,
                             help='path to folder with CSV files for user annotation')
    folder_parser.add_argument('-a','--annotations_directory',
                             help='path to folder to write user annotations of CSV files')
    folder_parser.add_argument("-m", "--max_lines", 
                            type = int, default=None, 
                            help="maximum lines to display to user for annotation, default is None which will display entire file")


    args = parser.parse_args()    

    # This is how you can tell which context was used
    if args.context == 'file':
        file_annotations = annotateCSV(args.filepath, 
                    args.max_lines)
        if args.output_file is not None:
            with open(args.output_file, 'w') as outfile:
                json.dump(file_annotations, outfile)
    elif args.context == 'folder':
        annotateCSVfolder(args.csv_directory, 
                            args.annotations_directory, 
                            args.max_lines)