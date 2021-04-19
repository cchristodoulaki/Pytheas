import os
import sys
import traceback
import cchardet as chardet
import codecs
from unidecode import unidecode

import pandas as pd
import numpy as np
import csv
# from pat_utilities import null_equivalent_values as null_equivalent
# import pat_utilities as pat
# from header_events import collect_events_on_row, collect_arithmetic_events_on_row
import copy
# from nltk import word_tokenize
# from nltk.corpus import stopwords
# stop = stopwords.words('french')+stopwords.words('english')+list(string.punctuation)
import string

from operator import add
from  functools import reduce

from langdetect import detect 
from langdetect import detect_langs
from langdetect import DetectorFactory 
DetectorFactory.seed = 0
import pprint
pp = pprint.PrettyPrinter(indent=4)
import math
import itertools

def decode_value(row):
    return row.apply(lambda value: unidecode(value) if type(value) is str else value)
"""
Returns the total number of lines and
the index of rows that are totally empty or only have delimiters (no values)
"""
def file_info(csv_tuples):
    blankindex = []
    lineindex=0
    for t in csv_tuples:        
        if len(t)==0 or sum(len(s.strip()) for s in t)==0:
            blankindex.append(lineindex)
        lineindex+=1
    return [lineindex, blankindex]

def split_lines_unquoted(txt):
    #https://stackoverflow.com/questions/24018577/parsing-a-string-in-python-how-to-split-newlines-while-ignoring-newline-inside
    s = txt.split('\n')
    res = []
    cnt = 0
    for item in s:
        if res and cnt % 2 == 1:
            res[-1] = res[-1] + '\n' + item
        else:
            res.append(item)
            cnt = 0
        cnt += item.count('"')
    return res

""" 
given a list of candidate delimiters and a list of ranked delimiters, 
return the candidate delimiter with the highest ranking
"""
def breaktie(candidate_delim, ranked_delimiters):
    indexlist = []
    for candidate in candidate_delim:
        #print(ranked_delimiters.index(candidate))
        indexlist.append(ranked_delimiters.index(candidate))
    indexlist.sort()
    delimiter = ranked_delimiters[indexlist[0]]
    return delimiter

"""
Seek non-singlecount consistent consecutive token count for each 
delimiter and select the delimiter with highest token count
if there is a tie break it with a ranking of known delimiters
"""
def discover_delimiter(filepath, encoding):
    # LOG
    # input('discover_delimiter')
    delimiter = ','
    ranked_delimiters = [',',';','\t','|', ':', '#', '^']
    delim_count= {}
    samplelimit = 25

    maxInt = sys.maxsize
    decrement = True

    while decrement:
        # decrease the maxInt value by factor 10 
        # as long as the OverflowError occurs.

        decrement = False
        try:
            csv.field_size_limit(maxInt)
        except OverflowError:
            maxInt = int(maxInt/10)
            decrement = True

    delimiter_idx_of_first_consistent_row= {}
    for delim in ranked_delimiters:
        # print('\n----------------\nChecking case for delim='+delim)        
        linecount = 0
        samplelines = []

        with codecs.open(filepath,'rU', encoding=encoding) as file:
            csvreader = csv.reader(file,delimiter= delim)
            # print('\n-------------------\n')
            line = next(csvreader, None)
            # input('\nline =' +str(line))
            while line !=None:
                
                linecount=linecount+1
                if linecount==samplelimit:
                    # print('reached limit = '+str(samplelimit))
                    break
                if linecount==1:
                    # print('First row')
                    #initialize counts
                    token_count = len(line)
                    # print('token_count='+str(token_count))
                    # print('\n-------------------\n')
                    line = next(csvreader, None)
                    # input('\nline =' +str(line))                    
                    continue
                else:
                    new_token_count = len(line)

                    # print('->new_token_count= '+str(new_token_count))
                    if new_token_count==0:
                        # print('\n-------------------\n')
                        line = next(csvreader, None)
                        # input('\nline =' +str(line))
                        continue
                                          
                    elif token_count!=1 and token_count == new_token_count:
                        delimiter_idx_of_first_consistent_row[delim]= linecount-1
                        # consecutive lines with same token count

                        # look ahead if another line exists, 
                        # if next line also has same token count save the count for delimiter
                        # print('\n-------------------\n')
                        line = next(csvreader, None)
                        # input('\nline =' +str(line))

                        if line!=None and len(line)>0:
                            #checking for third consecutive line
                            linecount=linecount+1
                            new_token_count = len(line)
                            # print('token_count='+str(token_count))
                            # print('new_token_count= '+str(new_token_count))
                            if new_token_count == token_count:
                                # print('FOUND CONSISTENT TOKEN COUNT')
                                # save this token count for this delim
                                delim_count[delim] = token_count
                                break
                            else:
                                #TOKEN COUNT WAS BY COINCIDENCE
                                #stop looking
                                delimiter_idx_of_first_consistent_row.pop(delim)
                                break

                        else:
                            # on the last two rows, no point in looking ahead
                            # save this token count for this delim
                            delim_count[delim] = token_count
                            break

                    token_count = new_token_count

                    line = next(csvreader, None)
                    # input('\nline =' +str(line))
    # LOG
    # input(delim_count)
    # if there is only one demiliter with highest score use that otherwise use tie breaker
    if len(delim_count.values())>0:
        maxValue = max(delim_count.values())
        cand_delims = [key for key in delim_count.keys() if delim_count[key]==maxValue]   
        # input(cand_delims)
        if len(cand_delims) == 0:
            # print('------------------ no delimiter--------------------')
            delimiter =  ''
            # print('Detected delimiter = "'+delimiter+'"')  
            return delimiter
        elif len(cand_delims) == 1:
            delimiter =  cand_delims[0]
            # print('Detected delimiter = "'+delimiter+'"')  
            return delimiter
        if delimiter_idx_of_first_consistent_row[cand_delims[0]]<delimiter_idx_of_first_consistent_row[cand_delims[1]]:
            delimiter = cand_delims[0]
            # print('Detected delimiter = "'+delimiter+'"')  
            return delimiter
        elif delimiter_idx_of_first_consistent_row[cand_delims[0]]>delimiter_idx_of_first_consistent_row[cand_delims[1]]:
            delimiter = cand_delims[1]
            # print('Detected delimiter = "'+delimiter+'"')  
            return delimiter
        else:
            delimiter =  breaktie(cand_delims, ranked_delimiters)
            # print('Detected delimiter = "'+delimiter+'"')  
            return delimiter
    else:
        return delimiter

def discard_file(filepath, encoding):
    with codecs.open(filepath,'rU', encoding=encoding) as fp:
        firstline = fp.readline()
        while len(firstline.strip())==0:
            firstline = fp.readline()
    badphrases = ['<','<!DOCTYPE', '<HTML','<?XML','<!doctype', '<html','<?xml', '{', '\'{', '"{'] 
    for phrase in badphrases:    
        if firstline.strip().startswith(phrase):
            print('-------------NASTY--------------')
            # print(firstline)
            return True
    return False


def isfloat(value):
    try:
        # input('-')
        float(str(value).strip())
        # input('--')
        return True
    except:
        # input('----')
        return False
def dequote(s):
    """
    If a string has single or double quotes around it, remove them.
    Make sure the pair of quotes match.
    If a matching pair of quotes is not found, return the string unchanged.
    """
    if len(s)>2 and (s[0] == s[-1]) and s.startswith(("'", '"')):
        return s[1:-1]
    return s
def detect_encoding(filepath):
    result={}
    # u = UniversalDetector()
    # with open(filepath, 'rb') as rawdata:
    #     u.feed(rawdata.read(10000))
    #     rawdata.flush()
    # u.close()
    # encoding_result = u.result
    # return encoding_result 
# use cchardet because it is faster than chardet
    with open(filepath, 'rb') as rawdata:
        msg = rawdata.read(50000)
        result = chardet.detect(msg)
        rawdata.flush()

    if result==None:
        result={}
    if result['encoding']==None:
        return result

    if 'encoding' not in result.keys():
        result['encoding']='utf-8-sig'
        result['confidence']=0

# https://github.com/buriy/python-readability/issues/42
    if result['encoding'].lower()=='big5':
        result['encoding']='big5hkscs'
    elif result['encoding'].lower()=='gb2312':
        result['encoding']='gb18030'
    elif result['encoding'].lower()=='ascii':
        result['encoding']='utf-8'
    elif result['encoding'].lower()=='iso-8859-1':
        result['encoding']='cp1252'

    return result

def merged_df(failure, all_csv_tuples):
    dataframes = []
    if failure==None and all_csv_tuples!=None and  len(all_csv_tuples)>0:      
        start_index = 0
        line_index = 0
        csv_tuples=[all_csv_tuples[0]]
        num_fields = len(all_csv_tuples[0])
        dataframes = []
        empty_lines = []
        line_index+=1
        while line_index<len(all_csv_tuples):
            csv_tuple = all_csv_tuples[line_index]
            if len(csv_tuple)==0:                
                csv_tuple = ['' for i in range(0, num_fields)]
                empty_lines.append(line_index)

            if len(csv_tuple)==num_fields:
                csv_tuples.append(csv_tuple)
                end_index = line_index+1
            else:            
                end_index = line_index

            if len(csv_tuple)!=num_fields or (len(csv_tuple)==num_fields and line_index+1 == len(all_csv_tuples)):
                
                dataframe = pd.DataFrame(csv_tuples)
                dataframe.index = list(range(start_index, end_index))
                dataframe.fillna(value=np.nan, inplace=True)
                dataframe.replace(to_replace=[None], value=np.nan, inplace=True)
                dataframe = dataframe.replace(r'^\s*$' , np.nan, regex=True)

                if start_index-1 in empty_lines or len(dataframes)==0:
                    dataframes.append(dataframe)
                else:
                    dataframes[-1] = dataframes[-1].append(dataframe)

                start_index= end_index
                if start_index<len(all_csv_tuples):
                    csv_tuples=[csv_tuple] 
                    num_fields = len(csv_tuple)
   
            line_index+=1            
 
    dataframe = pd.DataFrame()
    for df in dataframes:
        dataframe = dataframe.append(df)

    dataframe.reset_index(drop=True)

    last_idx = dataframe.last_valid_index()
    dataframe = dataframe.loc[:last_idx] 
    column_labels = dataframe.columns.values.tolist()
    column_labels.reverse() 
    for column in column_labels:
        if dataframe[column].isnull().all():
            dataframe = dataframe.drop([column], axis=1)
        else:
            break
      
    dataframe.columns = list(range(0,dataframe.shape[1]))
    dataframe = dataframe.apply(decode_value, axis=1)
    return dataframe
    
def get_dataframe(filepath, max_lines):

    all_csv_tuples= None
    failure= None
    all_csv_tuples, discovered_delimiter, discovered_encoding, encoding_language, encoding_confidence, failure, blank_lines_index, google_detected_lang = sample_file(filepath, max_lines)
        
    if max_lines is None:
        num_lines = 0
        blanklines=[]
        if failure==None:
            try:
                all_csv_tuples = []
                with codecs.open(filepath,'rU', encoding=discovered_encoding) as f:        
                    chunk = f.read()
                    if chunk:
                        for line in csv.reader(chunk.split("\n"), quotechar='"', delimiter= discovered_delimiter, skipinitialspace=True):
                            num_lines+=1
                            if len(line) == 0 or sum(len(s.strip()) for s in line)==0:
                                blanklines.append(num_lines-1)
                            all_csv_tuples.append(line) 
            except Exception as e:
                print(f'file_utilities.get_dataframe:{e}, filepath:{filepath}')   

    return merged_df(failure, all_csv_tuples)


def sample_file(filepath, max_batch= 100):
    batch = None
    discovered_delimiter = None
    discovered_encoding = None
    encoding_confidence = None
    encoding_language = None
    google_detected_lang= None
    failure = None

    blanklines=[]
    try:
        if os.path.exists(filepath)==False:
            failure= "file does not exist ¯\_(ツ)_/¯"
            print('\n~~~~~~~~~~~~~~~~~~~~~')
            print(filepath)
            print(failure)
            return batch, discovered_delimiter, discovered_encoding,encoding_language,encoding_confidence, failure, blanklines, google_detected_lang

        size_bytes = os.path.getsize(filepath)

        if size_bytes == 0:
            failure= "file is empty ¯\_(ツ)_/¯"
            print('\n~~~~~~~~~~~~~~~~~~~~~')
            print(filepath)
            print(failure)
            return batch, discovered_delimiter, discovered_encoding,encoding_language,encoding_confidence, failure, blanklines, google_detected_lang

        encoding_result = detect_encoding(filepath)
        # print('Encoding:'+str(encoding_result))
        discovered_encoding = encoding_result["encoding"]
        if discovered_encoding==None:
            failure = "chardet library failed to discover encoding ¯\_(ツ)_/¯"
            print('\n~~~~~~~~~~~~~~~~~~~~~')
            print(filepath)
            print(failure)
            return batch, discovered_delimiter, discovered_encoding,encoding_language,encoding_confidence, failure, blanklines, google_detected_lang
            
        encoding_confidence = encoding_result["confidence"]
        if "language" in encoding_result.keys():
            encoding_language = encoding_result["language"]
        else:
            
            encoding_language = ''

        if discard_file(filepath, discovered_encoding)==True:
            failure = "Illegal file format ¯\_(ツ)_/¯"
            print('\n~~~~~~~~~~~~~~~~~~~~~')
            print(filepath)
            print(failure)
            return batch, discovered_delimiter, discovered_encoding,encoding_language,encoding_confidence, failure, blanklines, google_detected_lang

        # discover delimiter
        discovered_delimiter = discover_delimiter(filepath, discovered_encoding)
        singletable = True
        batch=[]         
        lineindex=-1
   
        with codecs.open(filepath,'rU', encoding=discovered_encoding) as f:      
            chunk = f.read(min(size_bytes,100000))
            if chunk:
                # google_detected_lang = detect_lang(chunk)
                google_detected_lang = detect(chunk)

                for line in csv.reader(chunk.split("\n"), quotechar='"', delimiter= discovered_delimiter, skipinitialspace=True):
                    lineindex+=1
                    if len(line) == 0 or sum(len(s.strip()) for s in line)==0:
                        blanklines.append(lineindex)
                    batch.append(line)
                    if max_batch!=None and len(batch)>=max_batch:
                        break
            f.flush()
            
    except Exception as e:
        print('\n~~~~~~~~~~~~~~~~~~~~~')
        print(filepath)
        failure = str(e)
        print(failure)

    return batch, discovered_delimiter, discovered_encoding, encoding_language, encoding_confidence, failure, blanklines, google_detected_lang

def flatten(d):
    try:
        return reduce(add, d.values())
    except:
        return []