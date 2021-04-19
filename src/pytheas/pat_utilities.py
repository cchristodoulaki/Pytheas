import pandas as pd
import numpy as np
import csv
import os
import sys
sys.path.append('../')
from collections import Counter
import random
import string
import codecs
import json
import copy
import cchardet as chardet
import math
from pytheas.parsemathexpr import evaluate
import operator
from statistics import mean

# LOG
import pprint
pp = pprint.PrettyPrinter(indent=4)


strictly_null_equivalent= ['nil','data not available','not available','no data','no answer','nd','na','n/d','n/a','#n/a','n.a.','not applicable','sans objet','s.o.','so','s/o','-','--','.','..','...','*','void','0','redacted', 'confidential', 'confidentiel','unknown','inconnu','?']
null_equivalent_values = ['','null','nan','none']+strictly_null_equivalent

footnote_keywords = ['*','remarque', 'source', 'note', 'nota','not a', 'footnote']
aggregation_tokens = ['total', 'subtotal','totaux', 'totales', 'all','toute','tous','less than','moins de','moins que','more than', 'plus','plus de', 'plusieurs de', 'plus que', 'over','under','higher than','plus haut', 'lower than','plus bas', 'older than','plus vieux', 'younger than', 'plus jeune']
datatype_keywords = ['integer', 'string', 'text', 'numeric', 'boolean']
metadata_table_header_keywords = ['field name','field','description', 'example', 'data type']
# order is important!!!
#  cagr: https://www.investopedia.com/ask/answers/071014/what-formula-calculating-compound-annual-growth-rate-cagr-excel.asp
aggregation_functions = [('totaux','sum'), ('totales','sum'),('totale','sum'),('subtotal','sum'), ('total partiel','sum') ,('total','sum'), ('net change in', 'sum'), ('change in', 'sum')
                        ,('average','mean'), ('avg','mean')
                        ,('variation', 'difference'),('difference', 'difference')
                        ,('cagr', 'CAGR') ,('tcac', 'CAGR')
                        ,('var %',''), ('variance','')]



cell_rules = {
    "not_data":{
        "First_FW_Symbol_disagrees":{},
        "First_BW_Symbol_disagrees":{},
        "SymbolChain":{},
        "CC":{},

        "CONSISTENT_NUMERIC": {
            "name":"Below but not here: consistently ONE symbol = D"
            },
        "CONSISTENT_D_STAR": {
            "name":"Below but not here: consistently TWO symbols, the first is a digit"
            },
        "FW_SUMMARY_D": {
            "name":"Below but not here: two or above symbols in the FW summary, the first is a digit"
            },
        "BW_SUMMARY_D": {
            "name":"Below but not here: two or above symbols in the BW summary, the first is a digit"
            },
        "BROAD_NUMERIC": {
            "name":"Below but not here: all values digits, optionally have . or ,  or S"
            },    
        "FW_THREE_OR_MORE_NO_SPACE": {
            "name":"Below but not here: three or above symbols in FW summary that do not contain a  Space"
            },
        "BW_THREE_OR_MORE_NO_SPACE": {
            "name":"Below but not here: three or above symbols in BW summary that do not contain a  Space"
            },
        "CONSISTENT_SS_NO_SPACE": {
            "name":"Below but not here: consistently at least two symbols in the symbol set summary, none of which are S or _"
            },
        "FW_TWO_OR_MORE_OR_MORE_NO_SPACE": {
            "name":"Below but not here: two or above symbols in FW summary that do not contain a Space"
            },
        "BW_TWO_OR_MORE_OR_MORE_NO_SPACE": {
            "name":"Below but not here: two or above symbols in BW summary that do not contain a Space"
            },
        "CHAR_COUNT_UNDER_POINT1_MIN":{},
        "CHAR_COUNT_UNDER_POINT3_MIN":{},
        "CHAR_COUNT_UNDER_POINT5_MIN":{},

        "CHAR_COUNT_OVER_POINT5_MAX":{},
        "CHAR_COUNT_OVER_POINT6_MAX":{},
        "CHAR_COUNT_OVER_POINT7_MAX":{},
        "CHAR_COUNT_OVER_POINT8_MAX":{},
        "CHAR_COUNT_OVER_POINT9_MAX":{},
"NON_NUMERIC_CHAR_COUNT_DIFFERS_FROM_CONSISTENT":{}
        # "CHAR_COUNT_DIFFERS_FROM_CONSISTENT":{}
    },
    "data":{
            "VALUE_REPEATS_ONCE_BELOW": {
                "name":"Rule_1_a value repetition only once in values below me, skip the adjacent value"},
            "VALUE_REPEATS_TWICE_OR_MORE_BELOW": {
                "name":"Rule_1_b value repetition twice or more below me"},
            "ONE_ALPHA_TOKEN_REPEATS_ONCE_BELOW": {
                "name":"Rule_2_a: Only one alphabetic token from multiword value repeats below, and it repeats only once"},
            "ALPHA_TOKEN_REPEATS_TWICE_OR_MORE": {
                "name":"Rule_2_b: At least one alphabetic token from multiword value repeats below at least twice"},
            "CONSISTENT_NUMERIC_WIDTH": {
                "name":"Rule_3 consistently numbers with consistent digit count for all."},
            "CONSISTENT_NUMERIC": {
                "name":"Rule_4_a consistently ONE symbol = D"},
            "CONSISTENT_D_STAR": {
                "name":"Rule_4_b consistently TWO symbols, the first is a digit"},
            "FW_SUMMARY_D": {
                "name":"Rule_4_fw two or above symbols in the FW summary, the first is a digit"},
            "BW_SUMMARY_D": {
                "name":"Rule_4_bw two or above symbols in the BW summary, the first is a digit"},
            "BROAD_NUMERIC": {
                "name":"Rule_5 all values digits, optionally have . or ,  or S"},    
            "FW_THREE_OR_MORE_NO_SPACE": {
                "name":"Rule_6 three or above symbols in FW summary that do not contain a  Space"},
            "BW_THREE_OR_MORE_NO_SPACE": {
                "name":"Rule_7 three or above symbols in BW summary that do not contain a  Space"},
            "CONSISTENT_SS_NO_SPACE": {
                "name":"Rule_8 consistently at least two symbols in the symbol set summary, none of which are S or _"},
            "CONSISTENT_SC_TWO_OR_MORE": {
                "name":"Rule_10 two or more symbols consistent chain"},
            "FW_TWO_OR_MORE_OR_MORE_NO_SPACE": {
                "name":"Rule_11_fw two or above symbols in FW summary that do not contain a Space"},
            "BW_TWO_OR_MORE_OR_MORE_NO_SPACE": {
                "name":"Rule_11_bw two or above symbols in BW summary that do not contain a Space"},
            "FW_TWO_OR_MORE_OR_MORE_NO_SPACE_FIRST_TWO": {
                "name":"Rule_12_fw two or above symbols in FW summary, the first two do not contain a Space"},
            "BW_TWO_OR_MORE_OR_MORE_NO_SPACE_FIRST_TWO": {
                "name":"Rule_12_bw two or above symbols in BW summary, the first two do not contain a Space"},
            "FW_D5PLUS": {
                "name":"Rule_13_fw FW summary is [['D',count]], where count>=5"},
            "BW_D5PLUS": {
                "name":"Rule_13_bw BW summary is [['D',count]], where count>=5"},
            "FW_D1": {
                "name":"Rule_14_fw FW summary is [['D',1]]"},
            "BW_D1": {
                "name":"Rule_14_bw BW summary is [['D',1]]"},
            "FW_D4": {
                "name":"Rule_15_fw FW summary is [['D',4]]"},
            "BW_D4": {
                "name":"Rule_15_bw BW summary is [['D',4]]"},
            "FW_LENGTH_4PLUS": {
                "name":"Rule_17_fw four or more symbols in the FW summary"},
            "BW_LENGTH_4PLUS": {
                "name":"Rule_17_bw four or more symbols in the BW summary"},
            "CASE_SUMMARY_CAPS":{
                "name":"Rule_18 case summary is ALL_CAPS"},
            "CONSISTENT_CHAR_LENGTH":{
                "name":"this value and neighboring values are constant char length"
            },
            "CONSISTENT_SINGLE_WORD_CONSISTENT_CASE":{}

    }
}

line_rules = {
    "not_data":{
        "header":{
            #  "NON_ADJACENT_ARITHMETIC_SEQUENCE_3":{
            #      "name":""},
            #  "NON_ADJACENT_ARITHMETIC_SEQUENCE_4":{
            #      "name":""},
            #  "NON_ADJACENT_ARITHMETIC_SEQUENCE_5":{
            #      "name":""},
            #  "NON_ADJACENT_ARITHMETIC_SEQUENCE_6_plus":{
            #      "name":""},
             "ADJACENT_ARITHMETIC_SEQUENCE_2":{
                "name":""},
             "ADJACENT_ARITHMETIC_SEQUENCE_3":{
                 "name":""},
             "ADJACENT_ARITHMETIC_SEQUENCE_4":{
                 "name":""},
             "ADJACENT_ARITHMETIC_SEQUENCE_5":{
                 "name":""},
             "ADJACENT_ARITHMETIC_SEQUENCE_6_plus":{
                 "name":""},
             "RANGE_PAIRS_1":{
                 "name":""},
             "RANGE_PAIRS_2_plus":{
                 "name":""},
             "PARTIALLY_REPEATING_VALUES_length_2_plus":{
                 "name":""},
             "METADATA_LIKE_ROW":{
                 "name":""},
             "CONSISTENTLY_SLUG_OR_SNAKE":{
                 "name":""},
            "CONSISTENTLY_UPPER_CASE":{}
        },
        "aggregation":{
             "AGGREGATION_ON_ROW_WO_NUMERIC":{
                 "name":""},
             "AGGREGATION_ON_ROW_W_ARITH_SEQUENCE":{
                 "name":""}
            # ,"MULTIPLE_AGGREGATION_VALUES_ON_ROW":{
            #      "name":""} 
        },
        "other":{
            "UP_TO_FIRST_COLUMN_COMPLETE_CONSISTENTLY":{
                "name":"all lines consistently non-data from beginning of input, left-most column potentially non-null"},
            "STARTS_WITH_NULL":{
                "name":"line starts with null"},
            "NO_SUMMARY_BELOW":{
                "name":"no summary achieved below in any column"},
            "FOOTNOTE":{
                "name":"line resembles footnote"}
        }
    },
    "data":{
        "aggregation":{            
            "AGGREGATION_TOKEN_IN_FIRST_VALUE_OF_ROW":{
                "name":{"First value of a row (first column) contains an aggregation token, this is likely a summarizing data line"}
            }
        },
        "other":{
            "NULL_EQUIVALENT_ON_LINE_2_PLUS":{
                "name":"Two or more null equivalent values found on a line"
            },
            "ONE_NULL_EQUIVALENT_ON_LINE":{
                "name":"One null equivalent value found on line"
            }
        }
    }
}

class ColumnSampleSummary:
    def __init__(self,fw, bw, sc, ss, cc, nl,maxchar, minchar, maxtoken, mintoken,avgchar,avgtoken):
         self.fw_pattern_summary = fw
         self.bw_pattern_summary = bw
         self.strict_chain_summary = sc
         self.symbol_set_summary = ss
         self.consistent_case_summary = cc
         self.summary_strength = nl
         self.maxchar = maxchar
         self.minchar= minchar
         self.maxtoken = maxtoken
         self.mintoken = mintoken
         self.avgchar = avgchar
         self.avgtoken = avgtoken

    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)
        # return str(self.__dict__)

    def print_summary(self):
        pp.pprint(self.__dict__)

 


def normalize_decimals_numbers_predata(value, cand_pattern, cand_symbols, cand_case,cand_num_tokens, cand_num_chars, summary_patterns, summary_symbols, outlier_sensitive):

    new_summary_patterns = []
    new_summary_symbols = []
    new_value = value
    if 'D' in cand_symbols and cand_symbols.issubset(set(['D','.',',','S','-','+','~','(',')'])):
        if new_value.startswith('(') and new_value.endswith(')'):
            new_value= new_value.strip('(')
            new_value= new_value.strip(')')

        new_value = new_value.replace('.', '')
        new_value = new_value.replace(',', '')
        new_value = new_value.replace(' ', '')
        if new_value.startswith('~'):
            new_value= new_value.strip('~')
        if new_value.startswith('-'):
            new_value= new_value.strip('-')
        if new_value.startswith('+') or new_value.endswith('+'):
            new_value= new_value.strip('+')
        if new_value.startswith('>'):
            new_value= new_value.strip('>')
        if new_value.startswith('<'):
            new_value= new_value.strip('<')
        # print(value)
        # input('new_value='+str(new_value))            
        cand_pattern, cand_symbols, cand_case, cand_num_tokens, cand_num_chars = generate_pattern_symbols_and_case(new_value, outlier_sensitive)

    for symbols in summary_symbols:
        if (symbols==set() or ('D' in symbols and symbols.issubset(set(['D','.',',','S','-','+','~','>','<','(',')']))) )== False:
            return cand_pattern,cand_symbols,cand_case, cand_num_tokens, cand_num_chars, summary_patterns, summary_symbols
    
    for pattern in summary_patterns:
        symbolchain = [symbol_count[0] for symbol_count in pattern]
        indices = [i for i, x in enumerate(symbolchain) if x in ['-','+','~','>','<']]
        if len(indices)>1 or len(indices)==1 and indices[0]>0:
            new_summary_patterns.append(pattern)
            new_summary_symbols.append(set(symbolchain))
            continue

        if len(pattern)>0 :            
            digits = [symbol_count[1] for symbol_count in pattern if symbol_count[0] == 'D']
            digit_count = sum(digits)
            new_summary_patterns.append([['D',digit_count]])
            new_summary_symbols.append(set(['D']))
        else:
            new_summary_patterns.append(pattern)
            new_summary_symbols.append(set())

    # print('NORMALIZED:')
    # print('cand_pattern='+str(cand_pattern))
    # print('cand_symbols='+str(cand_symbols))
    # print('new_summary_patterns='+str(new_summary_patterns))
    # input('new_summary_symbols='+str(new_summary_symbols))
    return cand_pattern,cand_symbols,cand_case, cand_num_tokens, cand_num_chars, new_summary_patterns, new_summary_symbols

def normalize_decimals_numbers(summary_patterns, summary_symbols):
    new_summary_patterns = []
    new_summary_symbols = []
    
    for symbols in summary_symbols:
        if (symbols==set() or ('D' in symbols and symbols.issubset(set(['D','.',',','S','-','+','~','>','<','(',')']))) ) == False:
            return summary_patterns, summary_symbols
    
    for pattern in summary_patterns:
        symbolchain = [symbol_count[0] for symbol_count in pattern]
        indices = [i for i, x in enumerate(symbolchain) if x in ['-','+','~','>','<']]
        if len(indices)>1 or len(indices)==1 and indices[0]>0:
            new_summary_patterns.append(pattern)
            new_summary_symbols.append(set(symbolchain))
            continue        
        if len(pattern)>0:
            digits = [symbol_count[1] for symbol_count in pattern if symbol_count[0] == 'D']
            digit_count = sum(digits)
            new_summary_patterns.append([['D',digit_count]])
            new_summary_symbols.append(set(['D']))
        else:
            new_summary_patterns.append(pattern)
            new_summary_symbols.append(set())  

    # print('NORMALIZED:')
    # print('new_summary_patterns='+str(new_summary_patterns))
    # input('new_summary_symbols='+str(new_summary_symbols))
    return new_summary_patterns,new_summary_symbols 


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

# Discover tables in a dataframe
def discover_tables(df):
  
    nul_rows = list(df[df.isnull().all(axis=1)].index)

    list_of_dataframes = []
    for i in range(len(nul_rows) - 1):
        list_of_dataframes.append(df.iloc[nul_rows[i]+1:nul_rows[i+1],:])

    num_relations = list_of_dataframes

    # Remove null columns
    cleaned_tables = []
    for _df in list_of_dataframes:
        cleaned_tables.append(_df.dropna(axis=1, how='all'))

    tindex=0
    for t in cleaned_tables:
        tindex= tindex+1
        #print('\n------------------------ Relation_'+ str(tindex)+' --------------------------------\n')
        #print(t)
    #print('\n-----end discover_tables-------')
    return cleaned_tables

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
    

def split_metadata_data(csv_tuples, offset, sample_lines_limit, delimiter):
    print('\n---split_metadata_data\n')
    print('offset= '+str(offset))
    print('len(csv_tuples)='+str(len(csv_tuples)))
    print('sample_lines_limit='+str(sample_lines_limit))
    result = {"metadatastart":offset}
    metadatabuffer = []
    if offset+sample_lines_limit>=len(csv_tuples):
        limit = len(csv_tuples)#-1
    else:
        limit = offset+sample_lines_limit#-1
    # print('limit='+str(limit))
    samplelines = []
        
    line = csv_tuples[offset]
    # input('\noffset_'+str(offset)+'='+str(line))
    offset = offset+1
    #skip over any initial blank lines
    while((len(line)==0 or sum(len(s.strip()) for s in line)==0) and offset < limit):
        #line was blank
        line = csv_tuples[offset]
        # input('\noffset_'+str(offset)+'='+str(line)) 
        offset = offset+1

    #line is no longer blank, count tokens
    tokencount = len(line)
    # print('tokencount='+str(tokencount))
    while offset < limit:
        #get next line
        line = csv_tuples[offset]
        # input('\n->offset_'+str(offset)+'='+str(line))    
        offset = offset+1 
        #skip blank lines and return token count to 0       
        if line in ['\n', '\r\n'] or len(line)==0 or sum(len(s.strip()) for s in line)==0:
            while line in ['\n', '\r\n'] or (len(line)==0 or sum(len(s.strip()) for s in line)==0) and offset < limit:
                # line was blank, could be part of the metadata or a separator row
                # reinitialize token count and move on
                tokencount= 0

                line = csv_tuples[offset]
                # input('\noffset_'+str(offset)+'='+str(line))
                offset = offset+1        
        #count tokens in new line and compare    
        tokencountnew = len(line)
        # print('old tokencount= '+str(tokencount)+', new tokencount= '+str(tokencountnew))

        #token counts are consistent
        if tokencount!=0 and tokencount==tokencountnew:
            #tokencounts found = 1 twice, keep looking
            if tokencount==1:
                repeat=1;
                while repeat<5 and offset<limit and tokencount==tokencountnew:
                    # this is a candidate metadata case, keep looking for
                    #  more than 5 continuous rows total to label as data
                    repeat = repeat+1
                    # print('repeat ='+str(repeat))
                    tokencount = tokencountnew
                    # print('tokencount = tokencountnew='+str(tokencount))

                    line = csv_tuples[offset]
                    # input('\noffset_'+str(offset)+'='+str(line))
                    offset = offset+1
                    
                    tokencountnew = len(line)
                    # input('old tokencount= '+str(tokencount)+', new tokencount= '+str(tokencountnew))
                    
                    if tokencountnew!=1:
                        tokencount = tokencountnew                    
                        break
                continue
            #token count!0,1 twice, we found the data start!            
            else:# YAY consider solved
                # found consistent tokencount, consider as data 
                # from offset = '+str(offset-2)+' until EOF or next blank row
                result["metadataend"] = offset-3
                result["datastart"] = offset-2
                # print('result["datastart"]='+str(result["datastart"]))
                result["tokencount"] = tokencount
                # input(result)
                return result
        else:
            tokencount = tokencountnew
            metadatabuffer.append(line)
            
        samplelines.append(line)
    if "metadataend" not in result.keys():
        result["metadataend"] = offset-1

    # input(result)
    return result

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

"""
If row has empty values or duplicate values it cannot be a header
"""
def assess_candidate_header(datarow):
   
    datarow = list(datarow)
    #normalize
    datarow = [str(x).strip(' ').lower() for x in datarow]
    # print(datarow)
    # skip up to two first values if they are empty, 
    # assert the rest  are non-empty and unique
    if len(datarow)>2 and str(datarow[0]).strip().lower() in ['', ' ', 'none', 'nan'] and str(datarow[1]).strip().lower() in ['', ' ', 'none', 'nan']:
        candidate = datarow[2::]
    elif len(datarow)>1 and str(datarow[0]).strip().lower() in ['', ' ', 'none', 'nan']:
        candidate = datarow[1::]
    else:
        candidate = [str(elem).strip().lower() for elem in datarow]

    if '' in candidate or ' ' in candidate or 'nan' in candidate or 'none' in candidate:
        # print('empty space found')
        return False
    # print(len(set(candidate)))
    # print(len(candidate)) 
    if len(set(candidate))!=len(candidate):
        # print('duplicates found')
        return False
    #candidate is legal
    return True



def generate_pattern_summary(attribute_patterns):

    patterns = [p for p in attribute_patterns if len(p)>0].copy()
    # initialize the attribute pattern with the first value pattern
    if len(patterns)>0:
        summary_pattern = patterns[0].copy()
        
        consistent_symbol_chain = True
        for pattern in patterns:
            
            # print('--summary_pattern= '+str(summary_pattern))
            # input('--next pattern= '+str(pattern))

            if len(summary_pattern)==0:
                break
            if len(pattern)==0:
                continue
            for symbol_idx, symbol in enumerate(pattern):
                # make sure index exists in summary pattern
                if len(summary_pattern)>symbol_idx: #make sure index in bounds
                    # check if symbols agree
                    if symbol[0] == summary_pattern[symbol_idx][0]:
                        #check if counts disagree (if summary symbol has count)
                        if summary_pattern[symbol_idx][1]!=0 and symbol[1] != summary_pattern[symbol_idx][1]:
                            summary_pattern[symbol_idx][1] = 0
                        #else they agree so do nothing

                        # check if I am on last symbol of pattern and summary is longer, stop looking at this pattern and cut the summary here
                        if symbol_idx == len(pattern)-1 and len(pattern)<len(summary_pattern):
                            summary_pattern =  summary_pattern[0:symbol_idx+1]
                            consistent_symbol_chain = False
                            break                       
                    else: #symbols disagreed, remove everything from here on and go to next pattern
                        summary_pattern =  summary_pattern[0:symbol_idx]

                        consistent_symbol_chain = False
                        break
                else: #pattern is longer than summary, keep summary as is and stop looking
                    summary_pattern =  summary_pattern[0:symbol_idx+1]
                    consistent_symbol_chain = False
                    break
    else:
        summary_pattern = []
        consistent_symbol_chain = True
    # print('\n>>summary_pattern= '+str(summary_pattern))
    return summary_pattern, consistent_symbol_chain

def train_incremental_pattern(pattern, train_sig):
    train_pattern, consistent_symbol_chain = pattern

    if len(train_pattern)==0 or len(train_sig)==0:
        return train_pattern, consistent_symbol_chain

    for symbol_idx, symbol in enumerate(train_sig):
        # make sure index exists in summary pattern
        if len(train_pattern)>symbol_idx: #make sure index in bounds
            # check if symbols agree
            if symbol[0] == train_pattern[symbol_idx][0]:
                #check if counts disagree (if summary symbol has count)
                if train_pattern[symbol_idx][1]!=0 and symbol[1] != train_pattern[symbol_idx][1]:
                    train_pattern[symbol_idx][1] = 0
                #else they agree so do nothing

                # check if I am on last symbol of train_sig and pattern is longer, stop looking at this train_sig and cut the pattern here
                if symbol_idx == len(train_sig)-1 and len(train_sig)<len(train_pattern):
                    train_pattern =  train_pattern[0:symbol_idx+1]
                    consistent_symbol_chain = False
                    break                       
            else: #symbols disagreed, remove everything from here on and go to next pattern
                train_pattern =  train_pattern[0:symbol_idx]
                consistent_symbol_chain = False
                break
        else: #train_sig is longer than summary, keep summary as is and stop looking
            train_pattern =  train_pattern[0:symbol_idx+1]
            consistent_symbol_chain = False
            break    
    return train_pattern, consistent_symbol_chain

def generate_symbol_summary(attribute_symbols):
    #initialize symbol list
    attribute_symbols = [s for s in attribute_symbols if len(s)>0]
    if len(attribute_symbols)>0:
        summary_symbols = list(attribute_symbols[0])
        for symbol in list(attribute_symbols[0]):
            for symbolset in attribute_symbols:
                symbol_list = list(symbolset)
                if symbol not in symbolset:
                    summary_symbols.remove(symbol)
                    break
    else:
        summary_symbols = []
    return summary_symbols

def symbolset_incremental_pattern(pattern, symbolset):
    if len(symbolset)>0:
        return list(set(pattern).intersection(symbolset))
    else:
        return pattern

def generate_case_summary(attribute_cases):
    case_summary = []

    attribute_cases = [a for a in attribute_cases if a!='']
    if len(attribute_cases)>0:
        case_summary = attribute_cases[0]
    
        for case in attribute_cases:
            if case_summary != case:
                case_summary=''
                break
    # if case_summary=="MIX_CASE":
    #     case_summary = ''
    return case_summary
def consistent_symbol_sets_increment(consistent_ss, symbol_sig):

    consistent_symbol_sets, consistent_symbols = consistent_ss
    if len(symbol_sig)>0 and consistent_symbol_sets==True and symbol_sig != consistent_symbols:
        consistent_ss=(False,None)
    return consistent_ss
    
def token_repeats_increment(partof_multiword_value_repeats, candidate_tokens):                        
    for part in candidate_tokens:
        if part in partof_multiword_value_repeats.keys():
            partof_multiword_value_repeats[part]+=1
        else:
            partof_multiword_value_repeats[part]=0
    return partof_multiword_value_repeats

def summary_strength_increment(summary_strength, train_sig):
    if len(train_sig)>0:
        summary_strength+=1
    return summary_strength

def candidate_count_increment(candidate_count, value):
    if value in candidate_count.keys():
        candidate_count[value]+=1
    else:
        candidate_count[value]= 0    
    return candidate_count

def case_incremental_pattern(case_pattern, case):
    if case =='':
        return case_pattern
    elif case!=case_pattern:
        return ''
    else:
        return case_pattern

def generate_length_summary(column_lengths):
    column_lengths = np.asarray(column_lengths)
    length_summary = {}
    lengths= column_lengths[column_lengths!=0]
    if len(lengths)>0:
        length_summary["min"] = min(lengths)
        length_summary["max"] = max(lengths)
    else:
        length_summary["min"] = 0
        length_summary["max"] = 0
    return length_summary

def charlength_incremental_pattern(length_summary, char_length):
    if char_length>0:
        length_summary["min"] = min(length_summary["min"], char_length)
        length_summary["max"] = max(length_summary["max"], char_length)
    return length_summary
    
def keep_non_nulls(allvalues):
    #print('keep_non_nulls values: '+str(values))
    #remove trailing whitespace from all values and normalize

    values = []
    for x in allvalues:
        if x is not None:
            v = x.strip(' ').lower()
        else:
            v = ''
        values.append(v)
    
    #remove any null equivalents from list of values
    for null_equivalent in null_equivalent_values:
        try:
            while null_equivalent in values:
                # values.remove(null_equivalent)
                np.delete(values, null_equivalent)
                values = values[values!=null_equivalent]
        except ValueError:
            pass
    #print('normalized values: '+str(values))
    return values

def generate_pattern_symbols_and_case(value, outlier_sensitive):
    #print('\n generate_pattern_symbols_and_case for: \n'+value)
    value_lower = str(value).strip().lower()
    value_tokens = value_lower.split()
    if value==None or value_lower in null_equivalent_values:
        value= ''
        
    if len(value_lower)>0:
        for phrase in aggregation_tokens:
        # for phrase in ['total']:    
            if phrase in value_lower:
                value= ''
                break

    value = value.replace('\'', '')

    value_pattern = []
    value_symbols = set()
    try:
        if value.isupper():
            value_case = 'ALL_CAPS'
        elif value.islower():
            value_case = 'ALL_LOW'
        elif value.istitle():
            value_case = 'TITLE'
        elif value!='':
            value_case = 'MIX_CASE'
        else:
            value_case = ''
    except:
        value_case = ''    

    value= str(value).strip()
    value_tokens = len(value.split(' '))
    # input(value_tokens)

    value_characters = len(value)

    i = 0    
    while i<len(value):
        # input(f'value={value}\n\t-->i={i}, value_pattern={value_pattern}, value_symbols={value_symbols}')
        #print('i='+str(i))

        if i<(len(value)) and value[i].isalpha()  :
            letter_counter=0;
            while i<(len(value)) and  value[i].isalpha():
                #print('value['+str(i)+']='+str(value[i]))
                i+=1
                #print('i='+str(i))
                letter_counter+=1
            value_pattern.append(['A', letter_counter])
            value_symbols.add('A')

        elif i<(len(value)) and value[i].isspace():
            space_counter=0
            while i<(len(value)) and  value[i].isspace():
                i+=1
                space_counter+=1
            value_pattern.append(['S', space_counter])
            value_symbols.add('S')

        # ignore - if it is the first character followed by a digit    
        elif outlier_sensitive == True and i==0 and len(value)>1 and value[i]=='-' and value[i+1].isdigit():
            digit_counter=0;
            i+=1
            while i<(len(value)) and  value[i].isdigit():
                #print('value['+str(i)+']='+str(value[i]))
                i+=1
                #print('i='+str(i))
                digit_counter+=1
            value_pattern.append(['D', digit_counter])
            value_symbols.add('D')

        elif i<(len(value)) and value[i].isdigit():
            digit_counter=0;
            while i<(len(value)) and  value[i].isdigit():
                #print('value['+str(i)+']='+str(value[i]))
                i+=1
                #print('i='+str(i))
                digit_counter+=1
            value_pattern.append(['D', digit_counter])
            value_symbols.add('D')

        # Punctuation: !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~
        elif i<(len(value)) and value[i] in string.punctuation:
            punctuation_counter= 0
            punctuation = value[i]
            while i<(len(value)) and  value[i] == punctuation:
                i+=1
                punctuation_counter+=1
            value_pattern.append([punctuation, punctuation_counter])
            value_symbols.add(punctuation)

        elif i<(len(value)):
            unknown_counter= 0
            unknown = value[i]
            while i<(len(value)) and  value[i] == unknown:
                i+=1
                unknown_counter+=1
            value_pattern.append([unknown, unknown_counter])
            value_symbols.add(unknown)

        else:
            i+=1

    #print('\nvalue_pattern='+str(value_pattern))        
    #print('value_symbols='+str(value_symbols))
    return value_pattern, value_symbols, value_case, value_tokens, value_characters

def is_attribute_header(candidate, value_pattern_summary, value_symbol_summary):
    #print('candidate attribute header= '+str(candidate))
    cand_pattern, cand_symbols, cand_case = generate_pattern_and_symbols(candidate)
    #print('header patterns: '+str(cand_pattern)+', '+str(cand_symbols))
    if(len(value_pattern_summary)!=0):
        for i,tup in enumerate(value_pattern_summary):
            #if i<len(cand_pattern) and cand_pattern[i][0] == tup[0]:
            #there are more symbols in the value summary than the candidate header, so we can stop looking
            if i>=len(cand_pattern):
                return True

            if cand_pattern[i][0] == tup[0]:
                if tup[1]!=0:
                    if cand_pattern[i][1]!=tup[1]:
                        return True            
            else:
                return True
    # Cannot make a statement with certainty that this is a header
    return False

"""
Given a list of values 'row_values' discover values that are numeric.
if they are more than k and 
if they are sorted by increasing value and
if they increase with a consistent step
add the indexes to a list and return the list
row_values=['1','2001','2002','2003']
sample_symbols={0:[set(['D'])],1:[set(['D'])],2:[set(['D'])], 3:[set(['D'])]}
"""
def discover_incremental_values(row_values, sample_symbols):
    # aeval = Interpreter()
    incremental_value_idxs = []
    numeric_values= {}
    steps= []
    step_frequency = (0,0)
    
    for idx,value in enumerate(row_values):
        if value!=None :
            value = value.strip()
            if len(value)>0 and value[-1] == '%':
                value = value[0:-1]
            try:
                evaluate(value)
            except:
                continue
            if (value.count('-')==0 and isfloat(value)) or (value.count('-')==1 and len(sample_symbols[idx])>0 and (sample_symbols[idx][0]).issubset(set(['D',',','.','-','S'])) and isfloat(evaluate(value))):
                try:
                    ev =  evaluate(value)
                except:
                    ev = None

                if ev!=None:
                    try:
                        float_v = float(ev)
                        numeric_values[idx] = float_v
                    except:
                        continue

    # print('numeric_values='+str(numeric_values))
    numeric_value_list = []
    for key in sorted(numeric_values):
        number = numeric_values[key]
        if number not in numeric_value_list and number.is_integer():
            numeric_value_list.append(number)
    # input('\nnumeric_value_list='+str(numeric_value_list))
    step_pairs= {}
    if len(numeric_value_list)>1:
        for i in range(len(numeric_value_list)):
            value = numeric_value_list[-(i+1)]    
            k_indexes = list(range(0,(len(numeric_value_list)-(i+1))))
            k_indexes.sort(reverse=True)
            # print('k_indexes='+str(k_indexes))
            for k in k_indexes:
                # print(str(value) +'-'+str(numeric_value_list[k])+'='+str(value-numeric_value_list[k]))
                if value-numeric_value_list[k]>0:
                    step = value-numeric_value_list[k] 
                    if step in step_pairs.keys() and numeric_value_list[len(numeric_value_list)-(i+1)]<=numeric_value_list[step_pairs[step][-1]] and k<step_pairs[step][-1]:
                        step_pairs[step].append(k)
                    else:
                        step_pairs[step]=[len(numeric_value_list)-(i+1),k]
        # input('\nstep_pairs=')
        # pp.pprint(step_pairs)

        step_counter = {k:len(step_pairs[k]) for k in step_pairs.keys()}
        # print(step_counter)
        sorted_increment_counter_desc = sorted(step_counter.items(), key=lambda x: (-x[1], x[0]))
        # input('\nsorted_increment_counter_desc:')
        # pp.pprint(sorted_increment_counter_desc)
        if len(sorted_increment_counter_desc)>0:
            candidate_indexes = list(step_pairs[sorted_increment_counter_desc[0][0]])
            candidate_indexes.sort()
            # print(candidate_indexes)
            step_frequency = (0,0)
            if len(candidate_indexes)>1 and all_adjacent(candidate_indexes):
                step_frequency = sorted_increment_counter_desc[0]
        else:
            step_frequency = (0,0)
    else:
        step_frequency = (0,0)
    
    # input(step_frequency)  
    return step_frequency

def discover_incremental_values_at_least_one_nonadjacent(row_values, sample_symbols):
    # aeval = Interpreter()
    incremental_value_idxs = []
    numeric_values= {}
    steps= []
    step_frequency = (0,0)
    
    for idx,value in enumerate(row_values):
        if value!=None :
            value = value.strip()
            if len(value)>0 and value[-1] == '%':
                value = value[0:-1]
            try:
                evaluate(value)
            except:
                continue
            if (value.count('-')==0 and isfloat(value)) or (value.count('-')==1 and len(sample_symbols[idx])>0 and (sample_symbols[idx][0]).issubset(set(['D',',','.','-','S'])) and isfloat(evaluate(value))):
                try:
                    ev =  evaluate(value)
                except:
                    ev = None

                if ev!=None:
                    try:
                        float_v = float(ev)
                        numeric_values[idx] = float_v
                    except:
                        continue

    # print('numeric_values='+str(numeric_values))
    numeric_value_list = []
    for key in sorted(numeric_values):
        number = numeric_values[key]
        if number not in numeric_value_list and number.is_integer():
            numeric_value_list.append(number)
    # input('\nnumeric_value_list='+str(numeric_value_list))
    step_pairs= {}
    if len(numeric_value_list)>1:
        for i in range(len(numeric_value_list)):
            value = numeric_value_list[-(i+1)]    
            k_indexes = list(range(0,(len(numeric_value_list)-(i+1))))
            k_indexes.sort(reverse=True)
            # print('k_indexes='+str(k_indexes))
            for k in k_indexes:
                # print(str(value) +'-'+str(numeric_value_list[k])+'='+str(value-numeric_value_list[k]))
                if value-numeric_value_list[k]>0:
                    step = value-numeric_value_list[k] 
                    if step in step_pairs.keys() and numeric_value_list[len(numeric_value_list)-(i+1)]<=numeric_value_list[step_pairs[step][-1]] and k<step_pairs[step][-1]:
                        step_pairs[step].append(k)
                    else:
                        step_pairs[step]=[len(numeric_value_list)-(i+1),k]
        # input('\nstep_pairs=')
        # pp.pprint(step_pairs)

        step_counter = {k:len(step_pairs[k]) for k in step_pairs.keys()}
        # print(step_counter)
        sorted_increment_counter_desc = sorted(step_counter.items(), key=lambda x: (-x[1], x[0]))
        # input('\nsorted_increment_counter_desc:')
        # pp.pprint(sorted_increment_counter_desc)
        if len(sorted_increment_counter_desc)>0:
            step_frequency = sorted_increment_counter_desc[0]
            candidate_indexes = list(step_pairs[sorted_increment_counter_desc[0][0]])
            candidate_indexes.sort()
            # for idx in candidate_indexes:
                # print(numeric_value_list[idx])
            if len(candidate_indexes)>1 and all_adjacent(candidate_indexes):
                step_frequency = (0,0)
        else:
            step_frequency = (0,0)
    else:
        step_frequency = (0,0)
    
    return step_frequency

def all_adjacent(seq):
    d = seq[1] - seq[0]
    if d!=1:
        return False
    for index in range(len(seq) - 1):
        if not (seq[index + 1] - seq[index] == d):
            return False
    return True
# def discover_incremental_values(row_values, sample_symbols):
#     # aeval = Interpreter()
#     incremental_value_idxs = []
#     numeric_values= {}
#     steps= []
#     step_frequency = (0,0)
    
#     for idx,value in enumerate(row_values):
#         if value!=None :
#             value = value.strip()
#             if len(value)>0 and value[-1] == '%':
#                 value = value[0:-1]
#             try:
#                 evaluate(value)
#             except:
#                 continue
#             if (value.count('-')==0 and isfloat(value)) or (value.count('-')==1 and len(sample_symbols[idx])>0 and (sample_symbols[idx][0]).issubset(set(['D',',','.','-','S'])) and isfloat(evaluate(value))):
#                 try:
#                     ev =  evaluate(value)
#                 except:
#                     ev = None

#                 if ev!=None:
#                     try:
#                         float_v = float(ev)
#                         numeric_values[idx] = float_v
#                     except:
#                         continue

#     print('numeric_values='+str(numeric_values))
#     if len(numeric_values)>=3:
#         numeric_value_list = []
#         for key in sorted(numeric_values):
#         # for key in numeric_values:
#             numeric_value_list.append(numeric_values[key])
#         print('numeric_value_list='+str(numeric_value_list))
#         for i in range(1, len(numeric_values)):
#             if numeric_value_list[i]!=0:
#                 steps.append(numeric_value_list[i]-numeric_value_list[i-1])

#     consecutive = 0
#     consecutive_steps = []
#     print('steps='+str(steps))
    
#     for step_i, step in enumerate(steps):
#         if step_i>0 and step==steps[step_i-1]:
#             consecutive+=1
#             if step_i== len(steps)-1:
#                 consecutive_steps.append((steps[step_i-1],consecutive+1) )
#         if step_i>0 and step!=steps[step_i-1]:
#             if consecutive!=0 or step_i== len(steps)-1:
#                 consecutive_steps.append((steps[step_i-1],consecutive+1) )
#                 consecutive==0   

#     distinct_steps = list(set([x[0] for x in consecutive_steps]))
#     maxstepcount= {}
#     for step in distinct_steps:
#         maxstepcount[step] = max(consecutive_steps, key=lambda x:x[1])
#     if len(maxstepcount)>0:
#         step_frequency = sorted(Counter(maxstepcount).items(),key=operator.itemgetter(1), reverse=True)[0][1] 
#     else:
#         step_frequency = (0,0)
      
#     return step_frequency

def isfloat(value):
    try:
        # input('-')
        float(str(value).strip())
        # input('--')
        return True
    except:
        # input('----')
        return False

def identify_postmeta_candidates(dataframe):
    # print(dataframe)
    candidate_postmetastart_index = dataframe.shape[0]+1
    for index, row in dataframe.iloc[::-1].iterrows():
        datatuple = row.tolist()
        # print('\ndatatuple='+str(datatuple))
        # input('datatuple[1:]='+str(datatuple[1:]))

        if len(datatuple)>1 and len(datatuple[0])>0 and ('source' in (str(datatuple[0]).strip()).lower() or len([x for x in datatuple[1:] if x!=None and len(str(x).strip())>0])==0):
            # input('datatuple is a candidate')
            continue
        else:

            candidate_postmetastart_index = index+1
            # input('\ncandidate_postmetastart_index='+str(candidate_postmetastart_index))
            break  

    # input('\n->candidate_postmetastart_index='+str(candidate_postmetastart_index))
    return candidate_postmetastart_index

def discover_header_and_datastart(org_dataframe, candidate_size=20, trust_threshold=5, max_sample_size = 100):
    
    # initialize data start at the begining of the dataframe
    header_index = [0,0];

    null_equi_spotted = set()
    datastart_index = 0
    datastart_found = False
    nonzero_dataconf_seen= False
    confidence = 0
    row_column_summaries={}
    outlier_aware_summaries = {}
    outliers = {}
    pre_data_row_col_summaries= {}
    pre_data_row_disagreements= {}
    datastart_agreements = {}
    first_data_row_confidence = {}
    row_disagreements = {}
    row_agreements = {}
    col_summaries= []
    deciding_row_confidence= {}
    deciding_row_info= {}
    datastartinfo = ''
    header_detection_rule = ''
    sorted_rows = []
    first_row_data_row_column_summaries = []
    row_size = org_dataframe.shape[0]-1
    num_columns = org_dataframe.shape[1]
    postmeta_cand_startindex = org_dataframe.shape[0]
    dataframe_labels = []
    cand_postmeta_exists = False
    for column in org_dataframe:
        dataframe_labels.append(column)

    # figure out what the maximum number of candidate pre data rows is
    if row_size-candidate_size<0:
        #print('row_size-candidate_size<0')
        candidate_pre_data_rows = org_dataframe[0:row_size+1]            
    else:
        candidate_pre_data_rows = org_dataframe[0:candidate_size]
    #LOG
    # input(org_dataframe)

    # discover possible postmeta rows
    postmeta_cand_startindex = identify_postmeta_candidates(org_dataframe)

    if postmeta_cand_startindex<org_dataframe.shape[0]:
        cand_postmeta_exists = True
        cand_postmeta_rows = org_dataframe[postmeta_cand_startindex:]
        # input(cand_postmeta_rows)
        #print(list(range(postmeta_cand_startindex,dataframe.shape[0])))
        dataframe=org_dataframe.drop(org_dataframe.index[list(range(postmeta_cand_startindex,org_dataframe.shape[0]))])
        row_size = dataframe.shape[0]-1
    else:
        dataframe = org_dataframe
        
    #LOG
    # input(dataframe)
    # GENERATE SAMPLE UP TO max_sample_size rows (100)
    #for all rows until end or max_sample_size
    #generate value patterns, symbols and case flags per column
    column_index = 0
    candidate_valuerow_patterns = []
    candidate_valuerow_symbols = []
    candidate_valuerow_cases = []

    if row_size>max_sample_size:
        values_dataframe = dataframe[0:max_sample_size]
        last_sample_index = max_sample_size-1 #in case last line is aggregate
    else:
        if row_size<5:
            values_dataframe = dataframe
            last_sample_index = row_size
        else:
            values_dataframe = dataframe[0:-1]
            last_sample_index = row_size-1 #in case last line is aggregate

    # LOG
    input('values_dataframe:\n{}'.format(values_dataframe))

    # INITIALIZE samplerow patterns, symbols and cases')
    sample_patterns = {}
    sample_symbols = {}
    sample_cases = {}
    sample_token_len = {}
    sample_chars_len = {}

    for column in dataframe:
        column_value_patterns = []
        column_value_symbols = []
        column_value_cases = []
        column_value_token_lengths = []
        column_value_char_lengths = []

        values = values_dataframe[column].tolist()
        # print('values ['+str(column)+'] = '+str(values))
        columnvalues= values
        for value in columnvalues:
            
            if str(value).strip().lower() in null_equivalent_values:
                if str(value)!='0':
                    null_equi_spotted.add(str(value).strip().lower())
            pattern, symbols, case,value_num_tokens, value_num_chars  = generate_pattern_symbols_and_case(str(value).strip(), False)
            column_value_patterns.append(pattern)
            column_value_symbols.append(symbols)
            column_value_cases.append(case)
            column_value_token_lengths.append(value_num_tokens)
            column_value_char_lengths.append(value_num_chars)

        sample_patterns[dataframe_labels.index(column)] = column_value_patterns
        sample_symbols[dataframe_labels.index(column)] = column_value_symbols
        sample_cases[dataframe_labels.index(column)] = column_value_cases
        sample_token_len[dataframe_labels.index(column)] = column_value_token_lengths
        sample_chars_len[dataframe_labels.index(column)] = column_value_char_lengths

    #input('----> START EVALUATING CANDIDATES')
    first_disagreeing_row = {}
    for candidate_index, cand_predata_row in candidate_pre_data_rows.iterrows():
        # LOG
        input('\ncandidate on row_'+str(candidate_index)+': '+ str(cand_predata_row.tolist()))

        row_values = cand_predata_row.tolist()
        # #####################################################################################
        if candidate_index==0:
            for column in dataframe:
                all_values_summary,consistent_symbol_chain= generate_pattern_summary(sample_patterns[dataframe_labels.index(column)])
                bw_patterns =  [list(reversed(pattern)) for pattern in sample_patterns[dataframe_labels.index(column)]]
                pattern_BW_summary,_=generate_pattern_summary(bw_patterns)
    
                symbol_summary = generate_symbol_summary(sample_symbols[dataframe_labels.index(column)])
                case_summary = generate_case_summary(sample_cases[dataframe_labels.index(column)])
                summary_strength= sum(1 for x in sample_patterns[dataframe_labels.index(column)] if len(x)>0)

                if len(sample_chars_len[dataframe_labels.index(column)])>0:
                    maxchars = max(sample_chars_len[dataframe_labels.index(column)])
                    minchars = min(sample_chars_len[dataframe_labels.index(column)])
                    avgchars = mean(sample_chars_len[dataframe_labels.index(column)])
                else:
                    maxchars = 0
                    minchars = 0
                    avgchars = 0

                if len(sample_token_len[dataframe_labels.index(column)])>0:
                    maxtoken = max(sample_token_len[dataframe_labels.index(column)])
                    mintoken = min(sample_token_len[dataframe_labels.index(column)])
                    avgtoken = mean(sample_token_len[dataframe_labels.index(column)])
                else:
                    maxtoken = 0
                    mintoken = 0
                    avg_token = 0

                columnsummary = ColumnSampleSummary(all_values_summary,
                    pattern_BW_summary, consistent_symbol_chain,
                    symbol_summary, case_summary,
                    summary_strength,
                    maxchars,minchars,maxtoken, mintoken, avgchars, avgtoken                     
                )

                first_row_data_row_column_summaries.append(columnsummary) 

            #  check for evidence of a data row if first value of first row is not empty
            if len(cand_predata_row)>0:

                if str(cand_predata_row[dataframe_labels[0]]).strip() !='':
                    # calculate agreement from row 0 onwards

                    strong_agreement, agreeinging_columns, data_confidence = strong_pattern_agreement(values_dataframe,sample_patterns,sample_symbols, sample_cases, [row_values])
                    first_data_row_confidence[candidate_index]= data_confidence
                    if data_confidence>0:
                        nonzero_dataconf_seen= True

                    # LOG
                    print('confidence row_'+str(candidate_index)+' is DATA = '+str(data_confidence))
                    input(agreeinging_columns)

                    row_agreements[candidate_index]= agreeinging_columns
                    if strong_agreement:
                        #Data starts from first line, there is no header
                        header_index = [0,0]
                        datastart_index= 0

                        if first_data_row_confidence[candidate_index] >=0.9:
                            outlier_aware_summaries, outliers = generate_outlier_sensitive_column_summaries(values_dataframe)
                            if postmeta_cand_startindex!=dataframe.shape[0]:                                
                                dataend_index = eval_candidate_postmeta(cand_postmeta_rows,outlier_aware_summaries[0])#pre_data_row_col_summaries[0]
                                # input('\n\ndataend_index1='+str(dataend_index))
                            else:
                                dataend_index = org_dataframe.shape[0]
                            return datastart_index,  dataend_index, header_index, first_data_row_confidence[candidate_index], first_row_data_row_column_summaries,{}, agreeinging_columns, {},'data starts from first row in block', null_equi_spotted, sorted_rows, outlier_aware_summaries, outliers

                else:
                    first_data_row_confidence[candidate_index]=0
        #####################################################################################


        row_column_summaries[candidate_index] = []
        if candidate_index+trust_threshold>=row_size: ## if there are not enough rows to trust general summaries
            require_consistent_symbol_chain = True
        else:
            require_consistent_symbol_chain = False

        #remove top row from sample meta
        for column in dataframe:
            if len(sample_patterns[dataframe_labels.index(column)])>0:
                sample_patterns[dataframe_labels.index(column)].pop(0)
            if len(sample_symbols[dataframe_labels.index(column)])>0:
                sample_symbols[dataframe_labels.index(column)].pop(0)
            if len(sample_cases[dataframe_labels.index(column)])>0:
                sample_cases[dataframe_labels.index(column)].pop(0)
            if len(sample_token_len[dataframe_labels.index(column)])>0:
                sample_token_len[dataframe_labels.index(column)].pop(0)
            if len(sample_chars_len[dataframe_labels.index(column)])>0:
                sample_chars_len[dataframe_labels.index(column)].pop(0)

        # removed top row patterns if there are any to remove

        # if incremented last_sample_index in bounds add meta for that row to the end
        if last_sample_index+1 <row_size:
            last_sample_index = last_sample_index+1
            new_sample_row = dataframe.iloc[last_sample_index:last_sample_index+1,:]

            for column in dataframe:

                value = new_sample_row[column].values[0]
                if str(value).strip().lower() in null_equivalent_values:
                    value= ''
                if value==None:
                    value= ''
                pattern, symbols, case,value_num_tokens, value_num_chars = generate_pattern_symbols_and_case(value.strip(), False)
                sample_patterns[dataframe_labels.index(column)].append(pattern)
                sample_symbols[dataframe_labels.index(column)].append(symbols)
                sample_cases[dataframe_labels.index(column)].append(case)
                sample_token_len[dataframe_labels.index(column)].append(value_num_tokens)
                sample_chars_len[dataframe_labels.index(column)].append(value_num_chars)


        ## UP TO HERE WE HAVE GENERATED ALL META FOR CANDIDATE DATA ROWS
        ## Generate Value Row Summaries---------------------------------------------------------------
        columns_pattern_FW_summary = {}
        columns_pattern_BW_summary = {}
        columns_chain_consistency = {}
        columns_symbols_summary = {}
        columns_cases_summary = {}
        columns_maxchar= {}
        columns_minchar={}
        columns_maxtoken= {}
        columns_mintoken= {}
        summary_strength = {}# number of nonnull values that produced the summary of the column

        for column in dataframe:
            # input('\nCOLUMN_'+str(column))
            summary_strength[dataframe_labels.index(column)] = sum(1 for x in sample_patterns[dataframe_labels.index(column)] if len(x)>0)

            column_sample_patterns = copy.deepcopy(sample_patterns[dataframe_labels.index(column)])
            # input('column_sample_patterns='+str(column_sample_patterns))
            pattern_FW_summary, consistent_symbol_chain = generate_pattern_summary(column_sample_patterns)
            # input('pattern_FW_summary= '+str(pattern_FW_summary))
            bw_patterns =  [list(reversed(pattern)) for pattern in column_sample_patterns]
            pattern_BW_summary,_=generate_pattern_summary(bw_patterns)
            
            columns_pattern_FW_summary[dataframe_labels.index(column)] = pattern_FW_summary
            columns_pattern_BW_summary[dataframe_labels.index(column)] = pattern_BW_summary
            columns_chain_consistency[dataframe_labels.index(column)] = consistent_symbol_chain
            columns_symbols_summary[dataframe_labels.index(column)] = generate_symbol_summary(sample_symbols[dataframe_labels.index(column)])
            columns_cases_summary[dataframe_labels.index(column)] = generate_case_summary(sample_cases[dataframe_labels.index(column)])

            if len(sample_chars_len[dataframe_labels.index(column)])>0:
                maxchars = max(sample_chars_len[dataframe_labels.index(column)])
                minchars = min(sample_chars_len[dataframe_labels.index(column)])
                avgchars = mean(sample_chars_len[dataframe_labels.index(column)])
            else:
                maxchars = 0
                minchars = 0
                avgchars = 0
            if len(sample_token_len[dataframe_labels.index(column)])>0:
                maxtoken = max(sample_token_len[dataframe_labels.index(column)])
                mintoken = min(sample_token_len[dataframe_labels.index(column)])
                avgtoken = mean(sample_token_len[dataframe_labels.index(column)])
            else:
                maxtoken = 0
                mintoken = 0
                avg_token = 0

            columnsummary = ColumnSampleSummary(columns_pattern_FW_summary[dataframe_labels.index(column)],
             columns_pattern_BW_summary[dataframe_labels.index(column)], columns_chain_consistency[dataframe_labels.index(column)],
             columns_symbols_summary[dataframe_labels.index(column)], columns_cases_summary[dataframe_labels.index(column)],
             summary_strength[dataframe_labels.index(column)],
             maxchars,minchars,maxtoken, mintoken, avgchars, avgtoken
             )

            row_column_summaries[candidate_index].append(columnsummary)
        ###---------------------------------------------------------------------------------------------  
        # GENERATED SUMMARIES

        # Get the candidate predata row
        candidate_pdr = cand_predata_row.tolist() 

        #inspect maximum 40 columns
        max_columns =40
        disagreeing_column_count=0
        disagreeing_columns = {}

        first_disagreement_found = False
        column_iter_counter = 0
        first_value_on_row_isempty = False

        for column in dataframe:
            column_iter_counter+=1
            if column_iter_counter>max_columns:
                break
            # input('\nCOLUMN_'+str(column))
            candidate_value = candidate_pdr[dataframe_labels.index(column)]
            if dataframe_labels.index(column)==0 and (candidate_value=='' or candidate_value==None):
                first_value_on_row_isempty = True

            value_pattern_summary = columns_pattern_FW_summary[dataframe_labels.index(column)]
            
            value_pattern_BW_summary = columns_pattern_BW_summary[dataframe_labels.index(column)]
            value_chain_consistency = columns_chain_consistency[dataframe_labels.index(column)]
            value_symbol_summary = columns_symbols_summary[dataframe_labels.index(column)]                
            case_summary = columns_cases_summary[dataframe_labels.index(column)]
            
            
            if candidate_value=='' or candidate_value==None or str(candidate_value).strip()=='':# in ['','data not available', 'nan','not available', 'no data', 'no answer', 'nd', 'na', 'n/d', 'n/a','not applicable', '-','--','.', '..','...', 'null', 'none']:
                
                if candidate_index>0 and first_value_on_row_isempty == True:                    
                    prev_row = candidate_pre_data_rows[candidate_index-1:candidate_index].values.tolist()[0]
                    candidate_value= prev_row[dataframe_labels.index(column)]
                    if candidate_value=='' or candidate_value==None or str(candidate_value).strip().lower() in null_equivalent_values:
                        continue
                else:
                    continue
            
            cand_pattern, cand_symbols, cand_case, cand_num_tokens, cand_num_chars = generate_pattern_symbols_and_case(candidate_value.strip(), False)

            if (len(sample_patterns[dataframe_labels.index(column)])>1 and summary_strength[dataframe_labels.index(column)]>=2) or (len(sample_patterns[dataframe_labels.index(column)])==1 and summary_strength[dataframe_labels.index(column)]==1):
                disagreements = find_disagreement(value_pattern_summary, value_pattern_BW_summary, value_chain_consistency, cand_pattern, value_symbol_summary, cand_symbols, case_summary, cand_case)
                if len(disagreements)>0:                    
                    disagreeing_columns[dataframe_labels.index(column)]= {}

                    confidence = disagreement_confidence(disagreements, summary_strength[dataframe_labels.index(column)])
                    disagreeing_columns[dataframe_labels.index(column)]["confidence"] = float(confidence)
                    disagreeing_columns[dataframe_labels.index(column)]["disagreements"] = disagreements
                    disagreeing_column_count+=1
                    datastart_index = candidate_index+1

                    if first_disagreement_found==False:
                        first_disagreement_found=True

        # calculate agreement of all following rows in sample
        # calculate agreement from row candidate_index+1 onwards:
        next_row_values = candidate_pre_data_rows[candidate_index+1:candidate_index+2].values.tolist()
        strong_agreement, agreeing_columns, data_confidence = strong_pattern_agreement(dataframe.iloc[candidate_index+1::,:],sample_patterns,sample_symbols, sample_cases, next_row_values)

        #calculate summary disagreements for this row
        predata_row_confidence=calculate_row_summary_disagreement(disagreeing_columns, candidate_index, row_values, sample_symbols)     
        row_agreements[candidate_index+1] = agreeing_columns
        row_disagreements[candidate_index] = disagreeing_columns
        
        first_data_row_confidence[candidate_index+1] = combined_row_confidence(float(predata_row_confidence),float(data_confidence),candidate_index)
        # LOG
        print('confidence row_'+str(candidate_index+1)+' is DATA = '+str(data_confidence))
        print('confidence row_'+str(candidate_index)+ ' is PRE_DATA = '+str(predata_row_confidence))
        print('\ndisagreeing_columns = '+str(disagreeing_columns))
        print('\nagreeing_columns on next row = '+str(agreeing_columns))
        input('\n---->Row_'+str(candidate_index+1) +' is the first data row with combined confidence: '+str(first_data_row_confidence[candidate_index+1]))
        
        if first_disagreement_found==True and len(first_disagreeing_row.keys())==0:
            #print('\n\nFIRST DISAGREEMENT FOUND\n\n')
            first_disagreeing_row["disagreeing_columns"]= disagreeing_columns
            first_disagreeing_row["first_data_row_confidence"]= first_data_row_confidence[candidate_index+1]
            first_disagreeing_row["index"]= candidate_index            
            first_disagreeing_row["column_summaries"]= [json.dumps(x.__dict__) for  x in row_column_summaries[candidate_index]]#row_column_summaries[candidate_index]

        if  first_data_row_confidence[candidate_index+1]==1:# and assess_candidate_header(candidate_pdr):
            header_index, header_detection_rule = asses_candidate_predatarow(candidate_pre_data_rows, candidate_index)  
            datastart_found = True
            datastart_index = candidate_index+1
            break

        if nonzero_dataconf_seen== False and data_confidence>=0.9:
            # input('\nnonzero_dataconf_seen== False and data_confidence>=0.9\n')
            header_index, header_detection_rule = asses_candidate_predatarow(candidate_pre_data_rows, candidate_index)  
            datastart_found = True
            datastart_index = candidate_index+1
            break 
        elif nonzero_dataconf_seen== False and data_confidence>0:
            nonzero_dataconf_seen == True

        if  data_confidence>=0.9 or (data_confidence+predata_row_confidence)/2>=0.9:# and assess_candidate_header(candidate_pdr):
            # input('\ndata_confidence>=0.9\n')
            break #unlikely data start is after this line (it is here or before)

    # after all candidate predata rows were checked unsuccessfully, 
    # Look for highest row confidence!
    if datastart_found == False:
        # sort first_data_row_confidence by confidence descending index ascending
        sorted_rows = sorted(first_data_row_confidence.items(), key=lambda x: (-x[1], x[0]))
        if candidate_pre_data_rows.shape[0]==1:
            header_index=[0,0]
            datastart_index=0
        else:    
            for row_index, row_conf in sorted_rows:
                try:
                    nextrow= candidate_pre_data_rows[row_index:row_index+1].values.tolist()[0]
                    next_row_num_nulls = nextrow.count('')

                except:
                    pass

                if row_index-1 >=0:
                    thisrow = candidate_pre_data_rows[row_index-1:row_index].values.tolist()[0]            
                    this_row_num_nulls = thisrow.count('')
                    header_index, header_detection_rule = asses_candidate_predatarow(candidate_pre_data_rows, row_index-1)
                    datastart_index= row_index
                    break
                else:
                    header_detection_rule= "first line was data, no header exists"
                    datastart_index= row_index
                    header_index= [datastart_index,datastart_index]
                    break

    if datastart_index>0:
        # pre_data_row_col_summaries = [json.dumps(x.__dict__) for x in row_column_summaries[datastart_index-1]]
        pre_data_row_col_summaries = row_column_summaries[datastart_index-1]
        pre_data_row_disagreements = row_disagreements[datastart_index-1]
    else:
        pre_data_row_col_summaries = first_row_data_row_column_summaries

    if datastart_index in first_data_row_confidence.keys():
        deciding_row_confidence = first_data_row_confidence[datastart_index]
    else:
        deciding_row_confidence = 0    
    
    if len(row_agreements)>0 and datastart_index>0:
        datastart_agreements = row_agreements[datastart_index]
    else:
        datastart_agreements = {}
    # print('\n- RESULTS -\n\n')
    # print('\n datastart_index= '+str(datastart_index))
    # print('\n header_index= '+str(header_index))    
    # print('\n deciding_row_confidence= '+str(deciding_row_confidence))
    # print('\n pre_data_row_disagreements= '+str(pre_data_row_disagreements))
    # print('\n pre_data_row_col_summaries= '+str(pre_data_row_col_summaries))
    # print('\n data_row_agreements='+str(row_agreements[datastart_index]))
    # print('\nfirst_disagreeing_row= '+str(first_disagreeing_row))

    outlier_aware_summaries, outliers = generate_outlier_sensitive_column_summaries(values_dataframe[datastart_index:])
    if cand_postmeta_exists:
        dataend_index = eval_candidate_postmeta(cand_postmeta_rows,outlier_aware_summaries[0])#pre_data_row_col_summaries[0])
        # input('\n\ndataend_index2='+str(dataend_index))    
    else:
        dataend_index = org_dataframe.shape[0]

    return datastart_index,dataend_index,  header_index, deciding_row_confidence, pre_data_row_col_summaries,pre_data_row_disagreements,datastart_agreements, first_disagreeing_row, header_detection_rule, null_equi_spotted, sorted_rows, outlier_aware_summaries, outliers

def generate_outlier_sensitive_column_summaries(dataframe):
    # print('\n\ngenerate_outlier_sensitive_column_summaries:\n')
    # input(dataframe)
    outlier_aware_summaries = {}
    outliers_per_column = {}
    for ind, column in enumerate(dataframe.columns):
        values = dataframe.iloc[:,ind]
        outlier_labels, outlier_indexes = find_outliers(values)
        outliers_per_column[ind] = outlier_indexes
        values_sans_outliers =  values.drop(labels = outlier_labels)
        column_patterns = []
        column_symbols = []
        column_cases  = []
        column_char_lengths = []
        column_token_lengths = []

        for idx, value in values_sans_outliers.items():
            pattern, symbols, case, value_num_tokens, value_num_chars = generate_pattern_symbols_and_case(str(value).strip(), True)
            column_patterns.append(pattern)
            column_symbols.append(symbols)
            column_cases.append(case)
            column_char_lengths.append(value_num_chars)
            column_token_lengths.append(value_num_tokens)

        all_values_summary,consistent_symbol_chain= generate_pattern_summary(column_patterns)
        bw_patterns =  [list(reversed(pattern)) for pattern in column_patterns]
        pattern_BW_summary,_=generate_pattern_summary(bw_patterns)
        symbol_summary = generate_symbol_summary(column_symbols)
        case_summary = generate_case_summary(column_cases)
        summary_strength= sum(1 for x in column_patterns if len(x)>0)

        if len(column_char_lengths)>0:
            maxchars = max(column_char_lengths)
            minchars = min(column_char_lengths)
            avgchars = mean(column_char_lengths)
        else:
            maxchars = 0
            minchars = 0
            avgchars = 0

        if len(column_token_lengths)>0:
            maxtoken = max(column_token_lengths)
            mintoken = min(column_token_lengths)
            avgtoken = mean(column_token_lengths)
        else:
            maxtoken = 0
            mintoken = 0
            avg_token = 0

        columnsummary = ColumnSampleSummary(all_values_summary,
            pattern_BW_summary, consistent_symbol_chain,
            symbol_summary, case_summary,summary_strength,
            maxchars,minchars,maxtoken, mintoken, avgchars, avgtoken                     
        )
        outlier_aware_summaries[ind] = columnsummary
        # print('\nsummary:\n')
        # input(columnsummary)

    return outlier_aware_summaries, outliers_per_column



        
def find_outliers(values, outlier_ratio = 0.75):
    outlier_labels = []
    outlier_indexes = []
    label_to_ind = {}
    row_to_firstsymbol = {}
    row_to_lastsymbol = {}
    
    row_ind = 0
    for label, row in values.items():
        label_to_ind[label] = row_ind
        row_ind+=1
        # print('index= '+str(index))
        # input('row= '+row)

        value = str(row).strip()
        if value.lower() in null_equivalent_values:
            value = ''

        if len(value)>0:
            if value[0].isalpha()  :
                row_to_firstsymbol[label] = 'A'
            elif value[0].isdigit():
                row_to_firstsymbol[label] = 'D'
            else:
                if value[0] == '-' and len(value)>1 and value[1].isdigit():
                    row_to_firstsymbol[label] = 'D'
                else:
                    row_to_firstsymbol[label] = value[0]

            if value[-1].isalpha()  :
                row_to_lastsymbol[label] = 'A'
            elif value[-1].isdigit():
                row_to_lastsymbol[label] = 'D'
            else:
                row_to_lastsymbol[label] = value[0]

    c = Counter(row_to_firstsymbol.values())
    count_groupby = sorted(dict( c.items()).items(),key = lambda x:x[1], reverse=True)
    if len(count_groupby)>1:
        counts = [x[1] for x in count_groupby]
        totalvalues = sum(counts)

        if count_groupby[0][1]>=outlier_ratio*totalvalues:
            for x in count_groupby[1:]:
                idcs = [i for i,j in row_to_firstsymbol.items() if j==x[0]]
                outlier_labels.extend(idcs)

    c = Counter(row_to_lastsymbol.values())
    count_groupby = sorted(dict( c.items()).items(),key = lambda x:x[1], reverse=True)
    if len(count_groupby)>1:
        counts = [x[1] for x in count_groupby]
        totalvalues = sum(counts)
        if count_groupby[0][1]>=outlier_ratio*totalvalues:
            for x in count_groupby[1:]:
                idcs = [i for i,j in row_to_lastsymbol.items() if j==x[0]]
                outlier_labels.extend(idcs)

    if len(outlier_labels)>0:
        outlier_labels = list(set(outlier_labels))
        outlier_labels.sort()
        outlier_indexes = [label_to_ind[x] for x in outlier_labels]
        outlier_indexes.sort()

    return outlier_labels,  outlier_indexes


def eval_candidate_postmeta(cand_postmeta_rows,pre_data_row_firstcol_summary):
    # input('\n\n\neval_candidate_postmeta')
    postmetastart = cand_postmeta_rows.shape[0]
    for rowindex, row in cand_postmeta_rows.iterrows():
        postmetastart = rowindex
        # print('rowindex='+str(rowindex))
        rowvalues = row.values.tolist()
        value= str(rowvalues[0]).strip()
        # input(value)
        # input(pre_data_row_firstcol_summary)

        pattern, symbols, case, num_tokens, num_chars = generate_pattern_symbols_and_case(value, True)
        
        disagreements = find_disagreement(pre_data_row_firstcol_summary.fw_pattern_summary,
         pre_data_row_firstcol_summary.bw_pattern_summary, pre_data_row_firstcol_summary.strict_chain_summary,
         pattern,pre_data_row_firstcol_summary.symbol_set_summary,
         symbols, pre_data_row_firstcol_summary.consistent_case_summary, case)

        # print('\ndisagreements')
        # input(disagreements)
        # print('\nnum_chars='+str(num_chars))
        # print('pre_data_row_firstcol_summary.maxchar='+str(pre_data_row_firstcol_summary.maxchar))
        # print('1.2*pre_data_row_firstcol_summary.maxchar='+str(1.2*pre_data_row_firstcol_summary.maxchar))
        # TODO use agreements as well
        if len(disagreements)>0 or num_chars>1.2*pre_data_row_firstcol_summary.maxchar or 'note' in value.lower() or 'nota' in value.lower() or 'source' in value.lower():            
            return postmetastart 
    return postmetastart+1

def asses_candidate_predatarow(candidate_pre_data_rows, candidate_index):
    # print('asses_candidate_predatarow')
    header_detection_rule= ''
    # see if two rows above works as header with this row as explanations
    if candidate_index-1>=0:
        candidate_pdr = candidate_pre_data_rows[candidate_index-1:candidate_index].values[0] 
        # print(candidate_pdr)   
        if assess_candidate_header(candidate_pdr):
            header_index = [candidate_index-1, candidate_index+1] 
            header_detection_rule = 'second row above the first data row qualifies as a header (-2)'
            return header_index, header_detection_rule    

    # check if this row qualifies as a header
    # print('check if this row qualifies as a header')
    # print(candidate_pre_data_rows[candidate_index:candidate_index+1])
    candidate_pdr = candidate_pre_data_rows[candidate_index:candidate_index+1].values[0]
    # input('candidate_pdr='+str(candidate_pdr))
    if assess_candidate_header(candidate_pdr):
        header_index = [candidate_index, candidate_index+1] 
        header_detection_rule = 'first row above the first data row qualifies as a header (-1)'
        # print(header_detection_rule)
        return header_index, header_detection_rule

    #elif num_columns-this_row_num_nulls<= 0.7*(num_columns-next_row_num_nulls):#nonnulls!!!!
        #print('Evaluate this_row_non_nulls= '+str(num_columns-this_row_num_nulls)+'<=  next_row_non_nulls= 0.7*'+str(num_columns-next_row_num_nulls))
    #    datastart_found = False
    #    break
        
    elif str(candidate_pdr[0]).strip() == '' or str(candidate_pdr[0]).strip() == '' and str(candidate_pdr[1]).strip() == '':
        # print('\ntry two row combo')
        if candidate_index-1>=0:
            tworow_combo = combo_row(candidate_pre_data_rows[candidate_index-1:candidate_index+1])            
            if assess_candidate_header(tworow_combo):
                # print('\nhey')
                header_index = [candidate_index-1,candidate_index+1]
                header_detection_rule = 'two row combo with no spaces'
                # print(header_detection_rule)
                return header_index, header_detection_rule
            elif len(set(tworow_combo[2:]))==len(tworow_combo[2:]) and '' not in tworow_combo[2:]:
                header_index = [candidate_index-1,candidate_index+1]
                header_detection_rule = 'two row combo with up to two spaces in the beginning'
                # print(header_detection_rule)
                return header_index, header_detection_rule
        #that didnt work, see if three make a good combo
        # print('\n\nthat didnt work, see if three make a good combo\n')
        if candidate_index-2>=0:
            if assess_candidate_header(combo_row(candidate_pre_data_rows[candidate_index-2:candidate_index+1])):
                header_index = [candidate_index-2,candidate_index+1]
                header_detection_rule = 'three row combo'
                # print(header_detection_rule)
                return header_index, header_detection_rule
            else:
                header_index = [candidate_index+1,candidate_index+1]
                header_detection_rule = 'no header detected in the three rows above the data'
                # print(header_detection_rule)
                return header_index, header_detection_rule
        else:

            header_index = [candidate_index+1,candidate_index+1]
            header_detection_rule = 'only one candidate header above the data that didnt pass the header assessment'
            # print(header_detection_rule)
            return header_index, header_detection_rule
    else:#no header just metadata
        header_index = [candidate_index+1,candidate_index+1]
        header_detection_rule = 'no header detected'
        # print(header_detection_rule)
        return header_index, header_detection_rule

def dequote(s):
    """
    If a string has single or double quotes around it, remove them.
    Make sure the pair of quotes match.
    If a matching pair of quotes is not found, return the string unchanged.
    """
    if len(s)>2 and (s[0] == s[-1]) and s.startswith(("'", '"')):
        return s[1:-1]
    return s

def combo_row(rows):
    rows = rows.values
    combo_row = list(rows[0])
    for rowidx, row in enumerate(rows):
        if rowidx==0:
            continue
        buffer_row=rows[rowidx-1]
        buffer_row = [i.strip() if i!=None else '' for i in buffer_row]
        for idx, value in enumerate(buffer_row):
            if idx+1<len(buffer_row) and buffer_row[idx+1].strip() == '':
                buffer_row[idx+1] = dequote(value.strip())
        for idx, value in enumerate(row):
            if value == None:
                value = ''
            combo_row = [i.strip() if i!=None else '' for i in combo_row]
            if combo_row[idx].strip()!='':
               combo_row[idx] =  (dequote(combo_row[idx].strip())+' '+dequote(value.strip()).strip())
            else:
               combo_row[idx] =  (buffer_row[idx].strip()+' '+dequote(value.strip()).strip()) 
    return combo_row

"""
All Symbols on summary must be found in candidate to agree, and if counts are nonzero they must agree too.
Otherwise candidate and summary disagree
"""
def pattern_disagrees(value_pattern_summary, cand_pattern):
    if len(value_pattern_summary)>len(cand_pattern):
        return True
    else:
        for idx, tup in enumerate(value_pattern_summary):
            if tup[0]!=cand_pattern[idx][0]:
                return True
            else:
                if tup[1] !=0 and cand_pattern[idx][1] !=tup[1]:
                    return True
                else:
                    continue
    
    return False


    

def symbol_summary_disagrees(value_symbol_summary,cand_symbols ):
    #if candidate pattern does not contain ALL summary sumbols disagree (TRUE)
    for symbol in value_symbol_summary:
        if symbol not in list(cand_symbols):
            return True #at least one summary symbol found that does not exist in candidate
    return False

def symbol_chain_disagrees(value_symbol_chain,cand_symbol_chain):    
    if cand_symbol_chain != value_symbol_chain:
        return True

def find_disagreement(value_pattern_summary,value_pattern_BW_summary, value_chain_consistent, cand_pattern,value_symbol_summary, cand_symbols, case_summary, cand_case ):
    disagreements = []
    cand_symbol_chain = [x[0] for x in cand_pattern]
    value_symbol_chain = [x[0] for x in value_pattern_summary]

    if len(value_pattern_summary)!=0 and pattern_disagrees(value_pattern_summary, cand_pattern):
        disagreements.append("FW") 
        # print('FW disagreement')
        # print('\tFW_cand_pattern='+str(cand_pattern))
        # print('\tvalue_pattern_FW_summary='+str(value_pattern_summary))

    if len(value_pattern_BW_summary)!=0 and pattern_disagrees(value_pattern_BW_summary, list(reversed(cand_pattern))):
        disagreements.append("BW") 
        # print('BW disagreement')
        # print('\tBW_cand_pattern='+str(list(reversed(cand_pattern))))
        # print('\tvalue_pattern_BW_summary='+str(value_pattern_BW_summary))

    if value_chain_consistent and symbol_chain_disagrees(value_symbol_chain,cand_symbol_chain):
        disagreements.append("SC")
        # print('SC disagreement')
        # print('\tcand_symbol_chain='+str(cand_symbol_chain))
        # print('\tvalue_symbol_chain='+str(value_symbol_chain))

    if len(value_symbol_summary)!=0 and symbol_summary_disagrees(value_symbol_summary,cand_symbols):
        disagreements.append("SS") 
        # print('SS disagreement')
        # print('\tcand_symbols='+str(cand_symbols))    
        # print('\tvalue_symbol_summary='+str(value_symbol_summary))   

    if len(case_summary)!=0 and case_summary!=cand_case:
        disagreements.append("CC")
        # print('CC disagreement')
        # print('\tcand_case='+str(cand_case))
        # print('\tcase_summary='+str(case_summary))

    return disagreements

def decimal_pattern(column_symbols, column_patterns):
    decimal_spotted = False

    for value_index, value_symbols in enumerate(column_symbols):
        count_decimal = 0
        count_decimal = count_decimal+[x[0] for x in column_patterns[value_index]].count('.')
        count_decimal = count_decimal+[x[0] for x in column_patterns[value_index]].count(',')
        if len(value_symbols)==0:
            continue
        if 'D' in value_symbols and count_decimal==1 and value_symbols.issubset(set(['D',',','.','-','S'])) and (([x[0] for x in column_patterns[value_index]].count('-') ==1 and len(column_patterns[value_index])>0 and column_patterns[value_index][0][0]=='-') or ([x[0] for x in column_patterns[value_index]].count('-') ==0)):
            # column_patterns[value_index][0][0] 
            decimal_spotted = True

        elif ('D' in value_symbols and count_decimal==1 and value_symbols.issubset(set(['D',',','.','-','S'])) and (([x[0] for x in column_patterns[value_index]].count('-') ==1 and len(column_patterns[value_index])>0 and column_patterns[value_index][0][0]=='-') or ([x[0] for x in column_patterns[value_index]].count('-') ==0))) == False:
            return False

    if decimal_spotted:
        return True
    else:
        return False

def numeric_patterns(column_symbols, column_patterns):
    numeric_spotted = False
    # iterate over all cells
    for value_index, value_symbols in enumerate(column_symbols):
        if len(value_symbols)==0:# this cell was empty, move on
            continue

        if 'D' in value_symbols and value_symbols.issubset(set(['D',',','.','-','S'])) and (([x[0] for x in column_patterns[value_index]].count('-') ==1 and len(column_patterns[value_index])>0 and column_patterns[value_index][0][0]=='-') or ([x[0] for x in column_patterns[value_index]].count('-') ==0)):
            numeric_spotted = True #confirm i found a number, keep checking next values

        elif ('D' in value_symbols and value_symbols.issubset(set(['D',',','.','-','S'])) and (([x[0] for x in column_patterns[value_index]].count('-') ==1 and len(column_patterns[value_index])>0 and column_patterns[value_index][0][0]=='-') or ([x[0] for x in column_patterns[value_index]].count('-') ==0))) == False:
            return False # I saw something that isnt a number, don't bother checking anything else

    if numeric_spotted:
        return True
    else:
        return False



# def generate_all_patterns_numeric_summary(patterns_numeric, train_lengths):
def generate_all_numeric_sig_pattern(patterns_numeric, train_lengths):
    valid_indexes = [i for i, e in enumerate(train_lengths) if e != 0]
    if len(valid_indexes)>0:
        return np.all(np.array([patterns_numeric[i] for i in valid_indexes])), len(valid_indexes)
    return False, len(valid_indexes)

def numeric_train_incremental_pattern(numeric_train_sig, len_train_sig, column_is_numeric_pattern):
    column_is_numeric, len_valid_indexes = column_is_numeric_pattern

    if column_is_numeric==False:
        if len_valid_indexes==0 and len_train_sig>0:
            len_valid_indexes+=1
            return numeric_train_sig, len_valid_indexes   
        else:
            return column_is_numeric, len_valid_indexes
    else:
        if len_train_sig>0:
            len_valid_indexes+=1
            column_is_numeric = numeric_train_sig
        return column_is_numeric, len_valid_indexes 




def ratio_patterns(column_symbols):
    ratio_spotted = False
    for value_symbols in column_symbols:
        if len(value_symbols)==0:
            continue
        if 'D' in value_symbols and '%' in value_symbols and value_symbols.issubset(set(['D',',','.','S'])):
            ratio_spotted = True

        elif ('D' in value_symbols and '%' in value_symbols and value_symbols.issubset(set(['D',',','.','S']))) == False:
            return False

    if ratio_spotted:
        return True
    else:
        return False

def strong_pattern_agreement(values_dataframe,sample_patterns,sample_symbols, sample_cases,row_values, p=0.4, a=0.5, w=0.2):#, p=0.5, PA=1, PB=10):, a was 0.2
    strong_aggreement = False #initialize
    agreeing_columns={}
    count_value_repetition = 0
    dataframe_labels = []
    step_count= 0
    data_confidence=0

    for column in values_dataframe:
        dataframe_labels.append(column)

    for column in values_dataframe:
        agreeing_columns[dataframe_labels.index(column)]= {}
        non_empty_patterns=0
        column_patterns = sample_patterns[dataframe_labels.index(column)]
        column_symbols = sample_symbols[dataframe_labels.index(column)]

        if len(column_patterns)>0 and column_patterns[0]!=[]:
            for pattern in column_patterns:
                if pattern!=[]:
                    non_empty_patterns+=1
            #there is no point calculating agreement over one value, a single value always agrees with itself.
            #in addition, many tables have bilingual headers, so agreement between two header values is very common, require nij>=3
            if (len(column_patterns)>2 and non_empty_patterns>=3) or (len(column_patterns)==2 and non_empty_patterns==2):

                # POPULATION_WEIGHT = 1/(1+(PB/PA)*p**non_empty_patterns)
                POPULATION_WEIGHT = 1-(1-p)**non_empty_patterns
                columnvalues = values_dataframe.iloc[:,dataframe_labels.index(column)].tolist()
                candidate =columnvalues[0]

                if candidate==None  or str(candidate).strip() =='':
                    continue #No point looking for agreements, move along

                # value repeats in the rest of the column
                if len(columnvalues)>1 and columnvalues[1:].count(candidate)>=2:
                    agreeing_columns[dataframe_labels.index(column)]["rule"] = 'value repetition'
                    strong_aggreement = True
                    agreeing_columns[dataframe_labels.index(column)]["confidence"] = float(0.8 *POPULATION_WEIGHT)              
                    #dont bother checking for other conditions
                    continue

                all_values_summary,consistent_symbol_chain= generate_pattern_summary(column_patterns) 
                bw_patterns =  [list(reversed(pattern)) for pattern in column_patterns]
                pattern_BW_summary,_=generate_pattern_summary(bw_patterns)

                if  consistent_symbol_chain==True and len(all_values_summary)>=3 and (len(list(set([i[0] for i in all_values_summary]).difference(['A','D','S','_'])))>0):
                    strong_aggreement = True 
                    agreeing_columns[dataframe_labels.index(column)]= {}
                    agreeing_columns[dataframe_labels.index(column)]["confidence"]  = float(0.8*POPULATION_WEIGHT)
                    agreeing_columns[dataframe_labels.index(column)]["rule"] = 'strong FW pattern agreement'
                    continue

                
                if  consistent_symbol_chain==True and len(all_values_summary)>=2 and len([i[1] for i in all_values_summary if i[1]>=2])>0:
                    strong_aggreement = True 
                    agreeing_columns[dataframe_labels.index(column)]= {}
                    agreeing_columns[dataframe_labels.index(column)]["confidence"]  = float(0.8*POPULATION_WEIGHT)
                    agreeing_columns[dataframe_labels.index(column)]["rule"] = 'strong symbol chain agreement'
                    continue

                if  consistent_symbol_chain==True and len(all_values_summary)==1 and all_values_summary[0][0] == 'D' and all_values_summary[0][1]>=3:
                    strong_aggreement = True 
                    agreeing_columns[dataframe_labels.index(column)]= {}
                    agreeing_columns[dataframe_labels.index(column)]["confidence"]  = float(0.8*POPULATION_WEIGHT)
                    agreeing_columns[dataframe_labels.index(column)]["rule"] = 'consistent number of digits > 3'
                    continue

                if consistent_symbol_chain==True and len(all_values_summary) in [1,2] and  all_values_summary[0][0] == 'D':
                    agreeing_columns[dataframe_labels.index(column)]= {}
                    agreeing_columns[dataframe_labels.index(column)]["confidence"]  = float(0.7*POPULATION_WEIGHT)
                    agreeing_columns[dataframe_labels.index(column)]["rule"] = 'up to two symbols, the first is a digit'
                    continue

                if numeric_patterns(column_symbols, column_patterns):
                    agreeing_columns[dataframe_labels.index(column)]= {}
                    agreeing_columns[dataframe_labels.index(column)]["confidence"]  = float(0.7*POPULATION_WEIGHT)
                    agreeing_columns[dataframe_labels.index(column)]["rule"] = 'all values digits, optionally have . or , '
                    continue
                if ratio_patterns(column_symbols):
                    agreeing_columns[dataframe_labels.index(column)]= {}
                    agreeing_columns[dataframe_labels.index(column)]["confidence"]  = float(0.7*POPULATION_WEIGHT)
                    agreeing_columns[dataframe_labels.index(column)]["rule"] = 'all values digits with \%,  optionally have . or , '
                    continue
                if len(all_values_summary)>=3 and 'S' not in [x[0] for x in all_values_summary]:
                    agreeing_columns[dataframe_labels.index(column)]= {}
                    agreeing_columns[dataframe_labels.index(column)]["confidence"]  = float(0.6*POPULATION_WEIGHT)
                    agreeing_columns[dataframe_labels.index(column)]["rule"] = 'three or above symbols in FW summary that do not contain a Space'
                    continue
                if len(pattern_BW_summary)>=3 and 'S' not in [x[0] for x in pattern_BW_summary]:
                    agreeing_columns[dataframe_labels.index(column)]= {}
                    agreeing_columns[dataframe_labels.index(column)]["confidence"]  = float(0.6*POPULATION_WEIGHT)
                    agreeing_columns[dataframe_labels.index(column)]["rule"] = 'three or above symbols in BW summary that do not contain a Space'
                    continue
                if  len(sample_symbols)>=2 and {'S','_'}.intersection(sample_symbols)==0:
                    agreeing_columns[dataframe_labels.index(column)]= {}
                    agreeing_columns[dataframe_labels.index(column)]["confidence"]  = float(0.4*POPULATION_WEIGHT)
                    agreeing_columns[dataframe_labels.index(column)]["rule"] = "at least two symbols in the symbol summary, none of which are S or _"
                    continue
                if consistent_symbol_chain==True and len(all_values_summary)>=3:
                    agreeing_columns[dataframe_labels.index(column)]= {}
                    agreeing_columns[dataframe_labels.index(column)]["confidence"]  = float(0.3*POPULATION_WEIGHT)
                    agreeing_columns[dataframe_labels.index(column)]["rule"] = 'three or more symbols consistent chain'
                    continue

                if consistent_symbol_chain==True and len(all_values_summary) in [1,2]:
                    agreeing_columns[dataframe_labels.index(column)]= {}
                    agreeing_columns[dataframe_labels.index(column)]["confidence"]  = float(0.2*POPULATION_WEIGHT)
                    agreeing_columns[dataframe_labels.index(column)]["rule"] = 'one or two symbols consistent chain' #eg, words
                    continue
                
                if len(all_values_summary)>=2 and 'S' not in [x[0] for x in all_values_summary]:
                    agreeing_columns[dataframe_labels.index(column)]= {}
                    agreeing_columns[dataframe_labels.index(column)]["confidence"]  = float(0.2*POPULATION_WEIGHT)
                    agreeing_columns[dataframe_labels.index(column)]["rule"] = 'two or above symbols in FW summary that do not contain a Space'
                    continue
                if len(pattern_BW_summary)>=2 and 'S' not in [x[0] for x in pattern_BW_summary]:
                    agreeing_columns[dataframe_labels.index(column)]= {}
                    agreeing_columns[dataframe_labels.index(column)]["confidence"]  = float(0.2*POPULATION_WEIGHT)
                    agreeing_columns[dataframe_labels.index(column)]["rule"] = 'two or above symbols in BW summary that do not contain a Space'
                    continue
            
    column_agreement_confidence = [d['confidence'] for d in list(agreeing_columns.values()) if 'confidence' in d]
    score_counts = {x:column_agreement_confidence.count(x) for x in column_agreement_confidence}   
        
    prod_list=[]
    if len(score_counts)>0:
        for score, count in score_counts.items():
            prod_list.append((1-score)**count)

        if len(row_values)>0:
            step_increment, step_count = discover_incremental_values(row_values[0], sample_symbols)
            if step_increment!=0 and step_count>=3:
                INCREMENTAL_ATTRIBUTES_WEIGHT = a**(step_count-1)  
                data_confidence = w*(1- np.prod(prod_list))+(1-w)*INCREMENTAL_ATTRIBUTES_WEIGHT              
            else:
                data_confidence = 1- np.prod(prod_list)
    else:
        data_confidence=0

    return strong_aggreement, agreeing_columns, data_confidence

def combined_row_confidence(predata_row_confidence,data_confidence,candidate_index):
    DEPTH_WEIGHT = (1/2)**(candidate_index/10)
    max_c = max(predata_row_confidence,data_confidence)
    min_c = min(predata_row_confidence,data_confidence)

    # version 1 and 2 were rejected as they allow predata_row_confidence to influence the decision too much

    # formula version_1
    # combined_row_confidence = DEPTH_WEIGHT * min(1,max_c+min_c/2) #rejected

    # formula version 2
    # combined_row_confidence = DEPTH_WEIGHT * (max_c+(1-max_c)*min_c) #rejected

    # formula version 3
    # combined_row_confidence = DEPTH_WEIGHT * (max_c+min_c)/2 #this seems to work

    # formula version 4
    # This gives more weight to the confidence in agreements of columns, and takes into acount
    # disagreements if they are very strong

    # if data_confidence<predata_row_confidence:
    #     combined_row_confidence =  DEPTH_WEIGHT * data_confidence
    # else:
    #     combined_row_confidence =  DEPTH_WEIGHT * (predata_row_confidence+data_confidence)/2
    combined_row_confidence = DEPTH_WEIGHT * (predata_row_confidence+data_confidence)/2
    # combined_row_confidence =  DEPTH_WEIGHT *predata_row_confidence*data_confidence

    # use the geometric mean to smoothen out the results
    # combined_row_confidence =  DEPTH_WEIGHT *math.sqrt(predata_row_confidence*data_confidence)

    return float(combined_row_confidence)

def calculate_row_summary_disagreement(disagreeing_columns,lineindex,row_values, sample_symbols, a= 0.5, method='precise'):#a= 0.2
    predata_row_confidence = 0

    step_increment, step_count = discover_incremental_values(row_values, sample_symbols)
    INCREMENTAL_ATTRIBUTES_WEIGHT= 1
    if step_increment!=0 and step_count>=3:
        INCREMENTAL_ATTRIBUTES_WEIGHT = a**(step_count-1)

    column_confidence_values = [d['confidence'] for d in list(disagreeing_columns.values()) if 'confidence' in d]
    number_of_disagreeing_attributes = len(disagreeing_columns) 
    #approximation
    if len(column_confidence_values)!=0 and method=='avg':
        # using average
        predata_row_confidence = 1-(1-(sum(column_confidence_values) / len(column_confidence_values)))**number_of_disagreeing_attributes

    #exact
    elif len(column_confidence_values)!=0 and method=='precise':
    #demorgan, snorm, tconorm

        score_counts = {x:column_confidence_values.count(x) for x in column_confidence_values}   
        prod_list=[]
        for score, count in score_counts.items():
            prod_list.append((1-score)**count)
        predata_row_confidence = 1- np.prod(prod_list)*INCREMENTAL_ATTRIBUTES_WEIGHT

    elif len(column_confidence_values)==0:
        predata_row_confidence = 1-INCREMENTAL_ATTRIBUTES_WEIGHT

    return predata_row_confidence

def disagreement_confidence(disagreements, nonnull_values, p=0.4):#, p=0.5, PA=1, PB=10): 
    summary_disagreement_score = 0;
    if len(disagreements)==1 and disagreements[0] == "SC":
        summary_disagreement_score = 0.61
    elif  len(disagreements)==1 and (disagreements[0]=="FW" or disagreements[0]=="BW"  or disagreements[0]=="CC" or disagreements[0]=="SS"):
        summary_disagreement_score = 0.63
    elif len(disagreements)==2:
        summary_disagreement_score = 0.65
    elif len(disagreements)==3:
        summary_disagreement_score= 0.67
    elif len(disagreements)>=4:
        summary_disagreement_score = 0.69
    
    POPULATION_WEIGHT = 1-(1-p)**nonnull_values
    
    # if nonnull_values==0:
    #     POPULATION_WEIGHT = 0
    # else:
    #     POPULATION_WEIGHT = POPULATION_WEIGHT = 1/(1+(PB/PA)*p**nonnull_values)

    confidence = (summary_disagreement_score*POPULATION_WEIGHT)
    return float(confidence)

def discover_dataend_idx(candidate_datatuples):
    lenfirst = len(candidate_datatuples[0])
    tuples_reversed= list(reversed(candidate_datatuples))
    for tpl_idx, tpl in enumerate(tuples_reversed):
        if len(tpl) == lenfirst:
            return len(candidate_datatuples)-1-tpl_idx;

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

def row_patterns(attributes):
    attribute_info= {}
    for idx, attribute in enumerate(attributes):
        p, s, c, tl, cl = generate_pattern_symbols_and_case(attribute, False)
        attribute_info[idx]= {}
        attribute_info[idx]["pattern"]= p
        attribute_info[idx]["symbols"] = sorted(s)
        attribute_info[idx]["case"] = c
        attribute_info[idx]["token length"] = tl
        attribute_info[idx]["character length"] = cl
        
    return attribute_info

def process_file(filepath):
    # print('\n\n\n-_-_-_-_-_-_- FILE-_-_-_-_-_-_-\n')
    print('\nProcessing '+filepath+'\n')
    blanklines=[]
    unprocessed_files = []
    delimiter = ''
    post_metafiles = {}
    pre_metafiles = {}
    datafiles = []
    dataheaders = {}    
    tables = []
    premeta = {}
    postmeta = {}
    header = {}
    data = {}
    deciding_row = {}
    first_disagreeing_row = {}
    premeta_text = []
    failure=""
    null_equi_spotted= []
    size_bytes = 0# was 0
    num_rows = 0# was 0
    

    if os.path.exists(filepath)==False:
        failure= "file does not exist"
        return None, None, None, None, None, None,  failure,blanklines,null_equi_spotted
    size_bytes = os.path.getsize(filepath)

    if size_bytes == 0:
        failure= "file is empty"
        return None, None, None, None, None, None,  failure,blanklines,null_equi_spotted

    encoding_result = detect_encoding(filepath)
    print('Encoding:'+str(encoding_result))
    encoding = encoding_result["encoding"]
    if encoding==None:
        failure = "No encoding discovered"
        return None, None, None, None, None, None, failure,blanklines,null_equi_spotted
    encoding_confidence = encoding_result["confidence"]
    if "language" in encoding_result.keys():
        encoding_language = encoding_result["language"]
    else:
        encoding_language = ''

    if discard_file(filepath, encoding)==True:
        failure = "illegal file format"
        return None, None, None, None, None, None,  failure,blanklines,null_equi_spotted

    # discover delimiter
    delimiter = discover_delimiter(filepath, encoding)
    singletable = True
    batch=[]
    
    max_batch= 200
    lineindex=0
    with codecs.open(filepath,'rU', encoding=encoding) as f:        
        chunk = f.read(min(size_bytes,100000))
        if chunk:
            for line in csv.reader(chunk.split("\n"), quotechar='"', delimiter= delimiter, skipinitialspace=True):
                # print(line)
                lineindex+=1
                if len(line) ==0 or sum(len(s.strip()) for s in line)==0:
                    blanklines.append(lineindex)
                batch.append(line)
                if len(batch)>=max_batch:
                    break
            if len(blanklines)==0:
                # print('Assume this file has no blank lines')
                singletable = True 
            else:
                # print('blank line indexes = '+str(blanklines))
                singletable = False
        f.flush()

    # print('delimiter= '+delimiter)
    if singletable == False:        
        with codecs.open(filepath,'rU', encoding=encoding) as f:
            csv_reader = csv.reader(f, delimiter= delimiter, skipinitialspace=True, quotechar='"')
            csv_tuples = list(csv_reader)
        num_rows, file_blank_rowindex = file_info(csv_tuples)
        lastconsecutive = 0

        # input('file_blank_rowindex[-1]='+str(file_blank_rowindex[-1]))

        if len(file_blank_rowindex)>1:
            rev_blanklines = file_blank_rowindex.copy()
            rev_blanklines.reverse()
            for index, i in enumerate(rev_blanklines):
                if index+1<len(rev_blanklines) and i != rev_blanklines[index+1]+1:
                    lastconsecutive = index
                    break
                elif index+1==len(rev_blanklines) and i+1 == rev_blanklines[index-1]:
                    lastconsecutive = index
                    break

        if len(file_blank_rowindex)>1 and num_rows == file_blank_rowindex[-1]+1 and lastconsecutive>0:
            num_rows = file_blank_rowindex[-lastconsecutive]+1
            file_blank_rowindex = file_blank_rowindex[0:-lastconsecutive]           

        singletable=False
        if len(file_blank_rowindex)== 0:
            singletable = True

        metadataseen = False
        relation_counter = 1
        offset = 0
        sample_lines_limit = 25
        more_tables_in_file = True
        datastart = 0
        
        while more_tables_in_file and offset<num_rows:
            premeta = {}
            postmeta = {}
            header = {}
            data = {}
            deciding_row = {}
            first_disagreeing_row = {}
            premeta_text = []
            
            table_info = split_metadata_data(csv_tuples, offset, sample_lines_limit, delimiter)

            if "datastart" in table_info.keys():
                datastart = table_info["datastart"]
            else:
                datastart = table_info["metadataend"]


            if table_info["metadatastart"] != datastart:    
                print('/ / / / / / / WEEEEEEE This table has pre_metadata!')
                metadataseen = True
                metaend = table_info["metadataend"]
                metalines = []
                premeta_tuples = []
                for l in csv_tuples[table_info["metadatastart"]:metaend+1]:
                    metalines.append(delimiter.join(l).replace('\n', ' ').replace('\r', ''))
                    premeta_tuples.append([str(i).strip() for i in l])
                pre_metafiles[relation_counter] = {}

                # pre_metafiles[relation_counter]["lines"]=metalines
                pre_metafiles[relation_counter]["lines"]=premeta_tuples

                pre_metafiles[relation_counter]["file_index"]=table_info["metadatastart"]

                offset = metaend+1
                continue
            if sum(i > datastart for i in file_blank_rowindex)>0:
                print('\n(i) Not on the last table\n')
                metadataseen = False

                for lineindex in file_blank_rowindex:
                    if lineindex>datastart:
                        candidate_dataend = lineindex
                        break
                datalines = []
                candidate_datatuples = csv_tuples[datastart:candidate_dataend]                
                dataend = discover_dataend_idx(candidate_datatuples)

                if dataend == 0:
                    pre_meta= candidate_datatuples[0:1]
                    if relation_counter in pre_metafiles.keys():
                        for pre_metarow in [delimiter.join(x).replace('\n', ' ').replace('\r', '') for x in pre_meta]:
                            pre_metafiles[relation_counter]["lines"].append(pre_metarow)

                    else:
                        pre_metafiles[relation_counter]={}
                        pre_metafiles[relation_counter]["lines"]= [delimiter.join(x).replace('\n', ' ').replace('\r', '') for x in pre_meta]
                        pre_metafiles[relation_counter]["file_index"]= offset
                    offset =  datastart +2 
                    continue

                datatuples = candidate_datatuples[0:dataend+1]
                post_meta=csv_tuples[datastart+len(datatuples)+1:candidate_dataend]
                post_metafiles[relation_counter] = post_meta
                
                df = pd.DataFrame(datatuples)
                df = df.loc[:, (df != '').any(axis=0)]
                df = df.dropna(axis='columns', how='all')
                emptycolumns = []
                for column in df:
                    if all(np.where(df[column].str.strip()=='', True, False)):
                        emptycolumns.append(column)
                df=df.drop(emptycolumns, axis=1)
                datastart_index,dataend_index, headerindex, row_confidence, column_summaries, disagreeing_columns,agreeing_columns, first_disagreeing_row,header_detection_rule,null_equi_spotted, sorted_conf, outlier_aware_summaries, outliers = discover_header_and_datastart(df)
                # LOG
                print('column_summaries=')
                for sidx, summary in enumerate(column_summaries):
                    print('column_'+str(sidx))
                    summary.print_summary()

                dataframe = df[datastart_index:dataend_index]

                headertext=[]
                if headerindex[0]!=headerindex[1]:
                    headertext=df[headerindex[0]:headerindex[1]].values.tolist()
                if headerindex[1] == 0:
                    premetaend = datastart_index
                else:
                    premetaend = headerindex[0]
                metalines=[]
                premeta_tuples = []
                if relation_counter in pre_metafiles.keys():
                    # metalines = pre_metafiles[relation_counter]["lines"]
                    premeta_tuples = pre_metafiles[relation_counter]["lines"]
                    #ADD PRE DATA LINES TO PREMETA
                for row in df[0:premetaend].values.tolist():
                    # metalines.append( delimiter.join([str(i) for i in row]).replace('\n', ' ').replace('\r', ''))
                    premeta_tuples.append([str(i).strip() for i in row])


                if relation_counter not in pre_metafiles.keys():
                    pre_metafiles[relation_counter]={}

                # pre_metafiles[relation_counter]["lines"] = metalines
                pre_metafiles[relation_counter]["lines"] = premeta_tuples
                if "file_index" not in pre_metafiles[relation_counter].keys():
                    pre_metafiles[relation_counter]["file_index"] = offset
                # premeta_text = metalines 
                premeta_text = premeta_tuples              

                if relation_counter in pre_metafiles.keys() :
                    premeta["from_line_idx"]= pre_metafiles[relation_counter]["file_index"]

                premeta["to_line_idx"]= offset+premetaend
                # premeta["premeta_text"]= [delimiter.join(x) for x in csv_tuples[premeta["from_line_idx"]:premeta["to_line_idx"]]]#premeta_text
                premeta["premeta_text"]= csv_tuples[premeta["from_line_idx"]:premeta["to_line_idx"]]#premeta_text


                header["from_line_idx"]= offset+headerindex[0]
                header["to_line_idx"]= offset+headerindex[1]
                header["header_text"]= df[headerindex[0]: headerindex[1]].values.tolist()#csv_tuples[header["from_line_idx"]:header["to_line_idx"]]
                header["header_rule"]= header_detection_rule
                if headerindex[0]!= headerindex[1]:
                    header["combo"] = combo_row(df[headerindex[0]: headerindex[1]])                
                    header["attribute_patterns"] = row_patterns(header["combo"])

                data["from_line_idx"]= offset+datastart_index
                data["to_line_idx"]= offset+dataend+1
                data["dataframe"]= dataframe

                data["first_column_has_emptyvalues"]= False
                data["first_two_columns_have_emptyvalues"] = False
                try:
                    for x in dataframe.iloc[:,0].values:
                        if x== None or x.strip() == '':
                            data["first_column_has_emptyvalues"]= True
                            try:
                                for y in dataframe.iloc[:,1].values:
                                    if y== None or y.strip()== '':
                                        data["first_two_columns_have_emptyvalues"] = True
                                        break
                            except:
                                data["first_two_columns_have_emptyvalues"] = False
                            break
                except:
                    data["first_column_has_emptyvalues"]= False
                    data["first_two_columns_have_emptyvalues"] = False

                data["large_table"]=False
                data["num_columns"]= df.shape[1]
                
                data["five_lines"]= df[data["from_line_idx"]:min((df.shape[0]-data["from_line_idx"]-1),5)].to_csv(sep=delimiter, index=False,header=False)

                postmeta["from_line_idx"]= offset+datastart_index+dataframe.shape[0]
                postmeta["to_line_idx"]= candidate_dataend
                postmeta["postmeta_text"]= csv_tuples[postmeta["from_line_idx"]:postmeta["to_line_idx"]]#post_meta

                
                deciding_row["line_idx"]= offset+datastart_index
                deciding_row["disagreeing_columns"]=disagreeing_columns
                deciding_row["column_summaries"]= column_summaries
                deciding_row["agreeing_columns"]= agreeing_columns            
                deciding_row["row_confidence"]=row_confidence
                if len(first_disagreeing_row)>0:
                    first_disagreeing_row["line_idx"] = offset+first_disagreeing_row["index"]
                    first_disagreeing_row.pop("index")
                
                datatable = DataTable(premeta, header, data, postmeta, deciding_row, first_disagreeing_row, sorted_conf, outlier_aware_summaries,outliers)
                relation_counter+=1  

                offset =  candidate_dataend +1
                while offset in file_blank_rowindex:
                    offset+=1
                

            else: #read to end of file
                print('(i) On the last table\n')
                if singletable and metadataseen==False:
                    print('(i_a) singletable and pre_metadataseen==False')
                    df = pd.DataFrame(csv_tuples)
                    df = df.loc[:, (~df.isin(['',' ', None])).any(axis=0)]
                    df = df.dropna(axis='columns', how='all')
                    emptycolumns = []
                    for column in df:
                        if all(np.where(df[column].str.strip()=='', True, False)):
                            emptycolumns.append(column)
                    
                    df=df.drop(emptycolumns, axis=1)

                    datastart_index, dataend_index, headerindex, row_confidence, column_summaries,disagreeing_columns,agreeing_columns, first_disagreeing_row,header_detection_rule,null_equi_spotted, sorted_conf,outlier_aware_summaries, outliers= discover_header_and_datastart(df)
                    # LOG
                    print('datastart_index='+str(datastart_index))
                    input('column_summaries='+str(column_summaries))                    

                    dataframe = df[datastart_index:dataend_index]
                    candidate_dataend = num_rows
                    candidate_datatuples = csv_tuples[datastart:candidate_dataend]
                    dataend = discover_dataend_idx(candidate_datatuples)
                    if headerindex[0]!=headerindex[1]:
                        dataheaders[relation_counter] = df[headerindex[0]:headerindex[1]].values.tolist()
                    if headerindex[1] == 0:
                        premetaend = datastart_index
                    else:
                        premetaend = headerindex[0]
                    if relation_counter in pre_metafiles.keys():
                        # metalines = delimiter.join(pre_metafiles[relation_counter])
                        premeta_tuples = pre_metafiles[relation_counter]
                        for row in df[0:premetaend].values.tolist():
                            # metalines.append( delimiter.join([str(i) for i in row]).replace('\n', ' ').replace('\r', ''))
                            premeta_tuples.append([str(i).strip() for i in row])
                        # pre_metafiles[relation_counter] = metalines
                        pre_metafiles[relation_counter] = premeta_tuples

                    datatuples = csv_tuples[datastart:]
                    post_meta=csv_tuples[datastart+len(datatuples)+1::]

                    datatable = DataTable(premeta, header, data, postmeta, deciding_row, first_disagreeing_row, sorted_conf, outlier_aware_summaries, outliers)

                else:
                    print('(i_b) singletable and metadataseen==True')

                    metadataseen = False
                    if "datastart" not in table_info.keys():
                        datastart = table_info["metadataend"]
                        print('datastart = table_info["metadataend"]')
                    else:
                        datastart = table_info["datastart"]
                        print('datastart = table_info["datastart"]')
                        
                    candidate_dataend = num_rows
                    candidate_datatuples = csv_tuples[datastart:candidate_dataend]
                    dataend = discover_dataend_idx(candidate_datatuples)
                    datatuples = csv_tuples[datastart:]
                    post_meta=csv_tuples[datastart+len(datatuples)+1::]

                    df = pd.DataFrame(datatuples)
                    df = df.loc[:, (~df.isin(['',' ', None])).any(axis=0)]
                    df = df.dropna(axis='columns', how='all')
                    emptycolumns = []
                    for column in df:
                        if all(np.where(df[column].str.strip()=='', True, False)):
                            emptycolumns.append(column)
                    df=df.drop(emptycolumns, axis=1)

                    datastart_index,dataend_index, headerindex, row_confidence, column_summaries,disagreeing_columns,agreeing_columns, first_disagreeing_row, header_detection_rule,null_equi_spotted, sorted_conf, outlier_aware_summaries, outliers= discover_header_and_datastart(df)
                    # LOG
                    input('column_summaries='+str(column_summaries)) 

                    dataframe = df[datastart_index:dataend_index]                    
                    datafiles.append(dataframe)
                    
                    ### Headers
                    if headerindex[0]!=headerindex[1]:
                        dataheaders[relation_counter] = df[headerindex[0]:headerindex[1]].values.tolist()
                        headertext=df[headerindex[0]:headerindex[1]].values.tolist()
                    ## Pre-metadata
                    if headerindex[1] == 0:
                        premetaend = datastart_index
                    else:
                        premetaend = headerindex[0]

                    if relation_counter in pre_metafiles.keys():
                        # metalines = pre_metafiles[relation_counter]["lines"]
                        premeta_tuples = pre_metafiles[relation_counter]["lines"]
                        for row in df[0:premetaend].values.tolist():
                            if row!=None:
                                # metalines.append(delimiter.join([str(i) for i in row]).replace('\n', ' ').replace('\r', ''))
                                premeta_tuples.append([str(i).strip() for i in row])

                        # pre_metafiles[relation_counter]["lines"] = metalines
                        # premeta_text= metalines
                        pre_metafiles[relation_counter]["lines"] =premeta_tuples
                        premeta_text= premeta_tuples
                
                if relation_counter in pre_metafiles.keys() :
                    premeta["from_line_idx"]= pre_metafiles[relation_counter]["file_index"]
                else:
                    premeta["from_line_idx"]=offset

                premeta["to_line_idx"]= offset+premetaend
                # premeta["premeta_text"]= [delimiter.join(x) for x in csv_tuples[premeta["from_line_idx"]:premeta["to_line_idx"]]]
                premeta["premeta_text"]= csv_tuples[premeta["from_line_idx"]:premeta["to_line_idx"]]
                
                header["from_line_idx"]= offset+int(headerindex[0])
                header["to_line_idx"]= offset+int(headerindex[1])
                header["header_rule"]= header_detection_rule
                header["header_text"]=df[headerindex[0]: headerindex[1]].values.tolist() #csv_tuples[header["from_line_idx"]:header["to_line_idx"]]
                if headerindex[0]!= headerindex[1]:
                    header["combo"] = combo_row(df[headerindex[0]: headerindex[1]])                
                    header["attribute_patterns"] = row_patterns(header["combo"])
                

                data["from_line_idx"]= offset+datastart_index
                data["to_line_idx"]= offset+dataend+1
                data["dataframe"]= dataframe

                data["first_column_has_emptyvalues"]= False
                data["first_two_columns_have_emptyvalues"] = False
                try:
                    for x in dataframe.iloc[:,0].values:
                        if x== None or x.strip()== '':
                            data["first_column_has_emptyvalues"]= True
                            try:
                                for y in dataframe.iloc[:,1].values:
                                    if y== None or y.strip()== '':
                                        data["first_two_columns_have_emptyvalues"] = True
                                        break
                            except:
                                data["first_two_columns_have_emptyvalues"] = False
                            break
                except:
                    data["first_column_has_emptyvalues"]= False
                    data["first_two_columns_have_emptyvalues"] = False

                data["large_table"]=False
                data["num_columns"]= df.shape[1]
                data["five_lines"]= df[data["from_line_idx"]:min((df.shape[0]-data["from_line_idx"]-1),5)].to_csv(sep=delimiter, index=False,header=False)
                
                postmeta["from_line_idx"]= offset+datastart_index+dataframe.shape[0]#datastart+len(datatuples)
                postmeta["to_line_idx"]= candidate_dataend
                postmeta["postmeta_text"]= csv_tuples[postmeta["from_line_idx"]::]#post_meta

                deciding_row["line_idx"]= offset+datastart_index
                deciding_row["disagreeing_columns"]=disagreeing_columns
                deciding_row["column_summaries"]= column_summaries
                deciding_row["agreeing_columns"]= agreeing_columns            
                deciding_row["row_confidence"]=row_confidence
                if len(first_disagreeing_row)>0:
                    first_disagreeing_row["line_idx"] = offset+first_disagreeing_row["index"]
                    first_disagreeing_row.pop("index")
                else:
                    first_disagreeing_row= {}

                datatable = DataTable(premeta, header, data, postmeta, deciding_row, first_disagreeing_row, sorted_conf, outlier_aware_summaries, outliers)
                relation_counter+=1
                more_tables_in_file = False;

            # End of whole loop, add table information
            tables.append(datatable)
            if offset >=num_rows:
                break   
    else:
        print('\nKEEP IT SIMPLE\n')
        datatuples = batch[0:]
        
        df = pd.DataFrame(datatuples)
        df = df.loc[:, (~df.isin(['',' ', None])).any(axis=0)]
        df = df.dropna(axis='columns', how='all')
        emptycolumns = []
        for column in df:
            if all(np.where(df[column].str.strip()=='', True, False)):
                emptycolumns.append(column)
        df=df.drop(emptycolumns, axis=1)


        datastart_index,_, headerindex, row_confidence, column_summaries,disagreeing_columns,agreeing_columns, first_disagreeing_row, header_detection_rule,null_equi_spotted, sorted_conf, outlier_aware_summaries, outliers = discover_header_and_datastart(df)
        premeta["from_line_idx"]= 0
        premeta["to_line_idx"]= headerindex[0]
        # premeta["premeta_text"]= [delimiter.join(x) for x in batch[premeta["from_line_idx"]:premeta["to_line_idx"]]]
        premeta["premeta_text"]= batch[premeta["from_line_idx"]:premeta["to_line_idx"]]

        header["from_line_idx"]= headerindex[0]
        header["to_line_idx"]= datastart_index
        header["header_text"]= df[headerindex[0]: headerindex[1]].values.tolist()#batch[header["from_line_idx"]:header["to_line_idx"]] 
        header["header_rule"]= header_detection_rule
        if headerindex[0]!= headerindex[1]:
            header["combo"] = combo_row(df[headerindex[0]: headerindex[1]])                
            header["attribute_patterns"] = row_patterns(header["combo"])        

        
        data["from_line_idx"]= datastart_index
        data["large_table"]=True
        data["num_columns"]= df.shape[1]
        data["five_lines"]= df[data["from_line_idx"]:min((df.shape[0]-data["from_line_idx"]-1),5)].to_csv(sep=delimiter, index=False,header=False)
        dataframe = df[data["from_line_idx"]:-1]
        data["dataframe"]= dataframe
        data["first_column_has_emptyvalues"]= False
        data["first_two_columns_have_emptyvalues"] = False
        try:
            for x in dataframe.iloc[:,0].values:
                if x== None or x.strip()== '':
                    data["first_column_has_emptyvalues"]= True
                    try:
                        for y in dataframe.iloc[:,1].values:
                            if y== None or y.strip()== '':
                                data["first_two_columns_have_emptyvalues"] = True
                                break
                    except:
                        data["first_two_columns_have_emptyvalues"] = False
                    break
        except:
            data["first_column_has_emptyvalues"]= False
            data["first_two_columns_have_emptyvalues"] = False                
        
        postmeta["from_line_idx"]= 0
        postmeta["to_line_idx"]= 0
        postmeta["postmeta_text"]= ''

        deciding_row["line_idx"]= datastart_index
        deciding_row["disagreeing_columns"]=disagreeing_columns
        deciding_row["column_summaries"]= column_summaries
        deciding_row["agreeing_columns"]= agreeing_columns
        deciding_row["row_confidence"]=row_confidence

        if len(first_disagreeing_row)>0:
            first_disagreeing_row["line_idx"] = first_disagreeing_row["index"]
            first_disagreeing_row.pop("index")
        else:
            first_disagreeing_row= {}

        datatable = DataTable(premeta, header, data, postmeta, deciding_row, first_disagreeing_row, sorted_conf, outlier_aware_summaries, outliers)
        tables.append(datatable)

    return delimiter, num_rows, encoding, encoding_confidence, encoding_language, tables, failure, blanklines,null_equi_spotted
