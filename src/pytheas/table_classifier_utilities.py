import os
import sys
import traceback
import cchardet as chardet
import codecs
import pandas as pd
import numpy as np
import csv
from pytheas.pat_utilities import null_equivalent_values as null_equivalent
import pytheas.pat_utilities as pat
from pytheas.pat_utilities import line_rules
from pytheas.pat_utilities import cell_rules
from pytheas.header_events import collect_events_on_row, collect_arithmetic_events_on_row, header_row_with_aggregation_tokens
import copy
from nltk import word_tokenize
from nltk.corpus import stopwords
import string
stop = stopwords.words('french')+stopwords.words('english')+list(string.punctuation)
from langdetect import detect 
from langdetect import detect_langs
from langdetect import DetectorFactory 
DetectorFactory.seed = 0
import pprint
pp = pprint.PrettyPrinter(indent=4)
import math
import itertools


def eval_not_data_cell_rule(rule, repetitions_of_candidate, repetitions_of_neighbor, neighbor, value_pattern_summary, value_pattern_BW_summary, value_chain_consistent, value_symbol_summary, case_summary, length_summary, cand_pattern, cand_symbols, cand_case, cand_length, summary_strength, line_agreements, columnindex, line_index):

    cand_symbol_chain = [x[0] for x in cand_pattern]
    value_symbol_chain = [x[0] for x in value_pattern_summary]

    summary_min_length = length_summary["min"]
    summary_max_length = length_summary["max"]

    rule_fired = False
    if rule == "First_FW_Symbol_disagrees":        
        if len(value_pattern_summary)!=0 and cand_pattern[0][0]!=value_pattern_summary[0][0]:
            rule_fired = True

    elif rule == "First_BW_Symbol_disagrees":
        if len(value_pattern_BW_summary)!=0 and cand_pattern[-1][0]!=value_pattern_BW_summary[0][0]:
        # if (value_pattern_BW_summary[0][0] in [')','.'] or cand_pattern[-1][0] in [')','.'])==False:
            rule_fired = True 
    elif rule=="NON_NUMERIC_CHAR_COUNT_DIFFERS_FROM_CONSISTENT":
        if cand_length>0 and cand_length!= summary_min_length and summary_min_length==summary_max_length:
            if 'D' not in value_symbol_summary and 'D' not in value_symbol_summary:
                rule_fired = True 
            
    elif rule == "CHAR_COUNT_UNDER_POINT1_MIN":
        if cand_length>0 and cand_length<=0.1*summary_min_length:
            rule_fired = True

    elif rule == "CHAR_COUNT_UNDER_POINT3_MIN":
        if  cand_length>0 and cand_length<=0.3*summary_min_length:
            rule_fired = True

    elif rule == "CHAR_COUNT_UNDER_POINT5_MIN":
        if  cand_length>0 and cand_length<=0.5*summary_min_length:
            rule_fired = True

    elif rule == "CHAR_COUNT_UNDER_POINT6_MIN":
        if cand_length>0 and cand_length<=0.6*summary_min_length:
            rule_fired = True

    elif rule == "CHAR_COUNT_UNDER_POINT7_MIN":
        if  cand_length>0 and cand_length<=0.7*summary_min_length:
            rule_fired = True

    elif rule == "CHAR_COUNT_UNDER_POINT8_MIN":
        if  cand_length>0 and cand_length<=0.8*summary_min_length:
            rule_fired = True
            
    elif rule == "CHAR_COUNT_UNDER_POINT9_MIN":
        if cand_length>0 and cand_length<=0.9*summary_min_length:
            rule_fired = True

    elif rule == "CHAR_COUNT_OVER_POINT1_MAX":
        if  summary_max_length>0 and  cand_length>0 and cand_length>=1.1*summary_max_length:
            rule_fired = True

    elif rule == "CHAR_COUNT_OVER_POINT3_MAX":
        if  summary_max_length>0 and  cand_length>0 and cand_length>=1.3*summary_max_length:
            rule_fired = True

    elif rule == "CHAR_COUNT_OVER_POINT5_MAX":
        if  summary_max_length>0 and  cand_length>0 and cand_length>=1.5*summary_max_length:
            rule_fired = True

    elif rule == "CHAR_COUNT_OVER_POINT6_MAX":
        if  summary_max_length>0 and  cand_length>0 and cand_length>=1.6*summary_max_length:
            rule_fired = True

    elif rule == "CHAR_COUNT_OVER_POINT7_MAX":
        if  summary_max_length>0 and  cand_length>0 and cand_length>=1.7*summary_max_length:
            rule_fired = True

    elif rule == "CHAR_COUNT_OVER_POINT8_MAX":
        if  summary_max_length>0 and  cand_length>0 and cand_length>=1.8*summary_max_length:
            rule_fired = True

    elif rule == "CHAR_COUNT_OVER_POINT9_MAX":
        if  summary_max_length>0 and cand_length>0 and cand_length>=1.9*summary_max_length:
            rule_fired = True
                       
    elif rule == "SymbolChain":
        if summary_strength>1 and value_chain_consistent==True and pat.symbol_chain_disagrees(value_symbol_chain,cand_symbol_chain):
            # print(f'value_symbol_chain={value_symbol_chain}')
            # input(f'cand_symbol_chain={cand_symbol_chain}\n')
            rule_fired = True

    elif rule == "CC":
        if len(case_summary)!=0 and case_summary!=cand_case:
            rule_fired = True

    elif rule == "CONSISTENT_NUMERIC": 
    # "name":"Below but not here: consistently ONE symbol = D"
        if line_index+1 in line_agreements.keys() and line_index in line_agreements.keys():
            next_line_agreements = line_agreements[line_index+1]
            if columnindex in next_line_agreements.keys():
                value_below_agreements = next_line_agreements[columnindex]['agreements']
                if line_agreements[line_index][columnindex]['null_equivalent']==False and rule in value_below_agreements and columnindex in line_agreements[line_index].keys() and rule not in line_agreements[line_index][columnindex]['agreements']:
                    rule_fired = True 
                
            
    elif rule == "CONSISTENT_D_STAR": 
            # "name":"Below but not here: consistently TWO symbols, the first is a digit"
        if line_index+1 in line_agreements.keys() and line_index in line_agreements.keys():
            next_line_agreements = line_agreements[line_index+1]
            if columnindex in next_line_agreements.keys():
                value_below_agreements = next_line_agreements[columnindex]['agreements']
                if line_agreements[line_index][columnindex]['null_equivalent']==False and rule in value_below_agreements and columnindex in line_agreements[line_index].keys() and rule not in line_agreements[line_index][columnindex]['agreements']:
                    rule_fired = True            
            
    elif rule == "FW_SUMMARY_D": 
            # "name":"Below but not here: two or above symbols in the FW summary, the first is a digit"
        if line_index+1 in line_agreements.keys() and line_index in line_agreements.keys():
            next_line_agreements = line_agreements[line_index+1]
            if columnindex in next_line_agreements.keys():
                value_below_agreements = next_line_agreements[columnindex]['agreements']
                if line_agreements[line_index][columnindex]['null_equivalent']==False and rule in value_below_agreements and columnindex in line_agreements[line_index].keys() and rule not in line_agreements[line_index][columnindex]['agreements']:
                    rule_fired = True          

    elif rule == "BW_SUMMARY_D": 
            # "name":"Below but not here: two or above symbols in the BW summary, the first is a digit"
        if line_index+1 in line_agreements.keys() and line_index in line_agreements.keys():
            next_line_agreements = line_agreements[line_index+1]
            if columnindex in next_line_agreements.keys():
                value_below_agreements = next_line_agreements[columnindex]['agreements']
                if line_agreements[line_index][columnindex]['null_equivalent']==False and rule in value_below_agreements and columnindex in line_agreements[line_index].keys() and rule not in line_agreements[line_index][columnindex]['agreements']:
                    rule_fired = True   

    elif rule == "BROAD_NUMERIC": 
            # "name":"Below but not here: all values digits, optionally have . or ,  or S"
        if line_index+1 in line_agreements.keys() and line_index in line_agreements.keys():
            next_line_agreements = line_agreements[line_index+1]
            if columnindex in next_line_agreements.keys():
                value_below_agreements = next_line_agreements[columnindex]['agreements']
                if line_agreements[line_index][columnindex]['null_equivalent']==False and rule in value_below_agreements and columnindex in line_agreements[line_index].keys() and rule not in line_agreements[line_index][columnindex]['agreements']:
                    rule_fired = True   

    elif rule == "FW_THREE_OR_MORE_NO_SPACE": 
            # "name":"Below but not here: three or above symbols in FW summary that do not contain a  Space"
        if line_index+1 in line_agreements.keys() and line_index in line_agreements.keys():
            next_line_agreements = line_agreements[line_index+1]
            if columnindex in next_line_agreements.keys():
                value_below_agreements = next_line_agreements[columnindex]['agreements']
                if line_agreements[line_index][columnindex]['null_equivalent']==False and rule in value_below_agreements and columnindex in line_agreements[line_index].keys() and rule not in line_agreements[line_index][columnindex]['agreements']:
                    rule_fired = True   
                                            
    elif rule == "BW_THREE_OR_MORE_NO_SPACE": 
            # "name":"Below but not here: three or above symbols in BW summary that do not contain a  Space"
        if line_index+1 in line_agreements.keys() and line_index in line_agreements.keys():
            next_line_agreements = line_agreements[line_index+1]
            if columnindex in next_line_agreements.keys():
                value_below_agreements = next_line_agreements[columnindex]['agreements']
                if line_agreements[line_index][columnindex]['null_equivalent']==False and rule in value_below_agreements and columnindex in line_agreements[line_index].keys() and rule not in line_agreements[line_index][columnindex]['agreements']:
                    rule_fired = True   
                                          
    elif rule == "CONSISTENT_SS_NO_SPACE": 
            # "name":"Below but not here: consistently at least two symbols in the symbol set summary, none of which are S or _"
        if line_index+1 in line_agreements.keys() and line_index in line_agreements.keys():
            next_line_agreements = line_agreements[line_index+1]
            if columnindex in next_line_agreements.keys():
                value_below_agreements = next_line_agreements[columnindex]['agreements']
                if line_agreements[line_index][columnindex]['null_equivalent']==False and rule in value_below_agreements and columnindex in line_agreements[line_index].keys() and rule not in line_agreements[line_index][columnindex]['agreements']:
                    rule_fired = True   
                                            
    elif rule == "FW_TWO_OR_MORE_OR_MORE_NO_SPACE": 
            # "name":"Below but not here: two or above symbols in FW summary that do not contain a Space"
        if line_index+1 in line_agreements.keys() and line_index in line_agreements.keys():
            next_line_agreements = line_agreements[line_index+1]
            if columnindex in next_line_agreements.keys():
                value_below_agreements = next_line_agreements[columnindex]['agreements']
                if line_agreements[line_index][columnindex]['null_equivalent']==False and rule in value_below_agreements and columnindex in line_agreements[line_index].keys() and rule not in line_agreements[line_index][columnindex]['agreements']:
                    rule_fired = True   
                                            
    elif rule == "BW_TWO_OR_MORE_OR_MORE_NO_SPACE": 
            # "name":"Below but not here: two or above symbols in BW summary that do not contain a Space"
        if line_index+1 in line_agreements.keys() and line_index in line_agreements.keys():
            next_line_agreements = line_agreements[line_index+1]
            if columnindex in next_line_agreements.keys():
                value_below_agreements = next_line_agreements[columnindex]['agreements']
                if line_agreements[line_index][columnindex]['null_equivalent']==False and rule in value_below_agreements and columnindex in line_agreements[line_index].keys() and rule not in line_agreements[line_index][columnindex]['agreements']:
                    rule_fired = True   
                                 

    return rule_fired

def eval_data_cell_rule(rule, columnvalues, column_tokens, all_values_summary, consistent_symbol_chain, pattern_BW_summary, value_symbol_summary, column_symbols, column_patterns, case_summary, candidate_count, partof_multiword_value_repeats,candidate_tokens,consistent_symbol_set, all_patterns_numeric, max_values_lookahead=6):

    rule_fired = False
    
    candidate = columnvalues[0]
    # candidate_tokens= [t  for t in column_tokens[0] if any(c.isalpha() for c in t)]
    # consistent_symbol_set = is_consistent_symbol_sets(column_symbols)

    if candidate==None  or str(candidate).strip().lower() in null_equivalent:
        rule_fired= False

    elif rule == "VALUE_REPEATS_ONCE_BELOW": 
        rule_fired = False
        if len(columnvalues)>2:
            # candidate_count = np.count_nonzero(columnvalues[2:min(max_values_lookahead,len(columnvalues))] == candidate) #numpyarray implementation
            if candidate_count==1:        
                rule_fired = True    

    elif rule == "CONSISTENT_SINGLE_WORD_CONSISTENT_CASE": 
        rule_fired = False
        if consistent_symbol_chain and len(value_symbol_summary)==1 and value_symbol_summary[0]=='A' and case_summary in ['ALL_CAPS', 'ALL_LOW', 'TITLE']:
            rule_fired = True

    elif rule == "CONSISTENT_CHAR_LENGTH":
        rule_fired = False
        if consistent_symbol_chain==True and str(candidate).strip().lower() not in ['', ' ', 'nan', 'None']:
            rule_fired = True

    elif  rule == "VALUE_REPEATS_TWICE_OR_MORE_BELOW":
        rule_fired = False
        if len(columnvalues)>2:
            # candidate_count = np.count_nonzero(columnvalues[1:min(max_values_lookahead,len(columnvalues))] == candidate)
            if candidate_count>=2:        
                rule_fired = True

    elif rule == "ONE_ALPHA_TOKEN_REPEATS_ONCE_BELOW": 
        # "Rule_2_a: Only one alphabetic token from multiword value repeats below, and it repeats only once"
        rule_fired = False
        # if len(columnvalues)>2 and partof_multiword_value_repeats_once(candidate_tokens, column_tokens[2:min(max_values_lookahead,len(columnvalues))]):
        # for t in candidate_tokens:
        #     print(f'partof_multiword_value_repeats[t]={partof_multiword_value_repeats[t]}')
        if len(columnvalues)>2 and sum([partof_multiword_value_repeats[t] for t in candidate_tokens])==1:
            rule_fired = True        

    elif rule == "ALPHA_TOKEN_REPEATS_TWICE_OR_MORE": 
        # "Rule_2_b: At least one alphabetic token from multiword value repeats below at least twice"
        rule_fired = False
        # if len(columnvalues)>2 and partof_multiword_value_repeats_at_least_k(candidate_tokens, column_tokens[2:min(max_values_lookahead,len(columnvalues))], 2):
        if len(columnvalues)>2 and sum([partof_multiword_value_repeats[t] for t in candidate_tokens])>=2:            
            rule_fired = True 

    elif rule == "CONSISTENT_NUMERIC_WIDTH":
        # "Rule_3 consistently numbers with consistent digit count for all."
        rule_fired = False
        if  consistent_symbol_chain==True and len(all_values_summary)==1 and all_values_summary[0][0] == 'D' and all_values_summary[0][1]>0:
            rule_fired = True

    elif rule == "CONSISTENT_NUMERIC": 
        # "Rule_4_a consistently ONE symbol = D"
        rule_fired = False
        if consistent_symbol_chain==True and len(all_values_summary)==1 and  all_values_summary[0][0] == 'D':
            rule_fired = True
        # print(f'rule_fired={rule_fired}')

    elif rule == "CONSISTENT_D_STAR":  
        # "Rule_4_b consistently TWO symbols, the first is a digit"
        rule_fired = False
        if consistent_symbol_chain==True and len(all_values_summary) == 2 and all_values_summary[0][0] == 'D':
            rule_fired = True
            
    elif rule == "FW_SUMMARY_D": 
        # "Rule_4_fw two or above symbols in the FW summary, the first is a digit"
        rule_fired = False
        if len(all_values_summary)>=2 and all_values_summary[0][0] == 'D':
            rule_fired = True
    elif rule == "BW_SUMMARY_D": 
        #"Rule_4_bw two or above symbols in the BW summary, the first is a digit"
        rule_fired = False
        if len(pattern_BW_summary)>=2 and pattern_BW_summary[0][0] == 'D':
            rule_fired = True
    elif rule == "BROAD_NUMERIC": 
        #"Rule_5 all values digits, optionally have . or ,  or S"    
        rule_fired = False
        if all_patterns_numeric:#pat.numeric_patterns(column_symbols, column_patterns):
            rule_fired = True
    elif rule == "FW_THREE_OR_MORE_NO_SPACE": 
        #"Rule_6 three or above symbols in FW summary that do not contain a  Space"
        rule_fired = False
        if len(all_values_summary)>=3 and 'S' not in [x[0] for x in all_values_summary]:
            rule_fired = True
    elif rule == "BW_THREE_OR_MORE_NO_SPACE": 
        #"Rule_7 three or above symbols in BW summary that do not contain a  Space"
        rule_fired = False
        if len(pattern_BW_summary)>=3 and 'S' not in [x[0] for x in pattern_BW_summary]:
            rule_fired = True
    elif rule == "CONSISTENT_SS_NO_SPACE": 
        #"Rule_8 consistently at least two symbols in the symbol set summary, none of which are S or _"
        rule_fired = False
        if consistent_symbol_set and len(value_symbol_summary)>=2 and 'S' not in value_symbol_summary and '_' not in  value_symbol_summary:
            rule_fired = True

    elif rule == "CONSISTENT_SC_TWO_OR_MORE": 
        #"Rule_10 two or more symbols consistent chain"
        rule_fired = False
        if consistent_symbol_chain==True and len(all_values_summary)>=2:
            rule_fired = True

    elif rule == "FW_TWO_OR_MORE_OR_MORE_NO_SPACE": 
        #"Rule_11_fw two or above symbols in FW summary that do not contain a Space"
        rule_fired = False
        if len(all_values_summary)>=2 and 'S' not in [x[0] for x in all_values_summary]:
            rule_fired = True

    elif rule == "BW_TWO_OR_MORE_OR_MORE_NO_SPACE": 
        #"Rule_11_bw two or above symbols in BW summary that do not contain a Space"
        rule_fired = False
        if len(pattern_BW_summary)>=2 and 'S' not in [x[0] for x in pattern_BW_summary]:
            rule_fired = True

    elif rule == "FW_TWO_OR_MORE_OR_MORE_NO_SPACE_FIRST_TWO": 
        #"Rule_12_fw two or above symbols in FW summary, the first two do not contain a Space"
        rule_fired = False
        if len(all_values_summary)>=2 and 'S' not in [x[0] for x in all_values_summary[0:2]]:
            rule_fired = True
        
    elif rule == "BW_TWO_OR_MORE_OR_MORE_NO_SPACE_FIRST_TWO": 
        #"Rule_12_bw two or above symbols in BW summary, the first two do not contain a Space"
        rule_fired = False
        if len(pattern_BW_summary)>=2 and 'S' not in [x[0] for x in pattern_BW_summary[0:2]]:
            rule_fired = True

    elif rule == "FW_D5PLUS": 
        #"Rule_13_fw FW summary is [['D',count]], where count>=5"
        rule_fired = False
        if  len(all_values_summary)==1 and  all_values_summary[0][0]=='D' and all_values_summary[0][1]>=5:
            rule_fired = True

    elif rule == "BW_D5PLUS": 
        #"Rule_13_bw BW summary is [['D',count]], where count>=5"
        rule_fired = False
        if  len(pattern_BW_summary)==1 and  pattern_BW_summary[0][0]=='D' and pattern_BW_summary[0][1]>=5:
            rule_fired = True

    elif rule == "FW_D1": 
        #"Rule_14_fw FW summary is [['D',1]]"
        rule_fired = False
        if  len(all_values_summary)==1 and  all_values_summary[0][0]=='D' and all_values_summary[0][1]==1:
            rule_fired = True

    elif rule == "BW_D1": 
        #"Rule_14_bw BW summary is [['D',1]]"
        rule_fired = False
        if  len(pattern_BW_summary)==1 and  pattern_BW_summary[0][0]=='D' and pattern_BW_summary[0][1]==1:
            rule_fired = True

    elif rule == "FW_D4": 
        #"Rule_15_fw FW summary is [['D',4]]"
        rule_fired = False
        if  len(all_values_summary)==1 and  all_values_summary[0][0]=='D' and all_values_summary[0][1]==4:
            rule_fired = True

    elif rule == "BW_D4": 
        #"Rule_15_bw BW summary is [['D',4]]"
        rule_fired = False
        if  len(pattern_BW_summary)==1 and  pattern_BW_summary[0][0]=='D' and pattern_BW_summary[0][1]==4:
            rule_fired = True  

    elif rule == "FW_LENGTH_4PLUS": 
        #"Rule_17_fw four or more symbols in the FW summary"
        rule_fired = False
        if len(all_values_summary)>=4:
            rule_fired = True

    elif rule == "BW_LENGTH_4PLUS": 
        #"Rule_17_bw four or more symbols in the BW summary"
        rule_fired = False
        if len(pattern_BW_summary)>=4:
            rule_fired = True

    elif rule == "CASE_SUMMARY_CAPS":
        #"Rule_18 case summary is ALL_CAPS"
        rule_fired = False
        if case_summary == 'ALL_CAPS':
            rule_fired = True 

    elif rule == "CASE_SUMMARY_LOWER":
        rule_fired = False
        if case_summary == 'ALL_LOW':
            rule_fired = True   

    elif rule == "CASE_SUMMARY_TITLE":
        rule_fired = False
        if case_summary == 'TITLE':
            rule_fired = True 

    # elif rule =="NUMERIC_VALUE":
    #     rule_fired = False
    #     if case_summary == 'TITLE':
    #         rule_fired = True 

    return rule_fired


def normalize_value(value):
    #print('\n generate_pattern_symbols_and_case for: \n'+value)
    value = str(value).strip()
    value_lower = value.lower()
    if value==None or value_lower in pat.null_equivalent_values:
        value= ''

    if len(value_lower)>0:
        for phrase in pat.aggregation_tokens:
        # for phrase in ['total']:    
            if phrase in value_lower:
                value= ''
                break

    value = value.replace('\'', '')
    return value

def generate_case(value):
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
    return value_case


def generate_tokens(value):
    tokens=value.lower().split(' ')   
    return tokens

def generate_token_length(tokens):
    token_length = len(tokens)
    return token_length
def generate_character_length(value):
    character_length = len(value)
    return character_length

def generate_train(value, outlier_sensitive):  
    value_pattern = []
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

        elif i<(len(value)) and value[i].isspace():
            space_counter=0
            while i<(len(value)) and  value[i].isspace():
                i+=1
                space_counter+=1
            value_pattern.append(['S', space_counter])

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

        elif i<(len(value)) and value[i].isdigit():
            digit_counter=0;
            while i<(len(value)) and  value[i].isdigit():
                #print('value['+str(i)+']='+str(value[i]))
                i+=1
                #print('i='+str(i))
                digit_counter+=1
            value_pattern.append(['D', digit_counter])

        # Punctuation: !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~
        elif i<(len(value)) and value[i] in string.punctuation:
            punctuation_counter= 0
            punctuation = value[i]
            while i<(len(value)) and  value[i] == punctuation:
                i+=1
                punctuation_counter+=1
            value_pattern.append([punctuation, punctuation_counter])

        elif i<(len(value)):
            unknown_counter= 0
            unknown = value[i]
            while i<(len(value)) and  value[i] == unknown:
                i+=1
                unknown_counter+=1
            value_pattern.append([unknown, unknown_counter])

        else:
            i+=1   
    return value_pattern

def train_normalize_numbers(train):
   
    # for symbols in summary_symbols:
    #     if (symbols==set() or ('D' in symbols and symbols.issubset(set(['D','.',',','S','-','+','~','>','<','(',')']))) ) == False:
    #         return summary_patterns, summary_symbols
    
    symbolchain = [symbol_count[0] for symbol_count in train]
    symbolset = set(symbolchain)
    if ('D' in symbolset and symbolset.issubset(set(['D','.',',','S','-','+','~','>','<','(',')']))) == False:
        return train    
    indices = [i for i, x in enumerate(symbolchain) if x in ['-','+','~','>','<']]
    if len(indices)>1 or len(indices)==1 and indices[0]>0:
        return train        
    if len(train)>0:
        digits = [symbol_count[1] for symbol_count in train if symbol_count[0] == 'D']
        digit_count = sum(digits)
        return [['D',sum(digits)]]
    else:
        return train

def symbolset_normalize_numbers(symbolset):

    if symbolset.issubset(set(['D','.',',','S','-','+','~','>','<','(',')'])):
        return set(['D'])
    else:
        return symbolset
    

def generate_chain(train):
    return [t[0] for t in train]

def generate_symbolset(chain):
    return set(chain)


def eval_numeric_pattern(train):
    symbolset = set([t[0] for t in train])
    if 'D' in symbolset and symbolset.issubset(set(['D',',','.','-','S'])) and (([x[0] for x in train].count('-') ==1 and len(train)>0 and train[0][0]=='-') or ([x[0] for x in train].count('-') ==0)):
        return True
    else:
        return False
class TableSignatures:
    def __init__(self, dataframe = pd.DataFrame(), outlier_sensitive=False):
        if dataframe.empty==False:
            dataframe_labels = []
            for column in dataframe:
                dataframe_labels.append(column)        
            normalized_values = dataframe.applymap(normalize_value)
            character_length = normalized_values.applymap(generate_character_length)
            case = normalized_values.applymap(generate_case)
            tokens = normalized_values.applymap(generate_tokens)
            token_length = tokens.applymap(generate_token_length)        
            train = normalized_values.applymap(lambda cell: generate_train(cell, outlier_sensitive))
            bw_train = train.applymap(lambda cell: list(reversed(cell)))            
            chain = train.applymap(generate_chain)
            symbolset = chain.applymap(generate_symbolset)

            train_normalized_numbers = train.applymap(train_normalize_numbers)
            bw_train_normalized_numbers = train_normalized_numbers.applymap(lambda cell: list(reversed(cell)))
            symbolset_normalized_numbers = symbolset.applymap(symbolset_normalize_numbers)

            isnumber = symbolset_normalized_numbers.applymap(is_number)
            is_numeric_pattern = train.applymap(eval_numeric_pattern)

            is_aggregate = tokens.applymap(lambda cell: (len(cell)>0 and not set(cell).isdisjoint(pat.aggregation_tokens)))
            is_null_equivalent= normalized_values.applymap(lambda cell: (cell.lower() in pat.null_equivalent_values))
            # is_null_equivalent= normalized_values.applymap(lambda cell: (cell.lower() in pat.strictly_null_equivalent))
            
            self.all_normalized_values = normalized_values.to_numpy()
            self.all_column_character_lengths = character_length.to_numpy()
            self.all_column_cases = case.to_numpy()
            self.all_column_tokens = tokens.to_numpy()
            self.all_column_token_lengths = token_length.to_numpy()
            self.all_column_train = train.to_numpy()
            self.all_column_bw_train = bw_train.to_numpy()
            self.all_column_chain = chain.to_numpy()
            self.all_column_symbols = symbolset.to_numpy()
            self.all_column_isnumber = isnumber.to_numpy()
            self.all_column_is_numeric_train = is_numeric_pattern.to_numpy()
            self.train_normalized_numbers=train_normalized_numbers.to_numpy()
            self.bw_train_normalized_numbers = bw_train_normalized_numbers.to_numpy()
            self.symbolset_normalized_numbers=symbolset_normalized_numbers.to_numpy()
            self.is_aggregate = is_aggregate.to_numpy()
            self.is_null_equivalent = is_null_equivalent.to_numpy()

        else:

            self.all_normalized_values = np.array([])
            self.all_column_character_lengths = np.array([])
            self.all_column_cases = np.array([])
            self.all_column_tokens = np.array([])
            self.all_column_token_lengths = np.array([])
            self.all_column_train = np.array([])
            self.all_column_bw_train = np.array([])
            self.all_column_chain = np.array([])
            self.all_column_symbols = np.array([])

            self.all_column_isnumber = np.array([])
            self.all_column_is_numeric_train = np.array([])

            self.train_normalized_numbers = np.array([])
            self.bw_train_normalized_numbers = np.array([])
            self.symbolset_normalized_numbers=np.array([])

            self.is_aggregate = np.array([])
            self.is_null_equivalent = np.array([])

             
    def preview(self):
        print('\nnormalized_values.head():')
        print(pd.DataFrame(self.all_normalized_values).head())
        print('\ncharacter_length.head():')
        print(pd.DataFrame(self.all_column_character_lengths).head())
        print('\ncase.head():')
        print(pd.DataFrame(self.all_column_cases).head())
        print('\ntokens.head():')
        print(pd.DataFrame(self.all_column_tokens).head())
        print('\ntoken_length.head():')
        print(pd.DataFrame(self.all_column_token_lengths).head()) 
        print('\ntrain.head():')   
        print(pd.DataFrame(self.all_column_train).head())
        print('\nall_column_bw_train.head():')   
        print(pd.DataFrame(self.all_column_bw_train).head())
        print('\nchain.head():')
        print(pd.DataFrame(self.all_column_chain).head())
        print('\nsymbolset.head():')
        print(pd.DataFrame(self.all_column_symbols).head())
        print('\nall_column_isnumber.head():')
        print(pd.DataFrame(self.all_column_isnumber).head())
        print('\nall_column_is_numeric_pattern.head():')
        print(pd.DataFrame(self.all_column_is_numeric_train).head())
        print('\ntrain_normalized_numbers.head():')
        print(pd.DataFrame(self.train_normalized_numbers).head())
        print('\nsymbolset_normalized_numbers.head():')
        print(pd.DataFrame(self.symbolset_normalized_numbers).head())
        print('\nis_aggregate.head():')
        print(pd.DataFrame(self.is_aggregate).head())
        print('\nis_null_equivalent.head():')
        print(pd.DataFrame(self.is_null_equivalent).head())

    def reverse_slice(self, top, bottom):
        slice = TableSignatures()   
        slice.all_normalized_values = self.all_normalized_values[top:bottom+1][::-1]
        slice.all_column_character_lengths = self.all_column_character_lengths[top:bottom+1][::-1]
        slice.all_column_cases = self.all_column_cases[top:bottom+1][::-1]
        slice.all_column_tokens = self.all_column_tokens[top:bottom+1][::-1]
        slice.all_column_token_lengths = self.all_column_token_lengths[top:bottom+1][::-1]
        slice.all_column_train = self.all_column_train[top:bottom+1][::-1]
        slice.all_column_bw_train=self.all_column_bw_train[top:bottom+1][::-1]
        slice.all_column_chain = self.all_column_chain[top:bottom+1][::-1]
        slice.all_column_symbols = self.all_column_symbols[top:bottom+1][::-1] 
        slice.all_column_isnumber = self.all_column_isnumber[top:bottom+1][::-1] 
        slice.all_column_is_numeric_train = self.all_column_is_numeric_train[top:bottom+1][::-1] 
        slice.train_normalized_numbers= self.train_normalized_numbers[top:bottom+1][::-1] 
        slice.bw_train_normalized_numbers = self.bw_train_normalized_numbers[top:bottom+1][::-1] 
        slice.symbolset_normalized_numbers = self.symbolset_normalized_numbers[top:bottom+1][::-1]  
        slice.is_aggregate = self.is_aggregate[top:bottom+1][::-1] 
        slice.is_null_equivalent = self.is_null_equivalent[top:bottom+1][::-1] 
        return slice



def normalize_numeric(value):
    value = str(value).strip()
    if any(char.isdigit() for char in value) and any(char.isalpha() for char in value)==False:
        
        while value[0].isdigit()==False and value[0]!='-':
            value = value[1:]

        if value.endswith('%'):
                value = value.replace('$','')            

        for i,char in enumerate(value):
            if char.isdigit()==False and char not in [',',' ','.']:
                if i == 0 and char =='-':
                    continue
                else:
                    value = ''
                    break    
        if value!='':
            while ',' in value:
                value= value.replace(',', '')
            while '.' in value:
                value= value.replace('.', '')
            while ' ' in value:
                value = value.replace(' ', '')  
            value = int(value)
        else:
            value = np.nan
    else:
        value = np.nan
    return value

def discover_aggregation_scope(csv_file, aggregation_rows, cand_subheaders, predicted_subheaders, certain_data_indexes,pat_headers):
    subheader_scope={}
    cand_subhead_indexes = sorted(list(cand_subheaders.keys()))
    last_header_value = ''
    if len(pat_headers)>0:
        last_header_value = str(csv_file.loc[pat_headers[-1],:].tolist()[0])

    # pp.pprint(aggregation_rows)
    # input(f'\ncand_subhead_indexes={cand_subhead_indexes}\n')
    
    if len(aggregation_rows)>0 and len(certain_data_indexes)>0:
        agg_idxs = sorted(aggregation_rows.keys())
        first_aggregation_row = aggregation_rows[agg_idxs[0]]
        if len(agg_idxs)>0  and agg_idxs[0]!=certain_data_indexes[0]:
            # scope='up'
            
            for aggregation_idx in agg_idxs:
                scope_head_idx= None 
                aggregation = aggregation_rows[aggregation_idx]
                aggregation_function = aggregation['aggregation_function'] 

                aggregation_row = csv_file.loc[[aggregation_idx]].applymap(normalize_numeric)
                candidate_scope = csv_file.loc[[i for i in sorted(certain_data_indexes+cand_subhead_indexes) if i<aggregation_idx and i not in agg_idxs]].applymap(normalize_numeric)

                # input(f'{aggregation_idx}: cand_subhead_indexes={cand_subhead_indexes}')
                # print(f'\ncandidate_scope = \n{candidate_scope}')
                # print(f'\naggregation_row=\n{aggregation_row}\n')    
                # print(aggregation_rows[aggregation_idx])

                if aggregation_function=='sum':
                    for i in range(1, candidate_scope.shape[0]+1):

                        summed_rows = candidate_scope.loc[candidate_scope.index[-i:]].sum(axis = 0, skipna = True)#.replace({ 0:np.nan})
                        # print(f'\nsummed_rows=\n{pd.DataFrame([summed_rows])}')
                        if scope_head_idx==None:
                            # input(f'\n~~aggregation_row=\n{aggregation_row}\n')
                            if aggregation_row.iloc[0].eq(summed_rows).any(): 
                                scope_head_idx = candidate_scope.index[-i]
                                # input(f'\n\nscope_head_idx (first assignment) ={scope_head_idx}')
                                if i<candidate_scope.shape[0] and scope_head_idx-1 not in cand_subhead_indexes:
                                    continue

                        if scope_head_idx!=None:
                            cand_scope_head = scope_head_idx
                            # print(f'\tcand_scope_head={cand_scope_head}')

                            if cand_scope_head in candidate_scope.index:
                                summed_rows = candidate_scope.loc[cand_scope_head:].sum(axis = 0, skipna = True)#.replace({ 0:np.nan})
                            # print(f'\n\tsummed_rows1=\n{pd.DataFrame([summed_rows])}')

                            while cand_scope_head >= candidate_scope.index[0] and aggregation_row.iloc[0].eq(summed_rows).any():
                                # print(f'\ncandidate_scope = \n{candidate_scope}')
                                if cand_scope_head in candidate_scope.index:
                                    summed_rows = candidate_scope.loc[cand_scope_head:].sum(axis = 0, skipna = True)#.replace({ 0:np.nan})
                                # print(f'\n\t\tsummed_rows2=\n{pd.DataFrame([summed_rows])}')

                                cand_scope_head = cand_scope_head-1 #candidate_scope.index[-i]
                                # print(f'\t\tcand_scope_head={cand_scope_head}')

                                if aggregation_row.iloc[0].eq(summed_rows).any():                             
                                    # input(f'\n\n__scope_head_idx={scope_head_idx}')
                                    # print(f'\t\tcand_subhead_indexes={cand_subhead_indexes}')
                                    if cand_scope_head in cand_subhead_indexes:
                                        scope_head_idx = cand_scope_head+1
                                        # print(f'\t\t\tscope_head={scope_head_idx}\nBREAK WHILE')
                                        break  
                                else:
                                    break                               

                        if scope_head_idx!=None:
                            # print(f'\ncandidate_scope = \n{candidate_scope}')
                            # print(f'summed_rows=\n{pd.DataFrame([summed_rows])}')
                            # input(f'\n\n---> selected scope_head_idx={scope_head_idx}')
                            aggregation_rows[aggregation_idx]['scope']='UP'
                            aggregation_rows[aggregation_idx]['scope_head']=scope_head_idx
                            aggregation_rows[aggregation_idx]['scope_range']=list(range(scope_head_idx,aggregation_idx ))
                            # certain_data_indexes = list(set(certain_data_indexes)-set(aggregation_rows[aggregation_idx]['scope_range']))
                            # print(f'scope_head_idx-1={scope_head_idx-1}')
                            cand_subheader_rev = list(cand_subhead_indexes)[::-1]
                            for cand_subheader_idx in cand_subheader_rev:
                                if cand_subheader_idx>scope_head_idx:
                                    continue
                                value = cand_subheaders[cand_subheader_idx]
                                # input(f'check if {cand_subheader_idx}:{value.lower()} in {aggregation["aggregation_label"].lower()}')
                                if value.lower() in aggregation['aggregation_label'].lower():
                                    # print(f"\n{value.lower()} in {aggregation['aggregation_label'].lower()}")
                                    aggregation_rows[aggregation_idx]['context_label'] = value
                                    aggregation_rows[aggregation_idx]['subheader'] = cand_subheader_idx 
                                    subheader_scope[cand_subheader_idx]=list(range(scope_head_idx,aggregation_idx ))                              
                                    predicted_subheaders.append(cand_subheader_idx)
                                    cand_subheaders.pop(cand_subheader_idx) 
                                    cand_subhead_indexes.remove(cand_subheader_idx)
                                    aggregation_rows[aggregation_idx]['scope_head']=cand_subheader_idx+1
                                    aggregation_rows[aggregation_idx]['scope_range']=list(range(cand_subheader_idx+1,aggregation_idx ))   
                                    # input(aggregation_rows[aggregation_idx])                                
                                    break

                            
                            if 'context_label' not in aggregation_rows[aggregation_idx].keys() and last_header_value!='' and last_header_value.lower() in aggregation["aggregation_label"].lower():
                                # input(f'HEADER CHECK: is {pat_headers[-1]}:{last_header_value.lower()} in {aggregation["aggregation_label"].lower()} ')
                                aggregation_rows[aggregation_idx]['context_label'] = last_header_value
                                aggregation_rows[aggregation_idx]['scope_head']=pat_headers[-1]+1
                                aggregation_rows[aggregation_idx]['scope_range']=list(range(pat_headers[-1]+1,aggregation_idx ))
                            # input(aggregation_rows[aggregation_idx])

                            if 'subheader' not in aggregation_rows[aggregation_idx].keys() and cand_subhead_indexes!=None and aggregation_rows[aggregation_idx]['scope_head']-1 in cand_subhead_indexes:
                                # input(f'prediction is missing subheader, look for a subheader')
                                aggregation_rows[aggregation_idx]['subheader'] = aggregation_rows[aggregation_idx]['scope_head']-1
                                subheader_scope[aggregation_rows[aggregation_idx]['scope_head']-1]=list(range(aggregation_rows[aggregation_idx]['scope_head'],aggregation_idx))
                                if aggregation_rows[aggregation_idx]['scope_head']-1 not in predicted_subheaders:
                                    predicted_subheaders.append(aggregation_rows[aggregation_idx]['scope_head']-1)
                                if aggregation_rows[aggregation_idx]['scope_head']-1 in cand_subhead_indexes:
                                    cand_subhead_indexes.remove(aggregation_rows[aggregation_idx]['scope_head']-1)
                                    if aggregation_rows[aggregation_idx]['scope_head']-1 in cand_subheaders.keys():
                                        cand_subheaders.pop(aggregation_rows[aggregation_idx]['scope_head']-1)                                    
                                # input(f'predicted_subheaders {aggregation_idx}={predicted_subheaders}')
                                aggregation_rows[aggregation_idx]['context_label'] = csv_file.loc[aggregation_rows[aggregation_idx]['scope_head']-1].tolist()[0]

                            if 'context_label' not in aggregation_rows[aggregation_idx].keys():
                                aggregation_rows[aggregation_idx]['context_label'] = aggregation['aggregation_label']
                                cand_subheader_rev = list(cand_subhead_indexes)[::-1]
                                for cand_subheader_idx in cand_subheader_rev:
                                    if cand_subheader_idx>scope_head_idx:
                                        continue
                                    value = cand_subheaders[cand_subheader_idx]
                                    if value.lower() in aggregation['aggregation_label'].lower():
                                        aggregation_rows[aggregation_idx]['context_label'] = value
                                        aggregation_rows[aggregation_idx]['subheader'] = cand_subheader_idx 
                                        subheader_scope[cand_subheader_idx]=list(range(scope_head_idx,aggregation_idx ))                              
                                        predicted_subheaders.append(cand_subheader_idx)
                                        cand_subheaders.pop(cand_subheader_idx) 
                                        cand_subhead_indexes.remove(cand_subheader_idx) 
                                        aggregation_rows[aggregation_idx]['scope_head']=cand_subheader_idx+1
                                        aggregation_rows[aggregation_idx]['scope_range']=list(range(cand_subheader_idx+1,aggregation_idx ))                                  
                                        break

                                if 'context_label' not in aggregation_rows[aggregation_idx].keys() and last_header_value!='' and last_header_value.lower() in aggregation["aggregation_label"].lower():
                                    # input(f'HEADER CHECK: is {pat_headers[-1]}:{last_header_value.lower()} in {aggregation["aggregation_label"].lower()} ')
                                    aggregation_rows[aggregation_idx]['context_label'] = last_header_value
                                    aggregation_rows[aggregation_idx]['scope_head']=pat_headers[-1]+1
                                    aggregation_rows[aggregation_idx]['scope_range']=list(range(pat_headers[-1]+1,aggregation_idx ))

                            for di in aggregation_rows[aggregation_idx]['scope_range']:
                                if di in cand_subhead_indexes:
                                    cand_subhead_indexes.remove(di)
                                if di in cand_subheaders.keys():
                                    cand_subheaders.pop(di)

                            # print(f'aggregation_rows[{aggregation_idx}]=')
                            # pp.pprint(aggregation_rows[aggregation_idx])
                            # input(f'~-cand_subhead_indexes={cand_subheaders}')
                            break

                        if cand_subhead_indexes!=None and candidate_scope.index[-i]-1 in cand_subhead_indexes: 
                            scope_head_idx = candidate_scope.index[-i]
                            candidate_subheader_value = str(csv_file.loc[scope_head_idx-1].tolist()[0])
                            if candidate_subheader_value.lower() in aggregation['aggregation_label'].lower():                                
                                # print(f'summed_rows={pd.DataFrame([summed_rows])}')
                                # print(f'aggregation_row={aggregation_row}')
                                aggregation_rows[aggregation_idx]['subheader'] = scope_head_idx-1
                                subheader_scope[scope_head_idx-1]=list(range(scope_head_idx,aggregation_idx ))

                                if scope_head_idx-1 not in predicted_subheaders:
                                    predicted_subheaders.append(scope_head_idx-1)                                   

                                if scope_head_idx-1 in cand_subhead_indexes:
                                    aggregation_rows[aggregation_idx]['scope']='UP'
                                    aggregation_rows[aggregation_idx]['scope_head']=scope_head_idx
                                    aggregation_rows[aggregation_idx]['scope_range']=list(range(scope_head_idx,aggregation_idx ))
                                    for di in aggregation_rows[aggregation_idx]['scope_range']:
                                        if di in cand_subhead_indexes:
                                            cand_subhead_indexes.remove(di)
                                        if di in cand_subheaders.keys():
                                            cand_subheaders.pop(di)      

                                    certain_data_indexes = list(set(certain_data_indexes)-set(aggregation_rows[aggregation_idx]['scope_range']))
                                    cand_subhead_indexes.remove(scope_head_idx-1)
                                    if scope_head_idx-1 in cand_subheaders.keys():
                                        cand_subheaders.pop(scope_head_idx-1)
                                #     input(f'--cand_subhead_indexes={cand_subhead_indexes}')
                                # print(f'predicted_subheaders {aggregation_idx}={predicted_subheaders}')
                                aggregation_rows[aggregation_idx]['context_label'] = csv_file.loc[scope_head_idx-1].tolist()[0]
                                # print(f'aggregation_rows[{aggregation_idx}]=')
                                # pp.pprint(aggregation_rows[aggregation_idx])
                                break

        else:
            scope='down'
            
        # for aggregation_idx in aggregation_rows.keys():

    return aggregation_rows, certain_data_indexes, predicted_subheaders, cand_subhead_indexes,subheader_scope

# def predict_subheaders(csv_file, cand_data, predicted_pat_sub_headers, pat_blank_lines, pat_headers, args, rule_weights ):
#     # input(f'predicted_pat_sub_headers={predicted_pat_sub_headers}')
#     cand_subhead_indexes = list(set(predicted_pat_sub_headers+list(cand_data.index[cand_data.iloc[:,1:].isnull().all(1)])).intersection(set(cand_data.index)))
#     cand_subhead_indexes.sort()
#     # input(f'cand_subhead_indexes={cand_subhead_indexes}')

#     candidate_subheaders = {}
#     subheader_scope={}
#     certain_data_indexes = list(cand_data.index)
#     aggregation_rows={}
#     first_column_data_values= []


#     # print(f'csv_file=\n{csv_file}\n')
#     # print(f'cand_data=\n{cand_data}\n')
    
#     # for row in csv_file.loc[certain_data_indexes].itertuples():
#     for row in cand_data.loc[certain_data_indexes].itertuples():

#         first_value = str(row[1]).strip()    
#         # input(f'row_{row.Index}: first_value={first_value}')
#         first_value_tokens = first_value.lower().split()  
#         for aggregation_phrase in pat.aggregation_functions:
#             agg_index = first_value.lower().find(aggregation_phrase[0])
#             # print(f'{aggregation_phrase[0]} in {first_value.lower()}={aggregation_phrase[0] in first_value.lower()}')
#             if agg_index>-1:                
#                 aggregation_rows[row.Index]={}
#                 aggregation_rows[row.Index]['value']=first_value
#                 aggregation_rows[row.Index]['aggregation_function'] = aggregation_phrase[1]
#                 aggregation_rows[row.Index]['aggregation_phrase'] = aggregation_phrase[0]
#                 aggregation_rows[row.Index]['aggregation_label'] = first_value[:agg_index]+first_value[agg_index+len(aggregation_phrase[0]):]
#                 break

#         if row.Index not in aggregation_rows.keys() and first_value.lower() not in pat.null_equivalent_values and row.Index not in cand_subhead_indexes:
#             first_column_data_values.append(first_value)

#     certain_data_indexes = list(set(certain_data_indexes) - set(aggregation_rows.keys()) - set(cand_subhead_indexes))

#     for row in csv_file.loc[cand_subhead_indexes].itertuples():
#         first_value = str(row[1]).strip()
#         if first_value.lower() not in  ['nan', 'none', ''] and row.Index not in aggregation_rows.keys():
#             candidate_subheaders[row.Index] = first_value

#     cand_subhead_indexes = list(candidate_subheaders.keys())
#     # aggregation_rows, certain_data_indexes, predicted_pat_sub_headers, cand_subhead_indexes, subheader_scope = discover_aggregation_scope(csv_file, aggregation_rows, candidate_subheaders, predicted_pat_sub_headers, certain_data_indexes,pat_headers)   
#     aggregation_rows, certain_data_indexes, predicted_pat_sub_headers, cand_subhead_indexes, subheader_scope = discover_aggregation_scope(csv_file.loc[:cand_data.index[-1]], aggregation_rows, candidate_subheaders, predicted_pat_sub_headers, certain_data_indexes, pat_headers)
#     # input(f'subheader_scope={subheader_scope}')

#     if cand_subhead_indexes!=None and len(cand_subhead_indexes)>0:
#         first_column_value_patterns=[]
#         first_column_value_symbols=[]
#         first_column_value_cases=[]
#         first_column_value_token_lengths=[]
#         first_column_value_char_lengths=[]
#         first_column_value_tokens=[]

#         # print(f'\ncand_subhead_indexes={cand_subhead_indexes}\n')
#         # pp.pprint(first_column_data_values)
#         # input()
#         for value in first_column_data_values:
#             pattern, symbols, case, value_num_tokens, value_num_chars = pat.generate_pattern_symbols_and_case(str(value).strip(), args.outlier_sensitive)
#             first_column_value_patterns.append(pattern)
#             first_column_value_symbols.append(symbols)
#             first_column_value_cases.append(case)
#             first_column_value_token_lengths.append(value_num_tokens)
#             first_column_value_char_lengths.append(value_num_chars)
#             # first_column_value_tokens.append([i for i in word_tokenize(str(value).strip().lower()) if i not in stop and i.isalpha() and i not in pat.null_equivalent_values])
#             first_column_value_tokens.append([i for i in (str(value).strip().lower()).split() if i not in stop and all(j.isalpha() or j in string.punctuation for j in i) and i not in null_equivalent] )

#         if args.normalize_decimals==True:   
#             first_column_value_patterns,first_column_value_symbols=pat.normalize_decimals_numbers(first_column_value_patterns, first_column_value_symbols)

#         value_pattern_summary, value_chain_consistent = pat.generate_pattern_summary(first_column_value_patterns)            
#         summary_strength = sum(1 for x in first_column_value_patterns if len(x)>0)
#         bw_patterns = [list(reversed(pattern)) for pattern in first_column_value_patterns]

#         value_pattern_BW_summary,_=pat.generate_pattern_summary(bw_patterns)

#         # input(f'value_pattern_BW_summary={value_pattern_BW_summary}')

#         value_symbol_summary = pat.generate_symbol_summary(first_column_value_symbols)
#         case_summary = pat.generate_case_summary(first_column_value_cases)
#         length_summary = pat.generate_length_summary(first_column_value_char_lengths)

#         line_agreements = {}
#         line_agreements[1]={}
#         line_agreements[1][0]={}
#         line_agreements[1][0]['agreements']=[]
#         line_agreements[1][0]['null_equivalent']=False
#         for rule in pat.cell_rules["data"].keys():
#             rule_fired = False
#             # Don't bother looking for agreements if there are no patterns
#             non_empty_patterns=0
#             if len(first_column_value_patterns)>0:
#                 for pattern in first_column_value_patterns:
#                     if pattern!=[]:
#                         non_empty_patterns+=1

#                 #there is no point calculating agreement over one value, a single value always agrees with itself.
#                 #in addition, many tables have bilingual headers, so agreement between two header values is very common, require nij>=3
#                 if len(first_column_value_patterns)>=2 and non_empty_patterns>=2: 
#                     rule_fired = eval_data_cell_rule(rule, first_column_data_values, first_column_value_tokens, value_pattern_summary, value_chain_consistent, value_pattern_BW_summary, value_symbol_summary, first_column_value_symbols, first_column_value_patterns, case_summary, len(first_column_data_values))
#                     # if rule_fired and "_REPEATS_" not in rule and rule not in ['FW_SUMMARY_D', 'FW_TWO_OR_MORE_OR_MORE_NO_SPACE', 'FW_TWO_OR_MORE_OR_MORE_NO_SPACE_FIRST_TWO','BW_TWO_OR_MORE_OR_MORE_NO_SPACE', 'BW_TWO_OR_MORE_OR_MORE_NO_SPACE_FIRST_TWO', 'BW_LENGTH_4PLUS', 'FW_LENGTH_4PLUS']:
#                     if rule_fired and "_REPEATS_" not in rule:
#                         line_agreements[1][0]['agreements'].append(rule)


#         # input(f"\nfirst_column_data_cell_rules_fired={line_agreements[1][0]['agreements']}")
        
#         for row in csv_file.loc[cand_subhead_indexes].itertuples():
#             first_value = str(row[1]).strip()
#             if first_value.lower() in ['', 'nan', 'none', 'null']:
#                 continue
#             if first_value in first_column_data_values:
#                 continue
#             if row.Index-1 in pat_blank_lines or row.Index-1 in pat_headers:
#                 predicted_pat_sub_headers.append(row.Index)
#             else: 

#                 value_tokens = first_value.lower().split() 
#                 pattern, symbols, case, value_num_tokens, value_num_chars = pat.generate_pattern_symbols_and_case(str(first_value).strip(), args.outlier_sensitive)
#                 if args.normalize_decimals==True:                    
#                     column_patterns, column_symbols=pat.normalize_decimals_numbers( [pattern]+first_column_value_patterns,  [symbols]+first_column_value_symbols)
#                     # input(f'\nrow_{row.Index} {[pattern]+first_column_value_patterns}')
#                     # print(f'column_patterns={column_patterns}')
#                 value_pattern_summary, value_chain_consistent = pat.generate_pattern_summary(column_patterns) 

#                 # print(f'value_pattern_summary={value_pattern_summary}')       
#                 summary_strength = sum(1 for x in column_patterns if len(x)>0)           
#                 bw_patterns = [list(reversed(pattern)) for pattern in column_patterns]
#                 value_pattern_BW_summary,_=pat.generate_pattern_summary(bw_patterns)
#                 value_symbol_summary = pat.generate_symbol_summary(column_symbols)
#                 case_summary = pat.generate_case_summary([case]+ first_column_value_cases)
#                 length_summary = pat.generate_length_summary( [value_num_chars]+first_column_value_char_lengths) 
#                 column_values = [first_value] + first_column_data_values 
#                 column_tokens = [value_tokens] + first_column_value_tokens

#                 cand_subhead_data_cell_rules_fired = []
#                 line_agreements[0]={}
#                 line_agreements[0][0]={}
#                 line_agreements[0][0]['agreements']=[]
#                 line_agreements[0][0]['null_equivalent']=False
#                 for rule in pat.cell_rules["data"].keys():
#                     rule_fired = False            
#                     non_empty_patterns=0
#                     if len(column_patterns)>0 and first_value.lower() not in pat.null_equivalent_values:
#                         for pattern in column_patterns:
#                             if pattern!=[]:
#                                 non_empty_patterns+=1
#                         if len(column_patterns)>=2 and non_empty_patterns>=2: 
#                             rule_fired = eval_data_cell_rule(rule, column_values, column_tokens, value_pattern_summary, value_chain_consistent, value_pattern_BW_summary, value_symbol_summary, column_symbols, column_patterns, case_summary, len(column_values))
#                             if rule_fired==True and "_REPEATS_" not in rule:
#                                 line_agreements[0][0]['agreements'].append(rule)
#                                 # if row.Index == 8:
#                                 #     print(f'\n\trow_{row.Index}:column_values={column_values}')
#                                 #     print(f'\n\trow_{row.Index}:column_patterns={column_patterns}')
#                                 #     print(f'\n\trow_{row.Index}:value_pattern_BW_summary={value_pattern_BW_summary}')
#                                 #     input(f'\n\trow_{row.Index}:value_pattern_summary={value_pattern_summary}')
#                 # if row.Index == 8:
#                 #     print(f"\nrow_{row.Index}:{line_agreements[0][0]['agreements']}")

#                 value_disagreements = []
#                 disagreement_summary_strength=summary_strength-1
#                 if len(pattern)>0:                      
#                     repetitions_of_candidate = np.count_nonzero(column_values[1:]==first_value)
#                     neighbor=''
#                     try:
#                         neighbor = column_values[1]
#                         repetitions_of_neighbor = np.count_nonzero(column_values[2:]==neighbor)                        
#                     except:
#                         repetitions_of_neighbor = 0                
                    
#                     for rule in pat.cell_rules["not_data"].keys():
#                         rule_fired = False
#                         if disagreement_summary_strength>0 and ( all_numbers(column_symbols)==False or is_number(symbols)==False):              
#                             rule_fired = eval_not_data_cell_rule(rule, repetitions_of_candidate, repetitions_of_neighbor, neighbor, value_pattern_summary, value_pattern_BW_summary, value_chain_consistent, value_symbol_summary, case_summary, length_summary, pattern, symbols, case, value_num_chars, disagreement_summary_strength, line_agreements, 0, 0)
#                             if rule_fired == True and "_REPEATS_" not in rule:
#                                 value_disagreements.append(rule)

#                 #######################################################################v######
#                 #  DATA value classification 
#                 data_score = max_score(line_agreements[0][0]['agreements'], rule_weights['data_cell'], args.weight_lower_bound)
#                 POPULATION_WEIGHT = 1-(1-args.p)**(2*summary_strength)
#                 if data_score!=None:
#                     if args.summary_population_factor:
#                         cell_data_score = data_score*POPULATION_WEIGHT
#                     else:
#                         cell_data_score = data_score

#                 #######################################################################v######
#                 #  NOT DATA value classification        
#                 not_data_score = max_score(value_disagreements, rule_weights['not_data_cell'], args.not_data_weight_lower_bound)
#                 POPULATION_WEIGHT = 1-(1-args.p)**(2*disagreement_summary_strength)
#                 if not_data_score!=None:
#                     if args.summary_population_factor:
#                         cell_not_data_score = not_data_score*POPULATION_WEIGHT
#                     else:
#                         cell_not_data_score = not_data_score 

#                 # print(f"\n{row.Index} data rules fired ={line_agreements[1][0]['agreements']}, not data rules fired ={value_disagreements}\ncell_data_score={cell_data_score}, cell_not_data_score={cell_not_data_score}")
#                 if  cell_data_score> cell_not_data_score:# candidate subheader is definitely data, move along
#                     # if row.Index == 31:
#                     # print(f"row_{row.Index} agreements:{line_agreements[0][0]['agreements']}")
#                     # print(f'row_{row.Index} disagreements:{value_disagreements}')

#                     continue            

#                 if (row.Index-1 in predicted_pat_sub_headers and row.Index-2 in predicted_pat_sub_headers):
#                     continue  

#                 if row.Index!=cand_data.index[-1]:
#                     predicted_pat_sub_headers.append(row.Index)

#     # print(f'predicted_pat_sub_headers={predicted_pat_sub_headers}')              
#     for s_i, subheader in enumerate(predicted_pat_sub_headers):
#         if subheader not in subheader_scope.keys():
#             if s_i+1==len(predicted_pat_sub_headers):
#                 subheader_scope[subheader] = list(range(subheader+1,cand_data.index[-1]+1))
#             else:
#                 next_s_i = s_i+1
#                 while next_s_i<len(predicted_pat_sub_headers):
#                     next_subh = predicted_pat_sub_headers[next_s_i]
#                     if next_subh not in subheader_scope:
#                         subheader_scope[subheader] = list(range(subheader+1,next_subh))
#                         break
#                     next_s_i+=1            
            
#     return  aggregation_rows, subheader_scope

def only_strings(line):
    only_strings = True
    for value in line:
        print(f'value={str(value)}')
        if str(value).isalpha():
            continue
        else:
            only_strings = False
            break
    return only_strings

def contains_number(line):
    contains_number = False
    for value in line:
        _, symbols, _,_,_ = pat.generate_pattern_symbols_and_case(str(value).strip(), True)
        if symbols != set() and symbols.issubset(set(['D','.',',','S','-','+','~'])):
            contains_number=True
            break
    return contains_number


# def predict_subheaders_new(csv_file, cand_data, predicted_pat_sub_headers, pat_blank_lines, pat_headers, args, rule_weights):
#     cand_subhead_indexes = list(set(predicted_pat_sub_headers+list(cand_data.index[cand_data.iloc[:,1:].isnull().all(1)])    ))
#     candidate_subheaders = {}
#     subheader_scope={}
#     certain_data_indexes = list(cand_data.index)
#     aggregation_rows={}
#     first_column_data_values= []


#     # print(f'csv_file=\n{csv_file}\n')
#     # print(f'cand_data=\n{cand_data}\n')
    
#     for row in csv_file.loc[certain_data_indexes].itertuples():
#         first_value = str(row[1]).strip()    
#         # input(f'row_{row.Index}: first_value={first_value}')
#         first_value_tokens = first_value.lower().split()  
#         for aggregation_phrase in pat.aggregation_functions:
#             agg_index = first_value.lower().find(aggregation_phrase[0])
#             # print(f'{aggregation_phrase[0]} in {first_value.lower()}={aggregation_phrase[0] in first_value.lower()}')
            
#             if agg_index>-1 and contains_number(row[1:]):                
#                 aggregation_rows[row.Index]={}
#                 aggregation_rows[row.Index]['value']=first_value
#                 aggregation_rows[row.Index]['aggregation_function'] = aggregation_phrase[1]
#                 aggregation_rows[row.Index]['aggregation_phrase'] = aggregation_phrase[0]
#                 aggregation_rows[row.Index]['aggregation_label'] = first_value[:agg_index]+first_value[agg_index+len(aggregation_phrase[0]):]
#                 break

#         if row.Index not in aggregation_rows.keys() and first_value.lower() not in pat.null_equivalent_values and row.Index not in cand_subhead_indexes:
#             first_column_data_values.append(first_value)

#     certain_data_indexes = list(set(certain_data_indexes) - set(aggregation_rows.keys()) - set(cand_subhead_indexes))

#     for row in csv_file.loc[cand_subhead_indexes].itertuples():
#         first_value = str(row[1]).strip()
#         if first_value.lower() not in  ['nan', 'none', ''] and row.Index not in aggregation_rows.keys():
#             candidate_subheaders[row.Index] = first_value

#     cand_subhead_indexes = list(candidate_subheaders.keys())
    
#     aggregation_rows, certain_data_indexes, predicted_pat_sub_headers, cand_subhead_indexes, subheader_scope = discover_aggregation_scope(csv_file, aggregation_rows, candidate_subheaders, predicted_pat_sub_headers, certain_data_indexes,pat_headers)

#     if cand_subhead_indexes!=None and len(cand_subhead_indexes)>0:
#         first_column_value_patterns=[]
#         first_column_value_symbols=[]
#         first_column_value_cases=[]
#         first_column_value_token_lengths=[]
#         first_column_value_char_lengths=[]
#         first_column_value_tokens=[]

#         # print(f'\ncand_subhead_indexes={cand_subhead_indexes}\n')
#         # pp.pprint(first_column_data_values)
#         # input()
#         for value in first_column_data_values:
#             pattern, symbols, case, value_num_tokens, value_num_chars = pat.generate_pattern_symbols_and_case(str(value).strip(), args.outlier_sensitive)
#             first_column_value_patterns.append(pattern)
#             first_column_value_symbols.append(symbols)
#             first_column_value_cases.append(case)
#             first_column_value_token_lengths.append(value_num_tokens)
#             first_column_value_char_lengths.append(value_num_chars)
#             # first_column_value_tokens.append([i for i in word_tokenize(str(value).strip().lower()) if i not in stop and i.isalpha() and i not in pat.null_equivalent_values])
#             first_column_value_tokens.append([i for i in (str(value).strip().lower()).split() if i not in stop and all(j.isalpha() or j in string.punctuation for j in i) and i not in null_equivalent] )


#         if args.normalize_decimals==True:   
#             first_column_value_patterns,first_column_value_symbols=pat.normalize_decimals_numbers(first_column_value_patterns, first_column_value_symbols)

#         value_pattern_summary, value_chain_consistent = pat.generate_pattern_summary(first_column_value_patterns)            
#         summary_strength = sum(1 for x in first_column_value_patterns if len(x)>0)
#         bw_patterns = [list(reversed(pattern)) for pattern in first_column_value_patterns]
#         value_pattern_BW_summary,_=pat.generate_pattern_summary(bw_patterns)

#         # input(f'value_pattern_BW_summary={value_pattern_BW_summary}')

#         value_symbol_summary = pat.generate_symbol_summary(first_column_value_symbols)
#         case_summary = pat.generate_case_summary(first_column_value_cases)
#         length_summary = pat.generate_length_summary(first_column_value_char_lengths)

#         line_agreements = {}
#         line_agreements[1]={}
#         line_agreements[1][0]={}
#         line_agreements[1][0]['agreements']=[]
#         line_agreements[1][0]['null_equivalent']=False
#         for rule in pat.cell_rules["data"].keys():
#             rule_fired = False
#             # Don't bother looking for agreements if there are no patterns
#             non_empty_patterns=0
#             if len(first_column_value_patterns)>0:
#                 for pattern in first_column_value_patterns:
#                     if pattern!=[]:
#                         non_empty_patterns+=1

#                 #there is no point calculating agreement over one value, a single value always agrees with itself.
#                 #in addition, many tables have bilingual headers, so agreement between two header values is very common, require nij>=3
#                 if len(first_column_value_patterns)>=2 and non_empty_patterns>=2: 
#                     rule_fired = eval_data_cell_rule(rule, first_column_data_values, first_column_value_tokens, value_pattern_summary, value_chain_consistent, value_pattern_BW_summary, value_symbol_summary, first_column_value_symbols, first_column_value_patterns, case_summary, len(first_column_data_values))
#                     # if rule_fired and "_REPEATS_" not in rule and rule not in ['FW_SUMMARY_D', 'FW_TWO_OR_MORE_OR_MORE_NO_SPACE', 'FW_TWO_OR_MORE_OR_MORE_NO_SPACE_FIRST_TWO','BW_TWO_OR_MORE_OR_MORE_NO_SPACE', 'BW_TWO_OR_MORE_OR_MORE_NO_SPACE_FIRST_TWO', 'BW_LENGTH_4PLUS', 'FW_LENGTH_4PLUS']:
#                     if rule_fired and "_REPEATS_" not in rule:
#                         line_agreements[1][0]['agreements'].append(rule)


#         # input(f"\nfirst_column_data_cell_rules_fired={line_agreements[1][0]['agreements']}")
        
#         for row in csv_file.loc[cand_subhead_indexes].itertuples():
#             first_value = str(row[1]).strip()
#             if first_value.lower() in ['', 'nan', 'none', 'null']:
#                 continue
#             if first_value in first_column_data_values:
#                 continue
#             if row.Index-1 in pat_blank_lines or row.Index-1 in pat_headers:
#                 predicted_pat_sub_headers.append(row.Index)
#             else: 

#                 value_tokens = first_value.lower().split() 
#                 pattern, symbols, case, value_num_tokens, value_num_chars = pat.generate_pattern_symbols_and_case(str(first_value).strip(), args.outlier_sensitive)
#                 if args.normalize_decimals==True:                    
#                     column_patterns, column_symbols=pat.normalize_decimals_numbers( [pattern]+first_column_value_patterns,  [symbols]+first_column_value_symbols)
#                     # input(f'\nrow_{row.Index} {[pattern]+first_column_value_patterns}')
#                     # print(f'column_patterns={column_patterns}')
#                 value_pattern_summary, value_chain_consistent = pat.generate_pattern_summary(column_patterns) 

#                 # print(f'value_pattern_summary={value_pattern_summary}')       
#                 summary_strength = sum(1 for x in column_patterns if len(x)>0)           
#                 bw_patterns = [list(reversed(pattern)) for pattern in column_patterns]
#                 value_pattern_BW_summary,_=pat.generate_pattern_summary(bw_patterns)
#                 value_symbol_summary = pat.generate_symbol_summary(column_symbols)
#                 case_summary = pat.generate_case_summary([case]+ first_column_value_cases)
#                 length_summary = pat.generate_length_summary( [value_num_chars]+first_column_value_char_lengths) 
#                 column_values = [first_value] + first_column_data_values 
#                 column_tokens = [value_tokens] + first_column_value_tokens

#                 cand_subhead_data_cell_rules_fired = []
#                 line_agreements[0]={}
#                 line_agreements[0][0]={}
#                 line_agreements[0][0]['agreements']=[]
#                 line_agreements[0][0]['null_equivalent']=False
#                 for rule in pat.cell_rules["data"].keys():
#                     rule_fired = False            
#                     non_empty_patterns=0
#                     if len(column_patterns)>0 and first_value.lower() not in pat.null_equivalent_values:
#                         for pattern in column_patterns:
#                             if pattern!=[]:
#                                 non_empty_patterns+=1
#                         if len(column_patterns)>=2 and non_empty_patterns>=2: 
#                             rule_fired = eval_data_cell_rule(rule, column_values, column_tokens, value_pattern_summary, value_chain_consistent, value_pattern_BW_summary, value_symbol_summary, column_symbols, column_patterns, case_summary, len(column_values))
#                             if rule_fired==True and "_REPEATS_" not in rule:
#                                 line_agreements[0][0]['agreements'].append(rule)
#                                 # if row.Index == 8:
#                                 #     print(f'\n\trow_{row.Index}:column_values={column_values}')
#                                 #     print(f'\n\trow_{row.Index}:column_patterns={column_patterns}')
#                                 #     print(f'\n\trow_{row.Index}:value_pattern_BW_summary={value_pattern_BW_summary}')
#                                 #     input(f'\n\trow_{row.Index}:value_pattern_summary={value_pattern_summary}')
#                 # if row.Index == 8:
#                 #     print(f"\nrow_{row.Index}:{line_agreements[0][0]['agreements']}")

#                 value_disagreements = []
#                 disagreement_summary_strength=summary_strength-1
#                 if len(pattern)>0:                      
#                     repetitions_of_candidate = np.count_nonzero(column_values[1:]==first_value)
#                     neighbor=''
#                     try:
#                         neighbor = column_values[1]
#                         repetitions_of_neighbor = np.count_nonzero(column_values[2:]==neighbor)                        
#                     except:
#                         repetitions_of_neighbor = 0                
                    
#                     for rule in pat.cell_rules["not_data"].keys():
#                         rule_fired = False
#                         if disagreement_summary_strength>0 and ( all_numbers(column_symbols)==False or is_number(symbols)==False):              
#                             rule_fired = eval_not_data_cell_rule(rule, repetitions_of_candidate, repetitions_of_neighbor, neighbor, value_pattern_summary, value_pattern_BW_summary, value_chain_consistent, value_symbol_summary, case_summary, length_summary, pattern, symbols, case, value_num_chars, disagreement_summary_strength, line_agreements, 0, 0)
#                             if rule_fired == True and "_REPEATS_" not in rule:
#                                 value_disagreements.append(rule)

#                 #######################################################################v######
#                 #  DATA value classification 
#                 data_score = max_score(line_agreements[0][0]['agreements'], rule_weights['data_cell'], args.weight_lower_bound)
#                 POPULATION_WEIGHT = 1-(1-args.p)**(2*summary_strength)
#                 if data_score!=None:
#                     if args.summary_population_factor:
#                         cell_data_score = data_score*POPULATION_WEIGHT
#                     else:
#                         cell_data_score = data_score

#                 #######################################################################v######
#                 #  NOT DATA value classification        
#                 not_data_score = max_score(value_disagreements, rule_weights['not_data_cell'], args.not_data_weight_lower_bound)
#                 POPULATION_WEIGHT = 1-(1-args.p)**(2*disagreement_summary_strength)
#                 if not_data_score!=None:
#                     if args.summary_population_factor:
#                         cell_not_data_score = not_data_score*POPULATION_WEIGHT
#                     else:
#                         cell_not_data_score = not_data_score 

#                 # print(f"\n{row.Index} data rules fired ={line_agreements[1][0]['agreements']}, not data rules fired ={value_disagreements}\ncell_data_score={cell_data_score}, cell_not_data_score={cell_not_data_score}")
#                 if  cell_data_score> cell_not_data_score:# candidate subheader is definitely data, move along
#                     # if row.Index == 31:
#                     # print(f"row_{row.Index} agreements:{line_agreements[0][0]['agreements']}")
#                     # print(f'row_{row.Index} disagreements:{value_disagreements}')

#                     continue            

#                 if (row.Index-1 in predicted_pat_sub_headers and row.Index-2 in predicted_pat_sub_headers):
#                     continue  

#                 if row.Index!=cand_data.index[-1]:
#                     predicted_pat_sub_headers.append(row.Index)

#     # print(f'predicted_pat_sub_headers={predicted_pat_sub_headers}')              
#     for s_i, subheader in enumerate(predicted_pat_sub_headers):
#         if subheader not in subheader_scope.keys():
#             if s_i+1==len(predicted_pat_sub_headers):
#                 subheader_scope[subheader] = list(range(subheader+1,cand_data.index[-1]+1))
#             else:
#                 next_s_i = s_i+1
#                 while next_s_i<len(predicted_pat_sub_headers):
#                     next_subh = predicted_pat_sub_headers[next_s_i]
#                     if next_subh not in subheader_scope:
#                         subheader_scope[subheader] = list(range(subheader+1,next_subh))
#                         break
#                     next_s_i+=1            
            
#     return  aggregation_rows, subheader_scope


def aggregation_first_value_of_row(row_values):
    fired=False
    numeric_value_seen = False
    aggregation_keyword_in_first_value=False

    if len(row_values)>1:
        first_value = str(row_values[0]).strip().lower()
        for aggregation_keyword in ['total']:#, 'average', 'agv', 'mean', 'percentage', '(%)', 'difference'
            if aggregation_keyword in first_value:
                aggregation_keyword_in_first_value = True
                break
        if aggregation_keyword_in_first_value==True:
            values = [str(value).strip().lower() for value in row_values[1:]]
            for value in values:
                is_number = True
                for char in value:
                    if char.isdigit() or char=='.' or char==',' or char==' ':
                        continue
                    else:
                        is_number=False
                        break
                if is_number:    
                    numeric_value_seen = True
                    break
            if numeric_value_seen==True:
                fired = True
    return fired

def assess_data_line(row_values):
    data_line_events = []
    fired, times = line_has_null_equivalent(row_values)
    if fired:
        if times == 1:
            data_line_events.append("ONE_NULL_EQUIVALENT_ON_LINE")
        if times>=2:
            data_line_events.append("NULL_EQUIVALENT_ON_LINE_2_PLUS")

    if aggregation_first_value_of_row(row_values):
        data_line_events.append("AGGREGATION_TOKEN_IN_FIRST_VALUE_OF_ROW")

    if contains_datatype_keyword(row_values):
        data_line_events.append("CONTAINS_DATATYPE_CELL_VALUE")

    return  data_line_events

def contains_datatype_keyword(row_values):
    row_values = [str(elem).strip().lower()for elem in row_values] 
    fired=False
    if len(row_values)>1:
        for value in row_values[1:]:
            if value in pat.datatype_keywords:
                fired=True
                break
    return fired

def assess_non_data_line(row_values,before_data,all_summaries_empty,row_index,dataframe, min_left_non_null=1):
    not_data_line_events = []
    left_non_null = 0
    # nulls_on_line = sum([1 for i in row_values if str(i).strip().lower() in ['','nan','none','null']])
    # if len(row_values)==nulls_on_line:
    #     not_data_line_events.append('ALL_VALUES_STRICTLY_NULL')
    
    if before_data==True:
        nulls_seen = False
        row_values = [str(elem).strip().lower()for elem in row_values] 
                      
        for value in row_values:
            if nulls_seen == False:
                if value not in ['nan', 'none', '', ' ']:
                    left_non_null+=1
                else:
                    nulls_seen = True
            else:
                if value not in ['nan', 'none', '', ' ']:
                    before_data = False
                    break

    if before_data==True and left_non_null<=1:
        not_data_line_events.append('UP_TO_FIRST_COLUMN_COMPLETE_CONSISTENTLY') 

    if row_index+1!=dataframe.shape[0] and len(row_values)>0 and row_values[0].lower() in ['','none', 'nan']:
        if len(row_values)>1:
            DATA = False
            for value_index, value in enumerate(row_values[1:]):
                column = dataframe.iloc[row_index:min(row_index+5, dataframe.shape[0]), value_index+1:value_index+2]
                if str(value).strip().lower() not in ['', 'nan','none','null'] and column_complete(column):
                    DATA = True
            if DATA==False:
                not_data_line_events.append('STARTS_WITH_NULL')    
        else:
            not_data_line_events.append('STARTS_WITH_NULL')

    if row_index+1!=dataframe.shape[0] and all_summaries_empty==True:
        not_data_line_events.append('NO_SUMMARY_BELOW')

    if len(row_values)>0 and row_values[0].lower() not in ['','none', 'nan'] and len([i for i in row_values if i.lower() not in  ['','none', 'nan']])<=2:
        for footnote_phrase in pat.footnote_keywords:
            if row_values[0].lower().startswith(footnote_phrase):
                not_data_line_events.append("FOOTNOTE")
                break
        if len(row_values)>0 and row_values[-1].lower()=='#ref!' and "FOOTNOTE" not in not_data_line_events:
            not_data_line_events.append("FOOTNOTE")
    if metadata_header_keywords(row_values):
        not_data_line_events.append("METADATA_TABLE_HEADER_KEYWORDS")


    return not_data_line_events

def metadata_header_keywords(row_values):
    fired = False
    row_values = [str(value).strip().lower() for value in row_values]
    for value in row_values:
        if value in pat.metadata_table_header_keywords:
            fired = True
            break
        if value in ['_'.join((value).split(' ')) for value in pat.metadata_table_header_keywords]:
            fired = True
            break
        if value in [''.join((value).split(' ')) for value in pat.metadata_table_header_keywords]:
            fired = True
            break        
    return fired

def non_nulls_in_line(row_values):
    non_null_percentage = None
    non_nulls = 0
    for value in row_values:
        if value.strip().lower() not in null_equivalent:
            non_nulls+=1
    if len(row_values)>0:
        non_null_percentage = non_nulls/len(row_values)
    return non_nulls,non_null_percentage
    

def column_complete(column):    
    if column.isnull().any().any():
        return False
    else:
        return True

def max_confidence(line_scores):
    predata_row_confidence = 0
    if len(line_scores)>0:
        predata_row_confidence=max(line_scores)   
    return predata_row_confidence

def max_score(events,weights,weight_lower_bound):
    if len(events)>0:
        # print(disagreements)
        event_score = []
        for event in events:
            if weights[event]!=None and weights[event]>=weight_lower_bound:
                event_score.append((event, weights[event]))
        if len(event_score)>0:
            event_score.sort(key=lambda x: x[1], reverse=True)
            return event_score[0][1]
        else:
            return 0
    else:
        return 0

def all_numbers(column_symbols):
    for symbols in column_symbols:
        if len(symbols) == 0:
            continue
        if is_number(symbols)==False:
            return False
    return True

def is_number(symbols):
    if set(symbols) <= set(['D',' ','S','.']) and 'D' in symbols:
        return True
    else:
        return False

def predict_line_label(data_confidence, not_data_confidence):
    line = dict()
    line["value"] = dict()
    line_is_data_conf = 0

    if data_confidence<=not_data_confidence:            
        line["label"]='NOT_DATA'            
        line["value"]["confusion_index"] = confusion_index(not_data_confidence, data_confidence)
        line["value"]["confidence"] = not_data_confidence
        line["value"]["difference"] = not_data_confidence-data_confidence
        # line_is_data_conf = 0
    else:
        line["label"]='DATA' 
        line["value"]["confusion_index"]= line_is_data_conf
        line["value"]["confidence"]= data_confidence
        line["value"]["difference"]= data_confidence-not_data_confidence            
        line_is_data_conf = confusion_index(data_confidence,not_data_confidence)

    return line,  line_is_data_conf      

def predict_combined_data_confidences(dataframe, data_confidence, not_data_confidence, max_candidates):
    combined_data_line_confidences = {}
    line_predictions = {}   
    offset = dataframe.index[0]

    row_counter = -1
    # for row_index, candidate_data_row in dataframe.iterrows(): 
    for row_index in dataframe.index: 
        row_counter +=1
        if row_counter == offset+max_candidates:
            break
        if row_index not in data_confidence.keys() or row_index not in not_data_confidence.keys():
            break  
        prediction, line_is_data_conf = predict_line_label(data_confidence[row_index], not_data_confidence[row_index])
        line_predictions[row_index] = prediction
        combined_data_line_confidences[row_index]=line_is_data_conf

    return combined_data_line_confidences, line_predictions

def predict_line_confidences(dataframe, dataframe_patterns, args, dataframe_labels, groundtruth_key, line_agreements):
    
    #unpack arguments
    data_weights, not_data_weights, max_candidates, summary_population_factor, weight_input, impute_nulls, max_summary_strength, max_line_depth, max_attributes,ignore_left, p, outlier_sensitive, normalize_decimals, weight_lower_bound, not_data_weight_lower_bound = args

    #unpack patterns
    sample_patterns, sample_symbols, sample_cases, sample_token_len, sample_chars_len,sample_tokens=dataframe_patterns

    #################################
    # initialize
    not_data_line_confidences = {} 

    data_line_confidences = {}

    line_disagreements = {}
    before_data = True
    #
    #################################
    # input(dataframe)
    row_counter = -1
    for row_index, candidate_row in dataframe.iterrows():
        row_counter+=1
        # initialize
        line_disagreements[row_index] = {}
        # print('-row_index:'+str(row_index))
        if row_counter==max_candidates:
            break
        if row_index not in line_agreements.keys():
            # print('predict_line_confidences: label '+str(row_index)+' not in line_agreements')
            # pp.pprint(line_agreements)
            break

        candidate_row_disagreements = []
        candidate_row_agreements = []
        
        candidate_row_values = [str(elem) if elem is not None else elem for elem in candidate_row.tolist()]

        for column in dataframe: 
            columnindex = dataframe_labels.index(column)
            candidate_value = str(candidate_row_values[columnindex])
            value_disagreements = []

            #################################################v######################v######
            #  DATA value classification
            value_agreements = line_agreements[row_index][columnindex]['agreements']
            summary_strength = line_agreements[row_index][columnindex]['summary_strength']            

            # if there are no lines below me to check agreement, 
            # and line before me exists and was data
            # see impute agreements
            if row_counter>0 and (line_agreements[row_index][columnindex]['null_equivalent']==True or line_agreements[row_index][columnindex]['summary_strength']==1) and impute_nulls==True and row_index-1 in line_agreements.keys() and columnindex in line_agreements[row_index-1].keys()  and row_index-1 in data_line_confidences.keys() and data_line_confidences[row_index-1]>not_data_line_confidences[row_index-1]:
                value_agreements = line_agreements[row_index-1][columnindex]['agreements']
                summary_strength = line_agreements[row_index-1][columnindex]['summary_strength']
            if row_counter>0 and line_agreements[row_index][columnindex]['summary_strength']==0 and line_agreements[row_index][columnindex]['aggregate'] and row_index-2 in line_agreements.keys() and columnindex in line_agreements[row_index-2].keys() and row_index-2 in data_line_confidences.keys() and data_line_confidences[row_index-2]>not_data_line_confidences[row_index-2]:
                value_agreements = line_agreements[row_index-2][columnindex]['agreements']
                summary_strength = line_agreements[row_index-2][columnindex]['summary_strength']

            # otherwise, nothing was wrong, i can use my own damn agreements as initialized
            
            data_score = max_score(value_agreements,data_weights,weight_lower_bound)
            POPULATION_WEIGHT = 1-(1-p)**(2*summary_strength)
            if data_score!=None:
                if summary_population_factor:
                    candidate_row_agreements.append(data_score*POPULATION_WEIGHT)
                else:
                    candidate_row_agreements.append(data_score)    

            #################################################################################
            # NOT DATA value classification

            cand_pattern, cand_symbols, cand_case, cand_num_tokens, cand_num_chars = pat.generate_pattern_symbols_and_case(candidate_value.strip(), outlier_sensitive)
            column_patterns = copy.deepcopy(sample_patterns[columnindex][row_index+1:])
            column_symbols = copy.deepcopy(sample_symbols[columnindex][row_index+1:])
            column_cases = copy.deepcopy(sample_cases[columnindex][row_index+1:])
            column_lengths = copy.deepcopy(sample_chars_len[columnindex][row_index+1:])

            if max_summary_strength!=None:
                nonempty_patterns=0
                nonempty_patterns_idx = 0
                for nonempty_patterns_idx in range(0, min(len(column_patterns),max_line_depth)):
                    if len(column_patterns[nonempty_patterns_idx])>0:
                        nonempty_patterns+=1
                        if nonempty_patterns==max_summary_strength:
                            column_patterns = column_patterns[:nonempty_patterns_idx+1]
                            column_symbols = column_symbols[:nonempty_patterns_idx+1]
                            column_cases = column_cases[:nonempty_patterns_idx+1]
                            column_lengths = column_lengths[:nonempty_patterns_idx+1]
                            break

            if normalize_decimals==True:
                cand_pattern,cand_symbols,cand_case,cand_num_tokens,cand_num_chars,column_patterns,column_symbols = pat.normalize_decimals_numbers_predata(candidate_value.strip(),cand_pattern,cand_symbols,cand_case,cand_num_tokens,cand_num_chars,column_patterns,column_symbols, outlier_sensitive)
          
            value_pattern_summary, value_chain_consistent = pat.generate_pattern_summary(column_patterns)
            disagreement_summary_strength = sum(1 for x in column_patterns if len(x)>0)

            bw_patterns =  [list(reversed(pattern)) for pattern in column_patterns]
            value_pattern_BW_summary,_=pat.generate_pattern_summary(bw_patterns)
            value_symbol_summary = pat.generate_symbol_summary(column_symbols)
            case_summary = pat.generate_case_summary(column_cases)
            length_summary = pat.generate_length_summary(column_lengths)
            

            #If the value/imputed value is not empty or null-equivalent
            if len(cand_pattern)>0:
                columnvalues = dataframe.iloc[row_counter:,dataframe_labels.index(column)].tolist()
                if len(columnvalues)>1:
                    neighbor = columnvalues[1]
                    repetitions_of_candidate = np.count_nonzero(columnvalues[1:]==candidate_value)
                    if len(columnvalues)>2:
                        repetitions_of_neighbor = np.count_nonzero(columnvalues[2:]==neighbor)
                    else:
                        repetitions_of_neighbor = 0 
                else:
                    repetitions_of_candidate=0
                    repetitions_of_neighbor = 0  
                
                #check7
                if  disagreement_summary_strength>0 and (all_numbers(column_symbols)==False or is_number(cand_symbols)==False):
                    value_disagreements = find_disagreements(repetitions_of_candidate,repetitions_of_neighbor,neighbor,value_pattern_summary,value_pattern_BW_summary,value_chain_consistent,value_symbol_summary,case_summary,length_summary,cand_pattern,cand_symbols,cand_case,cand_num_chars,disagreement_summary_strength)

            column_patterns = None
            column_symbols = None
            column_cases = None
            column_lengths = None  
  
            if row_index+1 in line_agreements.keys() and columnindex in line_agreements[row_index+1].keys():
                value_below_agreements = line_agreements[row_index+1][columnindex]['agreements'] 
                for agreement in value_below_agreements:  
                    if line_agreements[row_index][columnindex]['null_equivalent']==False and line_agreements[row_index][columnindex]['aggregate']==False and agreement not in line_agreements[row_index][columnindex]['agreements'] and agreement in not_data_value_rules.keys():
                        value_disagreements.append(agreement)                        

            not_data_score = max_score(value_disagreements,not_data_weights,not_data_weight_lower_bound)

            POPULATION_WEIGHT = 1-(1-p)**(2*disagreement_summary_strength)
            if not_data_score!=None:
                if summary_population_factor:
                    candidate_row_disagreements.append(not_data_score*POPULATION_WEIGHT)
                else:
                    candidate_row_disagreements.append(not_data_score)
            line_disagreements[row_index][columnindex] = value_disagreements

            ########################################################################

        ################## finished processing all values in row ##################
        #################################################################################
        # NOT DATA row classification
        #

        line_not_data_evidence =  [score for score in candidate_row_disagreements] 
        if len(candidate_row_values)>0 and weight_input=='values_and_lines':
            line_disagreements[row_index]['line'] = []

            header_events_fired = collect_events_on_row(candidate_row_values)
            arithmetic_events_fired = collect_arithmetic_events_on_row(candidate_row_values)

            for header_event in header_events_fired:
                line_disagreements[row_index]['line'].append(header_event)
                if not_data_weights[header_event]!=None and not_data_weights[header_event]>not_data_weight_lower_bound:
                    line_not_data_evidence.append(not_data_weights[header_event])
            if len(arithmetic_events_fired)>0:
                arithmetic_event_score = max_score(arithmetic_events_fired,not_data_weights,not_data_weight_lower_bound)
                line_not_data_evidence.append(arithmetic_event_score)
            for sequence_event in arithmetic_events_fired:
                line_disagreements[row_index]['line'].append(sequence_event)

            all_summaries_empty = line_agreements[row_index]["all_summaries_empty"]

            if row_counter>0 and data_line_confidences[row_index-1]>not_data_line_confidences[row_index-1]:
                before_data=False

            if dataframe.shape[1]>1:
                not_data_line_rules_fired = assess_non_data_line(candidate_row_values,before_data,all_summaries_empty,row_counter,dataframe)
                if "UP_TO_FIRST_COLUMN_COMPLETE_CONSISTENTLY" not in not_data_line_rules_fired:
                    before_data = False
                for event in not_data_line_rules_fired:
                    line_disagreements[row_index]['line'].append(event)
                    if not_data_weights[event]!=None and not_data_weights[event]>not_data_weight_lower_bound:
                        line_not_data_evidence.append(not_data_weights[event]) 
        
        not_data_conf = probabilistic_sum(line_not_data_evidence)
        not_data_line_confidences[row_index] = not_data_conf

        #################################################################################
        # DATA row classification
        line_is_data_evidence =  [score for score in candidate_row_agreements] 
        if len(candidate_row_values)>0 and weight_input=='values_and_lines':
            line_agreements[row_index]['line'] = []
            line_is_data_events = assess_data_line(candidate_row_values)
            for rule in line_is_data_events:
                line_agreements[row_index]['line'].append(rule)
                if data_weights[rule]!=None and data_weights[rule]>weight_lower_bound:
                    line_is_data_evidence.append(data_weights[rule])
        # input('line_agreements='+str(line_agreements[row_index]))
        # calculate confidence that this row is data
        data_conf = probabilistic_sum(line_is_data_evidence)
        data_line_confidences[row_index] = data_conf

        # if row_index in [0,1,2]:
        #     print('\nline_agreements['+str(row_index)+']')
        #     pp.pprint(line_agreements[row_index])
        #     input() 
        #     print('\nline_disagreements['+str(row_index)+']')
        #     pp.pprint(line_disagreements[row_index])
        #     input()  
        #     print('\ndata_conf='+str(data_conf))  
        #     input('not_data_conf='+str(not_data_conf))           
    ##########################################################################################################################
        
    # print('\nnot_data_line_confidences:')
    # pp.pprint(not_data_line_confidences)
    return data_line_confidences, not_data_line_confidences, line_disagreements


def partof_multiword_value_repeats_once(candidate_tokens, column_tokens):  
    # print(column_tokens)  
    repeats_of_any_token = 0
    repeats_of_token= {}
    for candidate_token in candidate_tokens:
        repeats_of_token[candidate_token]=0

    for value_tokens in column_tokens:       
        if (set(candidate_tokens) & set(value_tokens)):
            for token in (set(candidate_tokens) & set(value_tokens)):
                repeats_of_token[token]+=1
    if sum([repeats_of_token[key] for key in repeats_of_token.keys()])==1:
        return True
    return False

def partof_multiword_value_repeats_at_least_k(candidate_tokens, column_tokens, k = 1):    
    repeats_of_any_token = 0
    for value_tokens in column_tokens:       
        if (set(candidate_tokens) & set(value_tokens)):
            # print('Rule_2_b':)
            # input(set(candidate_tokens) & set(value_tokens))
            repeats_of_any_token+=1
            if repeats_of_any_token==k:
                return True
    return False

def is_consistent_symbol_sets(column_symbols):
    consistent_symbol_set = True
    prev_symbol_set = set()
    if len(column_symbols)>1:
        prev_symbol_set = column_symbols[0]
        for ss in column_symbols[1:]:
            if len(prev_symbol_set)==0:
                prev_symbol_set= ss
                continue
            elif len(ss)==0:
                continue
            elif ss != prev_symbol_set:
                consistent_symbol_set = False
                prev_symbol_set = set()
    return consistent_symbol_set, prev_symbol_set

# def is_consistent_symbol_sets(column_symbols):
#     symbol_sets_list = [l for l in column_symbols if len(l)>0]
#     symbol_sets_set = set(symbol_sets_list)
#     if len(symbol_sets_set)<=1:
#         return True, symbol_sets_set
#     else:
#         return False, None

def find_disagreements(repetitions_of_candidate,repetitions_of_neighbor,neighbor,value_pattern_summary,value_pattern_BW_summary,value_chain_consistent,value_symbol_summary,case_summary,length_summary,cand_pattern,cand_symbols,cand_case, cand_length,summary_strength,original_value=True ):

    disagreements = []
    cand_symbol_chain = [x[0] for x in cand_pattern]
    value_symbol_chain = [x[0] for x in value_pattern_summary]
    summary_min_length = length_summary["min"]
    summary_max_length = length_summary["max"]

    # if len(value_pattern_summary)!=0 and pat.pattern_disagrees(value_pattern_summary, cand_pattern):
    #     disagreements.append("FW")

    # if len(value_pattern_BW_summary)!=0 and pat.pattern_disagrees(value_pattern_BW_summary, list(reversed(cand_pattern))):
    #     disagreements.append("BW") 

    if len(value_pattern_summary)!=0 and cand_pattern[0][0]!=value_pattern_summary[0][0]:
        # if (value_pattern_summary[0][0]=='(' or cand_pattern[0][0]=='(')==False:
        disagreements.append("First_FW_Symbol_disagrees")

    if len(value_pattern_BW_summary)!=0 and cand_pattern[-1][0]!=value_pattern_BW_summary[0][0]:
        # if (value_pattern_BW_summary[0][0] in [')','.'] or cand_pattern[-1][0] in [')','.'])==False:
        disagreements.append("First_BW_Symbol_disagrees") 

    # if len(value_pattern_summary)!=0 and cand_pattern[0][0]!=value_pattern_summary[0][0] and len(value_pattern_BW_summary)!=0 and cand_pattern[-1][0]!=value_pattern_BW_summary[0][0]  :
    #     disagreements.append("First_FW_AND_First_BW")
 
    if summary_strength>1 and value_chain_consistent==True and pat.symbol_chain_disagrees(value_symbol_chain,cand_symbol_chain):
        disagreements.append("SymbolChain")

    # if len(value_symbol_summary)!=0 and pat.symbol_summary_disagrees(value_symbol_summary,cand_symbols):
    #     disagreements.append("SS")   

    if len(case_summary)!=0 and case_summary!=cand_case:
        disagreements.append("CC")
    
    # if value_chain_consistent and len(value_pattern_summary)!=0 and pat.pattern_disagrees(value_pattern_summary, cand_pattern):
    #     disagreements.append("Consistent_smr_disagrees_w_fw")

    # if len(value_pattern_summary)!=0 and neighbor not in null_equivalent and repetitions_of_neighbor>=2 and repetitions_of_candidate==0:
    #     disagreements.append("repetitions")

    # if value_chain_consistent and len(value_pattern_summary)>=1 and pat.pattern_disagrees(value_pattern_summary, cand_pattern):
    #     disagreements.append("FW_disagrees_w_consistent_summary")
    return disagreements

def find_agreements(max_line_depth,columnvalues,column_tokens,all_values_summary,consistent_symbol_chain,pattern_BW_summary,value_symbol_summary,column_symbols, column_patterns, case_summary, max_values_lookahead=6 ):
    # all_values_summary example:[['A', 3], ['S', 1], ['A', 0], ['S', 1], ['A', 0], ['S', 1], ['A', 0]]

    agreements = []                 
    # print('columnvalues:'+str(columnvalues))
    candidate = columnvalues[0]
    candidate_tokens= column_tokens[0]
    # print('candidate:'+str(candidate))
    # print(candidate_tokens)
    consistent_symbol_set = is_consistent_symbol_sets(column_symbols)

    if candidate==None  or str(candidate).strip().lower() in null_equivalent:
        return agreements #No point looking for agreements, move along

    # value repeats in the rest of the column
    candidate_count = 0
    candidate_selectivity = 0

    if len(columnvalues)>1:
        candidate_count = columnvalues[1:min(max_values_lookahead,len(columnvalues))].count(candidate)
        # candidate_selectivity = candidate_count/len(columnvalues[1:min(max_values_lookahead,len(columnvalues))])
        if candidate_count>=2:
            # 'value repetition twice or more below me'
            agreements.append("Rule_1_b")

    if len(columnvalues)>2:
        candidate_count = columnvalues[2:min(max_values_lookahead,len(columnvalues))].count(candidate)        
        if candidate_count==1:
            # 'value repetition only once in values below me, skip the adjacent value'
            agreements.append("Rule_1_a")

    if  consistent_symbol_chain==True and len(all_values_summary)==1 and all_values_summary[0][0] == 'D' and all_values_summary[0][1]>0:
        agreements.append("Rule_3")

    if consistent_symbol_chain==True and len(all_values_summary)==1 and  all_values_summary[0][0] == 'D':
        # 'consistently ONE symbol = D'                
        agreements.append("Rule_4_a")

    if consistent_symbol_chain==True and len(all_values_summary) == 2 and all_values_summary[0][0] == 'D':
        # 'consistently TWO symbols, the first is a digit'                
        agreements.append("Rule_4_b")

    if len(all_values_summary)>=2 and all_values_summary[0][0] == 'D':
        # 'two or above symbols in the FW summary, the first is a digit'                
        agreements.append("Rule_4_fw")

    if len(pattern_BW_summary)>=2 and pattern_BW_summary[0][0] == 'D':
        # two or above symbols in the BW summary, the first is a digit'
        agreements.append("Rule_4_bw")

    if pat.numeric_patterns(column_symbols, column_patterns):
        # 'all values digits, optionally have . or , or S '
        agreements.append("Rule_5")

    if len(all_values_summary)>=3 and 'S' not in [x[0] for x in all_values_summary]:
        # 'three or above symbols in FW summary that do not contain a Space'
        agreements.append("Rule_6")

    if len(pattern_BW_summary)>=3 and 'S' not in [x[0] for x in pattern_BW_summary]:
        # 'three or above symbols in BW summary that do not contain a Space'
        agreements.append("Rule_7")

    if consistent_symbol_set and len(value_symbol_summary)>=2 and 'S' not in value_symbol_summary and '_' not in  value_symbol_summary:
        # "consistently at least two symbols in the symbol set summary, none of which are S or _"
        agreements.append("Rule_8")

    if consistent_symbol_chain==True and len(all_values_summary)>=2:
        # 'two or more symbols consistent chain'
        agreements.append("Rule_10")

    if len(all_values_summary)>=2 and 'S' not in [x[0] for x in all_values_summary]:
        # 'two or above symbols in FW summary that do not contain a Space'
        agreements.append("Rule_11_fw")

    if len(pattern_BW_summary)>=2 and 'S' not in [x[0] for x in pattern_BW_summary]:
        # 'two or above symbols in BW summary that do not contain a Space'
        agreements.append("Rule_11_bw")

    if len(all_values_summary)>=2 and 'S' not in [x[0] for x in all_values_summary[0:2]]:
        # 'two or above symbols in FW summary, the first two do do not contain a Space'
        agreements.append("Rule_12_fw")

    if len(pattern_BW_summary)>=2 and 'S' not in [x[0] for x in pattern_BW_summary[0:2]]:
        # two or above symbols in BW summary, the first two do not contain a Space
        agreements.append("Rule_12_bw")

    if  len(all_values_summary)==1 and  all_values_summary[0][0]=='D' and all_values_summary[0][1]>=5:
        agreements.append("Rule_13_fw")

    if  len(pattern_BW_summary)==1 and  pattern_BW_summary[0][0]=='D' and pattern_BW_summary[0][1]>=5:
        agreements.append("Rule_13_bw")

    if  len(all_values_summary)==1 and  all_values_summary[0][0]=='D' and all_values_summary[0][1]==1:
        agreements.append("Rule_14_fw")

    if  len(pattern_BW_summary)==1 and  pattern_BW_summary[0][0]=='D' and pattern_BW_summary[0][1]==1:
        agreements.append("Rule_14_bw")

    if  len(all_values_summary)==1 and  all_values_summary[0][0]=='D' and all_values_summary[0][1]==4:
        agreements.append("Rule_15_fw")

    if  len(pattern_BW_summary)==1 and  pattern_BW_summary[0][0]=='D' and pattern_BW_summary[0][1]==4:
        agreements.append("Rule_15_bw") 

    if len(all_values_summary)>=4:
        # four or more symbols in the FW summary
        agreements.append("Rule_17_fw")
    
    if len(pattern_BW_summary)>=4:
        # four or more symbols in the BW summary
        agreements.append("Rule_17_bw")

    if case_summary == 'ALL_CAPS':
        agreements.append("Rule_18")

    if len(columnvalues)>2 and partof_multiword_value_repeats_once(candidate_tokens, column_tokens[2:min(max_values_lookahead,len(columnvalues))]):
        #Only one alphabetic token from multiword value repeats below, and it repeats only once
        agreements.append('Rule_2_a')

    if len(columnvalues)>2 and partof_multiword_value_repeats_at_least_k(candidate_tokens, column_tokens[2:min(max_values_lookahead,len(columnvalues))], 2):
        #'At least one alphabetic token from multiword value repeats below at least twice'
        agreements.append('Rule_2_b')        

    return agreements  

def find_line_agreements(dataframe,dataframe_patterns, args, start_index, do_ignore_left):
    max_data,max_summary_strength,max_line_depth,max_attributes, ignore_left, outlier_sensitive, normalize_decimals = args

    line_agreements= {}
    dataframe_labels = []
    for column in dataframe:
        dataframe_labels.append(column)
    # create patterns for all values in dataframe...
    sample_patterns, sample_symbols, sample_cases, sample_token_len, sample_chars_len, sample_tokens = dataframe_patterns
    # input(dataframe)
    first_row_index = list(dataframe.index)[0]
    row_counter= -1
    for row_index, cand_row in dataframe.iterrows():
        row_counter+=1
        # input('row_index='+str(row_index))
        # undersampling
        if max_data!=0 and int(start_index)+max_data == row_index:
            break

        row_values = [str(elem) if elem is not None else elem for elem in cand_row.tolist()]
        null_equivalent_fired, times = line_has_null_equivalent(row_values)
        line_agreements[row_index]={}
        all_summaries_empty = True
        for column in dataframe:            
            agreements = []
            columnindex = dataframe_labels.index(column)
            line_agreements[row_index][columnindex] = {}
            line_agreements[row_index][columnindex]['agreements'] = agreements
            line_agreements[row_index][columnindex]['summary_strength'] = 0
            candidate_value = str(row_values[columnindex])

            value_lower = str(candidate_value).strip().lower()
            value_tokens = value_lower.split()
            line_agreements[row_index][columnindex]['aggregate'] = (len(value_tokens)>0 and not set(value_tokens).isdisjoint(pat.aggregation_tokens))

            if candidate_value.strip().lower() in null_equivalent:
                line_agreements[row_index][columnindex]['null_equivalent'] = True
            else:
                line_agreements[row_index][columnindex]['null_equivalent'] = False

            column_patterns = copy.deepcopy(sample_patterns[columnindex][row_index:])
            column_symbols = copy.deepcopy(sample_symbols[columnindex][row_index:])
            column_cases = copy.deepcopy(sample_cases[columnindex][row_index:])
            column_lengths = copy.deepcopy(sample_chars_len[columnindex][row_index:])
            column_tokens = copy.deepcopy(sample_tokens[columnindex][row_index:])
     

            if max_summary_strength!=None:
                nonempty_patterns=0
                nonempty_patterns_idx = 0
                for nonempty_patterns_idx in range(0, min(len(column_patterns),max_line_depth)):
                    if len(column_patterns[nonempty_patterns_idx])>0:
                        nonempty_patterns+=1
                        if nonempty_patterns==max_summary_strength:
                            column_patterns = column_patterns[:nonempty_patterns_idx+1]                            
                            column_symbols = column_symbols[:nonempty_patterns_idx+1]
                            column_cases = column_cases[:nonempty_patterns_idx+1]
                            column_lengths = column_lengths[:nonempty_patterns_idx+1]
                            column_tokens = column_tokens[:nonempty_patterns_idx+1]
                            break
            if normalize_decimals==True:   
                column_patterns,column_symbols=pat.normalize_decimals_numbers(column_patterns, column_symbols)


            value_pattern_summary, value_chain_consistent = pat.generate_pattern_summary(column_patterns)            
            line_agreements[row_index][columnindex]['summary_strength'] = sum(1 for x in column_patterns if len(x)>0)


            bw_patterns = [list(reversed(pattern)) for pattern in column_patterns]
            value_pattern_BW_summary,_=pat.generate_pattern_summary(bw_patterns)
            value_symbol_summary = pat.generate_symbol_summary(column_symbols)
            case_summary = pat.generate_case_summary(column_cases)
            length_summary = pat.generate_length_summary(column_lengths)

            if null_equivalent_fired==True or len(value_pattern_summary)>0 or len(value_pattern_BW_summary)>0 or len(value_symbol_summary)>0 or len(case_summary)>0:                
                all_summaries_empty= False

            
            # Don't bother looking for agreements if there are no patterns or if the value on this line gives an empty pattern
            non_empty_patterns=0
            # if len(column_patterns)>0 and column_patterns[0]!=[]:
            if len(column_patterns)>0 and value_lower not in null_equivalent:
                for pattern in column_patterns:
                    if pattern!=[]:
                        non_empty_patterns+=1

                #there is no point calculating agreement over one value, a single value always agrees with itself.
                #in addition, many tables have bilingual headers, so agreement between two header values is very common, require nij>=3
                if len(column_patterns)>=2 and non_empty_patterns>=2:
                    columnvalues = dataframe.iloc[row_counter:,dataframe_labels.index(column)].tolist()

                    agreements = find_agreements(max_line_depth, columnvalues, column_tokens, value_pattern_summary, value_chain_consistent, value_pattern_BW_summary, value_symbol_summary, column_symbols, column_patterns,case_summary) 

            line_agreements[row_index][columnindex]['agreements'] = agreements
            
            column_patterns = None
            column_symbols = None
            column_cases = None
            column_lengths = None

        line_agreements[row_index]["all_summaries_empty"]=all_summaries_empty

    return line_agreements

def sample_file(filepath,max_batch= 100):
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
            failure= "file does not exist"
            print('\n~~~~~~~~~~~~~~~~~~~~~')
            print(filepath)
            print(failure)
            return batch, discovered_delimiter, discovered_encoding,encoding_language,encoding_confidence, failure, blanklines, google_detected_lang

        size_bytes = os.path.getsize(filepath)

        if size_bytes == 0:
            failure= "file is empty"
            print('\n~~~~~~~~~~~~~~~~~~~~~')
            print(filepath)
            print(failure)
            return batch, discovered_delimiter, discovered_encoding,encoding_language,encoding_confidence, failure, blanklines, google_detected_lang

        encoding_result = detect_encoding(filepath)
        # print('Encoding:'+str(encoding_result))
        discovered_encoding = encoding_result["encoding"]
        if discovered_encoding==None:
            failure = "No encoding discovered"
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
            failure = "Illegal file format"
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
                    if len(line) ==0 or sum(len(s.strip()) for s in line)==0:
                        blanklines.append(lineindex)
                    batch.append(line)
                    if len(batch)>=max_batch:
                        break
            f.flush()
    except Exception as e:
        print('\n~~~~~~~~~~~~~~~~~~~~~')
        print(filepath)
        failure = str(e)
        print(failure)

    return batch, discovered_delimiter, discovered_encoding,encoding_language,encoding_confidence, failure, blanklines, google_detected_lang


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


def detect_encoding(filepath):
    result={}
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




def line_has_null_equivalent(row_values):
    line_has_null_equivalent = False
    null_equivalent_counts = 0
    for value in row_values:
        if str(value).strip().lower() in pat.strictly_null_equivalent:
            null_equivalent_counts+=1
            line_has_null_equivalent = True
    return line_has_null_equivalent,  null_equivalent_counts

def match_sequence(legal_label_sequence, line_predictions, offset,first_double_data_index,data_window, label_weight='confidence'):
    # input('\n[METHOD] table_classifier_utilities.match_sequence')
    match_weight= 0
    # print('legal_label_sequence='+str(legal_label_sequence))
    for legal_label_idx, legal_label in enumerate(legal_label_sequence):
        # print(f'prediction of {legal_label} at sequence index_{offset+legal_label_idx}:')
        # pp.pprint(line_predictions[offset+legal_label_idx])
        if legal_label_idx>=first_double_data_index+data_window:
            break
        if  line_predictions[offset+legal_label_idx]['label']==legal_label:
            # print(f"match_weight={match_weight}+{line_predictions[offset+legal_label_idx]['value'][label_weight]}")
            match_weight+=line_predictions[offset+legal_label_idx]['value'][label_weight]
        else:
            # print(f"match_weight={match_weight}-{line_predictions[offset+legal_label_idx]['value'][label_weight]}")
            match_weight -=line_predictions[offset+legal_label_idx]['value'][label_weight]
    return match_weight


def predict_last_data_line(dataframe, predicted_fdl):
    predicted_last_data_line = -1
    predicted_footnotes = []
    predicted_last_data_line_found=False

    # if predicted_fdl!=-1:
    if dataframe.shape[0]>0:
        predicted_last_data_line = dataframe.index[-1]

    footnote_start = None 

    if dataframe.shape[1]>2:
        idx = list(dataframe.index[dataframe.iloc[:,2:].isnull().all(1)])
        if len(idx)>0:
            idx.sort(reverse=True)
            idxs = []
            if idx[0]==dataframe.index[-1]:
                for i, v in enumerate(idx):
                    if v <= predicted_fdl:
                        break
                    if (i+1 <len(idx) and idx[i+1] == v-1) or i+1 == len(idx):
                        idxs.insert(0, v)
                    else:
                        idxs.insert(0, v)
                        break   

                nans = dataframe.loc[idxs]
                FOOTNOTE_KEYWORD_SPOTTED = False
                for row in nans.itertuples():
                    first_value = str(row[1]).strip().lower()

                    for footnote_keyword in pat.footnote_keywords:
                        if first_value.startswith(footnote_keyword):
                            FOOTNOTE_KEYWORD_SPOTTED = True
                            break
                    if '=' in first_value:
                        FOOTNOTE_KEYWORD_SPOTTED = True                        

                    if len(first_value)>5 and ( ( (first_value[0].isdigit() or first_value[0]=='a' ) and first_value[1] in [' ', '.', '/', ')', ']', ':'] ) or (first_value[0]=='(' and ( first_value[1].isdigit() or first_value[1]=='a' ) and first_value[2]==')') ):
                        FOOTNOTE_KEYWORD_SPOTTED = True

                    if FOOTNOTE_KEYWORD_SPOTTED == True:
                        footnote_start = row.Index
                        predicted_last_data_line = footnote_start-1
                        break     
                


    if footnote_start==None:
        for row in dataframe[::-1].itertuples():
            
            row_values = []
            if len(row)>1:
                row_values = row[1:]
            row_index = row.Index

            if row_index == predicted_fdl:
                predicted_last_data_line_found=True
                predicted_last_data_line=row_index
                break

            if len(row_values)>0:
                first_value = str(row_values[0]).strip()
                for aggregation_phrase in pat.aggregation_functions:#['total', 'average', 'subtotal']
                    if first_value.lower().startswith(aggregation_phrase[0]):
                        predicted_last_data_line_found=True
                        break 
                if len(row_values)>2 and len([i for i in row_values[:-1] if str(i).strip().lower() not in ['', 'nan', 'none', 'null'] ])==0 and str(row_values[-1]).strip().lower() not in ['', 'nan', 'none', 'null'] :
                    continue

                if len(first_value)>1 and first_value[0].isdigit() and first_value[1] not in ['.','*', ':', ' ', '/', ')', ',']:
                    predicted_last_data_line_found=True
                    predicted_last_data_line=row_index
                    break

                if len(row_values)>2:
                    for value in row_values[2:]:
                        if str(value).strip().lower() not in ['', 'nan', 'none', 'null']:
                            predicted_last_data_line_found=True
                            break   
                elif len(row_values)==2:
                    for value in row_values[1:]:
                        if str(value).strip().lower() not in ['', 'nan', 'none', 'null']:
                            predicted_last_data_line_found=True
                            break 
                else:
                    break
            if predicted_last_data_line_found==True:
                predicted_last_data_line=row_index
                break

    if predicted_last_data_line!=-1 and predicted_last_data_line<dataframe.shape[0]-1:
        predicted_footnotes = list(range(predicted_last_data_line+1,dataframe.shape[0]))

    # print(f'predicted_last_data_line={predicted_last_data_line}')
    # print(f'predicted_footnotes={predicted_footnotes}')

    return predicted_last_data_line, predicted_footnotes

# def collect_dataframe_rules(csv_file, args):

#     # input(f'csv_file=\n\n{csv_file}\n')

#     ignore_left = args.ignore_left
#     if args.ignore_left >= csv_file.shape[1]: # trying to ignore more columns than the file actually has!
#         ignore_left = csv_file.shape[1]-1 # ignore all except the last one 
    
#     sample_patterns, sample_symbols, sample_cases, sample_token_len, sample_chars_len, sample_tokens = get_dataframe_signatures(csv_file, args.outlier_sensitive)
#     dataframe_patterns = [sample_patterns, sample_symbols, sample_cases, sample_token_len, sample_chars_len, sample_tokens]
#     dataframe_patterns_copy = copy.deepcopy(dataframe_patterns) 

#     # try:
#     sample_patterns, sample_symbols, sample_cases, sample_token_len, sample_chars_len, sample_tokens = dataframe_patterns
#     dataframe_labels = []
#     for column in csv_file:
#         dataframe_labels.append(column)

#     line_agreements= {} 
#     line_disagreements= {} 

#     row_counter = -1
    
#     # print(csv_file.index)
    
#     for row in csv_file.itertuples():
#         # print(row)
#         line_index = row.Index
#         if len(row)>1:
#             row = row[1:]
#         else:
#             row = []

#         row_values = [str(elem) if elem is not None else elem for elem in row]

#         line_agreements[line_index]= {}          
#         row_counter+=1

#         null_equivalent_fired, times = line_has_null_equivalent(row_values)
#         all_summaries_empty = True

#         column_index = -1
#         for column in csv_file.columns:
#             column_index+=1
#             line_agreements[line_index][column_index]={}
#             line_agreements[line_index][column_index]["agreements"]=[]

#             candidate_value = str(csv_file.loc[line_index,column])
#             value_lower = str(candidate_value).strip().lower()
#             value_tokens = value_lower.split()  

#             is_aggregate = (len(value_tokens)>0 and not set(value_tokens).isdisjoint(pat.aggregation_tokens))
#             is_null_equivalent=(candidate_value.strip().lower() in pat.null_equivalent_values)

#             line_agreements[line_index][column_index]['null_equivalent']= is_null_equivalent
#             line_agreements[line_index][column_index]['aggregate']= is_aggregate

#             column_patterns = copy.deepcopy(sample_patterns[column_index][line_index:])
#             column_symbols = copy.deepcopy(sample_symbols[column_index][line_index:])
#             column_cases = copy.deepcopy(sample_cases[column_index][line_index:])
#             column_lengths = copy.deepcopy(sample_chars_len[column_index][line_index:])
#             column_tokens = copy.deepcopy(sample_tokens[column_index][line_index:]) 

#             if args.max_summary_strength!=None:
#                 nonempty_patterns=0
#                 nonempty_patterns_idx = 0
#                 for nonempty_patterns_idx in range(0, min(len(column_patterns),args.max_line_depth)):
#                     if len(column_patterns[nonempty_patterns_idx])>0:
#                         nonempty_patterns+=1
#                         if nonempty_patterns==args.max_summary_strength:
#                             column_patterns = column_patterns[:nonempty_patterns_idx+1]                            
#                             column_symbols = column_symbols[:nonempty_patterns_idx+1]
#                             column_cases = column_cases[:nonempty_patterns_idx+1]
#                             column_lengths = column_lengths[:nonempty_patterns_idx+1]
#                             column_tokens = column_tokens[:nonempty_patterns_idx+1]
#                             break

#             if args.normalize_decimals==True:   
#                 column_patterns,column_symbols = pat.normalize_decimals_numbers(column_patterns, column_symbols)

#             value_pattern_summary, value_chain_consistent = pat.generate_pattern_summary(column_patterns)            
#             summary_strength = sum(1 for x in column_patterns if len(x)>0)
#             line_agreements[line_index][column_index]['summary_strength']= summary_strength
#             bw_patterns = [list(reversed(pattern)) for pattern in column_patterns]
#             value_pattern_BW_summary,_=pat.generate_pattern_summary(bw_patterns)
#             value_symbol_summary = pat.generate_symbol_summary(column_symbols)
#             case_summary = pat.generate_case_summary(column_cases)
#             length_summary = pat.generate_length_summary(column_lengths)

#             if null_equivalent_fired==True or len(value_pattern_summary)>0 or len(value_pattern_BW_summary)>0 or len(value_symbol_summary)>0 or len(case_summary)>0:         
#                 all_summaries_empty= False

#             pat_data_cell_rules_fired = []

#             columnvalues = csv_file.loc[line_index:,column].tolist()
#             for rule in pat.cell_rules["data"].keys():
#                 rule_fired = False
#                 # Don't bother looking for agreements if there are no patterns or if the value on this line gives an empty pattern
#                 non_empty_patterns=0
#                 if len(column_patterns)>0 and value_lower not in pat.null_equivalent_values:
#                     for pattern in column_patterns:
#                         if pattern!=[]:
#                             non_empty_patterns+=1

#                     #there is no point calculating agreement over one value, a single value always agrees with itself.
#                     #in addition, many tables have bilingual headers, so agreement between two header values is very common, require nij>=3
#                     if len(column_patterns)>=2 and non_empty_patterns>=2:                                
#                         rule_fired = eval_data_cell_rule(rule, columnvalues, column_tokens, value_pattern_summary, value_chain_consistent, value_pattern_BW_summary, value_symbol_summary, column_symbols, column_patterns, case_summary)

#                 if rule_fired==True:
#                     line_agreements[line_index][column_index]["agreements"].append(rule)

#         line_agreements[line_index]["all_summaries_empty"] = all_summaries_empty
    
#     ##########################################################################################
#     ##########################################################################################
#     #################             EVALUATE NOT_DATA CELL RULES             ###################
#     ##########################################################################################
#     ##########################################################################################
#     # input('\nEVALUATE NOT_DATA CELL RULES')

#     row_counter = -1
#     for row in csv_file.itertuples():
#         line_index = row.Index
#         line_disagreements[line_index]={}
#         if len(row)>1:
#             row = row[1:]
#         else:
#             row = []

#         row_values = [str(elem) if elem is not None else elem for elem in row]
        
#         row_counter+=1
        
#         for column in csv_file.columns:
#             columnindex = dataframe_labels.index(column)
#             line_disagreements[line_index][columnindex]={}
#             line_disagreements[line_index][columnindex]["disagreements"]=[]
#             candidate_value = str(csv_file.loc[line_index,column])                                

#             value_lower = str(candidate_value).strip().lower()
#             value_tokens = value_lower.split() 
#             cand_pattern, cand_symbols, cand_case, cand_num_tokens, cand_num_chars = pat.generate_pattern_symbols_and_case(candidate_value.strip(), args.outlier_sensitive) 

#             column_patterns = copy.deepcopy(sample_patterns[columnindex][line_index+1:])
#             column_symbols = copy.deepcopy(sample_symbols[columnindex][line_index+1:])
#             column_cases = copy.deepcopy(sample_cases[columnindex][line_index+1:])
#             column_lengths = copy.deepcopy(sample_chars_len[columnindex][line_index+1:])

#             if args.max_summary_strength!=None:
#                 nonempty_patterns=0
#                 nonempty_patterns_idx = 0
#                 for nonempty_patterns_idx in range(0, min(len(column_patterns), args.max_line_depth)):
#                     if len(column_patterns[nonempty_patterns_idx])>0:
#                         nonempty_patterns+=1
#                         if nonempty_patterns==args.max_summary_strength:
#                             # try:
#                             column_patterns = column_patterns[:nonempty_patterns_idx+1]
#                             column_symbols = column_symbols[:nonempty_patterns_idx+1]
#                             column_cases = column_cases[:nonempty_patterns_idx+1]
#                             column_lengths = column_lengths[:nonempty_patterns_idx+1]
#                             # except:
#                             #     print(f'nonempty_patterns_idx={nonempty_patterns_idx}')
#                             #     print(f'column_patterns={column_patterns}')
#                             #     print(f'column_symbols={column_symbols}')
#                             #     print(f'column_cases={column_cases}')
#                             #     print(f'column_lengths={column_lengths}')
#                             #     input('Press Enter..')
#                             break

#             if args.normalize_decimals==True:
#                 cand_pattern, cand_symbols, cand_case, cand_num_tokens, cand_num_chars, column_patterns,column_symbols = pat.normalize_decimals_numbers_predata(candidate_value.strip(),cand_pattern,cand_symbols,cand_case,cand_num_tokens,cand_num_chars,column_patterns,column_symbols, args.outlier_sensitive)

#             value_pattern_summary, value_chain_consistent = pat.generate_pattern_summary(column_patterns)
#             disagreement_summary_strength = sum(1 for x in column_patterns if len(x)>0)
#             line_disagreements[line_index][column]['disagreement_summary_strength']  = disagreement_summary_strength
#             bw_patterns =  [list(reversed(pattern)) for pattern in column_patterns]
#             value_pattern_BW_summary,_=pat.generate_pattern_summary(bw_patterns)
#             value_symbol_summary = pat.generate_symbol_summary(column_symbols)
#             case_summary = pat.generate_case_summary(column_cases)
#             length_summary = pat.generate_length_summary(column_lengths)

#             pat_not_data_cell_rules_fired= []

#             if len(cand_pattern)>0:                      
#                 columnvalues = csv_file.loc[line_index:,column].tolist()
#                 repetitions_of_candidate = columnvalues[1:].count(candidate_value)
#                 neighbor=''
#                 try:
#                     neighbor = columnvalues[1]
#                     repetitions_of_neighbor = columnvalues[2:].count(neighbor)                        
#                 except:
#                     repetitions_of_neighbor = 0 

            
#             for rule in pat.cell_rules["not_data"].keys():
#                 rule_fired = False
#                 if len(cand_pattern)>0:                     
#                     if disagreement_summary_strength>0 and (all_numbers(column_symbols)==False or is_number(cand_symbols)==False):              
#                         rule_fired = eval_not_data_cell_rule(rule, repetitions_of_candidate, repetitions_of_neighbor, neighbor, value_pattern_summary, value_pattern_BW_summary, value_chain_consistent, value_symbol_summary, case_summary, length_summary, cand_pattern, cand_symbols, cand_case, cand_num_chars, disagreement_summary_strength, line_agreements, columnindex, line_index)
#                         if rule_fired == True:
#                             line_disagreements[line_index][columnindex]["disagreements"].append(rule)

#         #end processing column              
#         ########################################################################                


#         #Collect data line rules fired
#         line_is_data_events = assess_data_line(row_values)
#         pat_data_line_rules_fired= []   
#         line_agreements[line_index]['line'] = [] 
#         line_disagreements[line_index]['line'] = []   

#         for data_rule_category in pat.line_rules["data"].keys():
#             for rule in pat.line_rules["data"][data_rule_category].keys():
#                 rule_fired = False
#                 if rule in line_is_data_events:
#                     rule_fired = True
#                     line_agreements[line_index]['line'].append(rule)
            

#         # Collect not_data line rules fired

#         non_nulls,non_null_percentage = non_nulls_in_line(row_values)
#         all_summaries_empty = line_agreements[line_index]["all_summaries_empty"]        
#         header_events_fired = collect_events_on_row(row_values)

#         arithmetic_events_fired = collect_arithmetic_events_on_row(row_values)
#         arithmetic_sequence_fired=False
#         if len(arithmetic_events_fired)>0:
#             arithmetic_sequence_fired=True
#         header_row_with_aggregation_tokens_fired = header_row_with_aggregation_tokens(row_values, arithmetic_sequence_fired)
        
#         before_data = True
#         # if line_index in data_indexes :
#         #     before_data = False
#         not_data_line_rules_fired = []
#         if csv_file.shape[1]>1:
#             not_data_line_rules_fired = assess_non_data_line(row_values, before_data, all_summaries_empty, line_index, csv_file)

#         pat_not_data_line_rules_fired = []
#         for not_data_rule_category in pat.line_rules["not_data"].keys():
#             for rule in pat.line_rules["not_data"][not_data_rule_category].keys():
#                 rule_fired = False
#                 if rule in (not_data_line_rules_fired + header_events_fired + arithmetic_events_fired + header_row_with_aggregation_tokens_fired):
#                     rule_fired = True
#                     line_disagreements[line_index]['line'].append(rule)

#     # except Exception as e:
#     #     pass
 

#     return line_agreements, line_disagreements

def predict_fdl(dataframe, line_predictions, markov_approximation_probabilities, markov_model, data_window=2, combined_label_weight='confidence'):
    # print('\n\n---> [METHOD] table_classifier_utilities.predict_fdl>>\n')
    first_data_line_combined_data_predictions= {}
    first_data_line_combined_data_predictions["TotalFiles"] = 0
    first_data_line_combined_data_predictions["PredictedPositive"]=0 
    first_data_line_combined_data_predictions["RealPositive"]=0
    first_data_line_combined_data_predictions["RealNegative"]=0
    first_data_line_combined_data_predictions["TruePositive"]=0 
    first_data_line_combined_data_predictions["Success"]=0
    first_data_line_combined_data_predictions["Error"]=0 
    first_data_line= -1
    softmax= 0
    prod_softmax_prior= 0

    if len(line_predictions)>1:
        legal_sequences = {}
        legal_sequence_priors = {}
        k = len(line_predictions)
        offset = dataframe.index[0]

        # input(f'offset={offset}')
        ordered_prediction_labels = [line_predictions[line_index]['label'] for line_index in sorted(line_predictions.keys()) if line_index>=offset]

        # print('ordered_prediction_labels='+str(ordered_prediction_labels))
        # pp.pprint(line_predictions)
        # input()

        b = (label for label in ordered_prediction_labels)

        where = 0    # need this to keep track of original indices
        first_data_window_index= -1# this was 0 until Nov 13th
        for key, group in itertools.groupby(b):
            length = sum(1 for item in group)
            #length = len([*group])
            if length >= data_window:
                items = [where + i for i in range(length)]
                if key=='DATA':
                    first_data_window_index= items[0]
                    # print(f'{key}:{items}')
                    # print('first_data_window_index:'+str(first_data_window_index))
                    break
            where += length
        if  first_data_window_index==-1:
            try:
                first_data_window_index = ordered_prediction_labels.index('DATA')
                data_window=1
            except:
                first_data_window_index = -1

        k = first_data_window_index+data_window

        # print(f'k={k}')

        if first_data_window_index>=0:
            for sequence_id in range(0,k+1): 
                sequence_prior = 1 
                legal_sequences[sequence_id] = []
                # input(f'\nGenerate sequence {sequence_id}')
                while len(legal_sequences[sequence_id]) <  k-sequence_id:
                    legal_sequences[sequence_id].append('NOT_DATA')
                    if markov_model!=None:
                        if len(legal_sequences[sequence_id])==1:
                            if  markov_model=='first_order':
                                sequence_prior=sequence_prior*markov_approximation_probabilities['p_first_tables_start_not_data']

                            elif markov_model=='second_order':
                                sequence_prior=sequence_prior*markov_approximation_probabilities['prob_nd_given_start']
                                # print(f"markov_approximation_probabilities['prob_nd_given_start']={markov_approximation_probabilities['prob_nd_given_start']}")
                        else:                            
                            if  markov_model=='first_order':
                                sequence_prior=sequence_prior*markov_approximation_probabilities['prob_ndI_given_ndIminus1']

                            elif markov_model=='second_order' and len(legal_sequences[sequence_id])==2:
                                sequence_prior=sequence_prior*markov_approximation_probabilities['prob_nd_given_start_nd']
                                # print(f"markov_approximation_probabilities['prob_nd_given_start_nd']={markov_approximation_probabilities['prob_nd_given_start_nd']}")
                            elif markov_model=='second_order' and len(legal_sequences[sequence_id])>2:  
                                sequence_prior=sequence_prior*markov_approximation_probabilities['prob_nd_given_nd_nd']
                                # print(f"markov_approximation_probabilities['prob_nd_given_nd_nd']={markov_approximation_probabilities['prob_nd_given_nd_nd']}")

                while len(legal_sequences[sequence_id]) >= k-sequence_id and len(legal_sequences[sequence_id]) <  k:
                    legal_sequences[sequence_id].append('DATA')
                    if markov_model!=None:
                        if len(legal_sequences[sequence_id])==1:
                            if  markov_model=='first_order':
                                sequence_prior=sequence_prior*markov_approximation_probabilities['p_first_tables_start_data']

                            elif markov_model=='second_order':
                                sequence_prior=sequence_prior*markov_approximation_probabilities['prob_d_given_start']    
                                # print(f"markov_approximation_probabilities['prob_d_given_start']={markov_approximation_probabilities['prob_d_given_start']}")
                        else:
                            if  markov_model=='first_order' and legal_sequences[sequence_id].count('DATA')==1:
                                sequence_prior=sequence_prior*markov_approximation_probabilities['prob_dI_given_ndIminus1']

                            elif markov_model=='second_order' and len(legal_sequences[sequence_id])==2 and legal_sequences[sequence_id].count('DATA')==1:
                                sequence_prior=sequence_prior*markov_approximation_probabilities['prob_d_given_start_nd']
                                # print(f"markov_approximation_probabilities['prob_d_given_start_nd']={markov_approximation_probabilities['prob_d_given_start_nd']}")
                            elif markov_model=='second_order' and len(legal_sequences[sequence_id])==2 and legal_sequences[sequence_id].count('DATA')==2:
                                sequence_prior=sequence_prior*markov_approximation_probabilities['prob_d_given_start_d']
                                # print(f"markov_approximation_probabilities['prob_d_given_start_d']={markov_approximation_probabilities['prob_d_given_start_d']}")
                            elif markov_model=='second_order' and len(legal_sequences[sequence_id])>2 and legal_sequences[sequence_id].count('DATA')==1:  
                                sequence_prior=sequence_prior*markov_approximation_probabilities['prob_d_given_nd_nd']  
                                # print(f"markov_approximation_probabilities['prob_d_given_nd_nd']={markov_approximation_probabilities['prob_d_given_nd_nd'] }") 
                            elif markov_model=='second_order' and len(legal_sequences[sequence_id])>2 and legal_sequences[sequence_id].count('DATA')==2:  
                                sequence_prior=sequence_prior*markov_approximation_probabilities['prob_d_given_nd_d']
                                # print(f"markov_approximation_probabilities['prob_d_given_nd_d']={markov_approximation_probabilities['prob_d_given_nd_d']}")
                            elif markov_model=='second_order' and len(legal_sequences[sequence_id])>2 and legal_sequences[sequence_id].count('DATA')>=3:  
                                sequence_prior=sequence_prior*markov_approximation_probabilities['prob_d_given_d_d']
                                # print(f"markov_approximation_probabilities['prob_d_given_d_d']={markov_approximation_probabilities['prob_d_given_d_d']}")

                # print(f'sequence_prior={sequence_prior}')
                legal_sequence_priors[sequence_id]=sequence_prior
            
            # print('\nvalid sequences:')
            # pp.pprint(legal_sequences)
            # input()

            match_weight = {}
            for sequence_id, legal_label_sequence in legal_sequences.items():
                match_weight[sequence_id] = match_sequence(legal_label_sequence, 
                                                            line_predictions, 
                                                            offset, 
                                                            first_data_window_index, 
                                                            data_window,
                                                            combined_label_weight)
            
            # print('match_weight:')
            # pp.pprint(match_weight)
            # input()

            match_softmax = {}
            for sequence_id, legal_label_sequence in legal_sequences.items():    
                match_softmax[sequence_id] = math.exp(match_weight[sequence_id])/ sum([math.exp(v) for v in match_weight.values()])
            
            # print('match_softmax:')
            # pp.pprint(match_softmax)
            # input()
            
            sequence_final_probability = {}
            for sequence_id,sequence in legal_sequences.items():
                sequence_final_probability[sequence_id]=legal_sequence_priors[sequence_id]*match_softmax[sequence_id]
            
            # print('sequence_final_probability:')
            # pp.pprint(sequence_final_probability)
            # input()
            
            sorted_id_weight = sorted(sequence_final_probability.items(), key=lambda kv: (-kv[1],-kv[0]))
            winning_sequence_id = sorted_id_weight[0][0]
            winning_sequence = legal_sequences[sorted_id_weight[0][0]]
            
            # print('winning_sequence_id: '+str(winning_sequence_id))
            # input('winning_sequence:'+str(winning_sequence))

            softmax = match_softmax[winning_sequence_id]
            prod_softmax_prior= sequence_final_probability[winning_sequence_id]
            try:
                first_data_line = winning_sequence.index('DATA') + offset
            except:
                first_data_line= -1  

    first_data_line_combined_data_predictions['softmax'] = softmax
    first_data_line_combined_data_predictions['prod_softmax_prior'] = prod_softmax_prior

    # Calculate CONFIDENCE of First Data Line with old method
    avg_confidence, min_confidence = first_data_line_confidence(line_predictions, first_data_line)

    first_data_line_combined_data_predictions['avg_confidence'] = avg_confidence
    first_data_line_combined_data_predictions['min_confidence'] = min_confidence

    # print('first_data_line='+str(first_data_line))
    # input(f'\nfirst_data_line_combined_data_predictions={first_data_line_combined_data_predictions}')
    
    return first_data_line, first_data_line_combined_data_predictions   


def predict_fdl_old(dataframe, data_confidence,not_data_confidence, combined_data_line_confidences, line_predictions, x=0.6):
    # Discover First Data Line
    first_data_line_combined_data_predictions = {}
    first_data_line_combined_data_predictions["TotalFiles"] = 0
    first_data_line_combined_data_predictions["PredictedPositive"]=0 
    first_data_line_combined_data_predictions["RealPositive"]=0
    first_data_line_combined_data_predictions["RealNegative"]=0
    first_data_line_combined_data_predictions["TruePositive"]=0 
    first_data_line_combined_data_predictions["Success"]=0
    first_data_line_combined_data_predictions["Error"]=0 

    first_data_line = -1
    row_counter = -1

    # Discover First Data Line NEW
    for row_index, candidate_data_row in dataframe.iterrows():
        row_counter +=1
        if first_data_line == -1 and row_index in not_data_confidence.keys() and  not_data_confidence[row_index] == 0 :
            first_data_line = row_index    
        # this line is data
        if row_index in combined_data_line_confidences.keys() and combined_data_line_confidences[row_index]>0:
            if len(candidate_data_row)==sum([1 for i in candidate_data_row if str(i).strip().lower()  in ['','nan']])==0:
                continue

            first_data_line = row_index
            #if next line is not data
            if row_index+1 in combined_data_line_confidences.keys() and combined_data_line_confidences[row_index+1]==0:
                Data_diff = data_confidence[row_index]-not_data_confidence[row_index]
                NotData_diff = not_data_confidence[row_index+1]-data_confidence[row_index+1]
                if Data_diff>NotData_diff+x:
                    break 
                continue
            else: #next line is data too
                break
            

    # Calculate CONFIDENCE of First Data Line
    avg_confidence, min_confidence = first_data_line_confidence(line_predictions, first_data_line)

    first_data_line_combined_data_predictions['avg_confidence'] = avg_confidence
    first_data_line_combined_data_predictions['min_confidence'] = min_confidence

    # print('old_first_data_line='+str(first_data_line))
    # input('old_confidence='+str(first_data_line_combined_data_predictions['avg_confidence']))

    return first_data_line, first_data_line_combined_data_predictions
    
def confusion_index(a,b):
    # if a+b==0:
    if a==0:
        return 0
    else:
        # return (a-b)/(a+b)
        return (a-b)/a

def last_data_line_confidence(line_predictions, predicted_boundary, max_window=4):
    avg_confidence = {}
    avg_predicted_data=0 
    avg_predicted_not_data=0

    sorted_indexes = sorted(list(line_predictions.keys()), reverse=True)
    for method in ['confidence']:
        predicted_not_data = []
        predicted_data = []   

        for index in sorted_indexes:
            # lines predicted not data
            if index>predicted_boundary:
                # only look if you are within a window of the boundary
                if index<=predicted_boundary-max_window:
                    # correctly predicted not data
                    if line_predictions[index]['label'] == 'NOT_DATA':
                        predicted_not_data.append(line_predictions[index]['value'][method])
                    # incorrectly predicted not data
                    else:
                        predicted_not_data.append(-line_predictions[index]['value'][method])
            else:
                if len(predicted_data)==max_window:
                    break
                if line_predictions[index]['label'] == 'NOT_DATA':
                    predicted_data.append(- line_predictions[index]['value'][method])
                else:
                    predicted_data.append(line_predictions[index]['value'][method])

        data_window_weight= len(predicted_data)
        not_data_window_weight = len(predicted_not_data)

        if len(predicted_data)>0:
            avg_predicted_data = sum(predicted_data)/len(predicted_data) 

        if len(predicted_not_data)>0:
            avg_predicted_not_data = sum(predicted_not_data)/len(predicted_not_data)

        if (data_window_weight+not_data_window_weight)>0:
            avg_confidence[method] = max(0,(data_window_weight*(avg_predicted_data)+not_data_window_weight*(avg_predicted_not_data))/(data_window_weight+not_data_window_weight))
        else:
            avg_confidence[method]=0          

    return avg_confidence 
    
def first_data_line_confidence(line_predictions, first_data_line, max_window=4):

    # pp.pprint(line_predictions)
    # input(f'predicted first_data_line={first_data_line}')

    avg_confidence = {}
    min_confidence = {}
    avg_predicted_data=0 
    avg_predicted_not_data=0
    min_predicted_data=0
    min_predicted_not_data=0

    sorted_indexes = sorted(list(line_predictions.keys()))

    for method in ['confusion_index', 'confidence', 'difference']:
        # input(f'method={method}')
        predicted_not_data = []
        predicted_data = []   

        if first_data_line != -1: 
            for index in sorted_indexes:
                if index<first_data_line:
                    if index>=first_data_line-max_window:
                        if line_predictions[index]['label'] == 'NOT_DATA':
                            predicted_not_data.append(line_predictions[index]['value'][method])
                        else:
                            predicted_not_data.append(-line_predictions[index]['value'][method])
                else:
                    if len(predicted_data)==max_window:
                        break
                    if line_predictions[index]['label'] == 'NOT_DATA':
                        data_confidence = - line_predictions[index]['value'][method]
                    else:
                        data_confidence = line_predictions[index]['value'][method]
                    predicted_data.append(data_confidence) 
            
            data_window_weight= len(predicted_data)
            not_data_window_weight = len(predicted_not_data)

            if len(predicted_data)>0:
                avg_predicted_data = sum(predicted_data)/len(predicted_data) 
                min_predicted_data = min(predicted_data)

            if len(predicted_not_data)>0:
                avg_predicted_not_data = sum(predicted_not_data)/len(predicted_not_data)
                min_predicted_not_data = min(predicted_not_data)

            if (data_window_weight+not_data_window_weight)>0:
                avg_confidence[method] = max(0,(data_window_weight*(avg_predicted_data)+not_data_window_weight*(avg_predicted_not_data))/(data_window_weight+not_data_window_weight))
                min_confidence[method] = max(0,(data_window_weight*(min_predicted_data)+not_data_window_weight*(min_predicted_not_data))/(data_window_weight+not_data_window_weight))
            else:
                avg_confidence[method]=0
                min_confidence[method]=0

        else:
            avg_windows = []
            # for index in sorted_indexes:
            #     if line_predictions[index]['label'] == 'NOT_DATA':
            #         predicted_not_data.append(line_predictions[index]['value'][method])
            #     else:
            #         predicted_not_data.append(-line_predictions[index]['value'][method])
            if len(sorted_indexes)>1:
                # print(f'sorted_indexes[:-1]={sorted_indexes[:-1]}')
                for index in sorted_indexes[:-1]:
                    predicted_not_data= []
                    # print(f"\tline_predictions[{index}]={line_predictions[index]}")
                    if line_predictions[index]['label'] == 'NOT_DATA':
                        predicted_not_data.append(line_predictions[index]['value'][method])
                    else:
                        predicted_not_data.append(-line_predictions[index]['value'][method])

                    # print(f"\tline_predictions[{index+1}]={line_predictions[index]}")
                    if line_predictions[index+1]['label'] == 'NOT_DATA':
                        predicted_not_data.append(line_predictions[index+1]['value'][method])
                    else:
                        predicted_not_data.append(-line_predictions[index+1]['value'][method])
                    # print(f'predicted_not_data={predicted_not_data}')
                    avg_predicted_not_data=0    
                    if len(predicted_not_data)>0:
                        avg_predicted_not_data = sum(predicted_not_data)/len(predicted_not_data)
                        min_predicted_not_data = min(predicted_not_data)

                    avg_windows.append(min(0,avg_predicted_not_data))
            else:
                # print(f"avg_windows.append({line_predictions[sorted_indexes[0]]['value'][method]})")
                # print(f'[METHOD]first_data_line_confidence:\nline_predictions={line_predictions}\nsorted_indexes={sorted_indexes}')
                avg_windows.append(line_predictions[sorted_indexes[0]]['value'][method])

            avg_confidence[method] = max(0,min(avg_windows))
            min_confidence[method] = max(0,min(avg_windows))                                                 

    return avg_confidence, min_confidence


def assess_combo_header(candidate_header_dataframe):

    
    candidate_header = combo_row(candidate_header_dataframe)

    # print('Assess:')
    # print(candidate_header_dataframe)
    # print('Combination assessed:')
    # input(candidate_header)

    #if no nulls in candidate this is a good candidate

    if len([i for i in candidate_header if str(i).strip().lower() not in ['','nan', 'none', 'null']])== len(candidate_header):
        # input('>1')
        return True
    else:
        #if only one or two attributes, second must be complete
        if len(candidate_header)==1:
            return False
        if len(candidate_header)==2:
            if str(candidate_header[1]).strip().lower() in ['','nan', 'none', 'null']:
                return False
            else:
                # input('>2')
                return True

        #if three attributes, first may be incomplete
        if len(candidate_header)==3:
            if len([i for i in candidate_header[1:] if str(i).strip().lower() not in ['','nan', 'none', 'null']])== len(candidate_header[1:]):
                # input('>3')
                return True
            else:
                return False
        #if four attributes, first two or last two may be incomplete
        if len(candidate_header)==4:
            if  len([i for i in candidate_header[2:] if str(i).strip().lower() not in ['','nan', 'none', 'null']])== len(candidate_header[2:]):
                # input('>4a')
                return True
            elif len([i for i in candidate_header[:-2] if str(i).strip().lower() not in ['','nan', 'none', 'null']])== len(candidate_header[:-2]):
                # input('>4b')
                return True
            elif len([i for i in candidate_header[1:-1] if str(i).strip().lower() not in ['','nan', 'none', 'null']])== len(candidate_header[1:-1])  or len([i for i in candidate_header[1:] if str(i).strip().lower() not in ['','nan', 'none', 'null']])== len(candidate_header[1:]) or len([i for i in candidate_header[:-1] if str(i).strip().lower() not in ['','nan', 'none', 'null']])== len(candidate_header[:-1]):
                # input('>4b')
                return True                
            else:
                return False

        if len(candidate_header)>4 and len([i for i in candidate_header[2:-2] if str(i).strip().lower() not in ['','nan', 'none', 'null']]) == len([i for i in candidate_header[2:-2]]):
            # input('5')
            return True
        else: 
            return False

        return False  

def combo_row(rows):

    column_names = name_table_columns(rows)
    combo_row = []
    for csv_column, column in column_names.items():
        combo_row.append((" ".join([cell["value"].lower() for cell in column["column_header"]])).strip())
    return combo_row

def name_table_columns(rows):
    rows = rows.applymap(lambda v: pat.dequote(str(v).strip()) if str(v).strip().lower() not in ['','nan', 'none', 'null'] else '')
    column_names = {}
    for rowidx, row in rows.iterrows():
        #initialize with first row
        if rowidx==rows.index[0]:
            for csv_index, value in row.items():
                column_names[csv_index] = {'column_header':[],
                                            'table_column': row.index.get_loc(csv_index)
                                            }
                if value!='':
                    column_names[csv_index]['column_header'].append({'row':rowidx,
                                                                    'column':csv_index,
                                                                    'value':value,
                                                                    'index':len(column_names[csv_index]['column_header'])})                                          
        else:            

            buffer_row = {}
            for csv_index, value in rows.loc[previous_row].items():
               
                if csv_index !=rows.index[-1] and (rows.columns.get_indexer([csv_index])+1)[0] < len(rows.columns):
                    next_csv_column = rows.columns[(rows.columns.get_indexer([csv_index])+1)[0]]
                    while rows.loc[previous_row, next_csv_column] == '':
                        if rows.loc[previous_row, csv_index]!='':
                            buffer_row[next_csv_column] = {
                                'row':previous_row,
                                'column':csv_index,
                                'value': rows.loc[previous_row, csv_index],
                                'index':len(buffer_row)
                            }
                        if (rows.columns.get_indexer([next_csv_column])+1)[0] < len(rows.columns):
                            next_csv_column = rows.columns[(rows.columns.get_indexer([next_csv_column])+1)[0]]
                        else:
                            break

            for csv_index, value in row.items(): 
                if sum(cell['value'] !='' for cell in column_names[csv_index]['column_header'])==0:  
                    if csv_index in buffer_row.keys():  
                        column_names[csv_index]['column_header'].append(buffer_row[csv_index]) 
                if value!='':
                    column_names[csv_index]['column_header'].append({
                        'row':rowidx,
                        'column':csv_index,
                        'value':value,
                        'index':len(column_names[csv_index]['column_header'])
                        })

        previous_row = rowidx
    return column_names 

def pre_header_line(row_values,before_header):

    pre_header_line_events = []
    if len(row_values)==1 and row_values[0] not in ['nan', 'none', '', ' ']: 
        before_header = False

    if before_header == True and len(row_values)==2:
        nulls_seen = False
        row_values = [str(elem).strip().lower()for elem in row_values[1:]] 
                      
        for value in row_values:
            if value not in ['nan', 'none', '', ' ']:
                before_header = False
                break

    if before_header == True and len(row_values)>2:
        nulls_seen = False
        row_values = [str(elem).strip().lower()for elem in row_values[2:]] 
                      
        for value in row_values:
            if value not in ['nan', 'none', '', ' ']:
                before_header = False
                break

    if before_header==True:
        pre_header_line_events.append('UP_TO_SECOND_COLUMN_COMPLETE_CONSISTENTLY') 

    return pre_header_line_events

def predict_header_indexes(file_dataframe, first_data_line_annotated, table_counter):
    # print('\n\n\npredict_header_indexes:')
    # input(file_dataframe)
    #initialize
    predicted_pat_subheaders=[]

    # print('\nDrop empty columns')
    null_columns = file_dataframe.columns[file_dataframe.isna().all()].tolist()  
    file_dataframe = file_dataframe.drop(null_columns, axis=1)
    # input(file_dataframe)

    # print('Candidate headers:')
    dataframe = file_dataframe.loc[:first_data_line_annotated-1] 
    # print(f'Candidate headers={dataframe}')

    if len(file_dataframe.index)>0:
        last_row_label = file_dataframe.index[-1]
    else:
        last_row_label = first_data_line_annotated
    
    # print('last_row_label='+str(last_row_label))
    # print('Head of data (sample):')
    data = file_dataframe.loc[first_data_line_annotated:min(last_row_label,first_data_line_annotated+10)] 
    # print(data)

    # print('\n\nLines before data (cleaned):')
    null_columns = data.columns[data.isna().all()].tolist() 
    data = data.drop(null_columns, axis=1)
    dataframe = dataframe.drop(null_columns, axis=1)
    # input(dataframe)

    before_header = True
    top_header_candidate_index = 0
    for row_index, row in dataframe.iterrows():
        row_values = [str(elem) if elem is not None else elem for elem in row.tolist()]

        pre_header_events = pre_header_line(row_values,before_header)
        if "UP_TO_SECOND_COLUMN_COMPLETE_CONSISTENTLY" not in pre_header_events:
            before_header= False
            top_header_candidate_index = row_index
            break
    
    candidate_headers = dataframe.loc[top_header_candidate_index:]

    # print('\n\nLines searched for header:')
    # print(candidate_headers)

    predicted_header_indexes=[]
    empty_lines = []
    NON_DUPLICATE_HEADER_ACHIEVED = False
    if candidate_headers.shape[0]>0:
        non_empty_lines_assessed = 0
        NON_EMPTY_LINE_SEEN = False
        for reverse_index in range(1,candidate_headers.shape[0]+1):
            # print(candidate_headers.iloc[-reverse_index:].dropna(how='all', axis=0))
            if len(candidate_headers.iloc[-reverse_index:].dropna(how='all', axis=0))>6:
                break


            # print(f'reverse_index:{reverse_index}')
            #ignore first line above data if it was completely empty
            row_values = candidate_headers.iloc[-reverse_index].tolist()
            if len([i for i in row_values if str(i).strip().lower() not in ['','nan', 'none', 'null']])==0:
                empty_lines.append(first_data_line_annotated-reverse_index)
                if NON_DUPLICATE_HEADER_ACHIEVED == True:
                    break
                else:
                    continue
            if reverse_index>1 and len(row_values)>1 and str(row_values[0]).strip().lower() not in ['','nan', 'none', 'null'] and len([i for i in row_values[1:] if str(i).strip().lower() not in ['','nan', 'none', 'null']])==0 :
                if NON_EMPTY_LINE_SEEN == False:
                    empty_lines.append(first_data_line_annotated-reverse_index)
                if NON_DUPLICATE_HEADER_ACHIEVED == True:
                    break
                else:
                    continue

            if len(row_values)>2 and str(row_values[1]).strip().lower() not in ['','nan', 'none', 'null'] and len([i for i in row_values[2:] if str(i).strip().lower() not in ['','nan', 'none', 'null']])==0 :
                if NON_DUPLICATE_HEADER_ACHIEVED == True:
                    break
                else:
                    continue           
            NON_EMPTY_LINE_SEEN = True
            non_empty_lines_assessed+=1
            consider_header_dataframe = candidate_headers.iloc[-reverse_index:].drop(empty_lines,
                                                                                    axis=0)



            candidate_header = combo_row(consider_header_dataframe)

            # print('Assess:')
            # print(candidate_headers.iloc[-reverse_index:].drop(empty_lines))
            # print('Combination assessed:')
            # input(candidate_header)
            if NON_DUPLICATE_HEADER_ACHIEVED==True and reverse_index>1 and len(row_values)>2:
                extension = True
                for value_index,value in enumerate(row_values[2:]):                    
                    if str(value).strip().lower() not in ['','nan', 'none', 'null'] and str(candidate_headers.iloc[-reverse_index+1].tolist()[value_index+2]).strip().lower() in ['','nan', 'none', 'null']:
                        if len(row_values)>4 and (value_index == len(row_values[2:])-1 or value_index == len(row_values[2:])-2):
                            continue
                        extension = False
                        break
                if extension== True:
                    header = candidate_headers.iloc[-reverse_index:]
                    predicted_header_indexes = [x for x in list(header.index) if x not in empty_lines]

            elif assess_combo_header(consider_header_dataframe) == True:
                header = candidate_headers.iloc[-reverse_index:]
                predicted_header_indexes = [x for x in list(header.index) if x not in empty_lines]
            
            if (len(predicted_header_indexes)>0 and (has_duplicates(consider_header_dataframe)==False)):
               NON_DUPLICATE_HEADER_ACHIEVED = True 

            if non_empty_lines_assessed>4:
                break


        if len(predicted_header_indexes)>0:
            if predicted_header_indexes[0]-1 in dataframe.index:
                row_before = dataframe.loc[predicted_header_indexes[0]-1].tolist()
                if len(row_before)>1 and len([i for i in row_before[1:] if str(i).strip().lower() not in ['', 'nan','none','null']])>0 :
                    predicted_header_indexes.insert(0,predicted_header_indexes[0]-1)
                    # if len(dataframe.loc[predicted_header_indexes].dropna(thresh=1))>5:

            if len(predicted_header_indexes)>0:
                last_header_index = predicted_header_indexes[-1]      
                while len(predicted_header_indexes)>0:
                    first_value = str(file_dataframe.loc[last_header_index,file_dataframe.columns[0]]).strip()
                    if  len(dataframe.columns)>1 and file_dataframe.loc[last_header_index, 1:].isnull().values.all()==True and (first_value.startswith('(') and first_value.endswith(')'))==False:
                        predicted_pat_subheaders.append(last_header_index)
                        predicted_header_indexes = predicted_header_indexes[:-1]
                        last_header_index = predicted_header_indexes[-1]
                    else:
                        break
        else:
            if len(predicted_header_indexes)==0 and table_counter == 1 and first_data_line_annotated>0 and len(candidate_headers)>0:
                count_offset = 0
                for reverse_index in range(1,candidate_headers.shape[0]+1):
                    count_offset+=1
                    row_values = candidate_headers.iloc[-reverse_index].tolist()
                    if len([i for i in row_values if str(i).strip().lower() not in ['','nan', 'none', 'null']])>0:
                        predicted_header_indexes.append(first_data_line_annotated-count_offset)  
                        break  
    return predicted_header_indexes,predicted_pat_subheaders  

def has_duplicates(candidate_header_dataframe):
    candidate_header = combo_row(candidate_header_dataframe)

    non_empty_values = [i for i in candidate_header if str(i).strip().lower() not in ['', 'nan','none','null']]
    if len(non_empty_values) == len(set(non_empty_values)):
        return False
    else:
        return True 

def discover_next_table_temp(csv_file, file_offset, table_counter, line_agreements, line_disagreements, blank_lines, args):
    print(f'\nDiscovering table {table_counter}:\n')
    rule_weights = args.rule_weights
    discovered_table = None
    csv_file = csv_file.loc[file_offset:]
    if csv_file.empty:
        return discovered_table

    input(f'{csv_file}\nPress enter...')

    not_data_line_confidences ={}
    data_line_confidences = {}
    label_confidences = {}
    before_data= True
    row_counter = 0

    for row_index in csv_file.index:
        row_counter+=1
        label_confidences[row_index]={}
        candidate_row_agreements=[]
        candidate_row_disagreements=[]

        for column_index in csv_file.columns:
            #################################################v######################v######
            #  DATA value classification
            value_agreements = line_agreements[row_index][column_index]['agreements']
            summary_strength = line_agreements[row_index][column_index]['summary_strength']  

            # if there are no lines below me to check agreement, 
            # and line before me exists and was data
            # see impute agreements
            if row_index in line_agreements.keys() and (line_agreements[row_index][column_index]['null_equivalent']==True or line_agreements[row_index][column_index]['summary_strength']==1) and args.impute_nulls==True and row_index-1 in line_agreements.keys() and column_index in line_agreements[row_index-1].keys() and row_index-1 in data_line_confidences.keys() and data_line_confidences[row_index-1]>not_data_line_confidences[row_index-1]:
                value_agreements = line_agreements[row_index-1][column_index]['agreements']
                summary_strength = line_agreements[row_index-1][column_index]['summary_strength']
            if row_index in line_agreements.keys() and line_agreements[row_index][column_index]['summary_strength']==0 and line_agreements[row_index][column_index]['aggregate'] and row_index-2 in line_agreements.keys() and column_index in line_agreements[row_index-2].keys() and row_index-2 in data_line_confidences.keys() and data_line_confidences[row_index-2]>not_data_line_confidences[row_index-2]:
                value_agreements = line_agreements[row_index-2][column_index]['agreements']
                summary_strength = line_agreements[row_index-2][column_index]['summary_strength']

            # otherwise, nothing was wrong, i can use my own damn agreements as initialized

            data_score = max_score(value_agreements, rule_weights['data_cell'], args.weight_lower_bound)
            POPULATION_WEIGHT = 1-(1-args.p)**(2*summary_strength)
            if data_score!=None:
                if args.summary_population_factor:
                    candidate_row_agreements.append(data_score*POPULATION_WEIGHT)
                else:
                    candidate_row_agreements.append(data_score) 

            #######################################################################v######
            #  NOT DATA value classification        
            value_disagreements = line_disagreements[row_index][column_index]['disagreements']
            disagreement_summary_strength = line_disagreements[row_index][column_index]['disagreement_summary_strength'] 
            not_data_score = max_score(value_disagreements, rule_weights['not_data_cell'], args.not_data_weight_lower_bound)
            POPULATION_WEIGHT = 1-(1-args.p)**(2*disagreement_summary_strength)

            if not_data_score!=None:
                if args.summary_population_factor:
                    candidate_row_disagreements.append(not_data_score*POPULATION_WEIGHT)
                else:
                    candidate_row_disagreements.append(not_data_score)

            ########################################################################                

        #################################################################################
        # NOT DATA line weights
        line_not_data_evidence =  [score for score in candidate_row_disagreements] 
        if args.weight_input=='values_and_lines':
            # if table_counter>1:
            #     input(f'row_index={row_index}\ndata_line_confidences={data_line_confidences}\n')
            #     input(f'row_index={row_index}\nnot_data_line_confidences={not_data_line_confidences}\n')

            if row_index>0 and row_index-1 in data_line_confidences.keys() and data_line_confidences[row_index-1]>not_data_line_confidences[row_index-1]:
                before_data=False

            if csv_file.shape[1]>1:
                not_data_line_rules_fired = line_disagreements[row_index]['line']

                for event in not_data_line_rules_fired:
                    if event =="UP_TO_FIRST_COLUMN_COMPLETE_CONSISTENTLY" and before_data == False:
                        continue
                    if event.startswith("ADJACENT_ARITHMETIC_SEQUENCE") and rule_weights['not_data_line'][event]==None:
                        steps = event[-1]
                        if int(steps) in range(2,6):
                            event = event[:-1]+ str(int(steps)+1)
                    if rule_weights['not_data_line'][event]!=None and  rule_weights['not_data_line'][event]>args.not_data_weight_lower_bound:
                        line_not_data_evidence.append(rule_weights['not_data_line'][event]) 

        not_data_conf = probabilistic_sum(line_not_data_evidence)
        not_data_line_confidences[row_index] = not_data_conf    

        # DATA line weights          
        line_is_data_evidence =  [score for score in candidate_row_agreements] 
        if args.weight_input=='values_and_lines':
            line_is_data_events = line_agreements[row_index]['line']
            for rule in line_is_data_events:
                if rule_weights['data_line'][rule]!=None and rule_weights['data_line'][rule]>args.weight_lower_bound:
                    line_is_data_evidence.append(rule_weights['data_line'][rule])

        # calculate confidence that this row is data
        data_conf = probabilistic_sum(line_is_data_evidence)
        data_line_confidences[row_index] = data_conf 

        label_confidences[row_index]['DATA'] = data_conf
        label_confidences[row_index]['NOT-DATA'] = not_data_conf  

    combined_data_line_confidences, line_predictions = predict_combined_data_confidences(csv_file, data_line_confidences, not_data_line_confidences, args.max_candidates)
    # if table_counter>1:
    #     print(f'line_predictions={line_predictions}')
    pat_first_data_line, first_data_line_combined_data_predictions = predict_fdl(csv_file, data_line_confidences, not_data_line_confidences, combined_data_line_confidences, line_predictions, args.markov_approximation_probabilities, args.markov_model, 2, args.combined_label_weight)

    header_predictions= {}
    header_predictions['avg_confidence'] = first_data_line_combined_data_predictions['avg_confidence']
    header_predictions['min_confidence'] = first_data_line_combined_data_predictions['min_confidence']
    header_predictions['softmax'] = first_data_line_combined_data_predictions['softmax']
    header_predictions['prod_softmax_prior']= first_data_line_combined_data_predictions['prod_softmax_prior']

    
    predicted_pat_header_indexes= []
    candidate_pat_sub_headers= []
    predicted_pat_sub_headers= []

    # print(f'pat_first_data_line={pat_first_data_line}')
    
    if pat_first_data_line>0:
        predicted_pat_header_indexes, candidate_pat_sub_headers = predict_header_indexes(csv_file, pat_first_data_line, table_counter)
        # print(f'\npredicted_pat_header_indexes={predicted_pat_header_indexes}\n')
        predicted_pat_data_lines=[]
        discovered_table= {}
        discovered_table['table_counter']=table_counter
        discovered_table['data_start'] = pat_first_data_line
        discovered_table['fdl_confidence']= {}
        discovered_table['fdl_confidence']["avg_majority_confidence"]=float(first_data_line_combined_data_predictions["avg_confidence"]['confidence'])
        discovered_table['fdl_confidence']["avg_difference"]=float(first_data_line_combined_data_predictions["avg_confidence"]['difference'])
        discovered_table['fdl_confidence']["avg_confusion_index"]=float(first_data_line_combined_data_predictions["avg_confidence"]['confusion_index'])
        discovered_table['fdl_confidence']["softmax"]=float(first_data_line_combined_data_predictions['softmax'])
        discovered_table['header'] = predicted_pat_header_indexes
    
    return discovered_table        


def process_dataframe(table_counter, input_dataframe, args, delimiter):
    file_dataframe = input_dataframe[:]
    rule_weights=args.rule_weights

    impute_nulls= True
    predicted_tables = []
    if args.max_attributes!= None:
        if args.ignore_left!=None:
            max_attributes = args.max_attributes+args.ignore_left
        slice_idx = min(max_attributes,file_dataframe.shape[1])+1
        file_dataframe = file_dataframe.iloc[:,:slice_idx]

    dataframe_labels = []
    for column in file_dataframe:
        dataframe_labels.append(column)
    
    # input(file_dataframe)

    line_agreements, line_disagreements = collect_dataframe_rules(file_dataframe,args)
    csv_file = file_dataframe
    row_counter= 0
    label_confidences={}
    data_line_confidences= {}
    not_data_line_confidences= {}
    before_data= True
    for row_index in csv_file.index:
        row_counter+=1
        label_confidences[row_index]={}
        candidate_row_agreements=[]
        candidate_row_disagreements=[]

        for column_index in csv_file.columns:
            #################################################v######################v######
            #  DATA value classification
            value_agreements = line_agreements[row_index][column_index]['agreements']
            summary_strength = line_agreements[row_index][column_index]['summary_strength']  

            # if there are no lines below me to check agreement, 
            # and line before me exists and was data
            # see impute agreements
            if row_index in line_agreements.keys() and (line_agreements[row_index][column_index]['null_equivalent']==True or line_agreements[row_index][column_index]['summary_strength']==1) and args.impute_nulls==True and row_index-1 in line_agreements.keys() and column_index in line_agreements[row_index-1].keys() and row_index-1 in data_line_confidences.keys() and data_line_confidences[row_index-1]>not_data_line_confidences[row_index-1]:
                value_agreements = line_agreements[row_index-1][column_index]['agreements']
                summary_strength = line_agreements[row_index-1][column_index]['summary_strength']
            if row_index in line_agreements.keys() and line_agreements[row_index][column_index]['summary_strength']==0 and line_agreements[row_index][column_index]['aggregate'] and row_index-2 in line_agreements.keys() and column_index in line_agreements[row_index-2].keys() and row_index-2 in data_line_confidences.keys() and data_line_confidences[row_index-2]>not_data_line_confidences[row_index-2]:
                value_agreements = line_agreements[row_index-2][column_index]['agreements']
                summary_strength = line_agreements[row_index-2][column_index]['summary_strength']

            # otherwise, nothing was wrong, i can use my own damn agreements as initialized

            data_score = max_score(value_agreements, rule_weights['data_cell'], args.weight_lower_bound)
            POPULATION_WEIGHT = 1-(1-args.p)**(2*summary_strength)
            if data_score!=None:
                if args.summary_population_factor:
                    candidate_row_agreements.append(data_score*POPULATION_WEIGHT)
                else:
                    candidate_row_agreements.append(data_score) 

            #######################################################################v######
            #  NOT DATA value classification        
            value_disagreements = line_disagreements[row_index][column_index]['disagreements']
            disagreement_summary_strength = line_disagreements[row_index][column_index]['disagreement_summary_strength'] 
            not_data_score = max_score(value_disagreements, rule_weights['not_data_cell'], args.not_data_weight_lower_bound)
            POPULATION_WEIGHT = 1-(1-args.p)**(2*disagreement_summary_strength)

            if not_data_score!=None:
                if args.summary_population_factor:
                    candidate_row_disagreements.append(not_data_score*POPULATION_WEIGHT)
                else:
                    candidate_row_disagreements.append(not_data_score)

            ########################################################################                

        #################################################################################
        # NOT DATA line weights
        line_not_data_evidence =  [score for score in candidate_row_disagreements] 
        if args.weight_input=='values_and_lines':
            # if table_counter>1:
            #     input(f'row_index={row_index}\ndata_line_confidences={data_line_confidences}\n')
            #     input(f'row_index={row_index}\nnot_data_line_confidences={not_data_line_confidences}\n')

            if row_index>0 and row_index-1 in data_line_confidences.keys() and data_line_confidences[row_index-1]>not_data_line_confidences[row_index-1]:
                before_data=False

            if csv_file.shape[1]>1:
                not_data_line_rules_fired = line_disagreements[row_index]['line']

                for event in not_data_line_rules_fired:
                    if event =="UP_TO_FIRST_COLUMN_COMPLETE_CONSISTENTLY" and before_data == False:
                        continue
                    if event.startswith("ADJACENT_ARITHMETIC_SEQUENCE") and rule_weights['not_data_line'][event]==None:
                        steps = event[-1]
                        if int(steps) in range(2,6):
                            event = event[:-1]+ str(int(steps)+1)
                    if rule_weights['not_data_line'][event]!=None and  rule_weights['not_data_line'][event]>args.not_data_weight_lower_bound:
                        line_not_data_evidence.append(rule_weights['not_data_line'][event]) 

        not_data_conf = probabilistic_sum(line_not_data_evidence)
        not_data_line_confidences[row_index] = not_data_conf    

        # DATA line weights          
        line_is_data_evidence =  [score for score in candidate_row_agreements] 
        if args.weight_input=='values_and_lines':
            line_is_data_events = line_agreements[row_index]['line']
            for rule in line_is_data_events:
                if rule_weights['data_line'][rule]!=None and rule_weights['data_line'][rule]>args.weight_lower_bound:
                    line_is_data_evidence.append(rule_weights['data_line'][rule])

        # calculate confidence that this row is data
        data_conf = probabilistic_sum(line_is_data_evidence)
        data_line_confidences[row_index] = data_conf 

        label_confidences[row_index]['DATA'] = data_conf
        label_confidences[row_index]['NOT-DATA'] = not_data_conf  

    combined_data_line_confidences, line_predictions = predict_combined_data_confidences(csv_file, data_line_confidences, not_data_line_confidences, args.max_candidates)
    # if table_counter>1:
    #     print(f'line_predictions={line_predictions}')
    pat_first_data_line, first_data_line_combined_data_predictions = predict_fdl(csv_file, data_line_confidences, not_data_line_confidences, combined_data_line_confidences, line_predictions, args.markov_approximation_probabilities, args.markov_model, 2, args.combined_label_weight)
    
    combined_data_predictions = [] 
    for key in combined_data_line_confidences.keys():
        value = combined_data_line_confidences[key]
        if value>0:
            combined_data_predictions.append(key)
    top_context_boundary = 0
    # print('\npat_first_data_line = '+str(pat_first_data_line))
    if pat_first_data_line!=-1:
        table_counter+=1
        predicted_table = {}
        predicted_table['footnotes']= [] #initialize
        predicted_table['table_counter'] = table_counter
        predicted_table['fdl_confidence'] = {}
        predicted_table['fdl_confidence']["avg_majority_confidence"]=float(first_data_line_combined_data_predictions["avg_confidence"]['confidence'])
        predicted_table['fdl_confidence']["avg_difference"]=float(first_data_line_combined_data_predictions["avg_confidence"]['difference'])
        predicted_table['fdl_confidence']["avg_confusion_index"]=float(first_data_line_combined_data_predictions["avg_confidence"]['confusion_index'])
        predicted_table['fdl_confidence']["softmax"]=float(first_data_line_combined_data_predictions['softmax'])
        predicted_table['data_start'] = pat_first_data_line
        data_index_list = []

        last_data_line_index = pat_first_data_line
        while last_data_line_index in combined_data_line_confidences.keys() and line_predictions[last_data_line_index]['label']=='DATA':
            data_index_list.append(last_data_line_index)
            last_data_line_index+=1
        
        candidate_data = file_dataframe.loc[data_index_list,:]
        
        # input(f'candidate_data:\n{candidate_data}')
        predicted_table['data_end'] = None
           
        predicted_header_indexes, predicted_pat_subheaders  = predict_header_indexes(file_dataframe.loc[:last_data_line_index],pat_first_data_line, table_counter)
        # print(f'\npredicted_header_indexes={predicted_header_indexes}\n')

        predicted_table['header'] = {}
        predicted_table['header']['index_list'] = predicted_header_indexes
        predicted_table['header']['num_lines'] = len(predicted_header_indexes)
        cleaned_candidate_data = input_dataframe.loc[pat_first_data_line:last_data_line_index,:].dropna(axis=0, how='all',inplace=False)
        if len(predicted_header_indexes)>0:
            predicted_table['header']["from_line_idx"] = predicted_header_indexes[0]
            predicted_table['header']["to_line_idx"] = predicted_header_indexes[-1]+1
            
            cleaned_candidate_section = input_dataframe.loc[predicted_header_indexes[0]:last_data_line_index,:].dropna(axis=1, how='all',inplace=False)
            predicted_table['header']['combo'] = combo_row(file_dataframe.loc[predicted_header_indexes,:])
            predicted_table['five_lines'] = cleaned_candidate_section.loc[pat_first_data_line:min(cleaned_candidate_section.index[-1], pat_first_data_line+5), :].to_csv(sep=',', index=False,header=False)
        else:
            cleaned_candidate_section = cleaned_candidate_data.dropna(axis=1, how='all',inplace=False)
            predicted_table['five_lines'] = cleaned_candidate_section.iloc[:min(cleaned_candidate_section.shape[0], 5), :].to_csv(sep=',', index=False,header=False)
            predicted_table['header']['combo'] = []
        
        # print("predicted_table['five_lines']=")
        # input(predicted_table['five_lines'])

        if cleaned_candidate_data.shape[0] == 1 and len(predicted_header_indexes)==0 or cleaned_candidate_section.shape[1]==1:
            # print('Invalid Data')
            table_counter-=1
        else:
            if len(predicted_header_indexes)>0:
                # print('\n---------------\n-- HEADER\n---------------\n')
                # print(file_dataframe.loc[predicted_header_indexes,:])
                header_text = file_dataframe.loc[predicted_header_indexes,:].fillna('')
                predicted_table['header']['header_text']=header_text.values.tolist()

                # print("predicted_table['header_text']=")
                # input(predicted_table['header']['header_text'])
                meta_end = predicted_header_indexes[0]
            else:
                meta_end = pat_first_data_line
            predicted_table['has_premeta']= False
            if (len(predicted_header_indexes)==0 and pat_first_data_line!=0) or (len(predicted_header_indexes)>0 and top_context_boundary!=predicted_header_indexes[0]):
                predicted_table['has_premeta']= True
                predicted_table['premeta']= {}
                predicted_table['premeta']['from_line_idx'] = top_context_boundary
                predicted_table['premeta']['to_line_idx'] = meta_end
                premeta_text = file_dataframe.iloc[top_context_boundary:meta_end,:].fillna('')
                predicted_table['premeta']['premeta_text'] = premeta_text.values.tolist()
                
            # print('\n---------------\n-- DATA\n---------------\n')
            # print(cleaned_candidate_data.dropna(axis=1, how='all',inplace=False))
            predicted_table['dataframe']= cleaned_candidate_data.dropna(axis=1, how='all',inplace=False)
            predicted_table['num_columns']= cleaned_candidate_section.shape[1]
            predicted_tables.append(predicted_table)

    # input('\n\npredicted_tables')
    # pp.pprint(predicted_tables)

    return table_counter, predicted_tables, combined_data_predictions

def view_dataframe(filepath):
    all_csv_tuples= None
    failure= None
    resource_directory= '/home/christina/OPEN_DATA_CRAWL_2018/'
    filepath = os.path.join(resource_directory, filepath)
    all_csv_tuples, discovered_delimiter, discovered_encoding,encoding_language,encoding_confidence, failure, blank_lines_index, google_detected_lang = sample_file(filepath,100)
    dataframes = []
    if failure==None and all_csv_tuples!=None and  len(all_csv_tuples)>0:
        table_counter = 0
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
                dataframe.fillna(value=pd.np.nan, inplace=True)
                dataframe.replace(to_replace=[None], value=np.nan, inplace=True)
                dataframe = dataframe.replace(r'^\s*$' , pd.np.nan, regex=True)

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

    df = dataframe
    return df 

def process_file(filepath, args):
    
    discovered_delimiter= None
    num_rows= None
    discovered_encoding= None 
    encoding_confidence= None
    encoding_language= None
    google_detected_lang= None
    predicted_tables=[]
    failure= None
    blank_lines_index= None
    null_equi_spotted= []
    # input(filepath)
    all_csv_tuples, discovered_delimiter, discovered_encoding,encoding_language,encoding_confidence, failure, blank_lines_index, google_detected_lang = sample_file(filepath,100)

    # input('len(all_csv_tuples)='+str(len(all_csv_tuples)))
    # input(pd.DataFrame(all_csv_tuples))

    # print('\nFile encoding = '+str(discovered_encoding))
    # print('Encoding confidence = '+ str(encoding_confidence))
    # print('Encoding language = ' +str(encoding_language))
    # print('google_detected_lang='+str(google_detected_lang))

    if failure==None and all_csv_tuples!=None and  len(all_csv_tuples)>0:
        table_counter = 0
        start_index = 0
        line_index = 0
        csv_tuples=[all_csv_tuples[0]]
        num_fields = len(all_csv_tuples[0])
        dataframes = []
        # print('num_fields='+str(num_fields))
        # input('line_index_'+str(line_index)+': '+str(all_csv_tuples[0]))
        empty_lines = []
        line_index+=1
        while line_index<len(all_csv_tuples):
            csv_tuple = all_csv_tuples[line_index]
            if len(csv_tuple)==0:                
                csv_tuple = ['' for i in range(0, num_fields)]
                empty_lines.append(line_index)

            if len(csv_tuple)==num_fields:
                # print('line_index_'+str(line_index)+': '+str(csv_tuple))
                csv_tuples.append(csv_tuple)
                end_index = line_index+1
            else:            
                end_index = line_index

            if len(csv_tuple)!=num_fields or (len(csv_tuple)==num_fields and line_index+1 == len(all_csv_tuples)):
                
                dataframe = pd.DataFrame(csv_tuples)
                dataframe.index = list(range(start_index, end_index))
                dataframe.fillna(value=pd.np.nan, inplace=True)
                dataframe.replace(to_replace=[None], value=np.nan, inplace=True)
                dataframe = dataframe.replace(r'^\s*$' , pd.np.nan, regex=True)

                if start_index-1 in empty_lines or len(dataframes)==0:
                    dataframes.append(dataframe)
                else:
                    dataframes[-1] = dataframes[-1].append(dataframe)

                start_index= end_index
                if start_index<len(all_csv_tuples):
                    csv_tuples=[csv_tuple] 
                    num_fields = len(csv_tuple)                               

            line_index+=1 


        for dataframe in dataframes:
            # input(dataframe)
            try: 
                
                # input('\n\nPress enter to Search:\n')               
                table_counter, tables, predicted_combined_data = process_dataframe(table_counter, dataframe, args, discovered_delimiter)
                # pp.pprint(tables)
                predicted_tables+=tables

            except Exception as error:
                traceback.print_exc()
                print(error.__class__.__name__)
                print(error)
                failure = error
                break    

            
    # pp.pprint(predicted_tables)
    return discovered_delimiter, num_rows, discovered_encoding, encoding_confidence, encoding_language, predicted_tables, failure, blank_lines_index, null_equi_spotted, google_detected_lang