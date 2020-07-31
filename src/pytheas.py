#!/usr/bin/python3

import re
import subprocess
from multiprocessing import Pool
import codecs
import csv
import io
from datetime import timedelta
from timeit import default_timer as timer
import traceback
import os
import string

from dotmap import DotMap
from sortedcontainers import SortedDict

from sqlalchemy import create_engine
import numpy as np
import pandas as pd
from tqdm import tqdm

from nltk import word_tokenize
from nltk.corpus import stopwords

from psycopg2 import connect
from psycopg2.extras import Json
from psycopg2.extras import execute_values
import psycopg2

from table_classifier_utilities import TableSignatures, is_consistent_symbol_sets, predict_fdl, predict_line_label
from table_classifier_utilities import predict_combined_data_confidences, predict_header_indexes
from table_classifier_utilities import eval_data_cell_rule, eval_not_data_cell_rule, line_has_null_equivalent
from table_classifier_utilities import all_numbers, is_number, assess_data_line, assess_non_data_line
from table_classifier_utilities import non_nulls_in_line, discover_aggregation_scope, contains_number, eval_numeric_pattern
from header_events import collect_events_on_row, header_row_with_aggregation_tokens, collect_arithmetic_events_on_row
import pytheas_utilities as pytheas_util
from utilities import null_equivalent_values
import file_utilities

psycopg2.extensions.register_adapter(np.int64, psycopg2.extensions.AsIs)

stop = stopwords.words('french') + stopwords.words('english') + list(string.punctuation)


class Pytheas:
    def __init__(self):
        self.opendata_engine = None
        self.parameters = DotMap({
            "undersample_data_limit": 2,
            "max_candidates": 100,
            "max_summary_strength": 6,
            "max_line_depth": 30,
            "max_attributes": 20,
            "outlier_sensitive": True,
            "normalize_decimals": True,
            "impute_nulls": True,
            "ignore_left": 4,
            "summary_population_factor": True,
            "weight_input": 'values_and_lines',
            "weight_lower_bound": 0.4,
            "not_data_weight_lower_bound": 0.6,
            "p": 0.3,
            "markov_model": None,
            "markov_approximation_probabilities": None,
            "combined_label_weight": 'confidence'#one of [confidence,confusion_index,difference]
        })
        self.ignore_rules = {
            "cell":{
                "not_data": [],
                "data": []
            },
            "line":{
                "not_data": [],
                "data": []
            }
        }
        self.fuzzy_rules = dict()
        self.fuzzy_rules["cell"] = {
            "not_data":{
                "First_FW_Symbol_disagrees":{
                    "name":"",
                    "theme":"SYMBOLS"
                },
                "First_BW_Symbol_disagrees":{
                    "name":"",
                    "theme":"SYMBOLS"},
                "SymbolChain":{
                    "name":"",
                    "theme":"SYMBOLS"},
                "CC":{
                    "name":"",
                    "theme":"CASE"
                },
                "CONSISTENT_NUMERIC": {
                    "name":"Below but not here: consistently ONE symbol = D",
                    "theme":"SYMBOLS"
                },
                "CONSISTENT_D_STAR": {
                    "name":"Below but not here: consistently TWO symbols, the first is a digit",
                    "theme":"SYMBOLS"
                },
                "FW_SUMMARY_D": {
                    "name":"Below but not here: two or above symbols in the FW summary, the first is a digit",
                    "theme":"SYMBOLS"
                },
                "BW_SUMMARY_D": {
                    "name":"Below but not here: two or above symbols in the BW summary, the first is a digit",
                    "theme":"SYMBOLS"
                },
                "BROAD_NUMERIC": {
                    "name":"Below but not here: all values digits, optionally have . or ,  or S",
                    "theme":"SYMBOLS"
                },
                "FW_THREE_OR_MORE_NO_SPACE": {
                    "name":"Below but not here: three or above symbols in FW summary that do not contain a  Space",
                    "theme":"SYMBOLS"
                },
                "BW_THREE_OR_MORE_NO_SPACE": {
                    "name":"Below but not here: three or above symbols in BW summary that do not contain a  Space",
                    "theme":"SYMBOLS"
                },
                "CONSISTENT_SS_NO_SPACE": {
                    "name":"Below but not here: consistently at least two symbols in the symbol set summary, none of which are S or _",
                    "theme":"SYMBOLS"
                },
                "FW_TWO_OR_MORE_OR_MORE_NO_SPACE": {
                    "name":"Below but not here: two or above symbols in FW summary that do not contain a Space",
                    "theme":"SYMBOLS"
                },
                "BW_TWO_OR_MORE_OR_MORE_NO_SPACE": {
                    "name":"Below but not here: two or above symbols in BW summary that do not contain a Space",
                    "theme":"SYMBOLS"
                },
                "CHAR_COUNT_UNDER_POINT1_MIN":{
                    "name":"",
                    "theme":"LENGTH_CTXT"
                },
                "CHAR_COUNT_UNDER_POINT3_MIN":{
                    "name":"",
                    "theme":"LENGTH_CTXT"
                },
                "CHAR_COUNT_OVER_POINT5_MAX":{
                    "name":"",
                    "theme":"LENGTH_CTXT"
                },
                "CHAR_COUNT_OVER_POINT6_MAX":{
                    "name":"",
                    "theme":"LENGTH_CTXT"
                },
                "CHAR_COUNT_OVER_POINT7_MAX":{
                    "name":"",
                    "theme":"LENGTH_CTXT"
                },
                "CHAR_COUNT_OVER_POINT8_MAX":{
                    "name":"",
                    "theme":"LENGTH_CTXT"
                },
                "CHAR_COUNT_OVER_POINT9_MAX":{
                    "name":"",
                    "theme":"LENGTH_CTXT"
                },
                "NON_NUMERIC_CHAR_COUNT_DIFFERS_FROM_CONSISTENT":{
                    "name":"",
                    "theme":"LENGTH_CTXT"
                }
            },
            "data":{
                "VALUE_REPEATS_ONCE_BELOW": {
                    "name":"Rule_1_a value repetition only once in values below me, skip the adjacent value",
                    "theme":"VALUE_CTXT"
                },
                "VALUE_REPEATS_TWICE_OR_MORE_BELOW": {
                    "name":"Rule_1_b value repetition twice or more below me",
                    "theme":"VALUE_CTXT"
                },
                "ONE_ALPHA_TOKEN_REPEATS_ONCE_BELOW": {
                    "name":"Rule_2_a: Only one alphabetic token from multiword value repeats below, and it repeats only once",
                    "theme":"VALUE_CTXT"
                },
                "ALPHA_TOKEN_REPEATS_TWICE_OR_MORE": {
                    "name":"Rule_2_b: At least one alphabetic token from multiword value repeats below at least twice",
                    "theme":"VALUE_CTXT"
                },
                "CONSISTENT_NUMERIC_WIDTH": {
                    "name":"Rule_3 consistently numbers with consistent digit count for all.",
                    "theme":"SYMBOLS"
                },
                "CONSISTENT_NUMERIC": {
                    "name":"Rule_4_a consistently ONE symbol = D",
                    "theme":"SYMBOLS"
                },
                "CONSISTENT_D_STAR": {
                    "name":"Rule_4_b consistently TWO symbols, the first is a digit",
                    "theme":"SYMBOLS"
                },
                "FW_SUMMARY_D": {
                    "name":"Rule_4_fw two or above symbols in the FW summary, the first is a digit",
                    "theme":"SYMBOLS"
                },
                "BW_SUMMARY_D": {
                    "name":"Rule_4_bw two or above symbols in the BW summary, the first is a digit",
                    "theme":"SYMBOLS"
                },
                "BROAD_NUMERIC": {
                    "name":"Rule_5 all values digits, optionally have . or ,  or S",
                    "theme":"SYMBOLS"
                },
                "FW_THREE_OR_MORE_NO_SPACE": {
                    "name":"Rule_6 three or above symbols in FW summary that do not contain a  Space",
                    "theme":"SYMBOLS"
                },
                "BW_THREE_OR_MORE_NO_SPACE": {
                    "name":"Rule_7 three or above symbols in BW summary that do not contain a  Space",
                    "theme":"SYMBOLS"
                },
                "CONSISTENT_SS_NO_SPACE": {
                    "name":"Rule_8 consistently at least two symbols in the symbol set summary, none of which are S or _",
                    "theme":"SYMBOLS"
                },
                "CONSISTENT_SC_TWO_OR_MORE": {
                    "name":"Rule_10 two or more symbols consistent chain",
                    "theme":"SYMBOLS"
                },
                "FW_TWO_OR_MORE_OR_MORE_NO_SPACE": {
                    "name":"Rule_11_fw two or above symbols in FW summary that do not contain a Space",
                    "theme":"SYMBOLS"
                },
                "BW_TWO_OR_MORE_OR_MORE_NO_SPACE": {
                    "name":"Rule_11_bw two or above symbols in BW summary that do not contain a Space",
                    "theme":"SYMBOLS"
                },
                "FW_TWO_OR_MORE_OR_MORE_NO_SPACE_FIRST_TWO": {
                    "name":"Rule_12_fw two or above symbols in FW summary, the first two do not contain a Space",
                    "theme":"SYMBOLS"
                },
                "BW_TWO_OR_MORE_OR_MORE_NO_SPACE_FIRST_TWO": {
                    "name":"Rule_12_bw two or above symbols in BW summary, the first two do not contain a Space",
                    "theme":"SYMBOLS"
                },
                "FW_D5PLUS": {
                    "name":"Rule_13_fw FW summary is [['D',count]], where count>=5",
                    "theme":"SYMBOLS"
                },
                "BW_D5PLUS": {
                    "name":"Rule_13_bw BW summary is [['D',count]], where count>=5",
                    "theme":"SYMBOLS"
                },
                "FW_D1": {
                    "name":"Rule_14_fw FW summary is [['D',1]]",
                    "theme":"SYMBOLS"
                },
                "BW_D1": {
                    "name":"Rule_14_bw BW summary is [['D',1]]",
                    "theme":"SYMBOLS"
                },
                "FW_D4": {
                    "name":"Rule_15_fw FW summary is [['D',4]]",
                    "theme":"SYMBOLS"
                },
                "BW_D4": {
                    "name":"Rule_15_bw BW summary is [['D',4]]",
                    "theme":"SYMBOLS"
                },
                "FW_LENGTH_4PLUS": {
                    "name":"Rule_17_fw four or more symbols in the FW summary",
                    "theme":"SYMBOLS"
                },
                "BW_LENGTH_4PLUS": {
                    "name":"Rule_17_bw four or more symbols in the BW summary",
                    "theme":"SYMBOLS"
                },
                "CASE_SUMMARY_CAPS":{
                    "name":"Rule_18 case summary is ALL_CAPS",
                    "theme":"CASE"
                },
                "CONSISTENT_CHAR_LENGTH":{
                    "name":"this value and neighboring values are constant char length",
                    "theme":"LENGTH_CTXT"
                },
                "CONSISTENT_SINGLE_WORD_CONSISTENT_CASE":{
                    "name":"",
                    "theme":"CASE"
                }
            }
        }

        self.fuzzy_rules["line"] = {
            "not_data":{
                "ADJACENT_ARITHMETIC_SEQUENCE_2":{
                    "type":"header",
                    "name":"",
                    "theme":"VALUES_CTXT"
                },
                "ADJACENT_ARITHMETIC_SEQUENCE_3":{
                    "type":"header",
                    "name":"",
                    "theme":"VALUES_CTXT"
                },
                "ADJACENT_ARITHMETIC_SEQUENCE_4":{
                    "type":"header",
                    "name":"",
                    "theme":"VALUES_CTXT"
                },
                "ADJACENT_ARITHMETIC_SEQUENCE_5":{
                    "type":"header",
                    "name":"",
                    "theme":"VALUES_CTXT"
                },
                "ADJACENT_ARITHMETIC_SEQUENCE_6_plus":{
                    "type":"header",
                    "name":"",
                    "theme":"VALUES_CTXT"
                },
                "RANGE_PAIRS_1":{
                    "type":"header",
                    "name":"",
                    "theme":"VALUES_CTXT"
                },
                "RANGE_PAIRS_2_plus":{
                    "type":"header",
                    "name":"",
                    "theme":"VALUES_CTXT"
                },
                "PARTIALLY_REPEATING_VALUES_length_2_plus":{
                    "type":"header",
                    "name":"",
                    "theme":"VALUES_CTXT"
                },
                "METADATA_LIKE_ROW":{
                    "type":"header",
                    "name":"",
                    "theme":"SYMBOLS"
                },
                "CONSISTENTLY_SLUG_OR_SNAKE":{
                    "type":"header",
                    "name":"",
                    "theme":"CASE"
                },
                "CONSISTENTLY_UPPER_CASE":{
                    "type":"header",
                    "theme":"CASE"
                },
                "AGGREGATION_ON_ROW_WO_NUMERIC":{
                    "type":"aggregation",
                    "name":"",
                    "theme":"VALUES_CTXT"
                },
                "AGGREGATION_ON_ROW_W_ARITH_SEQUENCE":{
                    "type":"aggregation",
                    "name":"",
                    "theme":"VALUES_CTXT"
                },
                "UP_TO_FIRST_COLUMN_COMPLETE_CONSISTENTLY":{
                    "type":"other",
                    "name":"all lines consistently non-data from beginning of input, left-most column potentially non-null",
                    "theme":"SYMBOLS"
                },
                "STARTS_WITH_NULL":{
                    "type":"other",
                    "name":"line starts with null",
                    "theme":"SYMBOLS"
                },
                "NO_SUMMARY_BELOW":{
                    "type":"other",
                    "name":"no summary achieved below in any column",
                    "theme":"SYMBOLS"
                },
                "FOOTNOTE":{
                    "type":"other",
                    "name":"line resembles footnote",
                    "theme":"VALUES"
                },
                "METADATA_TABLE_HEADER_KEYWORDS":{
                    "type":"header",
                    "name":"Line has cells with metadata specifiers",
                    "theme":"VALUES"
                }
            },
            "data":{
                "AGGREGATION_TOKEN_IN_FIRST_VALUE_OF_ROW":{
                    "type":"aggregation",
                    "name":"First value of a row (first column) contains an aggregation token, this is likely a summarizing data line",
                    "theme":"VALUES_CTXT"
                },
                "NULL_EQUIVALENT_ON_LINE_2_PLUS":{
                    "type":"other",
                    "name":"Two or more null equivalent values found on a line",
                    "theme":"VALUES"
                },
                "ONE_NULL_EQUIVALENT_ON_LINE":{
                    "type":"other",
                    "name":"One null equivalent value found on line",
                    "theme":"VALUES"
                },
                "CONTAINS_DATATYPE_CELL_VALUE":{
                    "type":"other",
                    "name":"Line has cell value taken from datatype keyword list",
                    "theme":"VALUES"
                }
            }
        }


    def leave_rules_out(self, ignore_rules):
        self.ignore_rules = ignore_rules


    def predict_subheaders(self, csv_file, cand_data, predicted_pytheas_sub_headers, pytheas_blank_lines, pytheas_headers):
        ignore_rules = self.ignore_rules
        fuzzy_rules = self.fuzzy_rules
        args = self.parameters

        cand_subhead_indexes = predicted_pytheas_sub_headers
        cand_subhead_indexes.sort()

        candidate_subheaders = dict()
        subheader_scope = dict()
        certain_data_indexes = list(cand_data.index)
        aggregation_rows = dict()
        first_column_data_values = []

        for row in cand_data.loc[certain_data_indexes].itertuples():
            first_value = str(row[1]).strip()
            for aggregation_phrase in pytheas_util.aggregation_functions:
                agg_index = first_value.lower().find(aggregation_phrase[0])
                if agg_index > -1:
                    aggregation_rows[row.Index] = {}
                    aggregation_rows[row.Index]['value'] = first_value
                    aggregation_rows[row.Index]['aggregation_function'] = aggregation_phrase[1]
                    aggregation_rows[row.Index]['aggregation_phrase'] = aggregation_phrase[0]
                    aggregation_rows[row.Index]['aggregation_label'] = first_value[:agg_index] + first_value[agg_index + len(aggregation_phrase[0]):]
                    break

            if row.Index not in aggregation_rows.keys() and first_value.lower() not in null_equivalent_values and row.Index not in cand_subhead_indexes:
                first_column_data_values.append(first_value)

        certain_data_indexes = list(set(certain_data_indexes) - set(aggregation_rows.keys()) - set(cand_subhead_indexes))

        for row in csv_file.loc[cand_subhead_indexes].itertuples():
            first_value = str(row[1]).strip()
            if first_value.lower() not in  ['nan', 'none', ''] and row.Index not in aggregation_rows.keys():
                candidate_subheaders[row.Index] = first_value

        cand_subhead_indexes = list(candidate_subheaders.keys())
        aggregation_rows, certain_data_indexes, predicted_pytheas_sub_headers, cand_subhead_indexes, subheader_scope = discover_aggregation_scope(csv_file.loc[:cand_data.index[-1]], aggregation_rows, candidate_subheaders, predicted_pytheas_sub_headers, certain_data_indexes, pytheas_headers)

        if cand_subhead_indexes is not None and len(cand_subhead_indexes) > 0:
            first_column_value_patterns = []
            first_column_value_symbols = []
            first_column_value_cases = []
            first_column_value_token_lengths = []
            first_column_value_char_lengths = []
            first_column_value_tokens = []
            first_column_all_patterns_numeric = []

            for value in first_column_data_values:
                pattern, symbols, case, value_num_tokens, value_num_chars = pytheas_util.generate_pattern_symbols_and_case(str(value).strip(), args.outlier_sensitive)
                first_column_value_patterns.append(pattern)
                first_column_value_symbols.append(symbols)
                first_column_value_cases.append(case)
                first_column_value_token_lengths.append(value_num_tokens)
                first_column_value_char_lengths.append(value_num_chars)
                first_column_value_tokens.append([i for i in (str(value).strip().lower()).split() if i not in stop and all(j.isalpha() or j in string.punctuation for j in i) and i not in null_equivalent_values])
                first_column_all_patterns_numeric.append(eval_numeric_pattern(pattern))
            if args.normalize_decimals:
                first_column_value_patterns, first_column_value_symbols = pytheas_util.normalize_decimals_numbers(first_column_value_patterns, first_column_value_symbols)

            value_pattern_summary, value_chain_consistent = pytheas_util.generate_pattern_summary(first_column_value_patterns)
            summary_strength = sum(1 for x in first_column_value_patterns if len(x) > 0)
            bw_patterns = [list(reversed(pattern)) for pattern in first_column_value_patterns]
            value_pattern_BW_summary, _ = pytheas_util.generate_pattern_summary(bw_patterns)

            value_symbol_summary = pytheas_util.generate_symbol_summary(first_column_value_symbols)
            case_summary = pytheas_util.generate_case_summary(first_column_value_cases)
            length_summary = pytheas_util.generate_length_summary(first_column_value_char_lengths)
            all_patterns_numeric, _ = pytheas_util.generate_all_numeric_sig_pattern(first_column_all_patterns_numeric,
                                                                                    [len(t) for t in first_column_value_patterns])

            candidate_tokens = set()
            if len(first_column_value_tokens) > 0:
                candidate_tokens = {t  for t in first_column_value_tokens[0] if any(c.isalpha() for c in t)}
            candidate_count_for_value = 0
            if len(first_column_data_values) > 2:
                candidate_count_for_value = np.count_nonzero(first_column_data_values[2:min(args.max_summary_strength, len(first_column_data_values))] == str(value).strip())

            partof_multiword_value_repeats = dict()
            for part in candidate_tokens:
                partof_multiword_value_repeats[part] = 0
                for value_tokens in first_column_value_tokens:
                    if part in value_tokens:
                        partof_multiword_value_repeats[part] += 1
            consistent_symbol_sets, _ = is_consistent_symbol_sets(first_column_value_symbols)
            data_rules_fired = {}
            data_rules_fired[1] = {}
            data_rules_fired[1][0] = {}
            data_rules_fired[1][0]['agreements'] = []
            data_rules_fired[1][0]['null_equivalent'] = False
            for rule in fuzzy_rules['cell']['data'].keys():
                rule_fired = False
                # Don't bother looking for agreements if there are no patterns
                non_empty_patterns = 0
                if rule not in ignore_rules['cell']['data'] and len(first_column_value_patterns) > 0:
                    for pattern in first_column_value_patterns:
                        if pattern != []:
                            non_empty_patterns += 1

                    #there is no point calculating agreement over one value, a single value always agrees with itself.
                    #in addition, many tables have bilingual headers, so agreement between two header values is very common, require nij>=3
                    if len(first_column_value_patterns) >= 2 and non_empty_patterns >= 2:
                        rule_fired = eval_data_cell_rule(rule, first_column_data_values, value_pattern_summary, value_chain_consistent, value_pattern_BW_summary, value_symbol_summary, case_summary, candidate_count_for_value, partof_multiword_value_repeats, candidate_tokens, consistent_symbol_sets, all_patterns_numeric)
                        if rule_fired and "_REPEATS_" not in rule:
                            data_rules_fired[1][0]['agreements'].append(rule)

            for row in csv_file.loc[cand_subhead_indexes].itertuples():
                first_value = str(row[1]).strip()
                if first_value.lower() in ['', 'nan', 'none', 'null']:
                    continue
                if first_value in first_column_data_values:
                    continue
                if row.Index - 1 in pytheas_blank_lines or row.Index - 1 in pytheas_headers:
                    predicted_pytheas_sub_headers.append(row.Index)
                else:
                    value_tokens = first_value.lower().split()
                    pattern, symbols, case, value_num_tokens, value_num_chars = pytheas_util.generate_pattern_symbols_and_case(str(first_value).strip(), args.outlier_sensitive)
                    if args.normalize_decimals:
                        column_patterns, column_symbols = pytheas_util.normalize_decimals_numbers([pattern] + first_column_value_patterns, [symbols] + first_column_value_symbols)
                    value_pattern_summary, value_chain_consistent = pytheas_util.generate_pattern_summary(column_patterns)

                    summary_strength = sum(1 for x in column_patterns if len(x) > 0)
                    bw_patterns = [list(reversed(pattern)) for pattern in column_patterns]
                    value_pattern_BW_summary, _ = pytheas_util.generate_pattern_summary(bw_patterns)
                    value_symbol_summary = pytheas_util.generate_symbol_summary(column_symbols)
                    case_summary = pytheas_util.generate_case_summary([case] + first_column_value_cases)
                    length_summary = pytheas_util.generate_length_summary([value_num_chars] + first_column_value_char_lengths)
                    all_patterns_numeric, _ = pytheas_util.generate_all_numeric_sig_pattern([eval_numeric_pattern(pattern)] + first_column_all_patterns_numeric, [len(t) for t in column_patterns])
                    column_values = [first_value] + first_column_data_values
                    column_tokens = [value_tokens] + first_column_value_tokens

                    candidate_tokens = {t  for t in value_tokens if any(c.isalpha() for c in t)}
                    candidate_count_for_value = 0
                    if len(column_values) > 2:
                        candidate_count_for_value = np.count_nonzero(column_values[2:min(args.max_summary_strength, len(column_values))] == str(value).strip())

                    partof_multiword_value_repeats = dict()
                    for part in candidate_tokens:
                        partof_multiword_value_repeats[part] = 0
                        for value_tokens in column_tokens:
                            if part in value_tokens:
                                partof_multiword_value_repeats[part] += 1

                    consistent_symbol_sets, _ = is_consistent_symbol_sets(column_symbols)
                    data_rules_fired[0] = {}
                    data_rules_fired[0][0] = {}
                    data_rules_fired[0][0]['agreements'] = []
                    data_rules_fired[0][0]['null_equivalent'] = False
                    for rule in fuzzy_rules['cell']['data'].keys():
                        rule_fired = False
                        non_empty_patterns = 0
                        if len(column_patterns) > 0 and first_value.lower() not in null_equivalent_values:
                            for pattern in column_patterns:
                                if pattern != []:
                                    non_empty_patterns += 1
                            if len(column_patterns) >= 2 and non_empty_patterns >= 2:
                                rule_fired = eval_data_cell_rule(rule, column_values, value_pattern_summary,
                                                                 value_chain_consistent, value_pattern_BW_summary,
                                                                 value_symbol_summary,
                                                                 case_summary, candidate_count_for_value,
                                                                 partof_multiword_value_repeats, candidate_tokens,
                                                                 consistent_symbol_sets, all_patterns_numeric)
                                if rule_fired and "_REPEATS_" not in rule:
                                    data_rules_fired[0][0]['agreements'].append(rule)

                    value_disagreements = []
                    disagreement_summary_strength = summary_strength - 1
                    if len(pattern) > 0:
                        for rule in fuzzy_rules['cell']['not_data'].keys():
                            rule_fired = False
                            if rule not in ignore_rules['cell']['not_data'] and disagreement_summary_strength > 0 and (not all_numbers(column_symbols) or not is_number(symbols)):
                                rule_fired = eval_not_data_cell_rule(rule, value_pattern_summary, value_pattern_BW_summary, value_chain_consistent, value_symbol_summary, case_summary, length_summary, pattern, case, value_num_chars, disagreement_summary_strength, data_rules_fired, 0, 0)
                                if rule_fired and "_REPEATS_" not in rule:
                                    value_disagreements.append(rule)

                    #######################################################################v######
                    #  DATA value classification
                    data_score = max_score(data_rules_fired[0][0]['agreements'], fuzzy_rules['cell']['data'], args.weight_lower_bound)
                    POPULATION_WEIGHT = 1 - (1 - args.p)**(2 * summary_strength)
                    if data_score is not None:
                        if args.summary_population_factor:
                            cell_data_score = data_score * POPULATION_WEIGHT
                        else:
                            cell_data_score = data_score

                    #######################################################################v######
                    #  NOT DATA value classification
                    not_data_score = max_score(value_disagreements, fuzzy_rules['cell']['not_data'], args.not_data_weight_lower_bound)
                    POPULATION_WEIGHT = 1 - (1 - args.p)**(2 * disagreement_summary_strength)
                    if not_data_score is not None:
                        if args.summary_population_factor:
                            cell_not_data_score = not_data_score * POPULATION_WEIGHT
                        else:
                            cell_not_data_score = not_data_score

                    if  cell_data_score > cell_not_data_score:# candidate subheader is definitely data, move along
                        continue

                    if (row.Index - 1 in predicted_pytheas_sub_headers and row.Index - 2 in predicted_pytheas_sub_headers):
                        continue

                    if row.Index != cand_data.index[-1]:
                        predicted_pytheas_sub_headers.append(row.Index)

        for s_i, subheader in enumerate(predicted_pytheas_sub_headers):
            if subheader not in subheader_scope.keys():
                if s_i + 1 == len(predicted_pytheas_sub_headers):
                    subheader_scope[subheader] = list(range(subheader + 1, cand_data.index[-1] + 1))
                else:
                    next_s_i = s_i + 1
                    while next_s_i < len(predicted_pytheas_sub_headers):
                        next_subh = predicted_pytheas_sub_headers[next_s_i]
                        if next_subh not in subheader_scope:
                            subheader_scope[subheader] = list(range(subheader + 1, next_subh))
                            break
                        next_s_i += 1

        return  aggregation_rows, subheader_scope


    def connect_opendata_profile(self, db_cred):
        self.opendata_engine = create_engine(f'postgresql+psycopg2://{db_cred.user}:{db_cred.password}@localhost:{db_cred.port}/{db_cred.opendata_database}')


    def load_model(self, fuzzy_rules):
        self.fuzzy_rules = fuzzy_rules


    def train_rules(self, undersampled_cell_data, undersampled_cell_not_data, undersampled_line_data, undersampled_line_not_data):
        total_cell_instances = undersampled_cell_data.shape[0]
        total_line_instances = undersampled_line_data.shape[0]

        for rule in self.fuzzy_rules["cell"]["data"].keys():
            data_cell_predicted_positive = 0
            data_cell_true_positive = 0
            data_cell_false_positive = 0

            if rule.lower() in undersampled_cell_data.columns:
                data_cell_predicted_positive = undersampled_cell_data.query(f"{rule.lower()}==True").shape[0]
                data_cell_true_positive = undersampled_cell_data.query(f"{rule.lower()}==True and label=='DATA'").shape[0]
                data_cell_false_positive = undersampled_cell_data.query(f"{rule.lower()}==True and label!='DATA'").shape[0]

            rule_weight = None
            confidence = None

            if data_cell_predicted_positive != 0:
                rule_weight = data_cell_true_positive / data_cell_predicted_positive - (data_cell_false_positive / data_cell_predicted_positive)
                confidence = data_cell_true_positive / data_cell_predicted_positive

            coverage = data_cell_predicted_positive / total_cell_instances

            self.fuzzy_rules["cell"]["data"][rule]["weight"] = rule_weight
            self.fuzzy_rules["cell"]["data"][rule]["confidence"] = confidence
            self.fuzzy_rules["cell"]["data"][rule]["coverage"] = coverage

        for rule in self.fuzzy_rules["cell"]["not_data"].keys():
            not_data_cell_predicted_positive = 0
            not_data_cell_true_positive = 0
            not_data_cell_false_positive = 0

            if rule.lower() in undersampled_cell_not_data.columns:
                not_data_cell_predicted_positive = undersampled_cell_not_data.query(f"{rule.lower()}==True").shape[0]
                not_data_cell_true_positive = undersampled_cell_not_data.query(f"{rule.lower()}==True and label!='DATA'").shape[0]
                not_data_cell_false_positive = undersampled_cell_not_data.query(f"{rule.lower()}==True and label=='DATA'").shape[0]

            rule_weight = None
            confidence = None

            if not_data_cell_predicted_positive != 0:
                rule_weight = not_data_cell_true_positive / not_data_cell_predicted_positive - (not_data_cell_false_positive / not_data_cell_predicted_positive)
                confidence = not_data_cell_true_positive / not_data_cell_predicted_positive

            coverage = not_data_cell_predicted_positive / total_cell_instances

            self.fuzzy_rules["cell"]["not_data"][rule]["weight"] = rule_weight
            self.fuzzy_rules["cell"]["not_data"][rule]["confidence"] = confidence
            self.fuzzy_rules["cell"]["not_data"][rule]["coverage"] = coverage

        for rule in self.fuzzy_rules["line"]["data"].keys():
            data_line_predicted_positive = 0
            data_line_true_positive = 0
            data_line_false_positive = 0
            if rule.lower() in undersampled_line_data.columns:
                data_line_predicted_positive = undersampled_line_data.query(f"{rule.lower()}==True").shape[0]
                data_line_true_positive = undersampled_line_data.query(f"{rule.lower()}==True and label=='DATA'").shape[0]
                data_line_false_positive = undersampled_line_data.query(f"{rule.lower()}==True and label!='DATA'").shape[0]

            rule_weight = None
            confidence = None

            if data_line_predicted_positive != 0:
                rule_weight = data_line_true_positive / data_line_predicted_positive - (data_line_false_positive / data_line_predicted_positive)
                confidence = data_line_true_positive / data_line_predicted_positive

            coverage = data_line_predicted_positive / total_line_instances

            self.fuzzy_rules["line"]["data"][rule]["weight"] = rule_weight
            self.fuzzy_rules["line"]["data"][rule]["confidence"] = confidence
            self.fuzzy_rules["line"]["data"][rule]["coverage"] = coverage

        for rule in self.fuzzy_rules["line"]["not_data"].keys():
            not_data_line_predicted_positive = 0
            not_data_line_true_positive = 0
            not_data_line_false_positive = 0

            if rule.lower() in undersampled_line_not_data.columns:
                not_data_line_predicted_positive = undersampled_line_not_data.query(f"{rule.lower()}==True").shape[0]
                not_data_line_true_positive = undersampled_line_not_data.query(f"{rule.lower()}==True and label!='DATA'").shape[0]
                not_data_line_false_positive = undersampled_line_not_data.query(f"{rule.lower()}==True and label=='DATA'").shape[0]

            rule_weight = None
            confidence = None

            if not_data_line_predicted_positive != 0:
                rule_weight = not_data_line_true_positive / not_data_line_predicted_positive - (not_data_line_false_positive / not_data_line_predicted_positive)
                confidence = not_data_line_true_positive / not_data_line_predicted_positive

            coverage = not_data_line_predicted_positive / total_line_instances

            self.fuzzy_rules["line"]["not_data"][rule]["weight"] = rule_weight
            self.fuzzy_rules["line"]["not_data"][rule]["confidence"] = confidence
            self.fuzzy_rules["line"]["not_data"][rule]["coverage"] = coverage


    def discover_next_table(self, csv_file, file_offset, table_counter, data_rules_fired, not_data_rules_fired, blank_lines, headers_discovered, signatures):
        parameters = self.parameters
        discovered_table = None
        csv_file = csv_file.loc[file_offset:]
        if csv_file.empty:
            return discovered_table

        data_line_confidences, not_data_line_confidences = get_class_confidences(csv_file,
                                                                                 data_rules_fired,
                                                                                 not_data_rules_fired,
                                                                                 self.fuzzy_rules,
                                                                                 parameters)

        _, line_predictions = predict_combined_data_confidences(csv_file, data_line_confidences,
                                                                not_data_line_confidences, parameters.max_candidates)

        pytheas_first_data_line, first_data_line_combined_data_predictions = predict_fdl(csv_file,
                                                                                         line_predictions,
                                                                                         parameters.markov_approximation_probabilities,
                                                                                         parameters.markov_model,
                                                                                         2, parameters.combined_label_weight)

        header_predictions = {}
        header_predictions['avg_confidence'] = first_data_line_combined_data_predictions['avg_confidence']
        header_predictions['min_confidence'] = first_data_line_combined_data_predictions['min_confidence']
        header_predictions['softmax'] = first_data_line_combined_data_predictions['softmax']
        header_predictions['prod_softmax_prior'] = first_data_line_combined_data_predictions['prod_softmax_prior']

        predicted_pytheas_header_indexes = []
        candidate_pytheas_sub_headers = []
        predicted_pytheas_sub_headers = []

        if pytheas_first_data_line >= 0:
            predicted_pytheas_header_indexes, candidate_pytheas_sub_headers = predict_header_indexes(csv_file, pytheas_first_data_line, table_counter)
            candidate_pytheas_sub_headers.sort()
            for h in predicted_pytheas_header_indexes:
                headers_discovered[h] = ','.join(csv_file.loc[h].apply(str).tolist())

            if len(candidate_pytheas_sub_headers) > 0:
                data_section_start = min(min(candidate_pytheas_sub_headers), pytheas_first_data_line)
            else:
                data_section_start = pytheas_first_data_line

            predicted_pytheas_data_lines = []

            #     # First data line predicted
            #     #############################   END  ###############################################

            #     ############################## START ###############################################
            #     # Predict Last data line

            cand_data = csv_file.loc[data_section_start:]
            aggregation_rows, subheader_scope = self.predict_subheaders_new(csv_file, cand_data, candidate_pytheas_sub_headers,
                                                                            blank_lines, predicted_pytheas_header_indexes)

            predicted_pytheas_sub_headers = list(subheader_scope.keys())
            pytheas_last_data_line = pytheas_first_data_line
            pytheas_last_data_line, bottom_boundary_confidence = predict_last_data_line_top_down(csv_file,
                                                                                                 pytheas_first_data_line,
                                                                                                 data_line_confidences,
                                                                                                 not_data_line_confidences,
                                                                                                 self, subheader_scope,
                                                                                                 aggregation_rows,
                                                                                                 blank_lines, headers_discovered,
                                                                                                 signatures,
                                                                                                 data_rules_fired,
                                                                                                 not_data_rules_fired)

            for blank_line_idx in blank_lines:
                if blank_line_idx in predicted_pytheas_data_lines:
                    predicted_pytheas_data_lines.remove(blank_line_idx)

            discovered_table = {}
            discovered_table['top_boundary'] = file_offset
            discovered_table['bottom_boundary'] = len(csv_file)-1
            discovered_table['data_start'] = data_section_start
            discovered_table['data_end_confidence'] = bottom_boundary_confidence['confidence']

            discovered_table['fdl_confidence'] = dict()
            discovered_table['fdl_confidence']["avg_majority_confidence"] = float(first_data_line_combined_data_predictions["avg_confidence"]['confidence'])
            discovered_table['fdl_confidence']["avg_difference"] = float(first_data_line_combined_data_predictions["avg_confidence"]['difference'])
            discovered_table['fdl_confidence']["avg_confusion_index"] = float(first_data_line_combined_data_predictions["avg_confidence"]['confusion_index'])
            discovered_table['fdl_confidence']["softmax"] = float(first_data_line_combined_data_predictions['softmax'])
            discovered_table['header'] = predicted_pytheas_header_indexes
            predicted_pytheas_data_lines = list(range(data_section_start, pytheas_last_data_line+1))
            discovered_table['data_end'] = pytheas_last_data_line
            discovered_table['footnotes'] = []

        #     # Last data line predicted
        #     #############################   END  ###############################################
        if pytheas_first_data_line >= 0:

            cand_data = csv_file.loc[data_section_start:pytheas_last_data_line]
            candidate_pytheas_sub_headers = sorted(list(set(predicted_pytheas_sub_headers).intersection(set(predicted_pytheas_data_lines))))

            aggregation_scope, subheader_scope = self.predict_subheaders(csv_file, cand_data, candidate_pytheas_sub_headers, blank_lines, predicted_pytheas_header_indexes)
            predicted_pytheas_sub_headers = subheader_scope.keys()

            discovered_table['subheader_scope'] = subheader_scope
            discovered_table['aggregation_scope'] = aggregation_scope

            for subheaders_idx in predicted_pytheas_sub_headers:
                if subheaders_idx in predicted_pytheas_data_lines:
                    predicted_pytheas_data_lines.remove(subheaders_idx)
            discovered_table['footnotes'] = []
        return discovered_table


    def extract_tables(self, crawl_datafile_key, file_dataframe_trimmed, blank_lines, assume_multi_table=True):
        #initialize
        discovered_tables = SortedDict()
        try:
            signatures = TableSignatures(file_dataframe_trimmed, self.parameters.outlier_sensitive)
            data_rules_fired, not_data_rules_fired = collect_dataframe_rules(file_dataframe_trimmed, self, signatures)

            table_counter = 1
            file_offset = 0
            headers_discovered = dict()
            discovered_table = self.discover_next_table(file_dataframe_trimmed,
                                                        file_offset,
                                                        table_counter,
                                                        data_rules_fired,
                                                        not_data_rules_fired,
                                                        blank_lines,
                                                        headers_discovered, signatures)

            if assume_multi_table:
                while discovered_table is not None:
                    discovered_tables[table_counter] = discovered_table
                    for h in discovered_table["header"]:
                        headers_discovered[h] = ','.join(file_dataframe_trimmed.loc[h].apply(str).tolist())

                    table_counter += 1
                    file_offset = discovered_table["data_end"] + 1
                    discovered_table = self.discover_next_table(file_dataframe_trimmed,
                                                                file_offset,
                                                                table_counter,
                                                                data_rules_fired,
                                                                not_data_rules_fired,
                                                                blank_lines,
                                                                headers_discovered, signatures)

                    if discovered_table is not None:
                        if set(range(discovered_tables[table_counter - 1]["data_end"] + 1, discovered_table["data_start"])).issubset(set(blank_lines + list(set(range(discovered_table["top_boundary"], discovered_table["data_start"])) - set(discovered_table["header"])))):
                            discovered_table = merge_tables(discovered_tables[table_counter - 1], discovered_table)
                            file_offset = discovered_table["data_end"] + 1
                            table_counter -= 1

                        if table_counter - 1 in discovered_tables.keys() and len(discovered_tables[table_counter - 1]["footnotes"]) > 0:
                            discovered_tables[table_counter - 1]["footnotes"] = list(range(discovered_tables[table_counter - 1]["footnotes"][0],
                                                                                           discovered_table["top_boundary"]))
                        if table_counter - 1 in discovered_tables.keys():
                            discovered_tables[table_counter - 1]["bottom_boundary"] = discovered_table["top_boundary"] - 1
                            discovered_tables[table_counter - 1]["data_end_confidence"] = discovered_table["data_end_confidence"]

                        if discovered_table["data_end"] < file_dataframe_trimmed.shape[0] - 1:
                            discovered_table["footnotes"] = list(range(discovered_table["data_end"] + 1, file_dataframe_trimmed.shape[0]))
                        else:
                            discovered_table["footnotes"] = []
                        if len(discovered_table["footnotes"]) > 0:
                            discovered_table["bottom_boundary"] = discovered_table["footnotes"][-1]
                        else:
                            discovered_table["bottom_boundary"] = discovered_table["data_end"]
                if table_counter-1 in discovered_tables.keys() and discovered_tables[table_counter - 1]["data_end"] != file_dataframe_trimmed.shape[0]:
                    footnotes = sorted(list(set(range(discovered_tables[table_counter - 1]["data_end"] + 1, file_dataframe_trimmed.shape[0])) - set(blank_lines)))
                    discovered_tables[table_counter-1]["footnotes"] = footnotes
            else:
                if discovered_table is not None:
                    discovered_tables[table_counter] = discovered_table
                    if discovered_tables[table_counter]["data_end"] != file_dataframe_trimmed.shape[0]:
                        footnotes = sorted(list(set(range(discovered_tables[table_counter]["data_end"]+1, file_dataframe_trimmed.shape[0])) - set(blank_lines)))
                        discovered_tables[table_counter]["footnotes"] = footnotes

        except Exception as e:
            print(f'crawl_datafile_key={crawl_datafile_key} failed to process, {e}: {traceback.format_exc()}')

        return discovered_tables


    def collect_rule_activation(self, db_cred, num_processors, top_level_dir):
        assume_multi_tables = True
        con = connect(dbname=db_cred.database,
                      user=db_cred.user,
                      host='localhost',
                      password=db_cred.password,
                      port=db_cred.port)
        cur = con.cursor()

        cur.execute("""DROP TABLE IF EXISTS pat_cell_datapoints""")
        cur.execute("""DROP TABLE IF EXISTS pat_data_cell_rules""")
        cur.execute("""DROP TABLE IF EXISTS pat_not_data_cell_rules""")
        con.commit()

        cur.execute("""DROP TABLE IF EXISTS pat_line_datapoints""")
        cur.execute("""DROP TABLE IF EXISTS pat_data_line_rules""")
        cur.execute("""DROP TABLE IF EXISTS pat_not_data_line_rules""")
        con.commit()

        cur.execute("""DROP TABLE IF EXISTS pat_line_and_cell_rules""")
        con.commit()

        cur.execute("""CREATE TABLE pat_line_and_cell_rules (
                    crawl_datafile_key integer,
                    data_rules_fired json,
                    not_data_rules_fired json)""")
        con.commit()


        cur.execute("""SELECT count(1)
                        FROM ground_truth_2k_canada
                        WHERE annotations is not null
                        """)
        NINPUTS = cur.fetchone()[0]
        NPROC = min(num_processors, available_cpu_count())
        # Process files
        combined_data = []
        with Pool(processes=NPROC) as pool:
            with tqdm(total=NINPUTS) as pbar:
                for r in pool.imap_unordered(pytheas_rule_worker,
                                             generate_rule_annotation_tasks(self,
                                                                            top_level_dir,
                                                                            db_cred,
                                                                            assume_multi_tables)):
                    combined_data.append(r)
                    pbar.update(1)

        start_reduce = timer()
        pytheas_line_datapoints_DATA = []
        pytheas_cell_datapoints_DATA = []
        pytheas_data_line_rules_DATA = []
        pytheas_not_data_line_rules_DATA = []
        pytheas_data_cell_rules_DATA = []
        pytheas_not_data_cell_rules_DATA = []
        pytheas_line_and_cell_rules_DATA = []
        for res in combined_data:
            crawl_datafile_key = res[0]
            pytheas_line_datapoints_DATA.append(res[1])
            pytheas_cell_datapoints_DATA.append(res[2])
            pytheas_data_line_rules_DATA.append(res[3])
            pytheas_not_data_line_rules_DATA.append(res[4])
            pytheas_data_cell_rules_DATA.append(res[5])
            pytheas_not_data_cell_rules_DATA.append(res[6])
            data_rules_fired = Json(res[7])
            not_data_rules_fired = Json(res[8])
            pytheas_line_and_cell_rules_DATA.append((crawl_datafile_key,
                                                     data_rules_fired,
                                                     not_data_rules_fired))


        pytheas_line_datapoints = pd.concat(pytheas_line_datapoints_DATA)
        pytheas_cell_datapoints = pd.concat(pytheas_cell_datapoints_DATA)
        pytheas_data_line_rules = pd.concat(pytheas_data_line_rules_DATA)
        pytheas_not_data_line_rules = pd.concat(pytheas_not_data_line_rules_DATA)
        pytheas_data_cell_rules = pd.concat(pytheas_data_cell_rules_DATA)
        pytheas_not_data_cell_rules = pd.concat(pytheas_not_data_cell_rules_DATA)
        end_reduce = timer()
        print(f'\n-reduced in {timedelta(seconds=end_reduce - start_reduce)}')

        con = connect(dbname=db_cred.database,
                      user=db_cred.user,
                      host='localhost',
                      password=db_cred.password,
                      port=db_cred.port)
        cur = con.cursor()

        execute_values(cur, """INSERT INTO pat_line_and_cell_rules (
                        crawl_datafile_key, data_rules_fired, not_data_rules_fired)
                        VALUES %s""", pytheas_line_and_cell_rules_DATA
                      )
        con.commit()
        cur.close()
        con.close()

        pytheas_line_datapoints.head(0).to_sql('pat_line_datapoints', self.opendata_engine, if_exists='replace', index=False) #truncates the table
        conn = self.opendata_engine.raw_connection()
        cur = conn.cursor()
        output = io.StringIO()
        pytheas_line_datapoints.to_csv(output, sep='\t', header=False, index=False)
        output.seek(0)
        cur.copy_from(output, 'pat_line_datapoints') # , null="" null values become ''
        conn.commit()

        pytheas_cell_datapoints.head(0).to_sql('pat_cell_datapoints', self.opendata_engine, if_exists='replace', index=False) #truncates the table
        conn = self.opendata_engine.raw_connection()
        cur = conn.cursor()
        output = io.StringIO()
        pytheas_cell_datapoints.to_csv(output, sep='\t', header=False, index=False)
        output.seek(0)
        cur.copy_from(output, 'pat_cell_datapoints') # , null="" null values become ''
        conn.commit()

        pytheas_data_line_rules.head(0).to_sql('pat_data_line_rules', self.opendata_engine, if_exists='replace', index=False) #truncates the table
        conn = self.opendata_engine.raw_connection()
        cur = conn.cursor()
        output = io.StringIO()
        pytheas_data_line_rules.to_csv(output, sep='\t', header=False, index=False)
        output.seek(0)
        cur.copy_from(output, 'pat_data_line_rules') # , null="" null values become ''
        conn.commit()

        pytheas_not_data_line_rules.head(0).to_sql('pat_not_data_line_rules', self.opendata_engine, if_exists='replace', index=False) #truncates the table
        conn = self.opendata_engine.raw_connection()
        cur = conn.cursor()
        output = io.StringIO()
        pytheas_not_data_line_rules.to_csv(output, sep='\t', header=False, index=False)
        output.seek(0)
        cur.copy_from(output, 'pat_not_data_line_rules') # , null="" null values become ''
        conn.commit()

        pytheas_data_cell_rules.head(0).to_sql('pat_data_cell_rules', self.opendata_engine, if_exists='replace', index=False) #truncates the table
        conn = self.opendata_engine.raw_connection()
        cur = conn.cursor()
        output = io.StringIO()
        pytheas_data_cell_rules.to_csv(output, sep='\t', header=False, index=False)
        output.seek(0)
        cur.copy_from(output, 'pat_data_cell_rules') # , null="" null values become ''
        conn.commit()

        pytheas_not_data_cell_rules.head(0).to_sql('pat_not_data_cell_rules', self.opendata_engine, if_exists='replace', index=False) #truncates the table
        conn = self.opendata_engine.raw_connection()
        cur = conn.cursor()
        output = io.StringIO()
        pytheas_not_data_cell_rules.to_csv(output, sep='\t', header=False, index=False)
        output.seek(0)
        cur.copy_from(output, 'pat_not_data_cell_rules') # , null="" null values become ''
        conn.commit()

        con = connect(dbname=db_cred.database,
                      user=db_cred.user,
                      host='localhost',
                      password=db_cred.password,
                      port=db_cred.port)
        cur = con.cursor()
        cur.execute("""CREATE INDEX IF NOT EXISTS pat_cell_datapoints_idx on pat_cell_datapoints(crawl_datafile_key)""")
        cur.execute("""CREATE INDEX IF NOT EXISTS pat_line_datapoints_idx on pat_line_datapoints(crawl_datafile_key)""")

        cur.execute("""CREATE INDEX IF NOT EXISTS pat_data_cell_rules_idx on pat_data_cell_rules(crawl_datafile_key)""")
        cur.execute("""CREATE INDEX IF NOT EXISTS pat_not_data_cell_rules_idx on pat_not_data_cell_rules(crawl_datafile_key)""")

        cur.execute("""CREATE INDEX IF NOT EXISTS pat_data_line_rules_idx on pat_data_line_rules(crawl_datafile_key)""")
        cur.execute("""CREATE INDEX IF NOT EXISTS pat_not_data_line_rules_idx on pat_not_data_line_rules(crawl_datafile_key)""")

        con.commit()

        cur.close()
        con.close()


    def predict_subheaders_new(self, csv_file, cand_data, predicted_pytheas_sub_headers, pytheas_blank_lines, pytheas_headers):
        args = self.parameters
        fuzzy_rules = self.fuzzy_rules
        ignore_rules = self.ignore_rules

        cand_subhead_indexes = list(set(predicted_pytheas_sub_headers + list(cand_data.index[cand_data.iloc[:, 1:].isnull().all(1)])))
        candidate_subheaders = {}
        subheader_scope = {}
        certain_data_indexes = list(cand_data.index)
        aggregation_rows = {}
        first_column_data_values = []

        for row in csv_file.loc[certain_data_indexes].itertuples():
            first_value = str(row[1]).strip()
            for aggregation_phrase in pytheas_util.aggregation_functions:
                agg_index = first_value.lower().find(aggregation_phrase[0])

                if agg_index > -1 and contains_number(row[1:]):
                    aggregation_rows[row.Index] = {}
                    aggregation_rows[row.Index]['value'] = first_value
                    aggregation_rows[row.Index]['aggregation_function'] = aggregation_phrase[1]
                    aggregation_rows[row.Index]['aggregation_phrase'] = aggregation_phrase[0]
                    aggregation_rows[row.Index]['aggregation_label'] = first_value[:agg_index] + first_value[agg_index + len(aggregation_phrase[0]):]
                    break

            if row.Index not in aggregation_rows.keys() and first_value.lower() not in null_equivalent_values and row.Index not in cand_subhead_indexes:
                first_column_data_values.append(first_value)

        certain_data_indexes = list(set(certain_data_indexes) - set(aggregation_rows.keys()) - set(cand_subhead_indexes))

        for row in csv_file.loc[cand_subhead_indexes].itertuples():
            first_value = str(row[1]).strip()
            if first_value.lower() not in  ['nan', 'none', ''] and row.Index not in aggregation_rows.keys():
                candidate_subheaders[row.Index] = first_value

        cand_subhead_indexes = list(candidate_subheaders.keys())

        aggregation_rows, certain_data_indexes, predicted_pytheas_sub_headers, cand_subhead_indexes, subheader_scope = discover_aggregation_scope(csv_file, aggregation_rows, candidate_subheaders, predicted_pytheas_sub_headers, certain_data_indexes, pytheas_headers)

        if cand_subhead_indexes is not None and len(cand_subhead_indexes) > 0:
            first_column_value_patterns = []
            first_column_value_symbols = []
            first_column_value_cases = []
            first_column_value_token_lengths = []
            first_column_value_char_lengths = []
            first_column_value_tokens = []
            first_column_all_patterns_numeric = []

            for value in first_column_data_values:
                pattern, symbols, case, value_num_tokens, value_num_chars = pytheas_util.generate_pattern_symbols_and_case(str(value).strip(), args.outlier_sensitive)
                first_column_value_patterns.append(pattern)
                first_column_value_symbols.append(symbols)
                first_column_value_cases.append(case)
                first_column_value_token_lengths.append(value_num_tokens)
                first_column_value_char_lengths.append(value_num_chars)
                first_column_value_tokens.append([i for i in (str(value).strip().lower()).split() if i not in stop and all(j.isalpha() or j in string.punctuation for j in i) and i not in null_equivalent_values])
                first_column_all_patterns_numeric.append(eval_numeric_pattern(pattern))

            if args.normalize_decimals:
                first_column_value_patterns, first_column_value_symbols = pytheas_util.normalize_decimals_numbers(first_column_value_patterns, first_column_value_symbols)

            value_pattern_summary, value_chain_consistent = pytheas_util.generate_pattern_summary(first_column_value_patterns)
            summary_strength = sum(1 for x in first_column_value_patterns if len(x) > 0)
            bw_patterns = [list(reversed(pattern)) for pattern in first_column_value_patterns]
            value_pattern_BW_summary, _ = pytheas_util.generate_pattern_summary(bw_patterns)
            all_patterns_numeric, _ = pytheas_util.generate_all_numeric_sig_pattern(first_column_all_patterns_numeric,
                                                                                    [len(t) for t in first_column_value_patterns])
            value_symbol_summary = pytheas_util.generate_symbol_summary(first_column_value_symbols)
            case_summary = pytheas_util.generate_case_summary(first_column_value_cases)
            length_summary = pytheas_util.generate_length_summary(first_column_value_char_lengths)

            if len(first_column_value_tokens) > 0:
                candidate_tokens = {[t for t in first_column_value_tokens[0] if any(c.isalpha() for c in t)]}
            else:
                candidate_tokens = set()
            if len(first_column_data_values) > 2:
                candidate_count_of_value = np.count_nonzero(first_column_data_values[2:min(args.max_summary_strength, len(first_column_data_values))] == str(value).strip())
            else:
                candidate_count_of_value = 0

            partof_multiword_value_repeats = dict()
            for part in candidate_tokens:
                partof_multiword_value_repeats[part] = 0
                for value_tokens in first_column_value_tokens:
                    if part in value_tokens:
                        partof_multiword_value_repeats[part] += 1
            consistent_symbol_sets, _ = is_consistent_symbol_sets(first_column_value_symbols)
            data_rules_fired = {}
            data_rules_fired[1] = {}
            data_rules_fired[1][0] = {}
            data_rules_fired[1][0]['agreements'] = []
            data_rules_fired[1][0]['null_equivalent'] = False
            for rule in fuzzy_rules['cell']['data'].keys():
                rule_fired = False

                # Don't bother looking for agreements if there are no patterns
                non_empty_patterns = 0
                if rule not in ignore_rules['cell']['data']  and len(first_column_value_patterns) > 0:
                    for pattern in first_column_value_patterns:
                        if pattern != []:
                            non_empty_patterns += 1

                    #there is no point calculating agreement over one value, a single value always agrees with itself.
                    #in addition, many tables have bilingual headers, so agreement between two header values is very common, require nij>=3
                    if len(first_column_value_patterns) >= 2 and non_empty_patterns >= 2:
                        rule_fired = eval_data_cell_rule(rule, first_column_data_values,
                                                         value_pattern_summary, value_chain_consistent,
                                                         value_pattern_BW_summary, value_symbol_summary,
                                                         case_summary, candidate_count_of_value,
                                                         partof_multiword_value_repeats,
                                                         candidate_tokens, consistent_symbol_sets,
                                                         all_patterns_numeric)
                        if rule_fired and "_REPEATS_" not in rule:
                            data_rules_fired[1][0]['agreements'].append(rule)


            for row in csv_file.loc[cand_subhead_indexes].itertuples():
                first_value = str(row[1]).strip()
                if first_value.lower() in ['', 'nan', 'none', 'null']:
                    continue
                if first_value in first_column_data_values:
                    continue
                if row.Index - 1 in pytheas_blank_lines or row.Index - 1 in pytheas_headers:
                    predicted_pytheas_sub_headers.append(row.Index)
                else:

                    value_tokens = first_value.lower().split()
                    pattern, symbols, case, value_num_tokens, value_num_chars = pytheas_util.generate_pattern_symbols_and_case(str(first_value).strip(), args.outlier_sensitive)
                    if args.normalize_decimals:
                        column_patterns, column_symbols = pytheas_util.normalize_decimals_numbers([pattern] + first_column_value_patterns, [symbols] + first_column_value_symbols)
                    value_pattern_summary, value_chain_consistent = pytheas_util.generate_pattern_summary(column_patterns)

                    summary_strength = sum(1 for x in column_patterns if len(x) > 0)
                    bw_patterns = [list(reversed(pattern)) for pattern in column_patterns]
                    value_pattern_BW_summary, _ = pytheas_util.generate_pattern_summary(bw_patterns)
                    value_symbol_summary = pytheas_util.generate_symbol_summary(column_symbols)
                    case_summary = pytheas_util.generate_case_summary([case] + first_column_value_cases)
                    length_summary = pytheas_util.generate_length_summary([value_num_chars] + first_column_value_char_lengths)
                    all_patterns_numeric, _ = pytheas_util.generate_all_numeric_sig_pattern([eval_numeric_pattern(pattern)] + first_column_all_patterns_numeric,
                                                                                            [len(t) for t in column_patterns])
                    column_values = [first_value] + first_column_data_values
                    column_tokens = [value_tokens] + first_column_value_tokens

                    candidate_tokens = {t  for t in value_tokens if any(c.isalpha() for c in t)}
                    if len(column_values) > 2:
                        candidate_count_of_value = np.count_nonzero(column_values[2:min(args.max_summary_strength, len(column_values))] == str(value).strip())
                    else:
                        candidate_count_of_value = 0
                    partof_multiword_value_repeats = dict()
                    for part in candidate_tokens:
                        partof_multiword_value_repeats[part] = 0
                        for value_tokens in column_tokens:
                            if part in value_tokens:
                                partof_multiword_value_repeats[part] += 1

                    consistent_symbol_sets, _ = is_consistent_symbol_sets(column_symbols)
                    data_rules_fired[0] = {}
                    data_rules_fired[0][0] = {}
                    data_rules_fired[0][0]['agreements'] = []
                    data_rules_fired[0][0]['null_equivalent'] = False
                    for rule in fuzzy_rules['cell']['data'].keys():
                        rule_fired = False
                        non_empty_patterns = 0
                        if rule not in ignore_rules['cell']['data'] and len(column_patterns) > 0 and first_value.lower() not in null_equivalent_values:
                            for pattern in column_patterns:
                                if pattern != []:
                                    non_empty_patterns += 1
                            if len(column_patterns) >= 2 and non_empty_patterns >= 2:
                                rule_fired = eval_data_cell_rule(rule, column_values,
                                                                 value_pattern_summary, value_chain_consistent,
                                                                 value_pattern_BW_summary, value_symbol_summary,
                                                                 case_summary, candidate_count_of_value,
                                                                 partof_multiword_value_repeats,
                                                                 candidate_tokens, consistent_symbol_sets,
                                                                 all_patterns_numeric,)
                                if rule_fired and "_REPEATS_" not in rule:
                                    data_rules_fired[0][0]['agreements'].append(rule)

                    value_disagreements = []
                    disagreement_summary_strength = summary_strength - 1
                    if len(pattern) > 0:
                        for rule in fuzzy_rules['cell']['not_data'].keys():
                            rule_fired = False
                            if rule not in ignore_rules['cell']['not_data'] and disagreement_summary_strength > 0 and (not all_numbers(column_symbols) or not is_number(symbols)):
                                rule_fired = eval_not_data_cell_rule(rule, value_pattern_summary, value_pattern_BW_summary, value_chain_consistent, value_symbol_summary, case_summary, length_summary, pattern, case, value_num_chars, disagreement_summary_strength, data_rules_fired, 0, 0)
                                if rule_fired and "_REPEATS_" not in rule:
                                    value_disagreements.append(rule)

                    #######################################################################v######
                    #  DATA value classification
                    data_score = max_score(data_rules_fired[0][0]['agreements'], fuzzy_rules['cell']['data'], args.weight_lower_bound)
                    POPULATION_WEIGHT = 1 - (1 - args.p)**(2 * summary_strength)
                    if data_score is not None:
                        if args.summary_population_factor:
                            cell_data_score = data_score * POPULATION_WEIGHT
                        else:
                            cell_data_score = data_score

                    #######################################################################v######
                    #  NOT DATA value classification
                    not_data_score = max_score(value_disagreements, fuzzy_rules['cell']['not_data'], args.not_data_weight_lower_bound)
                    POPULATION_WEIGHT = 1 - (1 - args.p)**(2 * disagreement_summary_strength)
                    if not_data_score is not None:
                        if args.summary_population_factor:
                            cell_not_data_score = not_data_score * POPULATION_WEIGHT
                        else:
                            cell_not_data_score = not_data_score

                    if  cell_data_score > cell_not_data_score:# candidate subheader is definitely data, move along
                        continue

                    if (row.Index - 1 in predicted_pytheas_sub_headers and row.Index - 2 in predicted_pytheas_sub_headers):
                        continue

                    if row.Index != cand_data.index[-1]:
                        predicted_pytheas_sub_headers.append(row.Index)

        for s_i, subheader in enumerate(predicted_pytheas_sub_headers):
            if subheader not in subheader_scope.keys():
                if s_i + 1 == len(predicted_pytheas_sub_headers):
                    subheader_scope[subheader] = list(range(subheader + 1, cand_data.index[-1] + 1))
                else:
                    next_s_i = s_i + 1
                    while next_s_i < len(predicted_pytheas_sub_headers):
                        next_subh = predicted_pytheas_sub_headers[next_s_i]
                        if next_subh not in subheader_scope:
                            subheader_scope[subheader] = list(range(subheader + 1, next_subh))
                            break
                        next_s_i += 1

        return  aggregation_rows, subheader_scope


def pytheas_rule_worker(task):
    start = timer()
    top_level_dir, _, file_object, pytheas_classifier, assume_multi_tables, _ = task
    max_attributes = pytheas_classifier.parameters.max_attributes
    crawl_datafile_key = file_object[0]
    annotations = file_object[2]
    filepath = file_object[3]

    filepath = os.path.join(top_level_dir, filepath)
    file_dataframe = file_utilities.get_dataframe(filepath, 100)
    lines_in_file = len(file_dataframe)

    bottom_boundary = file_dataframe.shape[0] - 1 #initialize
    if not assume_multi_tables:
        if 'tables' in annotations.keys():
            for table in annotations['tables']:
                if 'data_start' in table.keys():
                    bottom_boundary = table['bottom_boundary']
                break

    file_dataframe = file_dataframe.loc[:bottom_boundary]
    file_dataframe_trimmed = file_dataframe.copy()

    if pytheas_classifier.parameters.max_attributes is not None:
        max_attributes = pytheas_classifier.parameters.max_attributes
        if pytheas_classifier.parameters.ignore_left is not None:
            max_attributes = pytheas_classifier.parameters.max_attributes + pytheas_classifier.parameters.ignore_left
        slice_idx = min(max_attributes, file_dataframe.shape[1]) + 1
        file_dataframe_trimmed = file_dataframe.iloc[:, :slice_idx]

    signatures = TableSignatures(file_dataframe_trimmed, pytheas_classifier.parameters.outlier_sensitive)

    data_rules_fired, not_data_rules_fired = collect_dataframe_rules(file_dataframe_trimmed, pytheas_classifier, signatures)
    pytheas_line_datapoints, pytheas_cell_datapoints, pytheas_data_line_rules, pytheas_not_data_line_rules, pytheas_data_cell_rules, pytheas_not_data_cell_rules, lines_in_sample = save_training_data(
        crawl_datafile_key, file_dataframe_trimmed, annotations, data_rules_fired, not_data_rules_fired, pytheas_classifier)

    end = timer()
    processing_time = end - start
    return crawl_datafile_key, pytheas_line_datapoints, pytheas_cell_datapoints, pytheas_data_line_rules, pytheas_not_data_line_rules, pytheas_data_cell_rules, pytheas_not_data_cell_rules, data_rules_fired, not_data_rules_fired, lines_in_file, lines_in_sample, processing_time


def merge_tables(table_head, table_tail):
    # update the previous table
    table_head["data_end"] = table_tail["data_end"]
    table_head["subheader_scope"] = {**table_head["subheader_scope"], **table_tail["subheader_scope"]}
    for subheader in list(set(range(table_tail["top_boundary"], table_tail["data_start"])) - set(table_tail["header"])):
        table_head["subheader_scope"][subheader] = dict()
    table_head["aggregation_scope"] = {**table_head["aggregation_scope"], **table_tail["aggregation_scope"]}
    return table_head

def get_class_confidences(file_dataframe_trimmed, data_rules_fired, not_data_rules_fired, fuzzy_rules, parameters):
    data_line_confidences = dict()
    not_data_line_confidences = dict()
    label_confidences = dict()

    column_indexes = file_dataframe_trimmed.columns
    before_data = True
    for row_index in file_dataframe_trimmed.index:
        label_confidences[row_index] = dict()
        candidate_row_agreements = list()
        candidate_row_disagreements = list()

        for column_index in column_indexes:
            #############################################################################
            #  DATA value classification
            value_agreements = data_rules_fired[row_index][column_index]['agreements']
            summary_strength = data_rules_fired[row_index][column_index]['summary_strength']

            # if there are no lines below me to check agreement,
            # and line before me exists and was data
            # see impute agreements
            if (row_index in data_rules_fired.keys() and data_rules_fired[row_index][column_index]['null_equivalent'] or data_rules_fired[row_index][column_index]['summary_strength'] == 1) and parameters.impute_nulls and row_index - 1 in data_rules_fired.keys() and column_index in data_rules_fired[row_index - 1].keys() and row_index - 1 in data_line_confidences.keys() and data_line_confidences[row_index - 1] > not_data_line_confidences[row_index - 1]:
                value_agreements = data_rules_fired[row_index-1][column_index]['agreements']
                summary_strength = data_rules_fired[row_index-1][column_index]['summary_strength']
            if row_index in data_rules_fired.keys() and data_rules_fired[row_index][column_index]['summary_strength'] == 0 and data_rules_fired[row_index][column_index]['aggregate'] and row_index - 2 in data_rules_fired.keys() and column_index in data_rules_fired[row_index - 2].keys() and row_index - 2 in data_line_confidences.keys() and data_line_confidences[row_index - 2] > not_data_line_confidences[row_index - 2]:
                value_agreements = data_rules_fired[row_index-2][column_index]['agreements']
                summary_strength = data_rules_fired[row_index-2][column_index]['summary_strength']

            # otherwise, nothing was wrong, i can use my own agreements as initialized
            data_score = max_score(value_agreements, fuzzy_rules["cell"]["data"], parameters.weight_lower_bound)
            POPULATION_WEIGHT = 1 - (1 - parameters.p)**(2 * summary_strength)
            if data_score is not None:
                if parameters.summary_population_factor:
                    candidate_row_agreements.append(data_score * POPULATION_WEIGHT)
                else:
                    candidate_row_agreements.append(data_score)

            #######################################################################v######
            #  NOT DATA value classification
            value_disagreements = not_data_rules_fired[row_index][column_index]['disagreements']
            disagreement_summary_strength = not_data_rules_fired[row_index][column_index]['disagreement_summary_strength']
            not_data_score = max_score(value_disagreements, fuzzy_rules["cell"]["not_data"], parameters.not_data_weight_lower_bound)
            POPULATION_WEIGHT = 1 - (1 - parameters.p)**(2 * disagreement_summary_strength)

            if not_data_score is not None:
                if parameters.summary_population_factor:
                    candidate_row_disagreements.append(not_data_score * POPULATION_WEIGHT)
                else:
                    candidate_row_disagreements.append(not_data_score)

            ########################################################################

        #################################################################################
        # NOT DATA line weights
        line_not_data_evidence = candidate_row_disagreements.copy()
        if parameters.weight_input == 'values_and_lines':
            if row_index - 1 in data_line_confidences.keys() and data_line_confidences[row_index - 1] > not_data_line_confidences[row_index - 1]:
                before_data = False
            if file_dataframe_trimmed.shape[1] > 1:
                not_data_line_rules_fired = not_data_rules_fired[row_index]['line']
                for event in not_data_line_rules_fired:
                    if event == "UP_TO_FIRST_COLUMN_COMPLETE_CONSISTENTLY" and not before_data:
                        continue
                    if event.startswith("ADJACENT_ARITHMETIC_SEQUENCE") and fuzzy_rules["line"]["not_data"][event]["weight"] is None:
                        steps = event[-1]
                        if steps.isdigit() and int(steps) in range(2, 6):
                            event = event[:-1] + str(int(steps) + 1)
                    if event in fuzzy_rules["line"]["not_data"].keys() and fuzzy_rules["line"]["not_data"][event]["weight"] is not None and fuzzy_rules["line"]["not_data"][event]["weight"] > parameters.not_data_weight_lower_bound:
                        line_not_data_evidence.append(fuzzy_rules["line"]["not_data"][event]["weight"])

        # DATA line weights
        line_is_data_evidence = candidate_row_agreements.copy()
        if parameters.weight_input == 'values_and_lines':
            line_is_data_events = data_rules_fired[row_index]['line']
            for rule in line_is_data_events:
                if fuzzy_rules["line"]["data"][rule]["weight"] is not None and fuzzy_rules["line"]["data"][rule]["weight"] > parameters.weight_lower_bound:
                    line_is_data_evidence.append(fuzzy_rules["line"]["data"][rule]["weight"])

        # calculate confidence that this row is data
        data_conf = probabilistic_sum(line_is_data_evidence)
        data_line_confidences[row_index] = data_conf

        # calculate confidence that this row is not data
        not_data_conf = probabilistic_sum(line_not_data_evidence)
        not_data_line_confidences[row_index] = not_data_conf

        label_confidences[row_index]['DATA'] = data_conf
        label_confidences[row_index]['NOT-DATA'] = not_data_conf

    return data_line_confidences, not_data_line_confidences


def predict_last_data_line_top_down(dataframe, predicted_fdl, data_confidence, not_data_confidence, model,
                                    subheader_scope, aggregation_rows, blank_lines, headers_discovered, signatures,
                                    downwards_data_rules_fired,
                                    downwards_not_data_rules_fired):
    args = model.parameters
    fuzzy_rules = model.fuzzy_rules

    predicted_pytheas_sub_headers = list(subheader_scope.keys())
    certain_data = []
    certain_data_widths = []
    data_predictions = dict()

    data_rules_fired = {}
    not_data_rules_fired = {}
    predicted_ldl = predicted_fdl

    data = pd.DataFrame()
    candidate_data = dataframe.loc[predicted_fdl:]
    probation = []

    if args.max_attributes is not None:
        max_attributes = args.max_attributes
        if args.ignore_left is not None:
            max_attributes = args.max_attributes+args.ignore_left
        slice_idx = min(max_attributes, candidate_data.shape[1]) + 1
        candidate_data = candidate_data.iloc[:, :slice_idx]

    line_counter = 0
    patterns = Patterns()
    for line_label, line in candidate_data.iterrows():
        line_counter += 1
        row_values = [str(elem) if elem is not None else elem for elem in line.tolist()]
        first_value = row_values[0].lower()
        if first_value.startswith('"') and first_value.endswith('"'):
            first_value = first_value[1:-1]

        IS_DATA = False
        FOOTNOTE_FOUND = False
        data_conf = 0
        not_data_conf = 0

        if line_label not in blank_lines:
            if line_label in certain_data:
                IS_DATA = True
                data_conf = 1

            if line_label in predicted_pytheas_sub_headers:
                for footnote_keyword in pytheas_util.footnote_keywords:
                    if first_value.startswith(footnote_keyword):
                        FOOTNOTE_FOUND = True
                        not_data_conf = 1
                        break
                if len(first_value) > 5 and (((first_value[0] == '1' or first_value[0] == 'a') and first_value[1] in [' ', '.', '/', ')', ']', ':']) or (first_value[0] == '(' and (first_value[1].isdigit() or first_value[1] == 'a') and first_value[2] == ')')):
                    FOOTNOTE_FOUND = True
                    not_data_conf = 1

                if not FOOTNOTE_FOUND:
                    not_data_conf = 1
                    prediction, _ = predict_line_label(data_conf, not_data_conf)
                    data_predictions[line_label] = prediction
                    continue

            if len(row_values) > 0 and row_values[0].lower() not in ['', 'none', 'nan'] and ((len(row_values) > 1 and len([i for i in row_values[1:] if i.lower() not in  ['', 'none', 'nan']]) == 0) or (len(row_values) > 2 and len([i for i in row_values[2:] if i.lower() not in  ['', 'none', 'nan']]) == 0)):
                IS_DATA = True
                for footnote_keyword in pytheas_util.footnote_keywords:
                    if first_value.startswith(footnote_keyword):
                        FOOTNOTE_FOUND = True
                        break

                if '=' in first_value:
                    FOOTNOTE_FOUND = True

                if len(first_value) > 5 and ((first_value[0] == '1' or first_value[0] == 'a') and first_value[1] in [' ', '.', '/', ')', ']', ':']):
                    FOOTNOTE_FOUND = True

                if len(first_value) > 5 and (first_value[0] == '('  and (first_value[1] == '1' or first_value[1] == 'a')  and first_value[2] == ')'):
                    FOOTNOTE_FOUND = True

                if FOOTNOTE_FOUND:
                    IS_DATA = False
                    not_data_conf = 1
                    prediction, _ = predict_line_label(data_conf, not_data_conf)
                    data_predictions[line_label] = prediction
                    break

            # for the first 3 lines rely on the classification from first data line search.
            if line_counter <= 3 or line_label in aggregation_rows:
                data = pd.DataFrame([line],
                                    index=[line_label]).append(data)
                certain_data_widths.append(non_empty_values(line))
                predicted_ldl = line_label
                data_conf = 1
                prediction, _ = predict_line_label(data_conf, not_data_conf)
                data_predictions[line_label] = prediction
                continue
            else:
                if line.isnull().values.all():
                    prediction, _ = predict_line_label(data_conf, not_data_conf)
                    data_predictions[line_label] = prediction
                    continue

                if ','.join(line.apply(str).tolist()) in set(headers_discovered.values()):
                    IS_DATA = False
                else:
                    data_rules_fired, not_data_rules_fired, patterns = collect_line_rules(line,
                                                                                          predicted_fdl,
                                                                                          line_label,
                                                                                          data,
                                                                                          signatures,
                                                                                          model,
                                                                                          data_rules_fired,
                                                                                          not_data_rules_fired,
                                                                                          patterns)
                    candidate_row_agreements = []
                    candidate_row_disagreements = []
                    for column in candidate_data:
                        #################################################v######################v######
                        #  DATA value classification
                        value_agreements = data_rules_fired[line_label][column]['agreements']
                        summary_strength = data_rules_fired[line_label][column]['summary_strength']

                        data_score = max_score(value_agreements, fuzzy_rules["cell"]["data"], args.weight_lower_bound)
                        POPULATION_WEIGHT = 1 - (1 - args.p)**(2 * summary_strength)
                        if data_score is not None:
                            if args.summary_population_factor:
                                candidate_row_agreements.append(data_score * POPULATION_WEIGHT)
                            else:
                                candidate_row_agreements.append(data_score)
                        #######################################################################v######
                        #  NOT DATA value classification
                        value_disagreements = not_data_rules_fired[line_label][column]['disagreements']
                        disagreement_summary_strength = not_data_rules_fired[line_label][column]['disagreement_summary_strength']
                        not_data_score = max_score(value_disagreements, fuzzy_rules["cell"]["not_data"], args.not_data_weight_lower_bound)
                        POPULATION_WEIGHT = 1 - (1 - args.p)**(2 * disagreement_summary_strength)

                        if not_data_score is not None:
                            if args.summary_population_factor:
                                candidate_row_disagreements.append(not_data_score * POPULATION_WEIGHT)
                            else:
                                candidate_row_disagreements.append(not_data_score)
                    #################################################################################
                    # NOT DATA line weights
                    line_not_data_evidence = candidate_row_disagreements.copy()
                    if args.weight_input == 'values_and_lines':
                        if candidate_data.shape[1] > 1:
                            not_data_line_rules_fired = downwards_not_data_rules_fired[line_label]['line']
                            for event in not_data_line_rules_fired:
                                if event == "UP_TO_FIRST_COLUMN_COMPLETE_CONSISTENTLY":
                                    continue
                                if event.startswith("ADJACENT_ARITHMETIC_SEQUENCE") and fuzzy_rules["line"]["not_data"][event]["weight"] is None:
                                    steps = event[-1]
                                    if steps.isdigit() and int(steps) in range(2, 6):
                                        event = event[:-1]+ str(int(steps) + 1)
                                if event in fuzzy_rules["line"]["not_data"].keys() and fuzzy_rules["line"]["not_data"][event]["weight"] is not None and fuzzy_rules["line"]["not_data"][event]["weight"] > args.not_data_weight_lower_bound:
                                    line_not_data_evidence.append(fuzzy_rules["line"]["not_data"][event]["weight"])

                    not_data_conf = probabilistic_sum(line_not_data_evidence)

                    # DATA line weights
                    line_is_data_evidence = candidate_row_agreements.copy()
                    if args.weight_input == 'values_and_lines':
                        line_is_data_events = downwards_data_rules_fired[line_label]['line']
                        for rule in line_is_data_events:
                            if fuzzy_rules["line"]["data"][rule]["weight"] is not None and fuzzy_rules["line"]["data"][rule]["weight"] > args.weight_lower_bound:
                                line_is_data_evidence.append(fuzzy_rules["line"]["data"][rule]["weight"])
                    # calculate confidence that this row is data
                    data_conf = probabilistic_sum(line_is_data_evidence)

                    if (data_conf > 0 and data_conf >= not_data_conf):
                        IS_DATA = True
                    elif len(certain_data_widths) > 0 and non_empty_values(line) == max(certain_data_widths) and line_label - 1 in data.index:
                        IS_DATA = True
                    else:
                        prediction, _ = predict_line_label(data_confidence[line_label], not_data_confidence[line_label])
                        if line_label - 1 not in probation and line_label - 1 not in blank_lines and data_conf < not_data_conf and prediction['label'] != 'DATA':
                            probation.append(line_label)
                        elif (data_conf > 0 and data_conf >= not_data_conf) or (line_label - 1 not in probation and prediction['label'] == 'DATA'):
                            IS_DATA = True

            #--- end if blanklines
            if line_label in probation:
                prediction, _ = predict_line_label(data_conf, not_data_conf)
                data_predictions[line_label] = prediction
                continue

            if IS_DATA:
                predicted_ldl = line_label
                data = pd.DataFrame([line], index=[line_label]).append(data)
                certain_data_widths.append(non_empty_values(line))

            else:
                break
        else:
            not_data_conf = 1
            break

        prediction, _ = predict_line_label(data_conf, not_data_conf)
        data_predictions[line_label] = prediction

    bottom_boundary_confidence = last_data_line_confidence(data_predictions, predicted_ldl)
    return predicted_ldl, bottom_boundary_confidence


def collect_dataframe_rules(csv_file, model, signatures):
    args = model.parameters
    fuzzy_rules = model.fuzzy_rules
    ignore_rules = model.ignore_rules

    dataframe_labels = []
    for column in csv_file:
        dataframe_labels.append(column)

    data_rules_fired = {}
    not_data_rules_fired = {}
    row_counter = -1

    for row in csv_file.itertuples():
        line_index = row.Index
        if len(row) > 1:
            row = row[1:]
        else:
            row = []

        row_values = [str(elem) if elem is not None else elem for elem in row]
        null_equivalent_fired, _ = line_has_null_equivalent(row_values)

        data_rules_fired[line_index] = {}
        row_counter += 1
        all_summaries_empty = True #initialize

        n_lines = len(signatures.all_normalized_values)
        patterns = Patterns()
        for column_index, column in enumerate(csv_file.columns):
            data_rules_fired[line_index][column_index] = {}
            data_rules_fired[line_index][column_index]["agreements"] = []

            candidate_value = signatures.all_normalized_values[line_index, column_index]
            value_lower = candidate_value.lower()
            value_tokens = value_lower.split()

            is_aggregate = (len(value_tokens) > 0 and not set(value_tokens).isdisjoint(pytheas_util.aggregation_tokens))
            is_null_equivalent = (candidate_value.strip().lower() in null_equivalent_values)

            data_rules_fired[line_index][column_index]['null_equivalent'] = is_null_equivalent
            data_rules_fired[line_index][column_index]['aggregate'] = is_aggregate

            column_train_sigs = None
            column_symbols = None
            column_cases = None
            column_lengths = None
            column_tokens = None

            # we need a context window with up to args.max_summary_strength non empty values to generate a context pattern
            if args.max_summary_strength is not None:
                nonempty_patterns = 0
                nonempty_patterns_idx = 0
                for nonempty_patterns_idx in range(0, min(n_lines-line_index, args.max_line_depth)):
                    if len(signatures.all_column_train[line_index + nonempty_patterns_idx, column_index]) > 0:
                        nonempty_patterns += 1
                        if nonempty_patterns == args.max_summary_strength:
                            column_train_sigs = signatures.all_column_train[line_index:line_index + nonempty_patterns_idx + 1, column_index].tolist()
                            column_bw_train_sigs = signatures.all_column_bw_train[line_index:line_index + nonempty_patterns_idx + 1, column_index].tolist()
                            column_symbols = signatures.all_column_symbols[line_index:line_index + nonempty_patterns_idx + 1, column_index]
                            column_cases = signatures.all_column_cases[line_index:line_index + nonempty_patterns_idx + 1, column_index]
                            column_lengths = signatures.all_column_character_lengths[line_index:line_index + nonempty_patterns_idx + 1, column_index]
                            column_tokens = signatures.all_column_tokens[line_index:line_index + nonempty_patterns_idx + 1, column_index]
                            column_values = signatures.all_normalized_values[line_index:line_index + nonempty_patterns_idx + 1, column_index]
                            column_is_numeric_train = signatures.all_column_is_numeric_train[line_index:line_index + nonempty_patterns_idx + 1, column_index]
                            break

            if column_train_sigs is None:
                column_train_sigs = signatures.all_column_train[line_index:, column_index].tolist()
                column_bw_train_sigs = signatures.all_column_bw_train[line_index:, column_index].tolist()
                column_symbols = signatures.all_column_symbols[line_index:, column_index]
                column_cases = signatures.all_column_cases[line_index:, column_index]
                column_lengths = signatures.all_column_character_lengths[line_index:, column_index]
                column_tokens = signatures.all_column_tokens[line_index:, column_index]
                column_values = signatures.all_normalized_values[line_index:, column_index]
                column_is_numeric_train = signatures.all_column_is_numeric_train[line_index:, column_index]

            candidate_tokens = {t for t in column_tokens[0] if any(c.isalpha() for c in t)}

            patterns.data_initialize(column_index, candidate_value, candidate_tokens,
                                     column_values, column_tokens, column_train_sigs, column_bw_train_sigs,
                                     column_symbols, column_cases, column_lengths, column_is_numeric_train,
                                     args.max_summary_strength)

            # patterns of a window INCLUDING the cell we are on
            data_patterns = patterns.data[column_index]
            value_pattern_summary, value_chain_consistent = data_patterns['train']
            value_pattern_BW_summary, _ = data_patterns['bw_train']
            value_symbol_summary = data_patterns['symbolset']
            case_summary = data_patterns['case']
            length_summary = data_patterns['character_length']
            summary_strength = data_patterns['summary_strength']
            candidate_count_for_value = data_patterns['candidate_count'][candidate_value]
            partof_multiword_value_repeats = data_patterns['partof_multiword_value_repeats']
            consistent_symbol_sets, _ = data_patterns['consistent_symbol_sets']
            train_sigs_all_numeric, _ = data_patterns['column_is_numeric']
            data_rules_fired[line_index][column_index]['summary_strength'] = summary_strength

            if null_equivalent_fired or len(value_pattern_summary) > 0 or len(value_pattern_BW_summary) > 0 or len(value_symbol_summary) > 0 or len(case_summary) > 0:
                all_summaries_empty = False

            for rule in fuzzy_rules["cell"]["data"].keys():
                rule_fired = False
                # Don't bother looking for agreements if there are no patterns or if the value on this line gives an empty pattern
                non_empty_patterns = 0
                # CHECK RULE
                if rule not in ignore_rules["cell"]["data"] and len(column_train_sigs) > 0 and value_lower not in null_equivalent_values:
                    for pattern in column_train_sigs:
                        if pattern != []:
                            non_empty_patterns += 1

                    #there is no point calculating agreement over one value, a single value always agrees with itself.

                    if (len(column_train_sigs) >= 2 and non_empty_patterns >= 2) or (len(csv_file.index) > 0 and line_index == csv_file.index[-1]):
                        assert len(column_values) > 0

                        rule_fired = eval_data_cell_rule(rule, column_values, value_pattern_summary,
                                                         value_chain_consistent, value_pattern_BW_summary,
                                                         value_symbol_summary,
                                                         case_summary, candidate_count_for_value, partof_multiword_value_repeats,
                                                         candidate_tokens, consistent_symbol_sets, train_sigs_all_numeric)
                if rule_fired:
                    data_rules_fired[line_index][column_index]["agreements"].append(rule)

        data_rules_fired[line_index]["all_summaries_empty"] = all_summaries_empty

    ##########################################################################################
    ##########################################################################################
    #################             EVALUATE NOT_DATA CELL RULES             ###################
    ##########################################################################################
    ##########################################################################################

    row_counter = -1
    for row in csv_file.itertuples():
        line_index = row.Index
        not_data_rules_fired[line_index] = {}
        if len(row) > 1:
            row = row[1:]
        else:
            row = []
        row_values = [str(elem) if elem is not None else elem for elem in row]

        row_counter += 1

        for columnindex, column in enumerate(csv_file.columns):

            not_data_rules_fired[line_index][columnindex] = {}
            not_data_rules_fired[line_index][columnindex]["disagreements"] = []
            candidate_value = signatures.all_normalized_values[line_index, columnindex]

            value_lower = candidate_value.lower()
            value_tokens = value_lower.split()

            column_train_sigs = None
            column_symbols = None
            column_cases = None
            column_lengths = None

            if args.max_summary_strength is not None:
                nonempty_patterns = 0
                nonempty_patterns_idx = 0
                for nonempty_patterns_idx in range(0, min(n_lines - (line_index + 1), args.max_line_depth)):
                    if len(signatures.all_column_train[line_index + 1 + nonempty_patterns_idx, columnindex]) > 0:
                        nonempty_patterns += 1
                        if nonempty_patterns == args.max_summary_strength:
                            column_train_sigs = signatures.train_normalized_numbers[line_index + 1:line_index + 1 + nonempty_patterns_idx + 1, columnindex].tolist()
                            column_bw_train_sigs = signatures.bw_train_normalized_numbers[line_index + 1:line_index + nonempty_patterns_idx + 1, columnindex].tolist()
                            column_symbols = signatures.symbolset_normalized_numbers[line_index + 1:line_index + 1 + nonempty_patterns_idx + 1, columnindex]
                            column_cases = signatures.all_column_cases[line_index + 1:line_index + 1 + nonempty_patterns_idx + 1, columnindex]
                            column_lengths = signatures.all_column_character_lengths[line_index + 1:line_index + 1 + nonempty_patterns_idx + 1, columnindex]
                            break

            if column_train_sigs is None:
                column_train_sigs = signatures.train_normalized_numbers[line_index + 1:, columnindex].tolist()
                column_bw_train_sigs = signatures.bw_train_normalized_numbers[line_index + 1:, columnindex].tolist()
                column_symbols = signatures.symbolset_normalized_numbers[line_index + 1:, columnindex]
                column_cases = signatures.all_column_cases[line_index + 1:, columnindex]
                column_lengths = signatures.all_column_character_lengths[line_index + 1:, columnindex]

            disagreement_summary_strength = sum(1 for x in column_train_sigs if len(x) > 0)
            not_data_rules_fired[line_index][column]['disagreement_summary_strength'] = disagreement_summary_strength

            cand_pattern = signatures.train_normalized_numbers[line_index, columnindex]
            cand_case = signatures.all_column_cases[line_index, columnindex]
            cand_num_chars = signatures.all_column_character_lengths[line_index, columnindex]

            value_pattern_summary, value_chain_consistent = pytheas_util.generate_pattern_summary(column_train_sigs)
            value_pattern_BW_summary, _ = pytheas_util.generate_pattern_summary(column_bw_train_sigs)
            value_symbol_summary = pytheas_util.generate_symbol_summary(column_symbols)
            case_summary = pytheas_util.generate_case_summary(column_cases)
            length_summary = pytheas_util.generate_length_summary(column_lengths)

            for rule in fuzzy_rules["cell"]["not_data"].keys():
                rule_fired = False
                if rule not in ignore_rules["cell"]["not_data"] and len(cand_pattern) > 0:
                    if disagreement_summary_strength > 0 and (not np.all(signatures.all_column_isnumber[line_index:, columnindex])):
                        rule_fired = eval_not_data_cell_rule(rule,
                                                             value_pattern_summary,
                                                             value_pattern_BW_summary,
                                                             value_chain_consistent,
                                                             value_symbol_summary,
                                                             case_summary,
                                                             length_summary,
                                                             cand_pattern,
                                                             cand_case,
                                                             cand_num_chars,
                                                             disagreement_summary_strength,
                                                             data_rules_fired,
                                                             columnindex,
                                                             line_index)
                        if rule_fired:
                            not_data_rules_fired[line_index][columnindex]["disagreements"].append(rule)
        #end processing column
        ########################################################################
        #### COLLECT LINE RULES ####

        #1. Collect data line rules fired
        line_is_data_events = assess_data_line(row_values)
        data_rules_fired[line_index]['line'] = []
        not_data_rules_fired[line_index]['line'] = []

        for rule in fuzzy_rules["line"]["data"].keys():
            rule_fired = False
            if rule not in ignore_rules["line"]["data"] and rule in line_is_data_events:
                rule_fired = True
                data_rules_fired[line_index]['line'].append(rule)

        #2.  Collect not_data line rules fired
        all_summaries_empty = data_rules_fired[line_index]["all_summaries_empty"]
        header_events_fired = collect_events_on_row(row_values)
        arithmetic_events_fired = collect_arithmetic_events_on_row(row_values)

        arithmetic_sequence_fired = False
        if len(arithmetic_events_fired) > 0:
            arithmetic_sequence_fired = True
        header_row_with_aggregation_tokens_fired = header_row_with_aggregation_tokens(row_values, arithmetic_sequence_fired)

        before_data = True
        not_data_line_rules_fired = []
        if csv_file.shape[1] > 1:
            not_data_line_rules_fired = assess_non_data_line(row_values, before_data, all_summaries_empty, line_index, csv_file)

        for rule in fuzzy_rules["line"]["not_data"].keys():
            rule_fired = False
            if rule not in ignore_rules["line"]["not_data"] and rule in not_data_line_rules_fired + header_events_fired + arithmetic_events_fired + header_row_with_aggregation_tokens_fired:
                rule_fired = True
                not_data_rules_fired[line_index]['line'].append(rule)

    return data_rules_fired, not_data_rules_fired


def max_score(events, unit_class_fuzzy_rules, weight_lower_bound):
    if len(events) > 0:
        event_score = []
        for event in events:
            if unit_class_fuzzy_rules[event]["weight"] is not None and unit_class_fuzzy_rules[event]["weight"] >= weight_lower_bound:
                event_score.append((event, unit_class_fuzzy_rules[event]["weight"]))
        if len(event_score) > 0:
            event_score.sort(key=lambda x: x[1], reverse=True)
            return event_score[0][1]
        else:
            return 0
    else:
        return 0


def available_cpu_count():
    """ Number of available virtual or physical CPUs on this system, i.e.
    user/real as output by time(1) when called with an optimally scaling
    userspace-only program"""

    # cpuset
    # cpuset may restrict the number of *available* processors
    try:
        m = re.search(r'(?m)^Cpus_allowed:\s*(.*)$',
                      open('/proc/self/status').read())
        if m:
            res = bin(int(m.group(1).replace(',', ''), 16)).count('1')
            if res > 0:
                return res
    except IOError:
        pass

    # Python 2.6+
    try:
        import multiprocessing
        return multiprocessing.cpu_count()
    except (ImportError, NotImplementedError):
        pass

    # https://github.com/giampaolo/psutil
    try:
        import psutil
        return psutil.cpu_count()   # psutil.NUM_CPUS on old versions
    except (ImportError, AttributeError):
        pass

    # POSIX
    try:
        res = int(os.sysconf('SC_NPROCESSORS_ONLN'))

        if res > 0:
            return res
    except (AttributeError, ValueError):
        pass

    # Windows
    try:
        res = int(os.environ['NUMBER_OF_PROCESSORS'])

        if res > 0:
            return res
    except (KeyError, ValueError):
        pass

    # jython
    try:
        from java.lang import Runtime
        runtime = Runtime.getRuntime()
        res = runtime.availableProcessors()
        if res > 0:
            return res
    except ImportError:
        pass

    # BSD
    try:
        sysctl = subprocess.Popen(['sysctl', '-n', 'hw.ncpu'],
                                  stdout=subprocess.PIPE)
        scStdout = sysctl.communicate()[0]
        res = int(scStdout)

        if res > 0:
            return res
    except (OSError, ValueError):
        pass

    # Linux
    try:
        res = open('/proc/cpuinfo').read().count('processor\t:')

        if res > 0:
            return res
    except IOError:
        pass

    # Solaris
    try:
        pseudoDevices = os.listdir('/devices/pseudo/')
        res = 0
        for devs in pseudoDevices:
            if re.match(r'^cpuid@[0-9]+$', devs):
                res += 1

        if res > 0:
            return res
    except OSError:
        pass

    # Other UNIXes (heuristic)
    try:
        try:
            dmesg = open('/var/run/dmesg.boot').read()
        except IOError:
            dmesgProcess = subprocess.Popen(['dmesg'], stdout=subprocess.PIPE)
            dmesg = dmesgProcess.communicate()[0]

        res = 0
        while '\ncpu' + str(res) + ':' in dmesg:
            res += 1

        if res > 0:
            return res
    except OSError:
        pass
    raise Exception('Can not determine number of CPUs on this system')


def generate_rule_annotation_tasks(pytheas_model, top_level_dir, db_cred, assume_multi_tables):
    con = connect(dbname=db_cred.database, user=db_cred.user, host='localhost', password=db_cred.password, port=db_cred.port)
    cur = con.cursor()
    cur.execute("""SELECT crawl_datafile_key, groundtruth_key, annotations, original_path, failure
                    FROM ground_truth_2k_canada
                    WHERE annotations is not null
                    ORDER BY crawl_datafile_key
                    """)
    file_counter = 0
    for file_object in cur:
        file_counter += 1
        yield(top_level_dir, file_counter, file_object, pytheas_model, assume_multi_tables, db_cred)

    cur.close()
    con.close()


def save_training_data(crawl_datafile_key, file_dataframe_trimmed, annotations, data_rules_fired, not_data_rules_fired, pytheas_model):
    lines_in_sample = 0
    ### LINE###
    pytheas_line_datapoints_attribute_names = ['crawl_datafile_key',
                                               'line_index',
                                               'label',
                                               'all_summaries_empty']

    pytheas_line_datapoints_attribute_values = []

    pytheas_data_line_rules_attribute_names = ['crawl_datafile_key',
                                               'line_index',
                                               'label',
                                               'undersample']

    for rule in pytheas_model.fuzzy_rules['line']["data"].keys():
        pytheas_data_line_rules_attribute_names.append(rule.lower())

    pytheas_data_line_rules_attribute_values = []

    pytheas_not_data_line_rules_attribute_names = ['crawl_datafile_key',
                                                   'line_index',
                                                   'label',
                                                   'undersample']

    for rule in pytheas_model.fuzzy_rules['line']["not_data"].keys():
        pytheas_not_data_line_rules_attribute_names.append(str(rule.lower()))

    pytheas_not_data_line_rules_attribute_values = []

    ### CELL ###
    pytheas_cell_datapoints_attribute_names = ['crawl_datafile_key',
                                               'line_index',
                                               'column_index',
                                               'label']

    pytheas_cell_datapoints_attribute_values = []

    pytheas_data_cell_rules_attribute_names = ['crawl_datafile_key',
                                               'line_index',
                                               'column_index',
                                               'label',
                                               'aggregate',
                                               'summary_strength',
                                               'null_equivalent',
                                               'undersample']

    pytheas_data_cell_rules_attribute_values = []

    for rule in pytheas_model.fuzzy_rules["cell"]["data"].keys():
        pytheas_data_cell_rules_attribute_names.append(rule.lower())

    pytheas_not_data_cell_rules_attribute_names = ['crawl_datafile_key',
                                                   'line_index',
                                                   'column_index',
                                                   'label',
                                                   'disagreement_summary_strength',
                                                   'undersample']
    for rule in pytheas_model.fuzzy_rules["cell"]["not_data"].keys():
        pytheas_not_data_cell_rules_attribute_names.append(rule.lower())

    pytheas_not_data_cell_rules_attribute_values = []

    data_indexes = []
    header_indexes = []
    footnotes = []
    blank_lines = []
    sub_headers = []
    if 'tables' in annotations.keys():
        for table in annotations['tables']:
            if 'data_start' in table.keys():
                table_counter = table["table_counter"]
                data_indexes = table["data_indexes"]
                header_indexes = table["header"]
                footnotes = table['footnotes']
                sub_headers = table['subheaders']

                if 'blanklines' in table.keys():
                    blank_lines = table['blanklines']
                try:
                    for line_index in data_rules_fired.keys():
                        pytheas_data_line_rules_fired = []
                        pytheas_not_data_line_rules_fired = []

                        if line_index in blank_lines:
                            row_class = "BLANK"
                        elif (len(header_indexes) > 0 and line_index < header_indexes[0]) or (len(header_indexes) == 0 and len(data_indexes) > 0 and line_index < data_indexes[0]):
                            row_class = 'CONTEXT'
                        elif line_index in header_indexes:
                            row_class = 'HEADER'
                        elif line_index in data_indexes:
                            row_class = 'DATA'
                        elif line_index in footnotes:
                            row_class = 'FOOTNOTE'
                        elif line_index in sub_headers:
                            row_class = 'SUBHEADER'
                        else:
                            row_class = 'OTHER'

                        for column_label in file_dataframe_trimmed.columns:
                            pytheas_data_cell_rules_fired = []
                            pytheas_not_data_cell_rules_fired = []
                            undersample = False
                            if ('tables' in annotations.keys() and table_counter == 1 and len(annotations['tables']) > 0) and column_label >= pytheas_model.parameters.ignore_left and (len(data_indexes) > 0 and data_indexes[0] + pytheas_model.parameters.undersample_data_limit > line_index) and (line_index < file_dataframe_trimmed.shape[0] - 1):
                                undersample = True

                            cell_class = row_class
                            if str(file_dataframe_trimmed.loc[line_index, column_label]).lower() in ['' or 'nan']:
                                cell_class = 'BLANK'

                            pytheas_cell_datapoints_attribute_values.append((crawl_datafile_key,
                                                                         line_index,
                                                                         column_label,
                                                                         cell_class))
                            for rule in pytheas_model.fuzzy_rules['cell']["data"].keys():
                                rule_fired = False
                                if rule in data_rules_fired[line_index][column_label]['agreements']:
                                    rule_fired = True
                                pytheas_data_cell_rules_fired.append(rule_fired)
                            is_aggregate = data_rules_fired[line_index][column_label]['aggregate']
                            summary_strength = data_rules_fired[line_index][column_label]['summary_strength']
                            is_null_equivalent = data_rules_fired[line_index][column_label]['null_equivalent']
                            pytheas_data_cell_rules_attribute_values.append((crawl_datafile_key,
                                                                         line_index,
                                                                         column_label,
                                                                         cell_class,
                                                                         is_aggregate,
                                                                         summary_strength,
                                                                         is_null_equivalent,
                                                                         undersample) +
                                                                        tuple(pytheas_data_cell_rules_fired))
                            for rule in pytheas_model.fuzzy_rules['cell']["not_data"].keys():
                                rule_fired = False
                                if rule in not_data_rules_fired[line_index][column_label]['disagreements']:
                                    rule_fired = True
                                pytheas_not_data_cell_rules_fired.append(rule_fired)
                            disagreement_summary_strength = not_data_rules_fired[line_index][column_label]['disagreement_summary_strength']
                            pytheas_not_data_cell_rules_attribute_values.append((crawl_datafile_key,
                                                                             line_index,
                                                                             column_label,
                                                                             cell_class,
                                                                             disagreement_summary_strength,
                                                                             undersample) +
                                                                            tuple(pytheas_not_data_cell_rules_fired))

                        undersample = False
                        if ('tables' in annotations.keys() and table_counter == 1 and len(annotations['tables']) > 0) and (len(data_indexes) > 0 and data_indexes[0] + pytheas_model.parameters.undersample_data_limit > line_index) and (line_index < file_dataframe_trimmed.shape[0] - 1):
                            undersample = True
                            lines_in_sample += 1
                        else:     ################ ADDED
                            break ################
                        # flag which DATA line rules fired
                        for rule in pytheas_model.fuzzy_rules['line']["data"].keys():
                            rule_fired = False
                            if rule in data_rules_fired[line_index]['line']:
                                rule_fired = True
                            pytheas_data_line_rules_fired.append(rule_fired)

                        pytheas_data_line_rules_attribute_values.append((crawl_datafile_key,
                                                                     line_index,
                                                                     row_class,
                                                                     undersample) +
                                                                    tuple(pytheas_data_line_rules_fired))
                        # flag which NOT DATA line rules fired
                        for rule in pytheas_model.fuzzy_rules['line']["not_data"].keys():
                            rule_fired = False
                            if rule in not_data_rules_fired[line_index]['line']:
                                rule_fired = True
                            pytheas_not_data_line_rules_fired.append(rule_fired)

                        pytheas_not_data_line_rules_attribute_values.append((crawl_datafile_key,
                                                                         line_index,
                                                                         row_class,
                                                                         undersample) +
                                                                        tuple(pytheas_not_data_line_rules_fired))
                        all_summaries_empty = data_rules_fired[line_index]["all_summaries_empty"]
                        pytheas_line_datapoints_attribute_values.append((crawl_datafile_key,
                                                                     line_index,
                                                                     row_class,
                                                                     all_summaries_empty))
                except Exception:
                    print(f'Exception in crawl_datafile_key={crawl_datafile_key}')
                    traceback.print_exc()
                    traceback.print_stack()
            break

    pytheas_line_datapoints = pd.DataFrame(pytheas_line_datapoints_attribute_values, columns=pytheas_line_datapoints_attribute_names)
    pytheas_cell_datapoints = pd.DataFrame(pytheas_cell_datapoints_attribute_values, columns=pytheas_cell_datapoints_attribute_names)

    pytheas_data_line_rules = pd.DataFrame(pytheas_data_line_rules_attribute_values, columns=pytheas_data_line_rules_attribute_names)
    pytheas_not_data_line_rules = pd.DataFrame(pytheas_not_data_line_rules_attribute_values, columns=pytheas_not_data_line_rules_attribute_names)
    pytheas_data_cell_rules = pd.DataFrame(pytheas_data_cell_rules_attribute_values, columns=pytheas_data_cell_rules_attribute_names)
    pytheas_not_data_cell_rules = pd.DataFrame(pytheas_not_data_cell_rules_attribute_values, columns=pytheas_not_data_cell_rules_attribute_names)

    for rule in pytheas_model.fuzzy_rules["line"]["data"].keys():
        pytheas_data_line_rules[rule.lower()] = pytheas_data_line_rules[rule.lower()].astype('bool')

    for rule in pytheas_model.fuzzy_rules["line"]["not_data"].keys():
        pytheas_not_data_line_rules[rule.lower()] = pytheas_not_data_line_rules[rule.lower()].astype('bool')

    for rule in pytheas_model.fuzzy_rules["cell"]["data"].keys():
        pytheas_data_cell_rules[rule.lower()] = pytheas_data_cell_rules[rule.lower()].astype('bool')

    for rule in pytheas_model.fuzzy_rules["cell"]["not_data"].keys():
        pytheas_not_data_cell_rules[rule.lower()] = pytheas_not_data_cell_rules[rule.lower()].astype('bool')

    pytheas_data_cell_rules['crawl_datafile_key'] = pytheas_data_cell_rules['crawl_datafile_key'].astype('int')
    pytheas_data_cell_rules['line_index'] = pytheas_data_cell_rules['line_index'].astype('int')
    pytheas_data_cell_rules['column_index'] = pytheas_data_cell_rules['column_index'].astype('int')
    pytheas_data_cell_rules['summary_strength'] = pytheas_data_cell_rules['summary_strength'].astype('int')
    pytheas_data_cell_rules['aggregate'] = pytheas_data_cell_rules['aggregate'].astype('bool')
    pytheas_data_cell_rules['null_equivalent'] = pytheas_data_cell_rules['null_equivalent'].astype('bool')
    pytheas_data_cell_rules['undersample'] = pytheas_data_cell_rules['undersample'].astype('bool')

    pytheas_not_data_cell_rules['crawl_datafile_key'] = pytheas_not_data_cell_rules['crawl_datafile_key'].astype('int')
    pytheas_not_data_cell_rules['line_index'] = pytheas_not_data_cell_rules['line_index'].astype('int')
    pytheas_not_data_cell_rules['column_index'] = pytheas_not_data_cell_rules['column_index'].astype('int')
    pytheas_not_data_cell_rules['disagreement_summary_strength'] = pytheas_not_data_cell_rules['disagreement_summary_strength'].astype('int')
    pytheas_not_data_cell_rules['undersample'] = pytheas_not_data_cell_rules['undersample'].astype('bool')

    pytheas_data_line_rules['crawl_datafile_key'] = pytheas_data_line_rules['crawl_datafile_key'].astype('int')
    pytheas_data_line_rules['line_index'] = pytheas_data_line_rules['line_index'].astype('int')
    pytheas_data_line_rules['undersample'] = pytheas_data_line_rules['undersample'].astype('bool')

    pytheas_not_data_line_rules['crawl_datafile_key'] = pytheas_not_data_line_rules['crawl_datafile_key'].astype('int')
    pytheas_not_data_line_rules['line_index'] = pytheas_not_data_line_rules['line_index'].astype('int')
    pytheas_not_data_line_rules['undersample'] = pytheas_not_data_line_rules['undersample'].astype('bool')

    pytheas_cell_datapoints['crawl_datafile_key'] = pytheas_cell_datapoints['crawl_datafile_key'].astype('int')
    pytheas_cell_datapoints['line_index'] = pytheas_cell_datapoints['line_index'].astype('int')
    pytheas_cell_datapoints['column_index'] = pytheas_cell_datapoints['column_index'].astype('int')

    pytheas_line_datapoints['crawl_datafile_key'] = pytheas_line_datapoints['crawl_datafile_key'].astype('int')
    pytheas_line_datapoints['line_index'] = pytheas_line_datapoints['line_index'].astype('int')
    pytheas_line_datapoints['all_summaries_empty'] = pytheas_line_datapoints['all_summaries_empty'].astype('bool')

    return pytheas_line_datapoints, pytheas_cell_datapoints, pytheas_data_line_rules, pytheas_not_data_line_rules, pytheas_data_cell_rules, pytheas_not_data_cell_rules, lines_in_sample


def probabilistic_sum(line_scores):
    # product_form, demorgan, etc
    predata_row_confidence = 0
    if len(line_scores) > 0:
        score_counts = {x:line_scores.count(x) for x in line_scores}
        prod_list = []
        for score, count in score_counts.items():
            prod_list.append((1 - score)**count)
        predata_row_confidence = 1 - np.prod(prod_list)
    return predata_row_confidence


class Patterns:
    def __init__(self):
        self.data = dict()
        self.not_data = dict()


    def data_initialize(self, column_index, value, candidate_tokens, column_values, column_tokens, column_trains, column_bw_trains, column_symbols, column_cases, column_char_lengths, column_is_numeric_train, max_values_lookahead):
        self.data[column_index] = dict()
        self.data[column_index]['train'] = pytheas_util.generate_pattern_summary(column_trains)
        self.data[column_index]['bw_train'] = pytheas_util.generate_pattern_summary(column_bw_trains)
        self.data[column_index]['symbolset'] = pytheas_util.generate_symbol_summary(column_symbols)
        self.data[column_index]['case'] = pytheas_util.generate_case_summary(column_cases)
        self.data[column_index]['character_length'] = pytheas_util.generate_length_summary(column_char_lengths)
        self.data[column_index]['summary_strength'] = sum(1 for x in column_trains if len(x) > 0)
        self.data[column_index]['candidate_count'] = dict()
        self.data[column_index]['candidate_count'][value] = np.count_nonzero(column_values[2:min(max_values_lookahead, len(column_values))] == value)
        self.data[column_index]['consistent_symbol_sets'] = is_consistent_symbol_sets(column_symbols)
        self.data[column_index]['column_is_numeric'] = pytheas_util.generate_all_numeric_sig_pattern(column_is_numeric_train, [len(t) for t in column_trains])
        self.data[column_index]['partof_multiword_value_repeats'] = dict()
        for part in candidate_tokens:
            self.data[column_index]['partof_multiword_value_repeats'][part] = 0
            for value_tokens in column_tokens:
                if part in value_tokens:
                    self.data[column_index]['partof_multiword_value_repeats'][part] += 1


    def data_increment(self, column_index, value, train_sig, bw_train_sig, symbol_sig, case, num_chars_sig, numeric_train_sig, candidate_tokens):
        self.data[column_index]['train'] = pytheas_util.train_incremental_pattern(self.data[column_index]['train'], train_sig)
        self.data[column_index]['bw_train'] = pytheas_util.train_incremental_pattern(self.data[column_index]['bw_train'], bw_train_sig)
        self.data[column_index]['symbolset'] = pytheas_util.symbolset_incremental_pattern(self.data[column_index]['symbolset'], symbol_sig)
        self.data[column_index]['case'] = pytheas_util.case_incremental_pattern(self.data[column_index]['case'], case)
        self.data[column_index]['character_length'] = pytheas_util.charlength_incremental_pattern(self.data[column_index]['character_length'], num_chars_sig)
        self.data[column_index]['summary_strength'] = pytheas_util.summary_strength_increment(self.data[column_index]['summary_strength'], train_sig)
        self.data[column_index]['candidate_count'] = pytheas_util.candidate_count_increment(self.data[column_index]['candidate_count'], value)
        self.data[column_index]['partof_multiword_value_repeats'] = pytheas_util.token_repeats_increment(self.data[column_index]['partof_multiword_value_repeats'], candidate_tokens)
        self.data[column_index]['consistent_symbol_sets'] = pytheas_util.consistent_symbol_sets_increment(self.data[column_index]['consistent_symbol_sets'], symbol_sig)
        self.data[column_index]['column_is_numeric'] = pytheas_util.numeric_train_incremental_pattern(numeric_train_sig, len(train_sig), self.data[column_index]['column_is_numeric'])


    def not_data_initialize(self, column_index, column_trains, column_bw_trains, column_symbols,
                            column_cases, column_char_lengths, column_isnumber, train_sig, signatures_slice):
        self.not_data[column_index] = dict()
        self.not_data[column_index]['train'] = pytheas_util.generate_pattern_summary(column_trains)
        self.not_data[column_index]['bw_train'] = pytheas_util.generate_pattern_summary(column_bw_trains)
        self.not_data[column_index]['symbolset'] = pytheas_util.generate_symbol_summary(column_symbols)
        self.not_data[column_index]['case'] = pytheas_util.generate_case_summary(column_cases)
        self.not_data[column_index]['character_length'] = pytheas_util.generate_length_summary(column_char_lengths)
        self.not_data[column_index]['all_numbers'] = np.all(column_isnumber)
        self.not_data[column_index]['disagreement_summary_strength'] = sum(1 for x in column_trains if len(x) > 0)

        self.not_data[column_index]['candidate_count'] = dict()
        self.not_data[column_index]['neighbor_count'] = dict()

        if len(train_sig) > 0:
            value = signatures_slice.all_normalized_values[0, column_index]
            self.not_data[column_index]['candidate_count'][value] = 0
            context_values = signatures_slice.all_normalized_values[1:, column_index]
            self.not_data[column_index]['candidate_count'][value] = np.count_nonzero(context_values[1:] == value)
            neighbor = ''
            try:
                neighbor = context_values[1]
                self.not_data[column_index]['neighbor_count'][neighbor] = np.count_nonzero(context_values[2:] == neighbor)
            except:
                self.not_data[column_index]['neighbor_count'][neighbor] = 0


    def not_data_increment(self, column_index, signatures_slice):
        last_train = signatures_slice.train_normalized_numbers[1, column_index]
        last_bw_train_sig = signatures_slice.all_column_bw_train[1, column_index]
        last_symbol_sig = signatures_slice.all_column_symbols[1, column_index]
        last_case = signatures_slice.all_column_cases[1, column_index]
        last_num_chars_sig = signatures_slice.all_column_character_lengths[1, column_index]
        last_is_number = signatures_slice.all_column_isnumber[1, column_index]

        self.not_data[column_index]['train'] = pytheas_util.train_incremental_pattern(self.not_data[column_index]['train'], last_train)
        self.not_data[column_index]['bw_train'] = pytheas_util.train_incremental_pattern(self.not_data[column_index]['bw_train'], last_bw_train_sig)
        self.not_data[column_index]['symbolset'] = pytheas_util.symbolset_incremental_pattern(self.not_data[column_index]['symbolset'], last_symbol_sig)
        self.not_data[column_index]['case'] = pytheas_util.case_incremental_pattern(self.not_data[column_index]['case'], last_case)
        self.not_data[column_index]['character_length'] = pytheas_util.charlength_incremental_pattern(self.not_data[column_index]['character_length'], last_num_chars_sig)

        if len(last_symbol_sig) > 0:
            self.not_data[column_index]['all_numbers'] = np.all([self.not_data[column_index]['all_numbers'], last_is_number])

        if len(last_train) > 0:
            self.not_data[column_index]['disagreement_summary_strength'] += 1

        train_sig = signatures_slice.train_normalized_numbers[0, column_index]
        value = signatures_slice.all_normalized_values[0, column_index]

        if len(train_sig) > 0:
            context_values = signatures_slice.all_normalized_values[1:, column_index]
            self.not_data[column_index]['candidate_count'] = pytheas_util.candidate_count_increment(self.not_data[column_index]['candidate_count'], value)
            neighbor = ''
            try:
                neighbor = context_values[1]
                self.not_data[column_index]['neighbor_count'] = pytheas_util.candidate_count_increment(self.not_data[column_index]['neighbor_count'], neighbor)
            except:
                self.not_data[column_index]['neighbor_count'] = 0


def non_empty_values(df_row):
    last_idx = df_row.last_valid_index()
    return df_row.loc[:last_idx].shape[0]


def collect_line_rules(line, predicted_fdl, line_label, data, signatures, model, line_agreements, line_disagreements, patterns):
    ignore_rules = model.ignore_rules
    signatures_slice = signatures.reverse_slice(top=predicted_fdl,
                                                bottom=line_label)

    null_equivalent_fired = False
    times = sum(signatures.is_null_equivalent[line_label, :])
    if times > 0:
        null_equivalent_fired = True

    all_summaries_empty = True
    max_values_lookahead = data.shape[0]
    coherent_cells = dict()
    incoherent_cells = dict()
    for column_index, column in enumerate(line.index):
        coherent_cells[column] = {}
        incoherent_cells[column] = {}
        coherent_cells[column]["agreements"] = []
        incoherent_cells[column]["disagreements"] = []
        value = signatures.all_normalized_values[line_label, column_index]
        value_tokens = signatures.all_column_tokens[line_label, column_index]
        is_aggregate = signatures.is_aggregate[line_label, column_index]
        is_null_equivalent = signatures.is_null_equivalent[line_label, column_index]

        train_sig = signatures.all_column_train[line_label, column_index]
        bw_train_sig = signatures.all_column_bw_train[line_label, column_index]
        symbol_sig = signatures.all_column_symbols[line_label, column_index]
        case = signatures.all_column_cases[line_label, column_index]
        num_chars_sig = signatures.all_column_character_lengths[line_label, column_index]
        numeric_train_sig = signatures.all_column_is_numeric_train[line_label, column_index]
        is_num = signatures.all_column_isnumber[line_label, column_index]
        coherent_cells[column]['null_equivalent'] = is_null_equivalent
        coherent_cells[column]['aggregate'] = is_aggregate

        column_values = signatures_slice.all_normalized_values[:, column_index]
        column_tokens = signatures_slice.all_column_tokens[:, column_index]
        candidate_tokens = {t  for t in value_tokens if any(c.isalpha() for c in t)}
        column_trains = signatures_slice.train_normalized_numbers[:, column_index]
        column_symbols = signatures_slice.symbolset_normalized_numbers[:, column_index]
        column_bw_trains = signatures_slice.all_column_bw_train[:, column_index]
        column_cases = signatures_slice.all_column_cases[:, column_index]
        column_char_lengths = signatures_slice.all_column_character_lengths[:, column_index]
        column_is_numeric_train = signatures_slice.all_column_is_numeric_train[:, column_index]

        if column_index not in patterns.data.keys():
            patterns.data_initialize(column_index, value, candidate_tokens, column_values, column_tokens,
                                     column_trains, column_bw_trains, column_symbols, column_cases,
                                     column_char_lengths, column_is_numeric_train, max_values_lookahead)
        else:
            patterns.data_increment(column_index, value, train_sig, bw_train_sig, symbol_sig, case,
                                    num_chars_sig, numeric_train_sig, candidate_tokens)

        # patterns of a window INCLUDING the cell we are on
        data_patterns = patterns.data[column_index]
        value_pattern_summary, value_chain_consistent = data_patterns['train']
        value_pattern_BW_summary, _ = data_patterns['bw_train']
        value_symbol_summary = data_patterns['symbolset']
        case_summary = data_patterns['case']
        length_summary = data_patterns['character_length']
        summary_strength = data_patterns['summary_strength']
        candidate_count_of_value = data_patterns['candidate_count'][value]
        partof_multiword_value_repeats = data_patterns['partof_multiword_value_repeats']
        consistent_symbol_sets, _ = data_patterns['consistent_symbol_sets']
        train_sigs_all_numeric, _ = data_patterns['column_is_numeric']

        coherent_cells[column]['summary_strength'] = summary_strength

        if null_equivalent_fired or len(value_pattern_summary) > 0 or len(value_pattern_BW_summary) > 0 or len(value_symbol_summary) > 0 or len(case_summary) > 0:
            all_summaries_empty = False

        for rule in model.fuzzy_rules["cell"]["data"].keys():
            rule_fired = False
            # Don't bother looking for coherency if there are no patterns or if the value on this line gives an empty pattern
            if rule not in ignore_rules["cell"]["data"] and len(column_trains) > 0 and not is_null_equivalent:
                #there is no point calculating agreement over one value, a single value always agrees with itself.
                #in addition, many tables have bilingual headers, so agreement between two header values is very common, require nij>=3
                if len(column_trains) >= 2 and summary_strength >= 2:
                    rule_fired = eval_data_cell_rule(rule, column_values,
                                                     value_pattern_summary, value_chain_consistent,
                                                     value_pattern_BW_summary, value_symbol_summary,
                                                     case_summary, candidate_count_of_value,
                                                     partof_multiword_value_repeats,
                                                     candidate_tokens,
                                                     consistent_symbol_sets,
                                                     train_sigs_all_numeric)

            if rule_fired:
                coherent_cells[column]["agreements"].append(rule)

        ############################################ NOT DATA #####################################
        column_values = signatures_slice.all_normalized_values[1:, column_index]
        column_trains = signatures_slice.train_normalized_numbers[1:, column_index]
        column_bw_trains = signatures_slice.all_column_bw_train[1:, column_index]
        column_symbols = signatures_slice.symbolset_normalized_numbers[1:, column_index]
        column_cases = signatures_slice.all_column_cases[1:, column_index]
        column_char_lengths = signatures_slice.all_column_character_lengths[1:, column_index]
        column_isnumber = signatures_slice.all_column_isnumber[1:, column_index]

        if 'D' in symbol_sig and symbol_sig.issubset(set(['D', '.', ',', 'S', '-', '+', '~', '(', ')'])):
            train_sig = signatures.train_normalized_numbers[line_label, column_index]
            symbol_sig = signatures.symbolset_normalized_numbers[line_label, column_index]
            case = signatures.all_column_cases[line_label, column_index]
            num_chars_sig = signatures.all_column_character_lengths[line_label, column_index]
            ###############################################################################################
        if column_index not in patterns.not_data.keys():
            # initialize patterns
            patterns.not_data_initialize(column_index,
                                         column_trains,
                                         column_bw_trains,
                                         column_symbols,
                                         column_cases,
                                         column_char_lengths,
                                         column_isnumber,
                                         train_sig,
                                         signatures_slice)
        else:
            # increment patterns
            patterns.not_data_increment(column_index,
                                        signatures_slice)

        # patterns of a window that does NOT contain the cell we are on
        not_data_patterns = patterns.not_data[column_index]
        value_pattern_summary, value_chain_consistent = not_data_patterns['train']
        value_pattern_BW_summary, _ = not_data_patterns['bw_train']
        value_symbol_summary = not_data_patterns['symbolset']
        case_summary = not_data_patterns['case']
        length_summary = not_data_patterns['character_length']
        all_numbers_summary = not_data_patterns['all_numbers']
        disagreement_summary_strength = not_data_patterns['disagreement_summary_strength']

        incoherent_cells[column]['disagreement_summary_strength'] = disagreement_summary_strength

        for rule in model.fuzzy_rules["cell"]["not_data"].keys():
            rule_fired = False
            if rule not in ignore_rules["cell"]["not_data"] and len(train_sig) > 0:
                if disagreement_summary_strength > 0 and (not all_numbers_summary or not is_num):
                    rule_fired = eval_not_data_cell_rule(rule,
                                                         value_pattern_summary, value_pattern_BW_summary,
                                                         value_chain_consistent, value_symbol_summary, case_summary,
                                                         length_summary, symbol_sig, case,
                                                         num_chars_sig, disagreement_summary_strength, line_agreements,
                                                         column, line_label)
                    if rule_fired:
                        incoherent_cells[column]["disagreements"].append(rule)

    #Collect data line rules fired
    coherent_cells["all_summaries_empty"] = all_summaries_empty
    line_agreements[line_label] = coherent_cells
    line_disagreements[line_label] = incoherent_cells

    return line_agreements, line_disagreements, patterns


def last_data_line_confidence(line_predictions, predicted_boundary, max_window=4):
    avg_confidence = {}
    avg_predicted_data = 0
    avg_predicted_not_data = 0

    sorted_indexes = sorted(list(line_predictions.keys()), reverse=True)
    for method in ['confidence']:
        predicted_not_data = []
        predicted_data = []

        for index in sorted_indexes:
            # lines predicted not data
            if index > predicted_boundary:
                # only look if you are within a window of the boundary
                if index <= predicted_boundary - max_window:
                    # correctly predicted not data
                    if line_predictions[index]['label'] == 'NOT_DATA':
                        predicted_not_data.append(line_predictions[index]['value'][method])
                    # incorrectly predicted not data
                    else:
                        predicted_not_data.append(-line_predictions[index]['value'][method])
            else:
                if len(predicted_data) == max_window:
                    break
                if line_predictions[index]['label'] == 'NOT_DATA':
                    predicted_data.append(-line_predictions[index]['value'][method])
                else:
                    predicted_data.append(line_predictions[index]['value'][method])

        data_window_weight = len(predicted_data)
        not_data_window_weight = len(predicted_not_data)

        if len(predicted_data) > 0:
            avg_predicted_data = sum(predicted_data)/len(predicted_data)

        if len(predicted_not_data) > 0:
            avg_predicted_not_data = sum(predicted_not_data) / len(predicted_not_data)

        if (data_window_weight + not_data_window_weight) > 0:
            avg_confidence[method] = max(0, (data_window_weight * (avg_predicted_data) + not_data_window_weight * (avg_predicted_not_data)) / (data_window_weight + not_data_window_weight))
        else:
            avg_confidence[method] = 0

    return avg_confidence
