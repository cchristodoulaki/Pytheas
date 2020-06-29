import os
import sys
import csv
import codecs
import cchardet as chardet

from unidecode import unidecode

import numpy as np
import pandas as pd

from langdetect import detect
from langdetect import DetectorFactory
DetectorFactory.seed = 0

def decode_value(row):
    return row.apply(lambda value: unidecode(value) if isinstance(value, str) else value)


def get_dataframe(filepath, max_lines):
    all_csv_tuples = None
    failure = None
    all_csv_tuples, failure, = sample_file(filepath, max_lines)
    return merged_df(failure, all_csv_tuples)


def merged_df(failure, all_csv_tuples):
    dataframes = []
    if failure is None and all_csv_tuples is not None and len(all_csv_tuples) > 0:
        start_index = 0
        line_index = 0
        csv_tuples = [all_csv_tuples[0]]
        num_fields = len(all_csv_tuples[0])
        dataframes = []
        empty_lines = []
        line_index += 1
        while line_index < len(all_csv_tuples):
            csv_tuple = all_csv_tuples[line_index]
            if len(csv_tuple) == 0:
                csv_tuple = ['' for i in range(0, num_fields)]
                empty_lines.append(line_index)

            if len(csv_tuple) == num_fields:
                csv_tuples.append(csv_tuple)
                end_index = line_index + 1
            else:
                end_index = line_index

            if len(csv_tuple) != num_fields or (len(csv_tuple) == num_fields and line_index + 1 == len(all_csv_tuples)):

                dataframe = pd.DataFrame(csv_tuples)
                dataframe.index = list(range(start_index, end_index))
                dataframe.fillna(value=pd.np.nan, inplace=True)
                dataframe.replace(to_replace=[None], value=np.nan, inplace=True)
                dataframe = dataframe.replace(r'^\s*$', pd.np.nan, regex=True)

                if start_index - 1 in empty_lines or len(dataframes) == 0:
                    dataframes.append(dataframe)
                else:
                    dataframes[-1] = dataframes[-1].append(dataframe)

                start_index = end_index
                if start_index < len(all_csv_tuples):
                    csv_tuples = [csv_tuple]
                    num_fields = len(csv_tuple)

            line_index += 1

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

    dataframe.columns = list(range(0, dataframe.shape[1]))
    dataframe = dataframe.apply(decode_value, axis=1)
    return dataframe


def sample_file(filepath, max_batch=100):
    batch = None
    discovered_delimiter = None
    discovered_encoding = None
    encoding_confidence = None
    encoding_language = None
    google_detected_lang = None
    failure = None

    blanklines = []
    try:
        if not os.path.exists(filepath):
            failure = "file does not exist"
            print('\n~~~~~~~~~~~~~~~~~~~~~')
            print(filepath)
            print(failure)
            return batch, discovered_delimiter, discovered_encoding, encoding_language, encoding_confidence, failure, blanklines, google_detected_lang

        size_bytes = os.path.getsize(filepath)

        if size_bytes == 0:
            failure = "file is empty"
            print('\n~~~~~~~~~~~~~~~~~~~~~')
            print(filepath)
            print(failure)
            return batch, discovered_delimiter, discovered_encoding, encoding_language, encoding_confidence, failure, blanklines, google_detected_lang

        encoding_result = detect_encoding(filepath)
        discovered_encoding = encoding_result["encoding"]
        if discovered_encoding is None:
            failure = "No encoding discovered"
            print('\n~~~~~~~~~~~~~~~~~~~~~')
            print(filepath)
            print(failure)
            return batch, discovered_delimiter, discovered_encoding, encoding_language, encoding_confidence, failure, blanklines, google_detected_lang

        encoding_confidence = encoding_result["confidence"]
        if "language" in encoding_result.keys():
            encoding_language = encoding_result["language"]
        else:
            encoding_language = ''

        if discard_file(filepath, discovered_encoding):
            failure = "Illegal file format"
            print('\n~~~~~~~~~~~~~~~~~~~~~')
            print(filepath)
            print(failure)
            return batch, discovered_delimiter, discovered_encoding, encoding_language, encoding_confidence, failure, blanklines, google_detected_lang

        # discover delimiter
        discovered_delimiter = discover_delimiter(filepath, discovered_encoding)
        batch = []
        lineindex = -1

        with codecs.open(filepath, 'rU', encoding=discovered_encoding) as f:
            chunk = f.read(min(size_bytes, 100000))
            if chunk:
                google_detected_lang = detect(chunk)

                for line in csv.reader(chunk.split("\n"), quotechar='"', delimiter=discovered_delimiter, skipinitialspace=True):
                    lineindex += 1
                    if len(line) == 0 or sum(len(s.strip()) for s in line) == 0:
                        blanklines.append(lineindex)
                    batch.append(line)
                    if max_batch is not None and len(batch) >= max_batch:
                        break
            f.flush()

    except Exception as e:
        print('\n~~~~~~~~~~~~~~~~~~~~~')
        print(filepath)
        failure = str(e)
        print(failure)

    return batch, discovered_delimiter, discovered_encoding, encoding_language, encoding_confidence, failure, blanklines, google_detected_lang


def detect_encoding(filepath):
    result = {}
    # use cchardet because it is faster than chardet
    with open(filepath, 'rb') as rawdata:
        msg = rawdata.read(50000)
        result = chardet.detect(msg)
        rawdata.flush()

    if result is None:
        result = {}
    if result['encoding'] is None:
        return result

    if 'encoding' not in result.keys():
        result['encoding'] = 'utf-8-sig'
        result['confidence'] = 0

    # https://github.com/buriy/python-readability/issues/42
    if result['encoding'].lower() == 'big5':
        result['encoding'] = 'big5hkscs'
    elif result['encoding'].lower() == 'gb2312':
        result['encoding'] = 'gb18030'
    elif result['encoding'].lower() == 'ascii':
        result['encoding'] = 'utf-8'
    elif result['encoding'].lower() == 'iso-8859-1':
        result['encoding'] = 'cp1252'

    return result


def discard_file(filepath, encoding):
    with codecs.open(filepath, 'rU', encoding=encoding) as fp:
        firstline = fp.readline()
        while len(firstline.strip()) == 0:
            firstline = fp.readline()
    badphrases = ['<', '<!DOCTYPE', '<HTML', '<?XML', '<!doctype', '<html', '<?xml', '{', '\'{', '"{']
    for phrase in badphrases:
        if firstline.strip().startswith(phrase):
            print('-------------NASTY--------------')
            return True
    return False



def discover_delimiter(filepath, encoding):
    """
    Seek non-singlecount consistent consecutive token count for each
    delimiter and select the delimiter with highest token count
    if there is a tie break it with a ranking of known delimiters
    """
    delimiter = ','
    ranked_delimiters = [',', ';', '\t', '|', ':', '#', '^']
    delim_count = {}
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

    delimiter_idx_of_first_consistent_row = {}
    for delim in ranked_delimiters:
        linecount = 0

        with codecs.open(filepath, 'rU', encoding=encoding) as file:
            csvreader = csv.reader(file, delimiter=delim)
            line = next(csvreader, None)
            while line is not None:

                linecount = linecount + 1
                if linecount == samplelimit:
                    break

                if linecount == 1:
                    #initialize counts
                    token_count = len(line)
                    line = next(csvreader, None)
                    continue

                new_token_count = len(line)

                if new_token_count == 0:
                    line = next(csvreader, None)
                    continue
                elif token_count != 1 and token_count == new_token_count:
                    delimiter_idx_of_first_consistent_row[delim] = linecount - 1
                    # consecutive lines with same token count
                    # look ahead if another line exists,
                    # if next line also has same token count save the count for delimiter
                    line = next(csvreader, None)

                    if line is not None and len(line) > 0:
                        #checking for third consecutive line
                        linecount = linecount + 1
                        new_token_count = len(line)
                        if new_token_count == token_count:
                            # save this token count for this delim
                            delim_count[delim] = token_count
                            break

                        #TOKEN COUNT WAS BY COINCIDENCE
                        #stop looking
                        delimiter_idx_of_first_consistent_row.pop(delim)
                        break

                    # on the last two rows, no point in looking ahead
                    # save this token count for this delim
                    delim_count[delim] = token_count
                    break

                token_count = new_token_count

                line = next(csvreader, None)
    # LOG
    # if there is only one demiliter with highest score use that otherwise use tie breaker
    if len(delim_count.values()) > 0:
        maxValue = max(delim_count.values())
        cand_delims = [key for key in delim_count if delim_count[key] == maxValue]
        if len(cand_delims) == 0:
            delimiter = ''
            return delimiter
        if len(cand_delims) == 1:
            delimiter = cand_delims[0]
            return delimiter
        if delimiter_idx_of_first_consistent_row[cand_delims[0]] < delimiter_idx_of_first_consistent_row[cand_delims[1]]:
            delimiter = cand_delims[0]
            return delimiter
        if delimiter_idx_of_first_consistent_row[cand_delims[0]] > delimiter_idx_of_first_consistent_row[cand_delims[1]]:
            delimiter = cand_delims[1]
            return delimiter
        delimiter = breaktie(cand_delims, ranked_delimiters)
        return delimiter

    return delimiter


def breaktie(candidate_delim, ranked_delimiters):
    """
    given a list of candidate delimiters and a list of ranked delimiters,
    return the candidate delimiter with the highest ranking
    """
    indexlist = []
    for candidate in candidate_delim:
        indexlist.append(ranked_delimiters.index(candidate))
    indexlist.sort()
    return ranked_delimiters[indexlist[0]]


def flatten(d):
    try:
        return reduce(add, d.values())
    except:
        return []
