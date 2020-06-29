import string

import numpy as np

from utilities import null_equivalent_values

footnote_keywords = ['*', 'remarque', 'source', 'note', 'nota', 'not a', 'footnote']

aggregation_tokens = ['total', 'subtotal', 'totaux', 'totales', 'all', 'toute', 'tous', 'less than',
                      'moins de', 'moins que', 'more than', 'plus', 'plus de', 'plusieurs de',
                      'plus que', 'over', 'under', 'higher than', 'plus haut', 'lower than',
                      'plus bas', 'older than', 'plus vieux', 'younger than', 'plus jeune']

datatype_keywords = ['integer', 'string', 'text', 'numeric', 'boolean']

metadata_table_header_keywords = ['field name', 'field', 'description', 'example', 'data type']
# order is important!!!
#  cagr: https://www.investopedia.com/ask/answers/071014/what-formula-calculating-compound-annual-growth-rate-cagr-excel.asp
aggregation_functions = [('totaux', 'sum'), ('totales', 'sum'), ('totale', 'sum'), ('subtotal', 'sum'),
                         ('total partiel', 'sum'), ('total', 'sum'), ('net change in', 'sum'),
                         ('change in', 'sum'), ('average', 'mean'), ('avg', 'mean'),
                         ('variation', 'difference'), ('difference', 'difference'),
                         ('cagr', 'CAGR'), ('tcac', 'CAGR'), ('var %', ''), ('variance', '')]


def dequote(s):
    """
    If a string has single or double quotes around it, remove them.
    Make sure the pair of quotes match.
    If a matching pair of quotes is not found, return the string unchanged.
    """
    if len(s) > 2 and (s[0] == s[-1]) and s.startswith(("'", '"')):
        return s[1:-1]
    return s


def symbol_chain_disagrees(value_symbol_chain, cand_symbol_chain):
    return cand_symbol_chain != value_symbol_chain


def generate_pattern_symbols_and_case(value, outlier_sensitive):
    value_lower = str(value).strip().lower()
    value_tokens = value_lower.split()
    if value is None or value_lower in null_equivalent_values:
        value = ''

    if len(value_lower) > 0:
        for phrase in aggregation_tokens:
            if phrase in value_lower:
                value = ''
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
        elif value != '':
            value_case = 'MIX_CASE'
        else:
            value_case = ''
    except:
        value_case = ''

    value = str(value).strip()
    value_tokens = len(value.split(' '))

    value_characters = len(value)

    i = 0
    while i < len(value):
        if i < (len(value)) and value[i].isalpha():
            letter_counter = 0
            while i < (len(value)) and  value[i].isalpha():
                i += 1
                letter_counter += 1
            value_pattern.append(['A', letter_counter])
            value_symbols.add('A')

        elif i < (len(value)) and value[i].isspace():
            space_counter = 0
            while i < (len(value)) and  value[i].isspace():
                i += 1
                space_counter += 1
            value_pattern.append(['S', space_counter])
            value_symbols.add('S')

        # ignore - if it is the first character followed by a digit
        elif outlier_sensitive and i == 0 and len(value) > 1 and value[i] == '-' and value[i + 1].isdigit():
            digit_counter = 0
            i += 1
            while i < (len(value)) and value[i].isdigit():
                i += 1
                digit_counter += 1
            value_pattern.append(['D', digit_counter])
            value_symbols.add('D')

        elif i < (len(value)) and value[i].isdigit():
            digit_counter = 0
            while i < (len(value)) and  value[i].isdigit():
                i += 1
                digit_counter += 1
            value_pattern.append(['D', digit_counter])
            value_symbols.add('D')

        # Punctuation: !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~
        elif i < (len(value)) and value[i] in string.punctuation:
            punctuation_counter = 0
            punctuation = value[i]
            while i < (len(value)) and  value[i] == punctuation:
                i += 1
                punctuation_counter += 1
            value_pattern.append([punctuation, punctuation_counter])
            value_symbols.add(punctuation)

        elif i < (len(value)):
            unknown_counter = 0
            unknown = value[i]
            while i < (len(value)) and  value[i] == unknown:
                i += 1
                unknown_counter += 1
            value_pattern.append([unknown, unknown_counter])
            value_symbols.add(unknown)

        else:
            i += 1

    return value_pattern, value_symbols, value_case, value_tokens, value_characters


def normalize_decimals_numbers(summary_patterns, summary_symbols):
    new_summary_patterns = []
    new_summary_symbols = []

    for symbols in summary_symbols:
        if (symbols == set() or not ('D' in symbols and symbols.issubset({'D', '.', ',', 'S', '-', '+', '~', '>', '<', '(', ')'}))):
            return summary_patterns, summary_symbols

    for pattern in summary_patterns:
        symbolchain = [symbol_count[0] for symbol_count in pattern]
        indices = [i for i, x in enumerate(symbolchain) if x in ['-', '+', '~', '>', '<']]
        if len(indices) > 1 or len(indices) == 1 and indices[0] > 0:
            new_summary_patterns.append(pattern)
            new_summary_symbols.append(set(symbolchain))
            continue
        if len(pattern) > 0:
            digits = [symbol_count[1] for symbol_count in pattern if symbol_count[0] == 'D']
            digit_count = sum(digits)
            new_summary_patterns.append([['D', digit_count]])
            new_summary_symbols.append(set(['D']))
        else:
            new_summary_patterns.append(pattern)
            new_summary_symbols.append(set())

    return new_summary_patterns, new_summary_symbols


def generate_pattern_summary(attribute_patterns):
    patterns = [p for p in attribute_patterns if len(p) > 0].copy()
    # initialize the attribute pattern with the first value pattern
    if len(patterns) > 0:
        summary_pattern = patterns[0].copy()

        consistent_symbol_chain = True
        for pattern in patterns:
            if len(summary_pattern) == 0:
                break
            if len(pattern) == 0:
                continue
            for symbol_idx, symbol in enumerate(pattern):
                # make sure index exists in summary pattern
                if len(summary_pattern) > symbol_idx: #make sure index in bounds
                    # check if symbols agree
                    if symbol[0] == summary_pattern[symbol_idx][0]:
                        #check if counts disagree (if summary symbol has count)
                        if summary_pattern[symbol_idx][1] != 0 and symbol[1] != summary_pattern[symbol_idx][1]:
                            summary_pattern[symbol_idx][1] = 0
                        #else they agree so do nothing

                        # check if I am on last symbol of pattern and summary is longer, stop looking at this pattern and cut the summary here
                        if symbol_idx == len(pattern) - 1 and len(pattern) < len(summary_pattern):
                            summary_pattern = summary_pattern[0:symbol_idx + 1]
                            consistent_symbol_chain = False
                            break
                    else: #symbols disagreed, remove everything from here on and go to next pattern
                        summary_pattern = summary_pattern[0:symbol_idx]
                        consistent_symbol_chain = False
                        break
                else: #pattern is longer than summary, keep summary as is and stop looking
                    summary_pattern = summary_pattern[0:symbol_idx + 1]
                    consistent_symbol_chain = False
                    break
    else:
        summary_pattern = []
        consistent_symbol_chain = True
    return summary_pattern, consistent_symbol_chain


def generate_symbol_summary(attribute_symbols):
    #initialize symbol list
    attribute_symbols = [s for s in attribute_symbols if len(s) > 0]
    if len(attribute_symbols) > 0:
        summary_symbols = list(attribute_symbols[0])
        for symbol in list(attribute_symbols[0]):
            for symbolset in attribute_symbols:
                if symbol not in symbolset:
                    summary_symbols.remove(symbol)
                    break
        return summary_symbols
    return []


def generate_case_summary(attribute_cases):
    case_summary = ''
    attribute_cases = [a for a in attribute_cases if a != '']
    if len(attribute_cases) > 0:
        case_summary = attribute_cases[0]
        for case in attribute_cases:
            if case_summary != case:
                return ''
    return case_summary


def generate_length_summary(column_lengths):
    column_lengths = np.asarray(column_lengths)
    length_summary = {}
    lengths = column_lengths[column_lengths != 0]
    if len(lengths) > 0:
        length_summary["min"] = min(lengths)
        length_summary["max"] = max(lengths)
    else:
        length_summary["min"] = 0
        length_summary["max"] = 0
    return length_summary


def generate_all_numeric_sig_pattern(patterns_numeric, train_lengths):
    valid_indexes = [i for i, e in enumerate(train_lengths) if e != 0]
    if len(valid_indexes) > 0:
        return np.all(np.array([patterns_numeric[i] for i in valid_indexes])), len(valid_indexes)
    return False, len(valid_indexes)


def train_incremental_pattern(pattern, train_sig):
    train_pattern, consistent_symbol_chain = pattern

    if len(train_pattern) == 0 or len(train_sig) == 0:
        return train_pattern, consistent_symbol_chain

    for symbol_idx, symbol in enumerate(train_sig):
        # make sure index exists in summary pattern
        if len(train_pattern) > symbol_idx: #make sure index in bounds
            # check if symbols agree
            if symbol[0] == train_pattern[symbol_idx][0]:
                #check if counts disagree (if summary symbol has count)
                if train_pattern[symbol_idx][1] != 0 and symbol[1] != train_pattern[symbol_idx][1]:
                    train_pattern[symbol_idx][1] = 0
                #else they agree so do nothing

                # check if I am on last symbol of train_sig and pattern is longer, stop looking at this train_sig and cut the pattern here
                if symbol_idx == len(train_sig) - 1 and len(train_sig) < len(train_pattern):
                    train_pattern = train_pattern[0:symbol_idx + 1]
                    consistent_symbol_chain = False
                    break
            else: #symbols disagreed, remove everything from here on and go to next pattern
                train_pattern = train_pattern[0:symbol_idx]
                consistent_symbol_chain = False
                break
        else: #train_sig is longer than summary, keep summary as is and stop looking
            train_pattern = train_pattern[0:symbol_idx + 1]
            consistent_symbol_chain = False
            break
    return train_pattern, consistent_symbol_chain


def symbolset_incremental_pattern(pattern, symbolset):
    if len(symbolset) > 0:
        return list(set(pattern).intersection(symbolset))
    return pattern


def case_incremental_pattern(case_pattern, case):
    if case == '':
        return case_pattern
    if case != case_pattern:
        return ''
    return case_pattern


def charlength_incremental_pattern(length_summary, char_length):
    if char_length > 0:
        length_summary["min"] = min(length_summary["min"], char_length)
        length_summary["max"] = max(length_summary["max"], char_length)
    return length_summary


def token_repeats_increment(partof_multiword_value_repeats, candidate_tokens):
    for part in candidate_tokens:
        if part in partof_multiword_value_repeats.keys():
            partof_multiword_value_repeats[part] += 1
        else:
            partof_multiword_value_repeats[part] = 0
    return partof_multiword_value_repeats


def summary_strength_increment(summary_strength, train_sig):
    if len(train_sig) > 0:
        summary_strength += 1
    return summary_strength


def candidate_count_increment(candidate_count, value):
    if value in candidate_count.keys():
        candidate_count[value] += 1
    else:
        candidate_count[value] = 0
    return candidate_count


def consistent_symbol_sets_increment(consistent_ss, symbol_sig):
    consistent_symbol_sets, consistent_symbols = consistent_ss
    if len(symbol_sig) > 0 and consistent_symbol_sets and symbol_sig != consistent_symbols:
        consistent_ss = (False, None)
    return consistent_ss


def numeric_train_incremental_pattern(numeric_train_sig, len_train_sig, column_is_numeric_pattern):
    column_is_numeric, len_valid_indexes = column_is_numeric_pattern

    if not column_is_numeric:
        if len_valid_indexes == 0 and len_train_sig > 0:
            len_valid_indexes += 1
            return numeric_train_sig, len_valid_indexes
        return column_is_numeric, len_valid_indexes

    if len_train_sig > 0:
        len_valid_indexes += 1
        column_is_numeric = numeric_train_sig
    return column_is_numeric, len_valid_indexes
