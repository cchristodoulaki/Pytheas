import string
import math
import itertools

import pandas as pd
import numpy as np

import pytheas_utilities as pytheas
import utilities

class TableSignatures:
    def __init__(self, dataframe=pd.DataFrame(), outlier_sensitive=False):
        if not dataframe.empty:
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

            is_aggregate = tokens.applymap(lambda cell: (len(cell) > 0 and not set(cell).isdisjoint(pytheas.aggregation_tokens)))
            is_null_equivalent = normalized_values.applymap(lambda cell: (cell.lower() in pytheas.null_equivalent_values))

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
            self.train_normalized_numbers = train_normalized_numbers.to_numpy()
            self.bw_train_normalized_numbers = bw_train_normalized_numbers.to_numpy()
            self.symbolset_normalized_numbers = symbolset_normalized_numbers.to_numpy()
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
            self.symbolset_normalized_numbers = np.array([])

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
        reverse = TableSignatures()
        reverse.all_normalized_values = self.all_normalized_values[top:bottom + 1][::-1]
        reverse.all_column_character_lengths = self.all_column_character_lengths[top:bottom + 1][::-1]
        reverse.all_column_cases = self.all_column_cases[top:bottom + 1][::-1]
        reverse.all_column_tokens = self.all_column_tokens[top:bottom + 1][::-1]
        reverse.all_column_token_lengths = self.all_column_token_lengths[top:bottom + 1][::-1]
        reverse.all_column_train = self.all_column_train[top:bottom + 1][::-1]
        reverse.all_column_bw_train = self.all_column_bw_train[top:bottom + 1][::-1]
        reverse.all_column_chain = self.all_column_chain[top:bottom + 1][::-1]
        reverse.all_column_symbols = self.all_column_symbols[top:bottom + 1][::-1]
        reverse.all_column_isnumber = self.all_column_isnumber[top:bottom + 1][::-1]
        reverse.all_column_is_numeric_train = self.all_column_is_numeric_train[top:bottom + 1][::-1]
        reverse.train_normalized_numbers = self.train_normalized_numbers[top:bottom + 1][::-1]
        reverse.bw_train_normalized_numbers = self.bw_train_normalized_numbers[top:bottom + 1][::-1]
        reverse.symbolset_normalized_numbers = self.symbolset_normalized_numbers[top:bottom + 1][::-1]
        reverse.is_aggregate = self.is_aggregate[top:bottom + 1][::-1]
        reverse.is_null_equivalent = self.is_null_equivalent[top:bottom + 1][::-1]
        return reverse

def normalize_value(value):
    value = str(value).strip()
    value_lower = value.lower()
    if value is None or value_lower in pytheas.null_equivalent_values:
        value = ''

    if len(value_lower) > 0:
        for phrase in pytheas.aggregation_tokens:
            if phrase in value_lower:
                value = ''
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
        elif value != '':
            value_case = 'MIX_CASE'
        else:
            value_case = ''
    except:
        value_case = ''
    return value_case


def generate_tokens(value):
    tokens = value.lower().split(' ')
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
    while i < len(value):
        if i < (len(value)) and value[i].isalpha():
            letter_counter = 0
            while i < (len(value)) and  value[i].isalpha():
                i += 1
                letter_counter += 1
            value_pattern.append(['A', letter_counter])

        elif i < (len(value)) and value[i].isspace():
            space_counter = 0
            while i < (len(value)) and  value[i].isspace():
                i += 1
                space_counter += 1
            value_pattern.append(['S', space_counter])

        # ignore - if it is the first character followed by a digit
        elif outlier_sensitive and i == 0 and len(value) > 1 and value[i] == '-' and value[i + 1].isdigit():
            digit_counter = 0
            i += 1
            while i < (len(value)) and  value[i].isdigit():
                i += 1
                digit_counter += 1
            value_pattern.append(['D', digit_counter])

        elif i < (len(value)) and value[i].isdigit():
            digit_counter = 0
            while i < (len(value)) and  value[i].isdigit():
                i += 1
                digit_counter += 1
            value_pattern.append(['D', digit_counter])

        # Punctuation: !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~
        elif i < (len(value)) and value[i] in string.punctuation:
            punctuation_counter = 0
            punctuation = value[i]
            while i < (len(value)) and  value[i] == punctuation:
                i += 1
                punctuation_counter += 1
            value_pattern.append([punctuation, punctuation_counter])

        elif i < (len(value)):
            unknown_counter = 0
            unknown = value[i]
            while i < (len(value)) and  value[i] == unknown:
                i += 1
                unknown_counter += 1
            value_pattern.append([unknown, unknown_counter])

        else:
            i += 1

    return value_pattern


def generate_chain(train):
    return [t[0] for t in train]


def generate_symbolset(chain):
    return set(chain)


def train_normalize_numbers(train):
    symbolchain = [symbol_count[0] for symbol_count in train]
    symbolset = set(symbolchain)
    if not ('D' in symbolset and symbolset.issubset({'D', '.', ',', 'S', '-', '+', '~', '>', '<', '(', ')'})):
        return train
    indices = [i for i, x in enumerate(symbolchain) if x in ['-', '+', '~', '>', '<']]
    if len(indices) > 1 or len(indices) == 1 and indices[0] > 0:
        return train
    if len(train) > 0:
        digits = [symbol_count[1] for symbol_count in train if symbol_count[0] == 'D']
        return [['D', sum(digits)]]
    return train


def symbolset_normalize_numbers(symbolset):
    if symbolset.issubset({'D', '.', ',', 'S', '-', '+', '~', '>', '<', '(', ')'}):
        return {'D'}
    return symbolset


def is_number(symbols):
    return (set(symbols) <= {'D', ' ', 'S', '.'} and 'D' in symbols)


def eval_numeric_pattern(train):
    symbolset = {t[0] for t in train}
    if 'D' in symbolset and symbolset.issubset({'D', ',', '.', '-', 'S'}) and (([x[0] for x in train].count('-') == 1 and len(train) > 0 and train[0][0] == '-') or ([x[0] for x in train].count('-') == 0)):
        return True
    return False


def is_consistent_symbol_sets(column_symbols):
    consistent_symbol_set = True
    prev_symbol_set = set()
    if len(column_symbols) > 1:
        prev_symbol_set = column_symbols[0]
        for ss in column_symbols[1:]:
            if len(prev_symbol_set) == 0:
                prev_symbol_set = ss
                continue
            if len(ss) == 0:
                continue
            if ss != prev_symbol_set:
                consistent_symbol_set = False
                prev_symbol_set = set()
    return consistent_symbol_set, prev_symbol_set


def predict_fdl(dataframe, line_predictions, markov_approximation_probabilities, markov_model, data_window=2, combined_label_weight='confidence'):
    first_data_line_combined_data_predictions = {}
    first_data_line_combined_data_predictions["TotalFiles"] = 0
    first_data_line_combined_data_predictions["PredictedPositive"] = 0
    first_data_line_combined_data_predictions["RealPositive"] = 0
    first_data_line_combined_data_predictions["RealNegative"] = 0
    first_data_line_combined_data_predictions["TruePositive"] = 0
    first_data_line_combined_data_predictions["Success"] = 0
    first_data_line_combined_data_predictions["Error"] = 0
    first_data_line = -1
    softmax = 0
    prod_softmax_prior = 0

    if len(line_predictions) > 1:
        legal_sequences = {}
        legal_sequence_priors = {}
        k = len(line_predictions)
        offset = dataframe.index[0]

        ordered_prediction_labels = [line_predictions[line_index]['label'] for line_index in sorted(line_predictions.keys()) if line_index >= offset]

        b = (label for label in ordered_prediction_labels)

        where = 0    # need this to keep track of original indices
        first_data_window_index = -1
        for key, group in itertools.groupby(b):
            length = sum(1 for item in group)
            if length >= data_window:
                items = [where + i for i in range(length)]
                if key == 'DATA':
                    first_data_window_index = items[0]
                    break
            where += length
        if  first_data_window_index == -1:
            try:
                first_data_window_index = ordered_prediction_labels.index('DATA')
                data_window = 1
            except:
                first_data_window_index = -1

        k = first_data_window_index + data_window

        if first_data_window_index >= 0:
            for sequence_id in range(0, k + 1):
                sequence_prior = 1
                legal_sequences[sequence_id] = []
                while len(legal_sequences[sequence_id]) < k - sequence_id:
                    legal_sequences[sequence_id].append('NOT_DATA')
                    if markov_model is not None:
                        if len(legal_sequences[sequence_id]) == 1:
                            if markov_model == 'first_order':
                                sequence_prior = sequence_prior * markov_approximation_probabilities['p_first_tables_start_not_data']

                            elif markov_model == 'second_order':
                                sequence_prior = sequence_prior * markov_approximation_probabilities['prob_nd_given_start']
                        else:
                            if markov_model == 'first_order':
                                sequence_prior = sequence_prior * markov_approximation_probabilities['prob_ndI_given_ndIminus1']

                            elif markov_model == 'second_order' and len(legal_sequences[sequence_id]) == 2:
                                sequence_prior = sequence_prior * markov_approximation_probabilities['prob_nd_given_start_nd']
                            elif markov_model == 'second_order' and len(legal_sequences[sequence_id]) > 2:
                                sequence_prior = sequence_prior * markov_approximation_probabilities['prob_nd_given_nd_nd']

                while len(legal_sequences[sequence_id]) >= k - sequence_id and len(legal_sequences[sequence_id]) < k:
                    legal_sequences[sequence_id].append('DATA')
                    if markov_model is not None:
                        if len(legal_sequences[sequence_id]) == 1:
                            if markov_model == 'first_order':
                                sequence_prior = sequence_prior * markov_approximation_probabilities['p_first_tables_start_data']

                            elif markov_model == 'second_order':
                                sequence_prior = sequence_prior * markov_approximation_probabilities['prob_d_given_start']
                        else:
                            if markov_model == 'first_order' and legal_sequences[sequence_id].count('DATA') == 1:
                                sequence_prior = sequence_prior * markov_approximation_probabilities['prob_dI_given_ndIminus1']

                            elif markov_model == 'second_order' and len(legal_sequences[sequence_id]) == 2 and legal_sequences[sequence_id].count('DATA') == 1:
                                sequence_prior = sequence_prior * markov_approximation_probabilities['prob_d_given_start_nd']
                            elif markov_model == 'second_order' and len(legal_sequences[sequence_id]) == 2 and legal_sequences[sequence_id].count('DATA') == 2:
                                sequence_prior = sequence_prior * markov_approximation_probabilities['prob_d_given_start_d']
                            elif markov_model == 'second_order' and len(legal_sequences[sequence_id]) > 2 and legal_sequences[sequence_id].count('DATA') == 1:
                                sequence_prior = sequence_prior * markov_approximation_probabilities['prob_d_given_nd_nd']
                            elif markov_model == 'second_order' and len(legal_sequences[sequence_id]) > 2 and legal_sequences[sequence_id].count('DATA') == 2:
                                sequence_prior = sequence_prior * markov_approximation_probabilities['prob_d_given_nd_d']
                            elif markov_model == 'second_order' and len(legal_sequences[sequence_id]) > 2 and legal_sequences[sequence_id].count('DATA') >= 3:
                                sequence_prior = sequence_prior * markov_approximation_probabilities['prob_d_given_d_d']

                legal_sequence_priors[sequence_id] = sequence_prior

            match_weight = {}
            for sequence_id, legal_label_sequence in legal_sequences.items():
                match_weight[sequence_id] = match_sequence(legal_label_sequence,
                                                           line_predictions,
                                                           offset,
                                                           first_data_window_index,
                                                           data_window,
                                                           combined_label_weight)

            match_softmax = {}
            for sequence_id, legal_label_sequence in legal_sequences.items():
                match_softmax[sequence_id] = math.exp(match_weight[sequence_id]) / sum([math.exp(v) for v in match_weight.values()])

            sequence_final_probability = {}
            for sequence_id, _ in legal_sequences.items():
                sequence_final_probability[sequence_id] = legal_sequence_priors[sequence_id] * match_softmax[sequence_id]

            sorted_id_weight = sorted(sequence_final_probability.items(), key=lambda kv: (-kv[1], -kv[0]))
            winning_sequence_id = sorted_id_weight[0][0]
            winning_sequence = legal_sequences[sorted_id_weight[0][0]]

            softmax = match_softmax[winning_sequence_id]
            prod_softmax_prior = sequence_final_probability[winning_sequence_id]
            try:
                first_data_line = winning_sequence.index('DATA') + offset
            except:
                first_data_line = -1

    first_data_line_combined_data_predictions['softmax'] = softmax
    first_data_line_combined_data_predictions['prod_softmax_prior'] = prod_softmax_prior

    # Calculate CONFIDENCE of First Data Line with old method
    avg_confidence, min_confidence = first_data_line_confidence(line_predictions, first_data_line)

    first_data_line_combined_data_predictions['avg_confidence'] = avg_confidence
    first_data_line_combined_data_predictions['min_confidence'] = min_confidence

    return first_data_line, first_data_line_combined_data_predictions


def match_sequence(legal_label_sequence, line_predictions, offset, first_double_data_index, data_window, label_weight='confidence'):
    match_weight = 0
    for legal_label_idx, legal_label in enumerate(legal_label_sequence):
        if legal_label_idx >= first_double_data_index+data_window:
            break
        if  line_predictions[offset + legal_label_idx]['label'] == legal_label:
            match_weight += line_predictions[offset + legal_label_idx]['value'][label_weight]
        else:
            match_weight -= line_predictions[offset + legal_label_idx]['value'][label_weight]
    return match_weight


def first_data_line_confidence(line_predictions, first_data_line, max_window=4):
    avg_confidence = {}
    min_confidence = {}
    avg_predicted_data = 0
    avg_predicted_not_data = 0
    min_predicted_data = 0
    min_predicted_not_data = 0

    sorted_indexes = sorted(list(line_predictions.keys()))

    for method in ['confusion_index', 'confidence', 'difference']:
        predicted_not_data = []
        predicted_data = []

        if first_data_line != -1:
            for index in sorted_indexes:
                if index < first_data_line:
                    if index >= first_data_line - max_window:
                        if line_predictions[index]['label'] == 'NOT_DATA':
                            predicted_not_data.append(line_predictions[index]['value'][method])
                        else:
                            predicted_not_data.append(-line_predictions[index]['value'][method])
                else:
                    if len(predicted_data) == max_window:
                        break
                    if line_predictions[index]['label'] == 'NOT_DATA':
                        data_confidence = -line_predictions[index]['value'][method]
                    else:
                        data_confidence = line_predictions[index]['value'][method]
                    predicted_data.append(data_confidence)

            data_window_weight = len(predicted_data)
            not_data_window_weight = len(predicted_not_data)

            if len(predicted_data) > 0:
                avg_predicted_data = sum(predicted_data) / len(predicted_data)
                min_predicted_data = min(predicted_data)

            if len(predicted_not_data) > 0:
                avg_predicted_not_data = sum(predicted_not_data) / len(predicted_not_data)
                min_predicted_not_data = min(predicted_not_data)

            if (data_window_weight+not_data_window_weight) > 0:
                avg_confidence[method] = max(0, (data_window_weight * (avg_predicted_data) + not_data_window_weight * (avg_predicted_not_data)) / (data_window_weight + not_data_window_weight))
                min_confidence[method] = max(0, (data_window_weight * (min_predicted_data) + not_data_window_weight * (min_predicted_not_data)) / (data_window_weight + not_data_window_weight))
            else:
                avg_confidence[method] = 0
                min_confidence[method] = 0

        else:
            avg_windows = []
            if len(sorted_indexes) > 1:
                for index in sorted_indexes[:-1]:
                    predicted_not_data = []
                    if line_predictions[index]['label'] == 'NOT_DATA':
                        predicted_not_data.append(line_predictions[index]['value'][method])
                    else:
                        predicted_not_data.append(-line_predictions[index]['value'][method])

                    if line_predictions[index + 1]['label'] == 'NOT_DATA':
                        predicted_not_data.append(line_predictions[index + 1]['value'][method])
                    else:
                        predicted_not_data.append(-line_predictions[index + 1]['value'][method])
                    avg_predicted_not_data = 0
                    if len(predicted_not_data) > 0:
                        avg_predicted_not_data = sum(predicted_not_data) / len(predicted_not_data)
                        min_predicted_not_data = min(predicted_not_data)

                    avg_windows.append(min(0, avg_predicted_not_data))
            else:
                avg_windows.append(line_predictions[sorted_indexes[0]]['value'][method])

            avg_confidence[method] = max(0, min(avg_windows))
            min_confidence[method] = max(0, min(avg_windows))

    return avg_confidence, min_confidence


def confusion_index(a, b):
    if a == 0:
        return 0
    return (a - b) / a


def predict_line_label(data_confidence, not_data_confidence):
    line = dict()
    line["value"] = dict()
    line_is_data_conf = 0

    if data_confidence <= not_data_confidence:
        line["label"] = 'NOT_DATA'
        line["value"]["confusion_index"] = confusion_index(not_data_confidence, data_confidence)
        line["value"]["confidence"] = not_data_confidence
        line["value"]["difference"] = not_data_confidence - data_confidence
    else:
        line["label"] = 'DATA'
        line["value"]["confusion_index"] = line_is_data_conf
        line["value"]["confidence"] = data_confidence
        line["value"]["difference"] = data_confidence - not_data_confidence
        line_is_data_conf = confusion_index(data_confidence, not_data_confidence)

    return line, line_is_data_conf


def predict_combined_data_confidences(dataframe, data_confidence, not_data_confidence, max_candidates):
    combined_data_line_confidences = {}
    line_predictions = {}
    offset = dataframe.index[0]

    row_counter = -1
    for row_index in dataframe.index:
        row_counter += 1
        if row_counter == offset + max_candidates:
            break
        if row_index not in data_confidence.keys() or row_index not in not_data_confidence.keys():
            break
        prediction, line_is_data_conf = predict_line_label(data_confidence[row_index], not_data_confidence[row_index])
        line_predictions[row_index] = prediction
        combined_data_line_confidences[row_index] = line_is_data_conf

    return combined_data_line_confidences, line_predictions


def eval_data_cell_rule(rule, columnvalues, all_values_summary, consistent_symbol_chain, pattern_BW_summary, value_symbol_summary, case_summary, candidate_count, partof_multiword_value_repeats, candidate_tokens, consistent_symbol_set, all_patterns_numeric):
    rule_fired = False

    candidate = columnvalues[0]

    if candidate is None  or str(candidate).strip().lower() in pytheas.null_equivalent_values:
        rule_fired = False

    elif rule == "VALUE_REPEATS_ONCE_BELOW":
        if len(columnvalues) > 2:
            if candidate_count == 1:
                rule_fired = True

    elif rule == "CONSISTENT_SINGLE_WORD_CONSISTENT_CASE":
        if consistent_symbol_chain and len(value_symbol_summary) == 1 and value_symbol_summary[0] == 'A' and case_summary in ['ALL_CAPS', 'ALL_LOW', 'TITLE']:
            rule_fired = True

    elif rule == "CONSISTENT_CHAR_LENGTH":
        if consistent_symbol_chain and str(candidate).strip().lower() not in ['', ' ', 'nan', 'None']:
            rule_fired = True

    elif  rule == "VALUE_REPEATS_TWICE_OR_MORE_BELOW":
        if len(columnvalues) > 2:
            if candidate_count >= 2:
                rule_fired = True

    elif rule == "ONE_ALPHA_TOKEN_REPEATS_ONCE_BELOW":
        # "Rule_2_a: Only one alphabetic token from multiword value repeats below, and it repeats only once"
        if len(columnvalues) > 2 and sum([partof_multiword_value_repeats[t] for t in candidate_tokens]) == 1:
            rule_fired = True

    elif rule == "ALPHA_TOKEN_REPEATS_TWICE_OR_MORE":
        # "Rule_2_b: At least one alphabetic token from multiword value repeats below at least twice"
        if len(columnvalues) > 2 and sum([partof_multiword_value_repeats[t] for t in candidate_tokens]) >= 2:
            rule_fired = True

    elif rule == "CONSISTENT_NUMERIC_WIDTH":
        # "Rule_3 consistently numbers with consistent digit count for all."
        if consistent_symbol_chain and len(all_values_summary) == 1 and all_values_summary[0][0] == 'D' and all_values_summary[0][1] > 0:
            rule_fired = True

    elif rule == "CONSISTENT_NUMERIC":
        # "Rule_4_a consistently ONE symbol = D"
        if consistent_symbol_chain and len(all_values_summary) == 1 and  all_values_summary[0][0] == 'D':
            rule_fired = True
        # print(f'rule_fired={rule_fired}')

    elif rule == "CONSISTENT_D_STAR":
        # "Rule_4_b consistently TWO symbols, the first is a digit"
        if consistent_symbol_chain and len(all_values_summary) == 2 and all_values_summary[0][0] == 'D':
            rule_fired = True

    elif rule == "FW_SUMMARY_D":
        # "Rule_4_fw two or above symbols in the FW summary, the first is a digit"
        if len(all_values_summary) >= 2 and all_values_summary[0][0] == 'D':
            rule_fired = True
    elif rule == "BW_SUMMARY_D":
        #"Rule_4_bw two or above symbols in the BW summary, the first is a digit"
        if len(pattern_BW_summary) >= 2 and pattern_BW_summary[0][0] == 'D':
            rule_fired = True
    elif rule == "BROAD_NUMERIC":
        #"Rule_5 all values digits, optionally have . or ,  or S"
        if all_patterns_numeric:
            rule_fired = True
    elif rule == "FW_THREE_OR_MORE_NO_SPACE":
        #"Rule_6 three or above symbols in FW summary that do not contain a  Space"
        if len(all_values_summary) >= 3 and 'S' not in [x[0] for x in all_values_summary]:
            rule_fired = True
    elif rule == "BW_THREE_OR_MORE_NO_SPACE":
        #"Rule_7 three or above symbols in BW summary that do not contain a  Space"
        if len(pattern_BW_summary) >= 3 and 'S' not in [x[0] for x in pattern_BW_summary]:
            rule_fired = True
    elif rule == "CONSISTENT_SS_NO_SPACE":
        #"Rule_8 consistently at least two symbols in the symbol set summary, none of which are S or _"
        if consistent_symbol_set and len(value_symbol_summary) >= 2 and 'S' not in value_symbol_summary and '_' not in  value_symbol_summary:
            rule_fired = True

    elif rule == "CONSISTENT_SC_TWO_OR_MORE":
        #"Rule_10 two or more symbols consistent chain"
        if consistent_symbol_chain and len(all_values_summary) >= 2:
            rule_fired = True

    elif rule == "FW_TWO_OR_MORE_OR_MORE_NO_SPACE":
        #"Rule_11_fw two or above symbols in FW summary that do not contain a Space"
        if len(all_values_summary) >= 2 and 'S' not in [x[0] for x in all_values_summary]:
            rule_fired = True

    elif rule == "BW_TWO_OR_MORE_OR_MORE_NO_SPACE":
        #"Rule_11_bw two or above symbols in BW summary that do not contain a Space"
        if len(pattern_BW_summary) >= 2 and 'S' not in [x[0] for x in pattern_BW_summary]:
            rule_fired = True

    elif rule == "FW_TWO_OR_MORE_OR_MORE_NO_SPACE_FIRST_TWO":
        #"Rule_12_fw two or above symbols in FW summary, the first two do not contain a Space"
        if len(all_values_summary) >= 2 and 'S' not in [x[0] for x in all_values_summary[0:2]]:
            rule_fired = True

    elif rule == "BW_TWO_OR_MORE_OR_MORE_NO_SPACE_FIRST_TWO":
        #"Rule_12_bw two or above symbols in BW summary, the first two do not contain a Space"
        if len(pattern_BW_summary) >= 2 and 'S' not in [x[0] for x in pattern_BW_summary[0:2]]:
            rule_fired = True

    elif rule == "FW_D5PLUS":
        #"Rule_13_fw FW summary is [['D',count]], where count>=5"
        if  len(all_values_summary) == 1 and  all_values_summary[0][0] == 'D' and all_values_summary[0][1] >= 5:
            rule_fired = True

    elif rule == "BW_D5PLUS":
        #"Rule_13_bw BW summary is [['D',count]], where count>=5"
        if  len(pattern_BW_summary) == 1 and  pattern_BW_summary[0][0] == 'D' and pattern_BW_summary[0][1] >= 5:
            rule_fired = True

    elif rule == "FW_D1":
        #"Rule_14_fw FW summary is [['D',1]]"
        if  len(all_values_summary) == 1 and  all_values_summary[0][0] == 'D' and all_values_summary[0][1] == 1:
            rule_fired = True

    elif rule == "BW_D1":
        #"Rule_14_bw BW summary is [['D',1]]"
        if  len(pattern_BW_summary) == 1 and  pattern_BW_summary[0][0] == 'D' and pattern_BW_summary[0][1] == 1:
            rule_fired = True

    elif rule == "FW_D4":
        #"Rule_15_fw FW summary is [['D',4]]"
        if  len(all_values_summary) == 1 and  all_values_summary[0][0] == 'D' and all_values_summary[0][1] == 4:
            rule_fired = True

    elif rule == "BW_D4":
        #"Rule_15_bw BW summary is [['D',4]]"
        if  len(pattern_BW_summary) == 1 and  pattern_BW_summary[0][0] == 'D' and pattern_BW_summary[0][1] == 4:
            rule_fired = True

    elif rule == "FW_LENGTH_4PLUS":
        #"Rule_17_fw four or more symbols in the FW summary"
        if len(all_values_summary) >= 4:
            rule_fired = True

    elif rule == "BW_LENGTH_4PLUS":
        #"Rule_17_bw four or more symbols in the BW summary"
        if len(pattern_BW_summary) >= 4:
            rule_fired = True

    elif rule == "CASE_SUMMARY_CAPS":
        #"Rule_18 case summary is ALL_CAPS"
        if case_summary == 'ALL_CAPS':
            rule_fired = True

    elif rule == "CASE_SUMMARY_LOWER":
        if case_summary == 'ALL_LOW':
            rule_fired = True

    elif rule == "CASE_SUMMARY_TITLE":
        if case_summary == 'TITLE':
            rule_fired = True

    return rule_fired


def predict_header_indexes(file_dataframe, first_data_line_annotated, table_counter):
    #initialize
    predicted_pytheas_subheaders = []

    null_columns = file_dataframe.columns[file_dataframe.isna().all()].tolist()
    file_dataframe = file_dataframe.drop(null_columns, axis=1)

    dataframe = file_dataframe.loc[:first_data_line_annotated - 1]

    if len(file_dataframe.index) > 0:
        last_row_label = file_dataframe.index[-1]
    else:
        last_row_label = first_data_line_annotated

    data = file_dataframe.loc[first_data_line_annotated:min(last_row_label, first_data_line_annotated + 10)]

    null_columns = data.columns[data.isna().all()].tolist()
    data = data.drop(null_columns, axis=1)
    dataframe = dataframe.drop(null_columns, axis=1)

    before_header = True
    top_header_candidate_index = 0
    for row_index, row in dataframe.iterrows():
        row_values = [str(elem) if elem is not None else elem for elem in row.tolist()]

        pre_header_events = pre_header_line(row_values, before_header)
        if "UP_TO_SECOND_COLUMN_COMPLETE_CONSISTENTLY" not in pre_header_events:
            before_header = False
            top_header_candidate_index = row_index
            break

    candidate_headers = dataframe.loc[top_header_candidate_index:]

    predicted_header_indexes = []
    empty_lines = []
    NON_DUPLICATE_HEADER_ACHIEVED = False
    if candidate_headers.shape[0] > 0:
        non_empty_lines_assessed = 0
        NON_EMPTY_LINE_SEEN = False
        for reverse_index in range(1, candidate_headers.shape[0] + 1):
            if len(candidate_headers.iloc[-reverse_index:].dropna(how='all', axis=0)) > 6:
                break

            #ignore first line above data if it was completely empty
            row_values = candidate_headers.iloc[-reverse_index].tolist()
            if len([i for i in row_values if str(i).strip().lower() not in ['', 'nan', 'none', 'null']]) == 0:
                empty_lines.append(first_data_line_annotated-reverse_index)
                if NON_DUPLICATE_HEADER_ACHIEVED:
                    break
                continue
            if reverse_index > 1 and len(row_values) > 1 and str(row_values[0]).strip().lower() not in ['', 'nan', 'none', 'null'] and len([i for i in row_values[1:] if str(i).strip().lower() not in ['', 'nan', 'none', 'null']]) == 0:
                if not NON_EMPTY_LINE_SEEN:
                    empty_lines.append(first_data_line_annotated-reverse_index)
                if NON_DUPLICATE_HEADER_ACHIEVED:
                    break
                continue

            if len(row_values) > 2 and str(row_values[1]).strip().lower() not in ['', 'nan', 'none', 'null'] and len([i for i in row_values[2:] if str(i).strip().lower() not in ['', 'nan', 'none', 'null']]) == 0:
                if NON_DUPLICATE_HEADER_ACHIEVED:
                    break
                continue
            NON_EMPTY_LINE_SEEN = True
            non_empty_lines_assessed += 1
            consider_header_dataframe = candidate_headers.iloc[-reverse_index:].drop(empty_lines,
                                                                                     axis=0)
            if NON_DUPLICATE_HEADER_ACHIEVED and reverse_index > 1 and len(row_values) > 2:
                extension = True
                for value_index, value in enumerate(row_values[2:]):
                    if str(value).strip().lower() not in ['', 'nan', 'none', 'null'] and str(candidate_headers.iloc[-reverse_index + 1].tolist()[value_index + 2]).strip().lower() in ['', 'nan', 'none', 'null']:
                        if len(row_values) > 4 and (value_index in (len(row_values[2:]) - 1, len(row_values[2:]) - 2)):
                            continue
                        extension = False
                        break
                if extension:
                    header = candidate_headers.iloc[-reverse_index:]
                    predicted_header_indexes = [x for x in list(header.index) if x not in empty_lines]

            elif assess_combo_header(consider_header_dataframe):
                header = candidate_headers.iloc[-reverse_index:]
                predicted_header_indexes = [x for x in list(header.index) if x not in empty_lines]

            if (len(predicted_header_indexes) > 0 and not has_duplicates(consider_header_dataframe)):
                NON_DUPLICATE_HEADER_ACHIEVED = True

            if non_empty_lines_assessed > 4:
                break

        if len(predicted_header_indexes) > 0:
            if predicted_header_indexes[0] - 1 in dataframe.index:
                row_before = dataframe.loc[predicted_header_indexes[0] - 1].tolist()
                if len(row_before) > 1 and len([i for i in row_before[1:] if str(i).strip().lower() not in ['', 'nan', 'none', 'null']]) > 0:
                    predicted_header_indexes.insert(0, predicted_header_indexes[0] - 1)

            last_header_index = predicted_header_indexes[-1]
            while len(predicted_header_indexes) > 0:
                first_value = str(file_dataframe.loc[last_header_index, file_dataframe.columns[0]]).strip()
                if  len(dataframe.columns) > 1 and file_dataframe.loc[last_header_index, 1:].isnull().values.all() and (first_value.startswith('(') and not first_value.endswith(')')):
                    predicted_pytheas_subheaders.append(last_header_index)
                    predicted_header_indexes = predicted_header_indexes[:-1]
                    last_header_index = predicted_header_indexes[-1]
                else:
                    break
        else:
            if len(predicted_header_indexes) == 0 and table_counter == 1 and first_data_line_annotated > 0 and len(candidate_headers) > 0:
                count_offset = 0
                for reverse_index in range(1, candidate_headers.shape[0] + 1):
                    count_offset += 1
                    row_values = candidate_headers.iloc[-reverse_index].tolist()
                    if len([i for i in row_values if str(i).strip().lower() not in ['', 'nan', 'none', 'null']]) > 0:
                        predicted_header_indexes.append(first_data_line_annotated - count_offset)
                        break
    return predicted_header_indexes, predicted_pytheas_subheaders


def pre_header_line(row_values, before_header):
    pre_header_line_events = []
    if len(row_values) == 1 and row_values[0] not in ['nan', 'none', '', ' ']:
        before_header = False

    if before_header and len(row_values) == 2:
        row_values = [str(elem).strip().lower()for elem in row_values[1:]]

        for value in row_values:
            if value not in ['nan', 'none', '', ' ']:
                before_header = False
                break

    if before_header and len(row_values) > 2:
        row_values = [str(elem).strip().lower()for elem in row_values[2:]]

        for value in row_values:
            if value not in ['nan', 'none', '', ' ']:
                before_header = False
                break

    if before_header:
        pre_header_line_events.append('UP_TO_SECOND_COLUMN_COMPLETE_CONSISTENTLY')

    return pre_header_line_events


def combo_row(rows):
    rows = rows.values
    ret = list(rows[0])
    for rowidx, row in enumerate(rows):
        if rowidx == 0:
            continue
        buffer_row = rows[rowidx - 1]
        buffer_row = [str(i).strip() if str(i).strip().lower() not in ['', 'nan', 'none', 'null'] else '' for i in buffer_row]
        for idx, value in enumerate(buffer_row):
            if idx + 1 < len(buffer_row) and buffer_row[idx + 1].strip() == '':
                buffer_row[idx + 1] = pytheas.dequote(value.strip())
        for idx, value in enumerate(row):
            if str(value).strip().lower() in ['', 'nan', 'none', 'null']:
                value = ''
            value = str(value)
            ret = [str(i).strip().lower() if str(i).strip().lower() not in ['', 'nan', 'none', 'null'] else '' for i in ret]
            if ret[idx].strip() != '':
                ret[idx] = (pytheas.dequote(str(ret[idx]).strip()) + ' ' + pytheas.dequote(str(value).strip()).strip())
            else:
                ret[idx] = (buffer_row[idx].strip() + ' ' + pytheas.dequote(value.strip()).strip())
    return ret


def assess_combo_header(candidate_header_dataframe):
    candidate_header = combo_row(candidate_header_dataframe)

    #if no nulls in candidate this is a good candidate
    if len([i for i in candidate_header if str(i).strip().lower() not in ['', 'nan', 'none', 'null']]) == len(candidate_header):
        return True

    #if only one or two attributes, second must be complete
    if len(candidate_header) == 1:
        return False
    if len(candidate_header) == 2:
        if str(candidate_header[1]).strip().lower() in ['', 'nan', 'none', 'null']:
            return False
        return True

    #if three attributes, first may be incomplete
    if len(candidate_header) == 3:
        if len([i for i in candidate_header[1:] if str(i).strip().lower() not in ['', 'nan', 'none', 'null']]) == len(candidate_header[1:]):
            return True
        return False

    #if four attributes, first two or last two may be incomplete
    if len(candidate_header) == 4:
        if  len([i for i in candidate_header[2:] if str(i).strip().lower() not in ['', 'nan', 'none', 'null']]) == len(candidate_header[2:]):
            return True
        if len([i for i in candidate_header[:-2] if str(i).strip().lower() not in ['', 'nan', 'none', 'null']]) == len(candidate_header[:-2]):
            return True
        if len([i for i in candidate_header[1:-1] if str(i).strip().lower() not in ['', 'nan', 'none', 'null']]) == len(candidate_header[1:-1])  or len([i for i in candidate_header[1:] if str(i).strip().lower() not in ['', 'nan', 'none', 'null']]) == len(candidate_header[1:]) or len([i for i in candidate_header[:-1] if str(i).strip().lower() not in ['', 'nan', 'none', 'null']]) == len(candidate_header[:-1]):
            return True
        return False

    if len(candidate_header) > 4 and len([i for i in candidate_header[2:-2] if str(i).strip().lower() not in ['', 'nan', 'none', 'null']]) == len([i for i in candidate_header[2:-2]]):
        return True

    return False


def has_duplicates(candidate_header_dataframe):
    candidate_header = combo_row(candidate_header_dataframe)

    non_empty_values = [i for i in candidate_header if str(i).strip().lower() not in ['', 'nan', 'none', 'null']]
    if len(non_empty_values) == len(set(non_empty_values)):
        return False
    return True


def eval_not_data_cell_rule(rule, value_pattern_summary, value_pattern_BW_summary, value_chain_consistent, value_symbol_summary, case_summary, length_summary, cand_pattern, cand_case, cand_length, summary_strength, line_agreements, columnindex, line_index):
    cand_symbol_chain = [x[0] for x in cand_pattern]
    value_symbol_chain = [x[0] for x in value_pattern_summary]

    summary_min_length = length_summary["min"]
    summary_max_length = length_summary["max"]

    rule_fired = False
    if rule == "First_FW_Symbol_disagrees":
        if len(value_pattern_summary) != 0 and cand_pattern[0][0] != value_pattern_summary[0][0]:
            rule_fired = True

    elif rule == "First_BW_Symbol_disagrees":
        if len(value_pattern_BW_summary) != 0 and cand_pattern[-1][0] != value_pattern_BW_summary[0][0]:
            rule_fired = True

    elif rule == "NON_NUMERIC_CHAR_COUNT_DIFFERS_FROM_CONSISTENT":
        if cand_length > 0 and cand_length != summary_min_length and summary_min_length == summary_max_length:
            if 'D' not in value_symbol_summary:
                rule_fired = True

    elif rule == "CHAR_COUNT_UNDER_POINT1_MIN":
        if cand_length > 0 and cand_length <= 0.1 * summary_min_length:
            rule_fired = True

    elif rule == "CHAR_COUNT_UNDER_POINT3_MIN":
        if cand_length > 0 and cand_length <= 0.3 * summary_min_length:
            rule_fired = True

    elif rule == "CHAR_COUNT_UNDER_POINT5_MIN":
        if cand_length > 0 and cand_length <= 0.5 * summary_min_length:
            rule_fired = True

    elif rule == "CHAR_COUNT_UNDER_POINT6_MIN":
        if cand_length > 0 and cand_length <= 0.6 * summary_min_length:
            rule_fired = True

    elif rule == "CHAR_COUNT_UNDER_POINT7_MIN":
        if cand_length > 0 and cand_length <= 0.7 * summary_min_length:
            rule_fired = True

    elif rule == "CHAR_COUNT_UNDER_POINT8_MIN":
        if cand_length > 0 and cand_length <= 0.8 * summary_min_length:
            rule_fired = True

    elif rule == "CHAR_COUNT_UNDER_POINT9_MIN":
        if cand_length > 0 and cand_length <= 0.9 * summary_min_length:
            rule_fired = True

    elif rule == "CHAR_COUNT_OVER_POINT1_MAX":
        if summary_max_length > 0 and cand_length > 0 and cand_length >= 1.1 * summary_max_length:
            rule_fired = True

    elif rule == "CHAR_COUNT_OVER_POINT3_MAX":
        if summary_max_length > 0 and cand_length > 0 and cand_length >= 1.3 * summary_max_length:
            rule_fired = True

    elif rule == "CHAR_COUNT_OVER_POINT5_MAX":
        if summary_max_length > 0 and cand_length > 0 and cand_length >= 1.5 * summary_max_length:
            rule_fired = True

    elif rule == "CHAR_COUNT_OVER_POINT6_MAX":
        if summary_max_length > 0 and cand_length > 0 and cand_length >= 1.6 * summary_max_length:
            rule_fired = True

    elif rule == "CHAR_COUNT_OVER_POINT7_MAX":
        if summary_max_length > 0 and cand_length > 0 and cand_length >= 1.7 * summary_max_length:
            rule_fired = True

    elif rule == "CHAR_COUNT_OVER_POINT8_MAX":
        if summary_max_length > 0 and cand_length > 0 and cand_length >= 1.8 * summary_max_length:
            rule_fired = True

    elif rule == "CHAR_COUNT_OVER_POINT9_MAX":
        if summary_max_length > 0 and cand_length > 0 and cand_length >= 1.9 * summary_max_length:
            rule_fired = True

    elif rule == "SymbolChain":
        if summary_strength > 1 and value_chain_consistent and pytheas.symbol_chain_disagrees(value_symbol_chain, cand_symbol_chain):
            rule_fired = True

    elif rule == "CC":
        if len(case_summary) != 0 and case_summary != cand_case:
            rule_fired = True

    elif rule == "CONSISTENT_NUMERIC":
        # "name":"Below but not here: consistently ONE symbol = D"
        if line_index + 1 in line_agreements.keys() and line_index in line_agreements.keys():
            next_line_agreements = line_agreements[line_index + 1]
            if columnindex in next_line_agreements.keys():
                value_below_agreements = next_line_agreements[columnindex]['agreements']
                if not line_agreements[line_index][columnindex]['null_equivalent'] and rule in value_below_agreements and columnindex in line_agreements[line_index].keys() and rule not in line_agreements[line_index][columnindex]['agreements']:
                    rule_fired = True

    elif rule == "CONSISTENT_D_STAR":
        # "name":"Below but not here: consistently TWO symbols, the first is a digit"
        if line_index + 1 in line_agreements.keys() and line_index in line_agreements.keys():
            next_line_agreements = line_agreements[line_index + 1]
            if columnindex in next_line_agreements.keys():
                value_below_agreements = next_line_agreements[columnindex]['agreements']
                if not line_agreements[line_index][columnindex]['null_equivalent'] and rule in value_below_agreements and columnindex in line_agreements[line_index].keys() and rule not in line_agreements[line_index][columnindex]['agreements']:
                    rule_fired = True

    elif rule == "FW_SUMMARY_D":
        # "name":"Below but not here: two or above symbols in the FW summary, the first is a digit"
        if line_index + 1 in line_agreements.keys() and line_index in line_agreements.keys():
            next_line_agreements = line_agreements[line_index + 1]
            if columnindex in next_line_agreements.keys():
                value_below_agreements = next_line_agreements[columnindex]['agreements']
                if not line_agreements[line_index][columnindex]['null_equivalent'] and rule in value_below_agreements and columnindex in line_agreements[line_index].keys() and rule not in line_agreements[line_index][columnindex]['agreements']:
                    rule_fired = True

    elif rule == "BW_SUMMARY_D":
        # "name":"Below but not here: two or above symbols in the BW summary, the first is a digit"
        if line_index + 1 in line_agreements.keys() and line_index in line_agreements.keys():
            next_line_agreements = line_agreements[line_index + 1]
            if columnindex in next_line_agreements.keys():
                value_below_agreements = next_line_agreements[columnindex]['agreements']
                if not line_agreements[line_index][columnindex]['null_equivalent'] and rule in value_below_agreements and columnindex in line_agreements[line_index].keys() and rule not in line_agreements[line_index][columnindex]['agreements']:
                    rule_fired = True

    elif rule == "BROAD_NUMERIC":
        # "name":"Below but not here: all values digits, optionally have . or ,  or S"
        if line_index + 1 in line_agreements.keys() and line_index in line_agreements.keys():
            next_line_agreements = line_agreements[line_index + 1]
            if columnindex in next_line_agreements.keys():
                value_below_agreements = next_line_agreements[columnindex]['agreements']
                if not line_agreements[line_index][columnindex]['null_equivalent'] and rule in value_below_agreements and columnindex in line_agreements[line_index].keys() and rule not in line_agreements[line_index][columnindex]['agreements']:
                    rule_fired = True

    elif rule == "FW_THREE_OR_MORE_NO_SPACE":
        # "name":"Below but not here: three or above symbols in FW summary that do not contain a  Space"
        if line_index + 1 in line_agreements.keys() and line_index in line_agreements.keys():
            next_line_agreements = line_agreements[line_index + 1]
            if columnindex in next_line_agreements.keys():
                value_below_agreements = next_line_agreements[columnindex]['agreements']
                if not line_agreements[line_index][columnindex]['null_equivalent'] and rule in value_below_agreements and columnindex in line_agreements[line_index].keys() and rule not in line_agreements[line_index][columnindex]['agreements']:
                    rule_fired = True

    elif rule == "BW_THREE_OR_MORE_NO_SPACE":
        # "name":"Below but not here: three or above symbols in BW summary that do not contain a  Space"
        if line_index + 1 in line_agreements.keys() and line_index in line_agreements.keys():
            next_line_agreements = line_agreements[line_index + 1]
            if columnindex in next_line_agreements.keys():
                value_below_agreements = next_line_agreements[columnindex]['agreements']
                if not line_agreements[line_index][columnindex]['null_equivalent'] and rule in value_below_agreements and columnindex in line_agreements[line_index].keys() and rule not in line_agreements[line_index][columnindex]['agreements']:
                    rule_fired = True

    elif rule == "CONSISTENT_SS_NO_SPACE":
        # "name":"Below but not here: consistently at least two symbols in the symbol set summary, none of which are S or _"
        if line_index + 1 in line_agreements.keys() and line_index in line_agreements.keys():
            next_line_agreements = line_agreements[line_index + 1]
            if columnindex in next_line_agreements.keys():
                value_below_agreements = next_line_agreements[columnindex]['agreements']
                if not line_agreements[line_index][columnindex]['null_equivalent'] and rule in value_below_agreements and columnindex in line_agreements[line_index].keys() and rule not in line_agreements[line_index][columnindex]['agreements']:
                    rule_fired = True

    elif rule == "FW_TWO_OR_MORE_OR_MORE_NO_SPACE":
        # "name":"Below but not here: two or above symbols in FW summary that do not contain a Space"
        if line_index + 1 in line_agreements.keys() and line_index in line_agreements.keys():
            next_line_agreements = line_agreements[line_index + 1]
            if columnindex in next_line_agreements.keys():
                value_below_agreements = next_line_agreements[columnindex]['agreements']
                if not line_agreements[line_index][columnindex]['null_equivalent'] and rule in value_below_agreements and columnindex in line_agreements[line_index].keys() and rule not in line_agreements[line_index][columnindex]['agreements']:
                    rule_fired = True

    elif rule == "BW_TWO_OR_MORE_OR_MORE_NO_SPACE":
        # "name":"Below but not here: two or above symbols in BW summary that do not contain a Space"
        if line_index + 1 in line_agreements.keys() and line_index in line_agreements.keys():
            next_line_agreements = line_agreements[line_index + 1]
            if columnindex in next_line_agreements.keys():
                value_below_agreements = next_line_agreements[columnindex]['agreements']
                if not line_agreements[line_index][columnindex]['null_equivalent'] and rule in value_below_agreements and columnindex in line_agreements[line_index].keys() and rule not in line_agreements[line_index][columnindex]['agreements']:
                    rule_fired = True

    return rule_fired


def line_has_null_equivalent(row_values):
    ret = False
    count = 0
    for value in row_values:
        if str(value).strip().lower() in utilities.strictly_null_equivalent:
            count += 1
            ret = True
    return ret, count


def all_numbers(column_symbols):
    for symbols in column_symbols:
        if len(symbols) == 0:
            continue
        if not is_number(symbols):
            return False
    return True


def assess_data_line(row_values):
    data_line_events = []
    fired, times = line_has_null_equivalent(row_values)
    if fired:
        if times == 1:
            data_line_events.append("ONE_NULL_EQUIVALENT_ON_LINE")
        if times >= 2:
            data_line_events.append("NULL_EQUIVALENT_ON_LINE_2_PLUS")

    if aggregation_first_value_of_row(row_values):
        data_line_events.append("AGGREGATION_TOKEN_IN_FIRST_VALUE_OF_ROW")

    if contains_datatype_keyword(row_values):
        data_line_events.append("CONTAINS_DATATYPE_CELL_VALUE")

    return  data_line_events


def assess_non_data_line(row_values, before_data, all_summaries_empty, row_index, dataframe):
    not_data_line_events = []
    left_non_null = 0

    if before_data:
        nulls_seen = False
        row_values = [str(elem).strip().lower()for elem in row_values]

        for value in row_values:
            if not nulls_seen:
                if value not in ['nan', 'none', '', ' ']:
                    left_non_null += 1
                else:
                    nulls_seen = True
            else:
                if value not in ['nan', 'none', '', ' ']:
                    before_data = False
                    break

    if before_data and left_non_null <= 1:
        not_data_line_events.append('UP_TO_FIRST_COLUMN_COMPLETE_CONSISTENTLY')

    if row_index + 1 != dataframe.shape[0] and len(row_values) > 0 and row_values[0].lower() in ['', 'none', 'nan']:
        if len(row_values) > 1:
            DATA = False
            for value_index, value in enumerate(row_values[1:]):
                column = dataframe.iloc[row_index:min(row_index + 5, dataframe.shape[0]), value_index + 1:value_index + 2]
                if str(value).strip().lower() not in ['', 'nan', 'none', 'null'] and column_complete(column):
                    DATA = True
            if not DATA:
                not_data_line_events.append('STARTS_WITH_NULL')
        else:
            not_data_line_events.append('STARTS_WITH_NULL')

    if row_index + 1 != dataframe.shape[0] and all_summaries_empty:
        not_data_line_events.append('NO_SUMMARY_BELOW')

    if len(row_values) > 0 and row_values[0].lower() not in ['', 'none', 'nan'] and len([i for i in row_values if i.lower() not in  ['', 'none', 'nan']]) <= 2:
        for footnote_phrase in pytheas.footnote_keywords:
            if row_values[0].lower().startswith(footnote_phrase):
                not_data_line_events.append("FOOTNOTE")
                break
        if len(row_values) > 0 and row_values[-1].lower() == '#ref!' and "FOOTNOTE" not in not_data_line_events:
            not_data_line_events.append("FOOTNOTE")
    if metadata_header_keywords(row_values):
        not_data_line_events.append("METADATA_TABLE_HEADER_KEYWORDS")

    return not_data_line_events


def aggregation_first_value_of_row(row_values):
    fired = False
    numeric_value_seen = False
    aggregation_keyword_in_first_value = False

    if len(row_values) > 1:
        first_value = str(row_values[0]).strip().lower()
        for aggregation_keyword in ['total']:
            if aggregation_keyword in first_value:
                aggregation_keyword_in_first_value = True
                break
        if aggregation_keyword_in_first_value:
            values = [str(value).strip().lower() for value in row_values[1:]]
            for value in values:
                number = True
                for char in value:
                    if not (char.isdigit() or char == '.' or char == ',' or char == ' '):
                        number = False
                        break
                if number:
                    numeric_value_seen = True
                    break
            if numeric_value_seen:
                fired = True
    return fired


def contains_datatype_keyword(row_values):
    row_values = [str(elem).strip().lower() for elem in row_values]
    if len(row_values) > 1:
        for value in row_values[1:]:
            if value in pytheas.datatype_keywords:
                return True
    return False


def column_complete(column):
    return column.isnull().any().any()


def metadata_header_keywords(row_values):
    row_values = [str(value).strip().lower() for value in row_values]
    for value in row_values:
        if value in pytheas.metadata_table_header_keywords:
            return True
        if value in ['_'.join((value).split(' ')) for value in pytheas.metadata_table_header_keywords]:
            return True
        if value in [''.join((value).split(' ')) for value in pytheas.metadata_table_header_keywords]:
            return True
    return False


def non_nulls_in_line(row_values):
    non_null_percentage = None
    non_nulls = 0
    for value in row_values:
        if value.strip().lower() not in pytheas.null_equivalent_values:
            non_nulls += 1
    if len(row_values) > 0:
        non_null_percentage = non_nulls / len(row_values)
    return non_nulls, non_null_percentage


def discover_aggregation_scope(csv_file, aggregation_rows, cand_subheaders, predicted_subheaders, certain_data_indexes, pytheas_headers):
    subheader_scope = {}
    cand_subhead_indexes = sorted(list(cand_subheaders.keys()))
    last_header_value = ''
    if len(pytheas_headers) > 0:
        last_header_value = str(csv_file.loc[pytheas_headers[-1], :].tolist()[0])

    if len(aggregation_rows) > 0 and len(certain_data_indexes) > 0:
        agg_idxs = sorted(aggregation_rows.keys())
        if len(agg_idxs) > 0 and agg_idxs[0] != certain_data_indexes[0]:

            for aggregation_idx in agg_idxs:
                scope_head_idx = None
                aggregation = aggregation_rows[aggregation_idx]
                aggregation_function = aggregation['aggregation_function']

                aggregation_row = csv_file.loc[[aggregation_idx]].applymap(normalize_numeric)
                candidate_scope = csv_file.loc[[i for i in sorted(certain_data_indexes+cand_subhead_indexes) if i < aggregation_idx and i not in agg_idxs]].applymap(normalize_numeric)

                if aggregation_function == 'sum':
                    for i in range(1, candidate_scope.shape[0] + 1):
                        summed_rows = candidate_scope.loc[candidate_scope.index[-i:]].sum(axis=0, skipna=True)
                        if scope_head_idx is None:
                            if aggregation_row.iloc[0].eq(summed_rows).any():
                                scope_head_idx = candidate_scope.index[-i]
                                if i < candidate_scope.shape[0] and scope_head_idx - 1 not in cand_subhead_indexes:
                                    continue

                        if scope_head_idx is not None:
                            cand_scope_head = scope_head_idx

                            if cand_scope_head in candidate_scope.index:
                                summed_rows = candidate_scope.loc[cand_scope_head:].sum(axis=0, skipna=True)

                            while cand_scope_head >= candidate_scope.index[0] and aggregation_row.iloc[0].eq(summed_rows).any():
                                if cand_scope_head in candidate_scope.index:
                                    summed_rows = candidate_scope.loc[cand_scope_head:].sum(axis=0, skipna=True)

                                cand_scope_head = cand_scope_head - 1

                                if aggregation_row.iloc[0].eq(summed_rows).any():
                                    if cand_scope_head in cand_subhead_indexes:
                                        scope_head_idx = cand_scope_head + 1
                                        break
                                else:
                                    break

                        if scope_head_idx is not None:
                            aggregation_rows[aggregation_idx]['scope'] = 'UP'
                            aggregation_rows[aggregation_idx]['scope_head'] = scope_head_idx
                            aggregation_rows[aggregation_idx]['scope_range'] = list(range(scope_head_idx, aggregation_idx))
                            cand_subheader_rev = list(cand_subhead_indexes)[::-1]
                            for cand_subheader_idx in cand_subheader_rev:
                                if cand_subheader_idx > scope_head_idx:
                                    continue
                                value = cand_subheaders[cand_subheader_idx]
                                if value.lower() in aggregation['aggregation_label'].lower():
                                    aggregation_rows[aggregation_idx]['context_label'] = value
                                    aggregation_rows[aggregation_idx]['subheader'] = cand_subheader_idx
                                    subheader_scope[cand_subheader_idx] = list(range(scope_head_idx, aggregation_idx))
                                    predicted_subheaders.append(cand_subheader_idx)
                                    cand_subheaders.pop(cand_subheader_idx)
                                    cand_subhead_indexes.remove(cand_subheader_idx)
                                    aggregation_rows[aggregation_idx]['scope_head'] = cand_subheader_idx + 1
                                    aggregation_rows[aggregation_idx]['scope_range'] = list(range(cand_subheader_idx + 1, aggregation_idx))
                                    break

                            if 'context_label' not in aggregation_rows[aggregation_idx].keys() and last_header_value != '' and last_header_value.lower() in aggregation["aggregation_label"].lower():
                                aggregation_rows[aggregation_idx]['context_label'] = last_header_value
                                aggregation_rows[aggregation_idx]['scope_head'] = pytheas_headers[-1] + 1
                                aggregation_rows[aggregation_idx]['scope_range'] = list(range(pytheas_headers[-1] + 1, aggregation_idx))

                            if 'subheader' not in aggregation_rows[aggregation_idx].keys() and cand_subhead_indexes is not None and aggregation_rows[aggregation_idx]['scope_head'] - 1 in cand_subhead_indexes:
                                aggregation_rows[aggregation_idx]['subheader'] = aggregation_rows[aggregation_idx]['scope_head'] - 1
                                subheader_scope[aggregation_rows[aggregation_idx]['scope_head'] - 1] = list(range(aggregation_rows[aggregation_idx]['scope_head'], aggregation_idx))
                                if aggregation_rows[aggregation_idx]['scope_head'] - 1 not in predicted_subheaders:
                                    predicted_subheaders.append(aggregation_rows[aggregation_idx]['scope_head'] - 1)
                                if aggregation_rows[aggregation_idx]['scope_head'] - 1 in cand_subhead_indexes:
                                    cand_subhead_indexes.remove(aggregation_rows[aggregation_idx]['scope_head'] - 1)
                                    if aggregation_rows[aggregation_idx]['scope_head'] - 1 in cand_subheaders.keys():
                                        cand_subheaders.pop(aggregation_rows[aggregation_idx]['scope_head'] - 1)
                                aggregation_rows[aggregation_idx]['context_label'] = csv_file.loc[aggregation_rows[aggregation_idx]['scope_head'] - 1].tolist()[0]

                            if 'context_label' not in aggregation_rows[aggregation_idx].keys():
                                aggregation_rows[aggregation_idx]['context_label'] = aggregation['aggregation_label']
                                cand_subheader_rev = list(cand_subhead_indexes)[::-1]
                                for cand_subheader_idx in cand_subheader_rev:
                                    if cand_subheader_idx > scope_head_idx:
                                        continue
                                    value = cand_subheaders[cand_subheader_idx]
                                    if value.lower() in aggregation['aggregation_label'].lower():
                                        aggregation_rows[aggregation_idx]['context_label'] = value
                                        aggregation_rows[aggregation_idx]['subheader'] = cand_subheader_idx
                                        subheader_scope[cand_subheader_idx] = list(range(scope_head_idx, aggregation_idx))
                                        predicted_subheaders.append(cand_subheader_idx)
                                        cand_subheaders.pop(cand_subheader_idx)
                                        cand_subhead_indexes.remove(cand_subheader_idx)
                                        aggregation_rows[aggregation_idx]['scope_head'] = cand_subheader_idx + 1
                                        aggregation_rows[aggregation_idx]['scope_range'] = list(range(cand_subheader_idx + 1, aggregation_idx))
                                        break

                                if 'context_label' not in aggregation_rows[aggregation_idx].keys() and last_header_value != '' and last_header_value.lower() in aggregation["aggregation_label"].lower():
                                    aggregation_rows[aggregation_idx]['context_label'] = last_header_value
                                    aggregation_rows[aggregation_idx]['scope_head'] = pytheas_headers[-1] + 1
                                    aggregation_rows[aggregation_idx]['scope_range'] = list(range(pytheas_headers[-1] + 1, aggregation_idx))

                            for di in aggregation_rows[aggregation_idx]['scope_range']:
                                if di in cand_subhead_indexes:
                                    cand_subhead_indexes.remove(di)
                                if di in cand_subheaders.keys():
                                    cand_subheaders.pop(di)
                            break

                        if cand_subhead_indexes is not None and candidate_scope.index[-i] - 1 in cand_subhead_indexes:
                            scope_head_idx = candidate_scope.index[-i]
                            candidate_subheader_value = str(csv_file.loc[scope_head_idx - 1].tolist()[0])
                            if candidate_subheader_value.lower() in aggregation['aggregation_label'].lower():
                                aggregation_rows[aggregation_idx]['subheader'] = scope_head_idx - 1
                                subheader_scope[scope_head_idx - 1] = list(range(scope_head_idx, aggregation_idx))

                                if scope_head_idx - 1 not in predicted_subheaders:
                                    predicted_subheaders.append(scope_head_idx - 1)

                                if scope_head_idx - 1 in cand_subhead_indexes:
                                    aggregation_rows[aggregation_idx]['scope'] = 'UP'
                                    aggregation_rows[aggregation_idx]['scope_head'] = scope_head_idx
                                    aggregation_rows[aggregation_idx]['scope_range'] = list(range(scope_head_idx, aggregation_idx))
                                    for di in aggregation_rows[aggregation_idx]['scope_range']:
                                        if di in cand_subhead_indexes:
                                            cand_subhead_indexes.remove(di)
                                        if di in cand_subheaders.keys():
                                            cand_subheaders.pop(di)

                                    certain_data_indexes = list(set(certain_data_indexes) - set(aggregation_rows[aggregation_idx]['scope_range']))
                                    cand_subhead_indexes.remove(scope_head_idx - 1)
                                    if scope_head_idx - 1 in cand_subheaders.keys():
                                        cand_subheaders.pop(scope_head_idx - 1)
                                aggregation_rows[aggregation_idx]['context_label'] = csv_file.loc[scope_head_idx - 1].tolist()[0]
                                break

    return aggregation_rows, certain_data_indexes, predicted_subheaders, cand_subhead_indexes, subheader_scope


def normalize_numeric(value):
    value = str(value).strip()
    if any(char.isdigit() for char in value) and not any(char.isalpha() for char in value):

        while not value[0].isdigit() and value[0] != '-':
            value = value[1:]

        if value.endswith('%'):
            value = value.replace('$', '')

        for i, char in enumerate(value):
            if not char.isdigit() and char not in [',', ' ', '.']:
                if not(i == 0 and char == '-'):
                    value = ''
                    break
        if value != '':
            while ',' in value:
                value = value.replace(',', '')
            while '.' in value:
                value = value.replace('.', '')
            while ' ' in value:
                value = value.replace(' ', '')
            value = int(value)
        else:
            value = np.nan
    else:
        value = np.nan
    return value


def contains_number(line):
    for value in line:
        _, symbols, _, _, _ = pytheas.generate_pattern_symbols_and_case(str(value).strip(), True)
        if symbols != set() and symbols.issubset({'D', '.', ',', 'S', '-', '+', '~'}):
            return True
    return False
