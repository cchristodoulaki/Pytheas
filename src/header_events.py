from unidecode import unidecode

import string_utils

import nb_utilities as nb_util
import utilities


def collect_arithmetic_events_on_row(row_values):
    events = []

    fired, times = integer_sequence_adjacent(row_values)
    if fired:
        if times >= 6:
            events.append("ADJACENT_ARITHMETIC_SEQUENCE_6_plus")
            # input(row_values)
        elif times == 5:
            events.append("ADJACENT_ARITHMETIC_SEQUENCE_5")
        elif times == 4:
            events.append("ADJACENT_ARITHMETIC_SEQUENCE_4")
        elif times == 3:
            events.append("ADJACENT_ARITHMETIC_SEQUENCE_3")
        elif times == 2:
            events.append("ADJACENT_ARITHMETIC_SEQUENCE_2")

    return events


def header_row_with_aggregation_tokens(row_values, arithmetic_sequence_fired):
    header_row_with_aggregation_tokens_rules_fired = []
    if aggregation_on_row_wo_numeric(row_values):
        header_row_with_aggregation_tokens_rules_fired.append("AGGREGATION_ON_ROW_WO_NUMERIC")
    if aggregation_on_row_w_arith_sequence(row_values, arithmetic_sequence_fired):
        header_row_with_aggregation_tokens_rules_fired.append("AGGREGATION_ON_ROW_W_ARITH_SEQUENCE")

    return header_row_with_aggregation_tokens_rules_fired


def collect_events_on_row(row_values):
    events = []
    fired, times = range_pairs_on_row(row_values)
    if fired:
        if times >= 2:
            events.append("RANGE_PAIRS_2_plus")
        else:
            events.append("RANGE_PAIRS_" + str(times))

    fired, block_size = partially_repeating_values_on_row(row_values)
    if fired:
        if block_size >= 2:
            events.append("PARTIALLY_REPEATING_VALUES_length_2_plus")
        else:
            events.append("PARTIALLY_REPEATING_VALUES_length_1")

    if metadata_like_row(row_values):
        # row has no digits, at least one value enclosed by parenthesis or contains currency sign
        events.append("METADATA_LIKE_ROW")

    if consistently_slug_or_snake(row_values):
        events.append("CONSISTENTLY_SLUG_OR_SNAKE")

    if consistently_upper_case(row_values):
        events.append("CONSISTENTLY_UPPER_CASE")

    return events


def aggregation_on_row_w_arith_sequence(row_values, arithmetic_sequence_fired):
    aggregation_column_seen = False
    if arithmetic_sequence_fired:
        values = [str(value).strip().lower() if str(value).strip().lower() not in ['nan', 'none', 'null'] else '' for value in row_values]
        for value in values:
            for aggregation_keyword in ['total']:
                if aggregation_keyword in value:
                    aggregation_column_seen = True
                    break
            if  aggregation_column_seen:
                break
        if  aggregation_column_seen:
            return True
    return False


def integer_sequence_adjacent(row_values):
    event_occurred = False
    step_count = None
    numeric_values = []
    sequential_values_list = []
    sequential_values = []
    sequence_found = False
    if len(row_values) >= 2:
        for value in row_values:
            # generate symbols NOT sensitive to outliers
            _, symbols, _, _, _ = utilities.generate_pattern_symbols_and_case(str(value).strip(), False)
            if symbols == set(['D']):
                numeric_values.append(int(value))
            else:
                numeric_values.append(None)
        for i in range(len(numeric_values) - 1):
            if(numeric_values[i] is not None and numeric_values[i + 1] is not None and numeric_values[i] + 1 == numeric_values[i + 1]):
                sequential_values += [numeric_values[i]]
                sequence_found = True
            else:
                if sequence_found:
                    if i > 0 and numeric_values[i - 1] + 1 == numeric_values[i] and numeric_values[i] is not None:
                        sequential_values += [numeric_values[i]]
                    sequential_values_list.append(sequential_values)
                    sequence_found = False
                sequential_values = []

        if sequence_found:
            if numeric_values[-2] + 1 == numeric_values[-1] and numeric_values[-1] is not None:
                sequential_values += [numeric_values[-1]]
            sequential_values_list.append(sequential_values)

        if len(sequential_values_list) > 0:
            event_occurred = True
            _, max_sequence_length = FindMaxLength(sequential_values_list)
            if max_sequence_length > 1:
                event_occurred = True
                step_count = max_sequence_length - 1

    return event_occurred, step_count


def aggregation_on_row_wo_numeric(row_values):
    values = [str(value).strip().lower() if str(value).strip().lower() not in ['nan', 'none', 'null'] else '' for value in row_values]
    numeric_value_seen = False
    aggregation_column_seen = False

    for value in values:
        for aggregation_keyword in ['total']:
            if aggregation_keyword in value:
                aggregation_column_seen = True

        is_number = True
        for char in value:
            if char.isdigit() or char == '.' or char == ',' or char == ' ' or (len(value) > 1 and value.startswith('-') and value[1].isdigit()):
                continue
            is_number = False
            break
        if is_number:
            numeric_value_seen = True
    return aggregation_column_seen and not numeric_value_seen


def consistently_slug_case(row_values):
    slug_case_seen = False
    fired = True
    for value in row_values:
        if ' ' in unidecode(value).strip():
            fired = False
            break
        if '-' in unidecode(value).strip() and string_utils.is_slug(unidecode(value).strip().lower()):
            slug_case_seen = True
        else:
            if not unidecode(value).strip().isalpha():
                fired = False
                break

    if not slug_case_seen:
        return False
    return fired


def consistently_snake_case(row_values):
    snake_case_seen = False
    fired = True
    for value in row_values:
        if ' ' in unidecode(value).strip():
            fired = False
            break
        if not string_utils.is_snake_case(unidecode(value).strip().lower()):
            if not unidecode(value).strip().isalpha():
                fired = False
                break
        else:
            snake_case_seen = True

    if not snake_case_seen:
        return False
    return fired


def consistently_slug_or_snake(row_values):
    values = [str(value).strip()  if str(value).lower() not in ['nan', 'none'] else '' for value in row_values]

    if consistently_snake_case(values) or consistently_slug_case(values):
        return True

    return False


def range_pairs_on_row(row_values):
    range_attributes_counted = 0
    range_pair_event_occurred = False
    range_pair_attributes_counted = 0
    row_value_ranges = []

    for value in row_values:
        numeric_tokens_new, _ = nb_util.discover_tokens(value)
        range_tokens, _, _ = nb_util.discover_range_tokens(value, numeric_tokens_new)
        row_value_ranges.append(range_tokens)

        if len(range_tokens) > 0:
            range_attributes_counted += 1

    # check for pairs
    for idx, i in enumerate(row_value_ranges):
        if idx + 1 < len(row_value_ranges):
            part1 = i
            part2 = row_value_ranges[idx + 1]
            # check that both values had range tokens
            if len(part1) > 0 and len(part2) > 0:
                part1_rules = [t[1] for t in part1]
                part2_rules = [t[1] for t in part2]
                #check rule overlap
                if any(i in part1_rules for i in part2_rules):
                    range_pair_attributes_counted += 1
                    range_pair_event_occurred = True

    return range_pair_event_occurred, range_pair_attributes_counted


def partially_repeating_values_on_row(row_values):
    event_occurred = False
    value_set = set(row_values)
    repeating_value_seen = False
    condition_failed = False
    repeating_lengths = []
    repeating_seen_list = []

    #for each distinct value
    for value in value_set:
        if value is None or value.strip().lower() in ['', ' ', 'nan', 'none', 'null'] or  value.replace('.', '', 1).isdigit():
            continue

        repeating_seen = 0
        value_idxs = [i for i, val in enumerate(row_values) if val == value]
        repeating_length = get_num_repeating_values(value_idxs)
        start_idx = 0
        while start_idx+repeating_length < len(value_idxs):
            start_idx = start_idx + repeating_length
            next_repeating_length = get_num_repeating_values(value_idxs[start_idx:])
            if repeating_length != next_repeating_length:
                #it must be repeated with the same length  at least once to be valid
                if repeating_seen < 1:
                    condition_failed = True

                break

            repeating_seen += 1
            repeating_value_seen = True

        repeating_seen_list.append(repeating_seen)
        repeating_lengths.append(repeating_length)
    if len(repeating_seen_list) > 0:
        max_repeats = max(repeating_seen_list)
        if not condition_failed and repeating_value_seen and max(repeating_lengths) > 0  and max_repeats >= 2 and repeating_seen_list.count(max_repeats) >= 2:
            event_occurred = True
    else:
        max_repeats = 0
        event_occurred = False
    return event_occurred, repeating_seen_list.count(max_repeats)


def metadata_like_row(row_values):
    if not any(char.isdigit() for char in ''.join(row_values)):
        for value in row_values:
            if '$' in str(value) or '%' in str(value):
                return True
            if len(value) > 0 and str(value)[0] == '(' and str(value)[-1] == ')':
                return True
    return False


def consistently_upper_case(row_values):
    values = [str(value).strip() if str(value).lower() not in ['nan', 'none'] else '' for value in row_values]
    for value in values:
        if not unidecode(value).strip().isupper() or value == '':
            return False
    return True


def FindMaxLength(lst):
    maxList = max((x) for x in lst)
    maxLength = max(len(x) for x in lst)

    return maxList, maxLength


def get_num_repeating_values(value_idxs):
    seq_idxs = value_idxs[0:1]
    for i in value_idxs[1:]:
        if seq_idxs[-1] + 1 == i:
            seq_idxs.append(i)
        else:
            break
    return len(seq_idxs)
