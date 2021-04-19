import sys
import pytheas.utilities as utilities
import pytheas.nb_utilities as nb_util
import string_utils
from unidecode import unidecode

def collect_arithmetic_events_on_row(row_values):
    events = []
    # input(row_values)
    
    # fired,times = arithmetic_sequence_non_adjacent(row_values)
    # if fired == True:
    #     if times>=6:
    #         events.append("NON_ADJACENT_ARITHMETIC_SEQUENCE_6_plus")
    #     elif times == 5:
    #         events.append("NON_ADJACENT_ARITHMETIC_SEQUENCE_5")
    #     elif times == 4:
    #         events.append("NON_ADJACENT_ARITHMETIC_SEQUENCE_4")  
    #     elif times == 3:
    #         events.append("NON_ADJACENT_ARITHMETIC_SEQUENCE_3")              

    fired,times = integer_sequence_adjacent(row_values)
    if fired == True:
        if times>=6:
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


def arithmetic_sequence_adjacent(row_values,step_count_k=2):
    event_occurred = False
    step_increment = None
    step_count = None
    sample_symbols = {}
    for value_idx, value in enumerate(row_values):
        _, symbols, _,_, _ = utilities.generate_pattern_symbols_and_case(str(value).strip(), False)
        if value_idx not in sample_symbols.keys():
            sample_symbols[value_idx]=[]
        sample_symbols[value_idx].append(symbols)

    # example
    # row_values=['1','2001','2002','2003','2004']
    # sample_symbols={0:[set(['D'])],1:[set(['D'])],2:[set(['D'])], 3:[set(['D'])]}
    step_increment, step_count = utilities.discover_incremental_values(row_values, sample_symbols)
    # print('step_increment='+str(step_increment))
    # print('step_count='+str(step_count))
    # print('step_count_k threshold='+str(step_count_k))
    if step_increment!=0 and step_count>step_count_k:
        # INCREMENTAL RULE FIRES
        event_occurred= True
    # print('event_occurred='+str(event_occurred))

    return event_occurred,step_count
    
def arithmetic_sequence_non_adjacent(row_values,step_count_k=2):
    event_occurred = False
    step_increment = None
    step_count = None
    sample_symbols = {}
    for value_idx, value in enumerate(row_values):
        _, symbols, _,_, _ = utilities.generate_pattern_symbols_and_case(str(value).strip(), False)
        if value_idx not in sample_symbols.keys():
            sample_symbols[value_idx]=[]
        sample_symbols[value_idx].append(symbols)

    # example
    # row_values=['1','2001','2002','2003','2004']
    # sample_symbols={0:[set(['D'])],1:[set(['D'])],2:[set(['D'])], 3:[set(['D'])]}
    step_increment, step_count = utilities.discover_incremental_values_at_least_one_nonadjacent(row_values, sample_symbols)
    # print('step_increment='+str(step_increment))
    # print('step_count='+str(step_count))
    # print('step_count_k threshold='+str(step_count_k))
    if step_increment!=0 and step_count>step_count_k:
        # INCREMENTAL RULE FIRES
        event_occurred= True
    # print('event_occurred='+str(event_occurred))

    return event_occurred,step_count



def collect_new_rules(row_values):
    events= set()
    if any(char.isdigit() for char in ''.join(row_values))==False:
        for value in row_values:
            if '$' in str(value):
                events.add('DOLLAR_ON_LINE_NO_DIGITS')
            if '%' in str(value):
                events.add('MODULO_ON_LINE_NO_DIGITS')
            if len(value)>0 and str(value)[0]=='(' and str(value)[-1]==')':
                events.add('NO_DIGITS_VALUES_IN_PARENTHESIS')
    return list(events)



def collect_events_on_row(row_values):
    events = []
    fired, times = range_pairs_on_row(row_values)
    if fired == True:
        if times>=2:
            events.append("RANGE_PAIRS_2_plus")
        else:
            events.append("RANGE_PAIRS_"+str(times))

    fired, block_size = partially_repeating_values_on_row(row_values)
    if fired == True:
        if block_size>=2:
            events.append("PARTIALLY_REPEATING_VALUES_length_2_plus")
        else:
            events.append("PARTIALLY_REPEATING_VALUES_length_1")
    
    fired = metadata_like_row(row_values)# row has no digits, at least one value enclosed by parenthesis or contains currency sign
    if fired == True:
        events.append("METADATA_LIKE_ROW")
    
    fired = consistently_slug_or_snake(row_values)
    if fired == True:
        events.append("CONSISTENTLY_SLUG_OR_SNAKE")

    if consistently_upper_case(row_values):
        events.append("CONSISTENTLY_UPPER_CASE")

    return events

def consistently_upper_case(row_values):
    fired = True
    values = [str(value).strip()  if str(value).lower() not in ['nan', 'none'] else '' for value in row_values]
    for value in values:
        if unidecode(value).strip().isupper()==False or value=='':
            fired = False
            break
    return fired 

def metadata_like_row(row_values):
    event_occurred = False
    if any(char.isdigit() for char in ''.join(row_values))==False:
        for value in row_values:
            if '$' in str(value) or '%' in str(value):
                event_occurred = True
                break
            if len(value)>0 and str(value)[0]=='(' and str(value)[-1]==')':
                event_occurred = True
                break
    return event_occurred

def FindMaxLength(lst): 
    maxList = max((x) for x in lst) 
    maxLength = max(len(x) for x in lst ) 
  
    return maxList, maxLength

def integer_sequence_adjacent(row_values):
    event_occurred = False
    step_increment = None
    step_count = None
    sample_symbols = {}
    numeric_values= []
    sequential_values_list = []
    sequential_values = []
    sequence_found = False
    if len(row_values)>=2:
        for value_idx, value in enumerate(row_values):
            # generate symbols NOT sensitive to outliers
            _, symbols, _,_, _ = utilities.generate_pattern_symbols_and_case(str(value).strip(), False)
            if symbols==set(['D']):
                numeric_values.append(int(value))
            else:
                numeric_values.append(None)
        # input('numeric_values='+str(numeric_values))
        for i in range(len(numeric_values)-1):
            if(numeric_values[i]!=None and numeric_values[i+1]!=None and numeric_values[i]+1 == numeric_values[i+1]):
                sequential_values += [numeric_values[i]]
                sequence_found=True
            else:
                if sequence_found==True:
                    if i>0 and numeric_values[i-1]+1 == numeric_values[i] and numeric_values[i]!=None:
                        sequential_values += [numeric_values[i]]
                    sequential_values_list.append(sequential_values)
                    sequence_found = False
                sequential_values = []

        if sequence_found==True:            
            if numeric_values[-2]+1 == numeric_values[-1] and numeric_values[-1]!=None:
                sequential_values += [numeric_values[-1]]
            sequential_values_list.append(sequential_values)

        # input('sequential_values_list='+str(sequential_values_list))
        if len(sequential_values_list)>0:
            event_occurred = True        
            max_sequence, max_sequence_length = FindMaxLength(sequential_values_list)
            if max_sequence_length>1:
                event_occurred = True
                step_count = max_sequence_length-1

    return event_occurred,step_count 

# ['13', '2', '6', '999', 'OUT/EXT', 'West Isles', '710', '25984000', '50', '50', '70', '90', '60', '60', '60', '60', '50', '30', '20', '20', '30', '20', '10', '10', '30']
# ['1','2001','2002','2003','2004']
# ['1','2001','2004','2002','2003']
# ['1','2001','January','2002','January','2003']
def incremental_on_row(row_values,step_count_k=2):
    event_occurred = False
    step_increment = None
    step_count = None
    sample_symbols = {}
    for value_idx, value in enumerate(row_values):
        _, symbols, _,_, _ = utilities.generate_pattern_symbols_and_case(str(value).strip(), False)
        if value_idx not in sample_symbols.keys():
            sample_symbols[value_idx]=[]
        sample_symbols[value_idx].append(symbols)

    # example
    # row_values=['1','2001','2002','2003','2004']
    # sample_symbols={0:[set(['D'])],1:[set(['D'])],2:[set(['D'])], 3:[set(['D'])]}
    step_increment, step_count = utilities.discover_incremental_values(row_values, sample_symbols)
    # print('step_increment='+str(step_increment))
    # print('step_count='+str(step_count))
    # print('step_count_k threshold='+str(step_count_k))
    if step_increment!=0 and step_count>step_count_k:
        # INCREMENTAL RULE FIRES
        event_occurred= True
    # print('event_occurred='+str(event_occurred))

    return event_occurred,step_count 

def aggregation_on_row_wo_numeric(row_values):
    values = [str(value).strip().lower() if str(value).strip().lower() not in ['nan', 'none', 'null'] else '' for value in row_values]
    numeric_value_seen = False
    aggregation_column_seen = False
    fired = False

    for value in values:
        for aggregation_keyword in ['total']:#, 'average', 'agv', 'mean', 'percentage', '(%)', 'difference'
            if aggregation_keyword in value:
                aggregation_column_seen = True

        is_number = True
        for char in value:
            if char.isdigit() or char=='.' or char==',' or char==' ' or (len(value)>1 and value.startswith('-') and value[1].isdigit()):
                continue
            else:
                is_number=False
                break
        if is_number:    
            numeric_value_seen = True
    if aggregation_column_seen==True and numeric_value_seen==False:
        fired = True
    return fired

def aggregation_on_row_wo_numeric_trial(row_values):
    values = [str(value).strip().lower() if str(value).strip().lower() not in ['nan', 'none', 'null'] else '' for value in row_values]
    numeric_value_seen = False
    aggregation_column_seen = False
    fired = False

    for value in values:
        for aggregation_keyword in ['total']:#, 'average', 'agv', 'mean', 'percentage', '(%)', 'difference'
            if aggregation_keyword in value:
                aggregation_column_seen = True
                continue

        if any(i.isdigit() for i in value):
            numeric_value_seen=True

    if aggregation_column_seen==True and numeric_value_seen==False:
        fired = True
    return fired


def aggregation_on_row_w_arith_sequence(row_values, arithmetic_sequence_fired): 
    fired = False
    aggregation_column_seen = False
    if arithmetic_sequence_fired == True:
        values = [str(value).strip().lower() if str(value).strip().lower() not in ['nan', 'none', 'null'] else '' for value in row_values]
        for value in values:
            for aggregation_keyword in ['total']:#, 'average', 'agv', 'mean', 'percentage', '(%)', 'difference'
                if aggregation_keyword in value:
                    aggregation_column_seen = True
                    break
            if  aggregation_column_seen == True:
                break
        if  aggregation_column_seen == True:
            fired=True
    return fired      

def multiple_aggregation_values_on_row(row_values):
    fired = False
    aggregation_column_count=0

    values = [str(value).strip().lower() if str(value).strip().lower() not in ['nan', 'none', 'null'] else '' for value in row_values]
    for value in values:
        for aggregation_keyword in ['total']:#, 'average', 'agv', 'mean', 'percentage', '(%)', 'difference'
            if aggregation_keyword in value:
                aggregation_column_count+=1
                if aggregation_column_count>1:
                    fired = True
                    break
        if fired == True:
            break
    return fired


def header_row_with_aggregation_tokens(row_values, arithmetic_sequence_fired):
    header_row_with_aggregation_tokens_rules_fired = []
    if aggregation_on_row_wo_numeric(row_values):
        header_row_with_aggregation_tokens_rules_fired.append("AGGREGATION_ON_ROW_WO_NUMERIC")
    if aggregation_on_row_w_arith_sequence(row_values,arithmetic_sequence_fired):
        header_row_with_aggregation_tokens_rules_fired.append("AGGREGATION_ON_ROW_W_ARITH_SEQUENCE")
    # if multiple_aggregation_values_on_row(row_values):
    #     header_row_with_aggregation_tokens_rules_fired.append("MULTIPLE_AGGREGATION_VALUES_ON_ROW")

    return header_row_with_aggregation_tokens_rules_fired



def consistently_title_case(row_values):
    fired = True
    title_case_seen = False
    for value in row_values:
        if unidecode(value).strip().istitle()==False:
            fired = False
            break
        else:
            title_case_seen = True
    if title_case_seen ==False:
        fired = False             
    return fired

def consistently_slug_case(row_values):
    fired = True
    slug_case_seen = False
    for value in row_values:
        if ' ' in unidecode(value).strip():
            fired = False
            break
        if '-' in unidecode(value).strip() and string_utils.is_slug(unidecode(value).strip().lower())== True:
            slug_case_seen = True
        else:
            if unidecode(value).strip().isalpha()==False:
                fired = False
                break
            
    if slug_case_seen == False:
        fired = False        
    return fired 

def consistently_snake_case(row_values):
    fired = True
    snake_case_seen = False
    for value in row_values:       
        if ' ' in unidecode(value).strip():
            fired = False
            break
        if string_utils.is_snake_case(unidecode(value).strip().lower())==False:
            if unidecode(value).strip().isalpha()==False:
                fired = False
                break
        else:
            snake_case_seen = True
            
    if snake_case_seen == False:
        fired = False        
    return fired
def consistently_slug_or_snake(row_values):
    fired = False
    values = [str(value).strip()  if str(value).lower() not in ['nan', 'none'] else '' for value in row_values]
    
    if consistently_snake_case(values) or consistently_slug_case(values):
        fired = True

    return fired

def range_pairs_on_row(row_values):
    range_attributes_counted = 0
    range_pair_event_occurred = False
    range_pair_attributes_counted = 0
    row_value_ranges = []

    for value_idx, value in enumerate(row_values):
        numeric_tokens_new, non_numeric_tokens = nb_util.discover_tokens(value)
        range_tokens, numeric_tokens, remaining_tokens = nb_util.discover_range_tokens(value, numeric_tokens_new)
        row_value_ranges.append(range_tokens)

        if len(range_tokens)>0:
            range_attributes_counted+=1
    
    # check for pairs
    for idx,i in enumerate(row_value_ranges):
        if idx+1<len(row_value_ranges):
            part1 = i
            part2 = row_value_ranges[idx+1]
            # check that both values had range tokens
            if len(part1)>0 and len(part2)>0:
                part1_rules = [t[1] for t in part1]
                part2_rules = [t[1] for t in part2]
                #check rule overlap
                if any(i in part1_rules for i in part2_rules):
                    range_pair_attributes_counted+=1
                    range_pair_event_occurred= True

    return range_pair_event_occurred,range_pair_attributes_counted

def get_num_repeating_values(value_idxs):
    seq_idxs = value_idxs[0:1]
    for i in value_idxs[1:]:
        if seq_idxs[-1]+1==i:
            seq_idxs.append(i)
        else:
            break
    return len(seq_idxs)

def repeating_set_on_row(row_values):
    event_occurred = False
    for idx,value in enumerate(row_values):    
        if value in row_values[idx+1:]:
            # print('row_values[idx+1:]='+str(row_values[idx+1:]))
            next_idx = idx+row_values[idx+1:].index(value)
            # print('next_idx='+str(next_idx))
            remaining = row_values[next_idx:]
            # print('remaining='+str(remaining))
            sliced = remaining[:next_idx-idx]
            if next_idx>idx+1 and sliced==row_values[idx:next_idx]:
                #  RULE FIRES
                # input(row_values)
                event_occurred = True

    return event_occurred

def repeating_values_on_row(row_values):
    # print(row_values)
    event_occurred = False
    value_set = set(row_values)
    # print(value_set)
    repeating_value_seen = False
    condition_failed = False
    repeating_lengths = []
    repeating_seen_list = []

    for value in value_set:
        # print(value)
        if value == None or value.strip() == "":
            continue

        repeating_seen = 0
        if condition_failed:
            repeating_seen_list.append(repeating_seen)
            break
    
        value_idxs = [i for i,val in enumerate(row_values) if val==value]
        # print("value_idxs="+str(value_idxs))
        repeating_length = get_num_repeating_values(value_idxs)
        # print("repeating_length="+str(repeating_length))
        start_idx=0
        while start_idx+repeating_length<len(value_idxs):
            start_idx = start_idx+repeating_length
            next_repeating_length = get_num_repeating_values(value_idxs[start_idx:])
            # print("next_repeating_length="+str(next_repeating_length))
            if repeating_length!=next_repeating_length:
                condition_failed = True
                break
            else:
                repeating_seen+=1
                if repeating_seen>0:
                    repeating_value_seen = True
        repeating_seen_list.append(repeating_seen)
        repeating_lengths.append(repeating_length)

    if condition_failed==False and repeating_value_seen and max(repeating_lengths)>0  and max(repeating_seen_list)>1 and len(set(repeating_seen_list))==1:
        event_occurred = True
        # print("\n---REPEATING VALUES")
        # print(repeating_seen_list)
        # print(repeating_lengths)
        # input(row_values)

    return event_occurred

# TEST EXAMPLES:
    # row_values = [None, 'Number of\nRequests','Pages Disclosed',None,'Number of\nRequests','Pages Disclosed','Number of\nRequests','Pages Disclosed','Number of\nRequests','Pages Disclosed']
    # row_values = ['nan', 'Number of\nRequests', 'Pages Disclosed', 'January','Number of\nRequests', 'Pages Disclosed','February', 'Number of\nRequests', 'Pages Disclosed','March']

def partially_repeating_values_on_row(row_values):
    event_occurred = False
    value_set = set(row_values)
    repeating_value_seen = False
    condition_failed = False
    repeating_lengths = []
    repeating_seen_list = []

    #for each distinct value
    for value in value_set:
        if value == None or value.strip().lower() in ['',' ','nan','none','null'] or  value.replace('.','',1).isdigit():
            continue

        repeating_seen = 0    
        value_idxs = [i for i,val in enumerate(row_values) if val==value]
        repeating_length = get_num_repeating_values(value_idxs)
        start_idx=0
        while start_idx+repeating_length<len(value_idxs):
            start_idx = start_idx+repeating_length
            next_repeating_length = get_num_repeating_values(value_idxs[start_idx:])
            if repeating_length!=next_repeating_length:
                #it must be repeated with the same length  at least once to be valid 
                if repeating_seen<1:
                    condition_failed = True
                break
            else:
                repeating_seen+=1
                repeating_value_seen = True

        repeating_seen_list.append(repeating_seen)
        repeating_lengths.append(repeating_length)
    if len(repeating_seen_list)>0:
        max_repeats = max(repeating_seen_list)
        if condition_failed == False and repeating_value_seen and max(repeating_lengths)>0  and max_repeats>=2 and repeating_seen_list.count(max_repeats)>=2:
            event_occurred = True
    else:
        max_repeats= 0
        event_occurred = False
    return event_occurred, repeating_seen_list.count(max_repeats)
