import string


strictly_null_equivalent = ['nil', 'data not available', 'not available', 'no data', 'no answer', 'nd',
                            'na', 'n/d', 'n/a', '#n/a', 'n.a.', 'not applicable', 'sans objet', 's.o.',
                            'so', 's/o', '-', '--', '.', '..', '...', '*', 'void', '0', 'redacted',
                            'confidential', 'confidentiel', 'unknown', 'inconnu', '?']

null_equivalent_values = ['', 'null', 'nan', 'none'] + strictly_null_equivalent


def generate_pattern_symbols_and_case(value, outlier_sensitive):
    if value is None or str(value).strip().lower() in null_equivalent_values:
        value = ''

    value_pattern = []
    value_symbols = set()
    try:
        if value.isupper():
            value_case = 'ALL_CAPS'
        elif value.islower():
            value_case = 'ALL_LOW'
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
            while i < (len(value)) and value[i].isdigit():
                i += 1
                digit_counter += 1
            value_pattern.append(['D', digit_counter])
            value_symbols.add('D')

        # Punctuation: !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~
        elif i < (len(value)) and value[i] in string.punctuation:
            punctuation_counter = 0
            punctuation = value[i]
            while i < (len(value)) and value[i] == punctuation:
                i += 1
                punctuation_counter += 1
            value_pattern.append([punctuation, punctuation_counter])
            value_symbols.add(punctuation)

        elif i < (len(value)):
            unknown_counter = 0
            unknown = value[i]
            while i < (len(value)) and value[i] == unknown:
                i += 1
                unknown_counter += 1
            value_pattern.append([unknown, unknown_counter])
            value_symbols.add(unknown)

        else:
            i += 1
    return value_pattern, value_symbols, value_case, value_tokens, value_characters
