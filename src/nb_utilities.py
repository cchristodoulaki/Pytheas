import re
import string

import pandas as pd


REGX = re.compile(r'(?<![\d.])(?!\.\.)(?<![\d.][eE][+-])(?<![\d.][eE])(?<!\d[.,])'
                  r'' #---------------------------------
                  r'([+-]?)'
                  r'(?![\d,]*?\.[\d,]*?\.[\d,]*?)'
                  r'(?:0|,(?=0)|(?<!\d),)*'
                  r'(?:'
                  r'((?:\d(?!\.[1-9])|,(?=\d))+)[.,]?'
                  r'|\.(0)'
                  r'|((?<!\.)\.\d+?)'
                  r'|([\d,]+\.\d+?))'
                  r'0*'
                  r'' #---------------------------------
                  r'(?:'
                  r'([eE][+-]?)(?:0|,(?=0))*'
                  r'(?:'
                  r'(?!0+(?=\D|\Z))((?:\d(?!\.[1-9])|,(?=\d))+)[.,]?'
                  r'|((?<!\.)\.(?!0+(?=\D|\Z))\d+?)'
                  r'|([\d,]+\.(?!0+(?=\D|\Z))\d+?))'
                  r'0*'
                  r')?'
                  r'' #---------------------------------
                  r'(?![.,]?\d)'
                  )


def discover_tokens(input_value):
    value = input_value
    non_numeric_tokens = []
    numeric_tokenpairs = []
    numeric_tokens = []
    numeric_tokens_new = []

    try:
        if not pd.isna(input_value) and input_value is not None:
            for strmatch, modified, _ in dzs_numbs2(input_value):
                if strmatch is not None:
                    numeric_tokenpairs.append((strmatch, modified))
                    numeric_tokens.append(strmatch)

            last_index = 0
            token_to_stridx = {}
            for tok_idx, numeric_tok in enumerate(numeric_tokens):
                occurence_index_pairs = [(m.start(), m.end()) for m in re.finditer(numeric_tok, input_value)]

                i = 0
                s_idx = occurence_index_pairs[i][0]
                e_idx = occurence_index_pairs[i][1]

                while s_idx < last_index:
                    i += 1
                    s_idx = occurence_index_pairs[i][0]
                    e_idx = occurence_index_pairs[i][1]

                last_index = e_idx
                token_to_stridx[tok_idx] = [s_idx, e_idx]

            tok_idx = 0
            last_index = 0
            first_flag = True
            while tok_idx < len(numeric_tokens):
                numeric_tok = numeric_tokens[tok_idx]
                s_idx = token_to_stridx[tok_idx][0]
                e_idx = token_to_stridx[tok_idx][1]
                last_index = e_idx
                if e_idx < len(input_value) and input_value[e_idx].strip() == '':
                    while tok_idx + 1 < len(token_to_stridx) and token_to_stridx[tok_idx + 1][0] == token_to_stridx[tok_idx][1] + 1 and input_value[token_to_stridx[tok_idx][1]].strip() == "":
                        if (first_flag and re.match(r'^[1-9]\d{1,2}$', numeric_tok)) or (not first_flag and re.match(r'^\d{3}$', numeric_tokens[tok_idx + 1])):
                            numeric_tok = numeric_tok + input_value[token_to_stridx[tok_idx][1]:token_to_stridx[tok_idx + 1][1]]
                            tok_idx += 1
                            last_index = token_to_stridx[tok_idx][1]
                            first_flag = False
                        else:
                            break

                tok_idx += 1
                numeric_tokens_new.append(numeric_tok)

            if len(numeric_tokens_new) > 0:
                for token in numeric_tokens_new:
                    m = re.search(token, value)
                    try:
                        value = (value[:m.span()[0]] + ' ' + value[m.span()[1]:]).strip()
                    except Exception as e:
                        print(e)

            non_numeric_tokens = split_to_tokens(value)
    except Exception as e:
        print(e)
    return numeric_tokens_new, non_numeric_tokens


def dzs_numbs2(x, regx=REGX): # ds = detect and zeros-shave
    if not regx.findall(x):
        yield (None, None, None)
    for mat in regx.finditer(x):
        yield (mat.group(), ''.join(('0' if n.startswith('.') else '') + n for n in mat.groups('')), mat.groups(''))


def split_to_tokens(value):
    non_numeric_tokens = []
    if value is not None and value != '':
        snake_case = underscore(str(value))
        for punct in string.punctuation:
            snake_case = snake_case.replace(punct, " " + punct + " ")
        split_underscore = snake_case.replace('_', ' ').replace('\n', ' ').replace('\r', '').replace('\t', '')
        non_numeric_tokens = split_underscore.strip().lower().split(' ')
    return non_numeric_tokens


def underscore(word):
    """
    https://inflection.readthedocs.io/en/latest/_modules/inflection.html#underscore

    Make an underscored, lowercase form from the expression in the string.

    Example::

        >>> underscore("DeviceType")
        "device_type"

    As a rule of thumb you can think of :func:`underscore` as the inverse of
    :func:`camelize`, though there are cases where that does not hold::

        >>> camelize(underscore("IOError"))
        "IoError"

    """
    word = re.sub(r"([A-Z]+)([A-Z][a-z])", r'\1_\2', word)
    word = re.sub(r"([a-z\d])([A-Z])", r'\1_\2', word)
    return word.lower()


def discover_range_tokens(value, numeric_tokens):
    value = str(value).strip()
    out_range_tokens = []
    out_numeric_tokens = list(numeric_tokens)
    out_value_tokens = []

    # exclude long values to avoid getting lost here for ever (len(value) > 50)
    if len(value) > 50 or (len(value) > 2 and value[0] == '<' and value[-1] == '>'):
        return out_range_tokens, out_numeric_tokens, out_value_tokens

    if value is not None:
        try:
            value = value.replace('\n', ' ').replace('\r', '')
            snipped_value = value

            range_phrases = [r'(?:(mois depuis )\s*[a-zA-Z]*\s*(REGEX_TKN))',
                             r'(?:(mois depuis le )\s*[a-zA-Z]*\s*(REGEX_TKN))',
                             r'(?:(months since )\s*[a-zA-Z]*\s*(REGEX_TKN))',
                             r'(?:[><][=]?\s*((\W?\s*)|[a-zA-Z]*|\W?\s*[a-zA-Z]*)\s*(REGEX_TKN))',
                             r'(?:(more than )((\W?\s*)|[a-zA-Z]*|\W?\s*[a-zA-Z]*)\s*(REGEX_TKN))',
                             r'(?:(plus de )((\W?\s*)|[a-zA-Z]*|\W?\s*[a-zA-Z]*)\s*(REGEX_TKN))',
                             r'(?:(après )(REGEX_TKN))',
                             r'(?:(sur )((\W?\s*)|[a-zA-Z]*|\W?\s*[a-zA-Z]*)\s*(REGEX_TKN))',
                             r'(?:(REGEX_TKN years or older))',
                             r'(?:(less than )((\W?\s*)|[a-zA-Z]*|\W?\s*[a-zA-Z]*)(REGEX_TKN))',
                             r'(?:(fewer than )((\W?\s*)|[a-zA-Z]*|\W?\s*[a-zA-Z]*)\s*(REGEX_TKN))',
                             r'(?:(REGEX_TKN)\s*[a-zA-Z]*\s*( and under))',
                             r'(?:(under )((\W?\s*)|[a-zA-Z]*|\W?\s*[a-zA-Z]*)\s*(REGEX_TKN))',
                             r'(?:(moins de )((\W?\s*)|[a-zA-Z]*|\W?\s*[a-zA-Z]*)\s*(REGEX_TKN))',
                             r'(?:(REGEX_TKN)((\W?\s*)|[a-zA-Z]*|\W?\s*[a-zA-Z]*)\s+((et moins)|(or less)|(or more)|(and over)|(et plus)|(ou plus)))',
                             r'(?:((\W\s*)|[a-zA-Z]+|\W\s*[a-zA-Z]+)\s*(REGEX_TKN)\s+((et moins)|(or less)|(or more)|(and over)|(et plus)|(ou plus)))',
                             r'(?:(over )((\W?\s*)|[a-zA-Z]*|\W?\s*[a-zA-Z]*)\s*(REGEX_TKN))',
                             r'(?:(jusqu’en )((\W?\s*)|[a-zA-Z]*|\W?\s*[a-zA-Z]*)\s*(REGEX_TKN))',
                             r'(?:(up to )((\W?\s*)|[a-zA-Z]*|\W?\s*[a-zA-Z]*)\s*(REGEX_TKN))',
                             r'(?:((months)|(days)|(years))?\s*(since )[a-zA-Z]*\s*(REGEX_TKN))',
                             r'(?:[a-zA-Z]+\s*(-|( to )|( au ))\s*[a-zA-Z]+\s+(REGEX_TKN))',
                             r'(?:(REGEX_TKN)\s*-?\s*((year)|(month)|(day)|(ans)|(mois)|(jours)))'
                            ]

            range_phrases_2 = [r'(?:(^|\s+)(REGEX_TKN)([-–/])(REGEX_TKN)(\s+|$))',#date exceptions
                               r'(?:(^|\s+)((\W?\s*)|[^\W\d_]*|\W?\s*[^\W\d_]*)\s*(REGEX_TKN)\s*([-à]|(to)|(to under)|(to just under)|–|(up to))\s*\2\s*(REGEX_TKN)(\s+|$))',
                               r'(?:(^|\s+)(REGEX_TKN)\s*((\W?\s*)|[^\W\d_]*|\W?\s*[^\W\d_]*)\s*([-à]|(to)|(to under)|(to just under)|–|(up to))\s*(REGEX_TKN)\s*\3(\s+|$))',
                               r'(?:(^|\s+)(REGEX_TKN)\s*((a.m.)|(am)|(pm)|(p.m.))\s*((to)|)\s*(REGEX_TKN)\s*((a.m.)|(am)|(pm)|(p.m.)))',
                               r'(?:[^\W\d_]+\s*(REGEX_TKN)\s*(( au )|( to ))\s*[^\W\d_]+\s*(REGEX_TKN))',
                               r'(?:(REGEX_TKN)\s+/\s+(REGEX_TKN))'
                              ]

            numeric_token_pairs = get_sequential_pairs(out_numeric_tokens)
            available_numeric_token_pairs = copy.deepcopy(numeric_token_pairs)

            for range_phrase in range_phrases_2:
                if len(out_numeric_tokens) == 0:
                    break

                for numeric_token1, numeric_token2 in numeric_token_pairs:
                    if (numeric_token1, numeric_token2) not in available_numeric_token_pairs:
                        continue

                    phrase = range_phrase.replace("REGEX_TKN", numeric_token1, 1)
                    phrase = phrase.replace("REGEX_TKN", numeric_token2, 2)

                    m = re.search(phrase, snipped_value.lower())
                    if not m:
                        continue
                    phrase_start_idx = m.span()[0]
                    phrase_end_idx = m.span()[1]
                    if phrase_start_idx != -1:
                        phrase_found = m.group()
                        if range_phrase == r'(?:(^|\s+)(REGEX_TKN)([-–/])(REGEX_TKN)(\s+|$))':
                            if (len(numeric_token1) == 4 and len(numeric_token2) == 2):

                                if evaluate(numeric_token2) != evaluate(numeric_token1[-2:]) + 1:
                                    available_numeric_token_pairs.remove((numeric_token1, numeric_token2))
                                    snipped_value = re.sub("(?i)" + phrase_found, ' ', snipped_value, flags=re.I)
                                    continue

                            delimiter = m.groups()[2]
                            if phrase_start_idx > 0 and value[phrase_start_idx - 1] == delimiter:
                                available_numeric_token_pairs.remove((numeric_token1, numeric_token2))
                                snipped_value = re.sub("(?i)" + phrase_found, ' ', snipped_value, flags=re.I)
                                continue
                            if phrase_end_idx < len(value) and value[phrase_end_idx] == delimiter:
                                available_numeric_token_pairs.remove((numeric_token1, numeric_token2))
                                snipped_value = re.sub("(?i)" + phrase_found, ' ', snipped_value, flags=re.I)
                                continue

                        out_range_tokens.append((phrase_found, range_phrase))
                        snipped_value = re.sub("(?i)" + phrase_found, ' ', snipped_value, flags=re.I)
                        if numeric_token1 in out_numeric_tokens:
                            out_numeric_tokens.remove(str(numeric_token1))
                        if numeric_token2 in out_numeric_tokens:
                            out_numeric_tokens.remove(str(numeric_token2))
                        available_numeric_token_pairs.remove((numeric_token1, numeric_token2))

            for range_phrase in range_phrases:
                if len(out_numeric_tokens) == 0:
                    break
                for numeric_token in numeric_tokens:
                    if numeric_token not in out_numeric_tokens:
                        continue
                    phrase = range_phrase.replace("REGEX_TKN", numeric_token, 1)

                    m = re.search(phrase, snipped_value.lower())
                    if not m:
                        continue

                    phrase_start_idx = m.span()[0]
                    if phrase_start_idx != -1:
                        phrase_found = m.group()
                        out_range_tokens.append((phrase_found, range_phrase))
                        snipped_value = re.sub("(?i)" + phrase_found, ' ', snipped_value, flags=re.I)

                        if numeric_token in out_numeric_tokens:
                            out_numeric_tokens.remove(numeric_token)

            _, out_value_tokens = discover_tokens(snipped_value)
        except:
            return out_range_tokens, out_numeric_tokens, out_value_tokens

    return out_range_tokens, out_numeric_tokens, out_value_tokens
