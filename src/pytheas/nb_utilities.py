import re
import pandas as pd
from inflection import singularize

from nltk import word_tokenize
from nltk.corpus import stopwords
import string

import copy
# import regex
from psycopg2 import connect
import ast
import pandas as pd
from pytheas.pat_utilities import generate_pattern_symbols_and_case, generate_pattern_summary
from pytheas.parsemathexpr import evaluate

stop = stopwords.words('french')+stopwords.words('english')+list(string.punctuation)
null_equivalent =['','data not available','nan','not available','no data','no answer','nd','na','n/d','n/a','n.a.','not applicable','sans objet','s.o.','so','s/o','-','--','.','..','...','null','none','*','void','0','redacted', 'confidential', 'confidentiel']



def recreate_data_block(con, sql_table_name, endpoint_dbname,datatable_key, has_premeta,has_header, row_limit = 10):
    endpoint_con = connect(dbname=endpoint_dbname,user='christina', host = 'localhost', password='marathi', port = 5532)
    endpoint_cur = endpoint_con.cursor()
    endpoint_cur.execute("""SELECT * FROM """+sql_table_name+""" limit """+str(row_limit))
    tuples = []
    for row in endpoint_cur:
        tuples.append(list(row))
    data_df = pd.DataFrame(tuples)
    # input(data_df)

    endpoint_cur.close()
    endpoint_con.close()

    cur = con.cursor()
    header_rows = []

    if has_header:
        cur.execute("""SELECT header_text from dataheader where datatable = %s""", (datatable_key, ))
        res = cur.fetchone()
        if res:
            for row in res[0]:
                header_rows.append(row)
                
    header_df = pd.DataFrame(header_rows)
    num_rows_header = header_df.shape[0]
    combined_df = pd.concat([header_df,data_df], axis=0)

    pre_meta_tuples = []
    if has_premeta:
        cur.execute("""SELECT premeta_text from pre_metadata where datatable = %s""", (datatable_key, ))
        res = cur.fetchone()
        if res:
            pre_meta_tuples = ast.literal_eval(res[0])
    
    pre_meta_df = pd.DataFrame(pre_meta_tuples)
    num_rows_premeta = pre_meta_df.shape[0]
    header_index = [num_rows_premeta, num_rows_premeta+num_rows_header]
    num_rows_header = header_df.shape[0]

    first_data_row_index = num_rows_premeta + num_rows_header

    final_df = pd.concat([pre_meta_df,combined_df], axis=0).reset_index()
    final_df.drop('index', axis=1, inplace=True)

    # print(final_df)
    # input('headers=' +str(list(range(header_index[0], header_index[1]))))
    # input('first_data_row_index='+str(first_data_row_index))

    return final_df, list(range(header_index[0], header_index[1])), first_data_row_index


regx = re.compile(  '(?<![\d.])(?!\.\.)(?<![\d.][eE][+-])(?<![\d.][eE])(?<!\d[.,])'
                    '' #---------------------------------
                    '([+-]?)'
                    '(?![\d,]*?\.[\d,]*?\.[\d,]*?)'
                    '(?:0|,(?=0)|(?<!\d),)*'
                    '(?:'
                    '((?:\d(?!\.[1-9])|,(?=\d))+)[.,]?'
                    '|\.(0)'
                    '|((?<!\.)\.\d+?)'
                    '|([\d,]+\.\d+?))'
                    '0*'
                    '' #---------------------------------
                    '(?:'
                    '([eE][+-]?)(?:0|,(?=0))*'
                    '(?:'
                    '(?!0+(?=\D|\Z))((?:\d(?!\.[1-9])|,(?=\d))+)[.,]?'
                    '|((?<!\.)\.(?!0+(?=\D|\Z))\d+?)'
                    '|([\d,]+\.(?!0+(?=\D|\Z))\d+?))'
                    '0*'
                    ')?'
                    '' #---------------------------------
                    '(?![.,]?\d)'
                    )

# https://stackoverflow.com/questions/5917082/regular-expression-to-match-numbers-with-or-without-commas-and-decimals-in-text/5929469#5929469
def dzs_numbs(x,regx = regx): # ds = detect and zeros-shave
    if not regx.findall(x):
        yield ('No match,', 'No catched string,', 'No groups.')
    for mat in regx.finditer(x):
        yield (mat.group(), ''.join(mat.groups('')), mat.groups(''))

def dzs_numbs2(x,regx = regx): # ds = detect and zeros-shave
    if not regx.findall(x):
        # yield ('No match,', 'No catched string,', 'No groups.')
        yield (None,None,None)
    for mat in regx.finditer(x):
        yield (mat.group(),
               ''.join(('0' if n.startswith('.') else '')+n for n in mat.groups('')),
               mat.groups(''))



def test_discover_numeric_tokens():
    NS = ['  23456000and23456000. or23456000.000  00023456000 s000023456000.  000023456000.000 ',
      'arf 10000 sea10000.+10000.000  00010000-00010000. kant00010000.000 ',
      '  24:  24,  24.   24.000  24.000,   00024r 00024. blue 00024.000  ',
      '  8zoom8.  8.000  0008  0008. and0008.000  ',
      '  0   00000M0. = 000.  0.0  0.000    000.0   000.000   .000000   .0   ',
      '  .0000023456    .0000023456000   '
      '  .0005872    .0005872000   .00503   .00503000   ',
      '  .068    .0680000   .8   .8000  .123456123456    .123456123456000    ',
      '  .657   .657000   .45    .4500000   .7    .70000  0.0000023230000   000.0000023230000   ',
      '  0.0081000    0000.0081000  0.059000   0000.059000     ',
      '  0.78987400000 snow  00000.78987400000  0.4400000   00000.4400000   ',
      '  -0.5000  -0000.5000   0.90   000.90   0.7   000.7   ',
      '  2.6    00002.6   00002.60000  4.71   0004.71    0004.7100   ',
      '  23.49   00023.49   00023.490000  103.45   0000103.45   0000103.45000    ',
      '  10003.45067   000010003.45067   000010003.4506700 ',
      '  +15000.0012   +000015000.0012   +000015000.0012000    ',
      '  78000.89   000078000.89   000078000.89000    ',
      '  .0457e10   .0457000e10   00000.0457000e10  ',
      '   258e8   2580000e4   0000000002580000e4   ',
      '  0.782e10   0000.782e10   0000.7820000e10  ',
      '  1.23E2   0001.23E2  0001.2300000E2   ',
      '  432e-102  0000432e-102   004320000e-106   ',
      '  1.46e10and0001.46e10  0001.4600000e10   ',
      '  1.077e-300  0001.077e-300  0001.077000e-300   ',
      '  1.069e10   0001.069e10   0001.069000e10   ',
      '  105040.03e10  000105040.03e10  105040.0300e10    ',
      '  +286E000024.487900  -78.4500e.14500   .0140E789.  ',
      '  081,12.40E07,95.0120     0045,78,123.03500e-0.00  ',
      '  0096,78,473.0380e-0.    0008,78,373.066000E0.    0004512300.E0000  ',
      '  ..18000  25..00 36...77   2..8  ',
      '  3.8..9    .12500.     12.51.400  ',
      '  00099,111.8713000   -0012,45,83,987.26+0.000,099,88,44.or00,00,00.00must',
      '  00099,44,and   0000,099,88,44.bom',
      '00,000,00.587000  77,98,23,45.,  this,that ',
      '  ,111  145.20  +9,9,9  0012800  .,,.  1  100,000 ',
      '1,1,1.111  000,001.111   -999.  0.  111.110000  1.1.1.111  9.909,888'] 
 
    for ch in NS:
        print('string: '+repr(ch))
        for strmatch, modified, the_groups in dzs_numbs2(ch):
            print(strmatch.rjust(20),'',modified,'',the_groups)

def discover_tokens(input_value):
    # print('\n--\ndiscover_tokens:\ninput_value= '+str(input_value))

    value = input_value
    non_numeric_tokens = [] 
    numeric_tokenpairs = []
    numeric_tokens = []
    numeric_tokens_new = []
    
    try:
        if pd.isna(input_value)== False and input_value != None:
            for strmatch, modified, the_groups in dzs_numbs2(input_value):
                if strmatch != None:
                    numeric_tokenpairs.append((strmatch,modified))
                    numeric_tokens.append(strmatch)

            last_index = 0
            token_to_stridx = {}
            for tok_idx, numeric_tok in enumerate(numeric_tokens):
                occurence_index_pairs = [(m.start(),m.end()) for m in re.finditer(numeric_tok, input_value)]

                i=0
                s_idx = occurence_index_pairs[i][0]
                e_idx = occurence_index_pairs[i][1]

                while(s_idx<last_index):
                    i+=1
                    s_idx = occurence_index_pairs[i][0]
                    e_idx = occurence_index_pairs[i][1]
                
                last_index = e_idx 
                token_to_stridx[tok_idx]= [s_idx, e_idx]
            
            tok_idx = 0
            last_index = 0
            first_flag = True
            while tok_idx<len(numeric_tokens):
                numeric_tok = numeric_tokens[tok_idx]
                s_idx = token_to_stridx[tok_idx][0]
                e_idx = token_to_stridx[tok_idx][1]
                last_index = e_idx
                if e_idx<len(input_value) and input_value[e_idx].strip()=='' :
             
                    while tok_idx+1<len(token_to_stridx) and token_to_stridx[tok_idx+1][0] == token_to_stridx[tok_idx][1]+1 and input_value[token_to_stridx[tok_idx][1]].strip()=="":
                        if (first_flag ==True and re.match(r'^[1-9]\d{1,2}$', numeric_tok)) or (first_flag ==False and re.match(r'^\d{3}$',numeric_tokens[tok_idx+1])):
                            numeric_tok = numeric_tok+input_value[token_to_stridx[tok_idx][1]:token_to_stridx[tok_idx+1][1]]
                            tok_idx+=1
                            last_index = token_to_stridx[tok_idx][1]
                            first_flag = False
                        else:
                            break

                tok_idx+=1               
                numeric_tokens_new.append(numeric_tok)

            if len(numeric_tokens_new)>0:
                for token in numeric_tokens_new:
                    m = re.search(token, value)
                    try:                        
                        value = (value[:m.span()[0]]+' '+value[m.span()[1]:]).strip()
                    except Exception as e:
                        print(e)

            non_numeric_tokens = split_to_tokens(value)
    except Exception as e:
        print(e)
    # print('numeric_tokens_new='+str(numeric_tokens_new))
    return numeric_tokens_new, non_numeric_tokens

def test_discover_range_tokens():
    test_strings = [
        'LOT 1  SECTION 26  TOWNSHIP 2  PLAN LMP31877  NWD',
        'http://archivesdemontreal.com/greffe/vues-aeriennes-archives/jpeg/CarteIndex-1947-1949.jpg',
        ' -1947-1949.',
        '-1947-1949.jpg',
        'http://archivesdemontreal.com/greffe/vues-aeriennes-archives/jpeg/VM97-3_7P1-01.jpg',
        'FY10-11',
        '500-17-048861-093',
        '31/05',
        '14-17/06',
        'DAUID2006',
        'PR-HRUID2013',
        'contracting-address-street-1',
        'All-Person-Consumption-Intake-g-per-person-P50',
        'Selected sites of cancer (ICD-O-3)',
        'Revenu total 15 et moins',
        'Period of immigration - 2006 to 2010 (distribution 2016)',
        'Mois depuis septembre 1998',
        '2009/2010-COMP',
        'Unemployment(000s) 2003/04',
        'Jan-Mar 2016 Total',
        '5-year total',
        'JULY 8 to AUGUST 11',
        'FÉVRIER 5 au MARS 10',
        'Under $10,000 - All Returns/Moins de 10 000$ - Toutes les déclarations',
        'Under $5,000',
        '$20,000 to just under $ 40,000',
        'Describe the capture rate of mercury switches collected as reported by the program in each year up to 2007 (%) or up to the year in which you became subject to the Notice./Indiquez le taux (%) de saisie annuel d’interrupteurs au mercure jusqu’en 2007 ',
        'Months since September 1998',                    
        '2033-12-01',
        '07/31/2014',
        '3rd Bracket  (>$87,907 & <=$136,270)',
        
        'avril 7 au mai 11',
        'Broad Age Groups: 65 years and over (Percent Change 2001 to 2016)',
        '80 000$à89 999$',
        '50-54',
        '$250,000 and over - All Returns/250 000$ et plus - Toutes les déclarations',
        
        'De 15 000 $ à 19 999 $',
        '2000/2001',    
        '20$USD - 30$USD ',    
        '$USD 20 - $USD 30 ',
        '$20 to $30',
        '1997 to 2016'   ,                
        '2 à 4 semaines  de retard',
        'The number of mercury switches collected as reported in each year up to 2007',
        '20$CAD – 30$CAD'
    ]

    for value in test_strings:
        print("\n-------\nvalue = "+str(value)+"\n")
        numeric_tokens_new, non_numeric_tokens = discover_tokens(value)
        # print('possible numeric_tokens: '+str(numeric_tokens_new))

        range_tokens, numeric_tokens, remaining_tokens = discover_range_tokens(value, numeric_tokens_new)
        input("""
- - - > range_tokens = """+str(range_tokens)+"""
        
        numeric_tokens = """+str(numeric_tokens)+"""
        remaining_tokens = """+str(remaining_tokens))




def discover_range_tokens(value, numeric_tokens):
    value = str(value).strip()
    # print('value='+str(value))
    # input('numeric_tokens='+str(numeric_tokens))
    out_range_tokens = []
    out_numeric_tokens = list(numeric_tokens)
    out_value_tokens = []

    # exclude long values to avoid getting lost here for ever (len(value) > 50)
    if len(value) > 50 or (len(value)>2 and value[0]=='<' and value[-1]=='>'):
        return out_range_tokens,out_numeric_tokens,out_value_tokens

    if value != None:
        try:
            value = value.replace('\n', ' ').replace('\r', '')
            snipped_value = value

            range_phrases = ['(?:(mois depuis )\s*[a-zA-Z]*\s*(REGEX_TKN))', 
                            '(?:(mois depuis le )\s*[a-zA-Z]*\s*(REGEX_TKN))',
                            '(?:(months since )\s*[a-zA-Z]*\s*(REGEX_TKN))',
                            '(?:[><][=]?\s*((\W?\s*)|[a-zA-Z]*|\W?\s*[a-zA-Z]*)\s*(REGEX_TKN))',    
                            '(?:(more than )((\W?\s*)|[a-zA-Z]*|\W?\s*[a-zA-Z]*)\s*(REGEX_TKN))', 
                            '(?:(plus de )((\W?\s*)|[a-zA-Z]*|\W?\s*[a-zA-Z]*)\s*(REGEX_TKN))', 
                            # '(?:(REGEX_TKN)((\W?\s*)|[a-zA-Z]*|\W?\s*[a-zA-Z]*)\s*( and over))',
                            # '(?:(REGEX_TKN)((\W?\s*)|[a-zA-Z]*|\W?\s*[a-zA-Z]*)\s*( et plus))',#'$250,000 and over - All Returns/250 000$ et plus - Toutes les déclarations'
                            # '(?:(REGEX_TKN)((\W?\s*)|[a-zA-Z]*|\W?\s*[a-zA-Z]*)\s*( ou plus))',
                            '(?:(après )(REGEX_TKN))',
                            '(?:(sur )((\W?\s*)|[a-zA-Z]*|\W?\s*[a-zA-Z]*)\s*(REGEX_TKN))',
                            # '(?:(REGEX_TKN)\s*((\W?\s*)|[a-zA-Z]*|\W?\s*[a-zA-Z]*)( et plus))',
                            '(?:(REGEX_TKN years or older))',
                            '(?:(less than )((\W?\s*)|[a-zA-Z]*|\W?\s*[a-zA-Z]*)(REGEX_TKN))',
                            '(?:(fewer than )((\W?\s*)|[a-zA-Z]*|\W?\s*[a-zA-Z]*)\s*(REGEX_TKN))',
                            '(?:(REGEX_TKN)\s*[a-zA-Z]*\s*( and under))',
                            '(?:(under )((\W?\s*)|[a-zA-Z]*|\W?\s*[a-zA-Z]*)\s*(REGEX_TKN))',
                            '(?:(moins de )((\W?\s*)|[a-zA-Z]*|\W?\s*[a-zA-Z]*)\s*(REGEX_TKN))',

                            '(?:(REGEX_TKN)((\W?\s*)|[a-zA-Z]*|\W?\s*[a-zA-Z]*)\s+((et moins)|(or less)|(or more)|(and over)|(et plus)|(ou plus)))',
                            '(?:((\W\s*)|[a-zA-Z]+|\W\s*[a-zA-Z]+)\s*(REGEX_TKN)\s+((et moins)|(or less)|(or more)|(and over)|(et plus)|(ou plus)))',

                            '(?:(over )((\W?\s*)|[a-zA-Z]*|\W?\s*[a-zA-Z]*)\s*(REGEX_TKN))',
                            '(?:(jusqu’en )((\W?\s*)|[a-zA-Z]*|\W?\s*[a-zA-Z]*)\s*(REGEX_TKN))',
                            '(?:(up to )((\W?\s*)|[a-zA-Z]*|\W?\s*[a-zA-Z]*)\s*(REGEX_TKN))',
                            '(?:((months)|(days)|(years))?\s*(since )[a-zA-Z]*\s*(REGEX_TKN))',
                            '(?:[a-zA-Z]+\s*(-|( to )|( au ))\s*[a-zA-Z]+\s+(REGEX_TKN))',
                            '(?:(REGEX_TKN)\s*-?\s*((year)|(month)|(day)|(ans)|(mois)|(jours)))'          
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
                if len(out_numeric_tokens)==0:
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
                    if phrase_start_idx!=-1:
                        phrase_found = m.group()
                        if range_phrase =='(?:(^|\s+)(REGEX_TKN)([-–/])(REGEX_TKN)(\s+|$))':
                            if (len(numeric_token1)== 4 and len(numeric_token2)==2):

                                if evaluate(numeric_token2) != evaluate(numeric_token1[-2:])+1:
                                    available_numeric_token_pairs.remove((numeric_token1,numeric_token2))
                                    snipped_value= re.sub("(?i)"+phrase_found,' ', snipped_value, flags=re.I)
                                    continue   
                                
                                    
                            # if (len(numeric_token1)== 2 and len(numeric_token2)==2) and int(numeric_token1) > int(numeric_token2):
                            #     available_numeric_token_pairs.remove((numeric_token1,numeric_token2))
                            #     snipped_value= re.sub("(?i)"+phrase_found,' ', snipped_value, flags=re.I)
                            #     continue
                            delimiter = m.groups()[2]
                            if phrase_start_idx>0 and value[phrase_start_idx-1] ==delimiter:
                                available_numeric_token_pairs.remove((numeric_token1,numeric_token2))
                                snipped_value= re.sub("(?i)"+phrase_found,' ', snipped_value, flags=re.I)
                                continue
                            if phrase_end_idx<len(value) and value[phrase_end_idx] ==delimiter:
                                available_numeric_token_pairs.remove((numeric_token1,numeric_token2))
                                snipped_value= re.sub("(?i)"+phrase_found,' ', snipped_value, flags=re.I)
                                continue

                        out_range_tokens.append((phrase_found,range_phrase))
                        snipped_value= re.sub("(?i)"+phrase_found,' ', snipped_value, flags=re.I)
                        if numeric_token1 in out_numeric_tokens: 
                            out_numeric_tokens.remove(str(numeric_token1))
                        if numeric_token2 in out_numeric_tokens:
                            out_numeric_tokens.remove(str(numeric_token2))
                        available_numeric_token_pairs.remove((numeric_token1,numeric_token2))


            for range_phrase in range_phrases:
                # input('\nrange_phrase='+range_phrase)
                if len(out_numeric_tokens)==0:
                    # print("\nno more tokens available -('_')-")
                    break
                for numeric_token in numeric_tokens:
                    # print('\navailable numeric_tokens='+str(out_numeric_tokens))
                    # print('numeric_token='+numeric_token)
                    
                    if numeric_token not in out_numeric_tokens:
                        continue
                    phrase = range_phrase.replace("REGEX_TKN", numeric_token, 1)
                    
                    m = re.search(phrase, snipped_value.lower())
                    if not m:
                        # print('\t'+numeric_token+' didnt work')
                        continue

                    phrase_start_idx = m.span()[0]
                    # print('SUCCESS!')
                    # print('phrase = '+phrase)
                    if phrase_start_idx!=-1:
                        phrase_found = m.group()
                        out_range_tokens.append((phrase_found,range_phrase))
                        # print('phrase_found='+str(phrase_found))
                        snipped_value= re.sub("(?i)"+phrase_found,' ', snipped_value, flags=re.I)

                        # snipped_value = snipped_value.replace(phrase_found, ' ', 1)
                        if numeric_token in out_numeric_tokens: 
                            out_numeric_tokens.remove(numeric_token)                
                    # print('out_range_tokens='+str(out_range_tokens))

            _, out_value_tokens = discover_tokens(snipped_value)
        except:
            return out_range_tokens,out_numeric_tokens,out_value_tokens

    return out_range_tokens,out_numeric_tokens,out_value_tokens


def get_sequential_pairs(tokens):
    pairs = []
    for i, token in enumerate(tokens):
        if i+1<len(tokens):
            pairs.append((token, tokens[i+1]))
    return pairs



def split_to_tokens(value):
    non_numeric_tokens = []
    if value!=None and value!='':
        snake_case = underscore(str(value))
        # print('snake_case='+str(snake_case))
        for punct in string.punctuation:
            snake_case = snake_case.replace(punct, " "+punct+" ")
        # print('snake_case='+str(snake_case))
        split_underscore = snake_case.replace('_', ' ').replace('\n', ' ').replace('\r', '').replace('\t', '')
        # print('split_underscore='+str(split_underscore))
        non_numeric_tokens=split_underscore.strip().lower().split(' ')
        # input('non_numeric_tokens='+str(non_numeric_tokens))
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
    # word = word.replace("-", "_")
    return word.lower()

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
    try:
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass 
    return False
