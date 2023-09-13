import re
from xml.dom import minidom
from datetime import date, time, datetime
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import spacy

from spacy import displacy
from spacy.matcher import PhraseMatcher
from spacy.tokens import Span
from spacy.matcher import Matcher
from negspacy.negation import Negex
import medspacy
import word2number
from word2number import w2n
from tqdm import tqdm

# ### Initialize Spacy Pipeline ###

from spacy.lang.en import English

import scispacy
from spacy.pipeline import EntityRuler
from spacy.language import Language

from spacy.util import filter_spans

from negspacy.termsets import termset

row_names_dia = ['Dia_ROS', 'dia_mention_Interval_History', 'dia_mention_Previous_note', 'dia_mention_remaining_notes']
row_names_stool_fre = ['stool_fre_ROS', 'stool_fre_mention_Interval_History', 'stool_fre_mention_Previous_note', 'stool_fre_mention_remaining_notes' ]
row_names_bristol = ['Bristol_ROS', 'bristol_mention_Interval_History', 'bristol_mention_Previous_note', 'bristol_mention_remaining_notes']
row_names_cr = ['CR_ROS', 'CR_mention_Interval_History', 'CR_mention_Previous_note', 'CR_mention_remaining_notes']
row_names_well = ['Well_ROS', 'well_mention_Interval_History', 'well_mention_Previous_note', 'well_mention_remaining_notes']


def return_diarrhea_spacy():
    ts = termset("en_clinical")
    nlp_spacy = spacy.load('en_core_web_sm', disable = ['ner'])
    ruler = nlp_spacy.add_pipe("entity_ruler")
    sentencizer = nlp_spacy.add_pipe('sentencizer')
    mw_current = ['24 hour', ]
    ruler.add_patterns([
                    #{"label": 'PAIN', "pattern": [{"TEXT": {"REGEX": r"(a|A)bdominal"}}, {"TEXT": {"REGEX": r"(p|P)ain"}}]},
                    #{"label": 'WELL', "pattern": [{"TEXT": {"REGEX": r"(F|f)eel[a-zA-Z]*"}}, {"TEXT": {"REGEX": r"(W|w)ell[a-zA-Z]*"}}]},
                    #{"label": 'WELL', "pattern": [{"TEXT": {"REGEX": r"(C|c)linical"}}, {"TEXT": {"REGEX": r"(R|r)emission"}}]},
                    #{"label": 'BM', "pattern": [{"TEXT": {"REGEX": r"(B|b)owel"}},  {"TEXT": {"REGEX": r"(M|m)ovement[a-zA-Z]*"}}]},
                    #{"label": 'BM', "pattern": [{"TEXT": {"REGEX": "(B|b)(M|m)"}}]},
                    #{"label": 'BM', "pattern": [{"TEXT": {"REGEX": r"(B|b)\.(M|m)\."}}]},
                    #{'label': 'Diarrhea', 'pattern': [{"TEXT": {"REGEX": "(d|D)iarrhea[a-zA-Z]*"}}]},
                    #{'label': 'Diarrhea', 'pattern': [{"TEXT": {"REGEX": "(s|S)tool[a-zA-Z]*"}}]},
    
                    {"label": 'CURRENT', "pattern": [{"TEXT": {"REGEX": "(c|C)urrent[a-zA-Z]*"}}]},
                    {"label": 'CURRENT', "pattern": [{"TEXT": {"REGEX": "(t|T)oday[a-zA-Z]*"}}]},
                    {"label": 'CURRENT', "pattern": [{"TEXT": {"REGEX": "(y|Y)esterday[a-zA-Z]*"}}]},
                    {"label": 'CURRENT', "pattern": [{"TEXT": {"REGEX": "(p|P)resent( |,)"}}]},
                    {"label": 'CURRENT', "pattern": [{"TEXT": {"REGEX": "(n|N)ow"}}]},
                    #{"label": 'FORMED', "pattern": [{"TEXT": {"REGEX": "(F|f)ormed"}}]},
                    
                   ])
    ts.remove_patterns({"pseudo_negations": ['no further', 'not able to be', 'not certain if', 'not certain whether', 'not necessarily', 'without any further', 'without difficulty', 'without further', 'might not', 'not only', 'no increase', 'no significant change', 'no change', 'no definite change', 'not extend', 'not cause']})

### additional term sets can be added to negspacy-- for full documentation see: https://pypi.org/project/negspacy/

    ts.add_patterns({
            "preceding_negations": ["hasn't noticed", 'non', 'non-','poorly' , "no", "none", "no further", 'somewhat',  "deny", 
                                    "No significant", "hasnt had any", "hasn't had", "hasn't",'hasnt', 'resolution', 'denies'],
            "following_negations": ["resolved", "subsided", "none now", 'erratic'],
            "pseudo_negations": ["does not typically", 'almost resolved', 'non bloody'],
            "termination": ['endorses', 'and only', 'scant', 'intermittent', 'with', 'then', 'now']
    
             })

    nlp_spacy.add_pipe('negex', config={"neg_termset":ts.get_patterns()}, last = True)
    return nlp_spacy

# ### Custom Spacy Components ###

@Language.component("Multi_Word_NER")
def Multi_Word_NER(doc):
    
  
    
    #text = doc.text
    #print(doc)
   
    
    #doc = nlp_spacy(doc.text)
    
    diarrhea_bag_of_words = ['(diarrhea)', 
                '(water[a-zA-Z]*[ ]{1,}(stool[a-zA-Z]*|bm[a-zA-Z]*|bowel[ ]{1,}movement[a-zA-Z]*))',
                '(normal[a-zA-Z]*[ ]{1,}(stool[a-zA-Z]*|bm[a-zA-Z]*|bowel[ ]{1,}movement[a-zA-Z]*))',
                '(liquid[a-zA-Z]*[ ]{1,}(stoo[a-zA-Z]*l|bm[a-zA-Z]*|bowel[a-zA-Z]*[ ]{1,}movement[a-zA-Z]*))', 
                '(loose[a-zA-Z]*[ ]{1,}(stool[a-zA-Z]*|bm[a-zA-Z]*|bowel[ ]{1,}movement[a-zA-Z]*))', 
                '(frequent[a-zA-Z]*[ ]{1,}(stool[a-zA-Z]*|bm[a-zA-Z]*|bowel[ ]{1,}movement[a-zA-Z]*))', 
                '(semi[ ]{1,}form[a-zA-Z]*[ ]{1,}(stool[a-zA-Z]*|bm[a-zA-Z]*|bowel[ ]{1,}movement[a-zA-Z]*))',
                '(partial[a-zA-Z]*[ ]{1,}form[a-zA-Z]*[ ]{1,}(stool[a-zA-Z]*|bm[a-zA-Z]*|bowel[ ]{1,}movement[a-zA-Z]*))',
                '(somewhat[a-zA-Z]*[ ]{1,}form[a-zA-Z]*[ ]{1,}(stool[a-zA-Z]*|bm[a-zA-Z]*|bowel[ ]{1,}movement[a-zA-Z]*))',
                '(poor[a-zA-Z]*[ ]{1,}form[a-zA-Z]*[ ]{1,}(stool[a-zA-Z]*|bm[a-zA-Z]*|bowel[ ]{1,}movement[a-zA-Z]*))', 
                '[^ ](form[a-zA-Z]*[ ]{1,}(stool[a-zA-Z]*|bm[a-zA-Z]*|bowel[ ]{1,}movement[a-zA-Z]*))',
                '(unform[a-zA-Z]*[ ]{1,}(stool[a-zA-Z]*|bm[a-zA-Z]*|bowel[ ]{1,}movement[a-zA-Z]*))', '(constipa[a-zA-Z]*)'
                  ]
    
    diarrhea_negation =  ['(?:[ ])(form[a-zA-Z]*[ ]{1,}(stool[a-zA-Z]*|bm[a-zA-Z]*|bowel[ ]{1,}movement[a-zA-Z]*))']
                       
    diarrhea_implied = ['(constipa[a-zA-Z]*)', 
                      '(normal[a-zA-Z]*[ ]{1,}(stool[a-zA-Z]*|bm[a-zA-Z]*|bowel[ ]{1,}movement[a-zA-Z]*))']

    bag_of_words_gen = ['(bm[a-zA-Z]*)', '(bowel[ ]{1,}movement[a-zA-Z]*)', '(stool[a-zA-Z]*)']
    
    positive_indicators = [r'loose[a-zA-Z]*', r'water[a-zA-Z]*', r'frequent[a-zA-Z]*', 
                           r'liquid[a-zA-Z]*', 'semi[ ]{1,}form[a-zA-Z]*',  r'unform[a-zA-Z]*', r'soft']
    
    adj_negation = [r'^form[a-zA-Z]*', 'normal', ' hard']
    
    Well_bag_of_words =   ['(doing[ ]{1,}well)', 
                    '(asymptomatic[a-zA-Z]*)', 
                    '((no|w/o)[ ]{1,}sx[a-zA-Z]*)', 
                    '(no[ ]{1,}complaint)',
                    '(normal[ ]{1,}(bm[a-zA-Z]*|bowel[a-zA-Z]*))', 
                    '(feel[a-zA-Z]*[ ]{1,}well)'] 
    
    Well_bag_of_words_neg = ['(gi[ ]{1,}(issue[a-zA-Z]*|complaint[a-zA-Z]*|symptom[a-zA-Z]*))']
    
    
    cr_bag_of_words =   ['(clin[a-zA-Z]*[ ]{1,}rem[a-zA-Z]*)', '(quiescent)', '(symptomatic[ ]{1,}remission)'] 
   
    
    for ent_category, Label in zip([diarrhea_bag_of_words, diarrhea_negation, 
                                    bag_of_words_gen, diarrhea_implied, Well_bag_of_words, 
                                    Well_bag_of_words_neg, cr_bag_of_words], 
                                   ['Diarrhea', 'Diarrhea_neg', 'Diarrhea_gen',
                                    'Diarrhea_imp', 'WELL', 'WELL_neg', 'Clin_Rem']):
        
        #print([token.text for token in doc])
        original_ents = list(doc.ents)
    
        mwt_ents = []
        indexes = []
        #track_ents = []
        Label_gen = []
        
        #print('orriginalents', original_ents)
        
        for regex in ent_category:
            for match in re.finditer(regex, doc.text, re.IGNORECASE):
                #print(match.span(1))
                #print([token.text for token in doc])
                start, end = match.span(1)
                #print('start', start, end)
                span = doc.char_span(start, end, alignment_mode = 'expand')
                
                #print(match)
                if Label == 'Diarrhea_gen':
                    #print('here',  [token.text for token in doc if not len(token.ent_type_) != 0])
                    
                    pos_result = check_word(positive_indicators, [token.text for token in doc if not len(token.ent_type_) != 0], span.start)
                    #print(pos_result)
                    neg_result = check_word(adj_negation, [token.text for token in doc if not len(token.ent_type_) != 0], span.start)
                    
                    #print(neg_result)
                    
                    
                    if pos_result[0] == True:
                        
                        Label_gen.append('Diarrhea')
                        
                    elif neg_result[0] == True:
                        
                        #print('here')
                        Label_gen.append('Diarrhea_neg')
                    else:
                        continue
                        
                elif Label == 'Diarrhea_neg':
                    
                    pos_result = check_word(positive_indicators, [token.text for token in doc], span.start)
                    
                    if pos_result[0] == True:
                        
                        continue
                    
                    
                #print('jerrr')
                start, end = match.span(1)
                #print('start', start, end)
                span = doc.char_span(start, end, alignment_mode = 'expand')
                #print(span)
                
                #print(span)
                if span is not None:
                    mwt_ents.append((span.start, span.end, span.text))
                    #print(mwt_ents)
                    token_locations = (span.start, span.end)
                    indexes.append(token_locations)
                    #track_ents.append((start, end))
        
        
        mwt_ents.sort(key=lambda x : x[1])

        indexes.sort(key=lambda x : x[1])
        
        
        if len(mwt_ents) != 0 and Label == 'Diarrhea_gen':
            #print('hererere')
            for ent, Label_g in zip(mwt_ents, Label_gen):
                #print(ent, Label_g)
                start, end, name = ent
                
                if end - start != 0:
                    per_ent = Span(doc, start, end, label = Label_g)
                    original_ents.append(per_ent)
                
            filtered = filter_spans(original_ents)
            doc.ents = filtered
            
            
            count = 0
            
            for (start, end), Label_g in zip(indexes, Label_gen):
                start -= count
                end -= count
                
                with doc.retokenize() as retokenizer:
                    if end - start > 1:
                        
                        attrs = {'POS': 'NOUN', 'ENT_TYPE': Label_g}
                        retokenizer.merge(doc[start:end], attrs = attrs)
                        count += end - start - 1 
                    else:
                        continue
                #count += 1 
        
        elif len(mwt_ents) != 0:
            for ent in mwt_ents:
                start, end, name = ent
                if end - start != 0:
                    per_ent = Span(doc, start, end, label = Label)
                    original_ents.append(per_ent)
                else:
                    continue


            filtered = filter_spans(original_ents)
            doc.ents = filtered
            
            count = 0
            
            for start, end in indexes:
                #print(start, end)
                with doc.retokenize() as retokenizer:
                    if end - start > 1:
                        start -= count
                        end -= count
                        attrs = {'POS': 'NOUN', 'ENT_TYPE': Label}
                        retokenizer.merge(doc[start:end], attrs = attrs)
                        count += end - start - 1 
                    else:
                        pass
                #count += 1
        
        else:
            continue





        
           

    return doc

@Language.component("Bristol_Score_NER")
def Bristol_Score_NER(doc):
    
    #print(doc)
    
    bag_of_words_bm = ['((bristol)|(bss))']
    
    bag_of_words_fre = [r'([0-9]+[ ]{0,}\-[ ]{0,}[0-9]+)[^.]{,10}per[ ]{1,}day', 
                            r'([0-9]+[ ]{0,}\-?[ ]{0,}[0-9]+)', 
                            r'([0-9]+[ ]{1,}[0-9]+)', 
                            r'~[ ]{0,}([0-9]+)',
                            r'(?<!\d|/|\.|&|%)([0-9]{1,2})(?=[^0-9\/])(?![ ]{0,}(week|day|year|month|hour|%|&))', ' (one) ', ' (two) ', 
                            ' (three) ', ' (four) ', ' (five) ', ' (six) ', ' (seven) ', ' (eight) ', ' (nine) ']
    
    #negative_identifiers = [' test[a-zA-Z]*', 'study', 'studie[a-zA-Z]*', 'o&p', 'gram']
    
   
    bm_spans = []
    final_matches = []
    track = []
    number_spans = []
    track_spans = []
    temp_spans = []
    indexes = []
    temp = []
    
    count = 0

    track_matches = []
    
    original_ents = list(doc.ents)
    
    for word_bm in bag_of_words_bm:
            
        try:
            trial = re.finditer(word_bm, doc.text, re.IGNORECASE)
            
            for match in trial:
                #print(match, match.span(1))
                if match.span(1) not in temp_spans:
                    #temp.append((match.group(1), match.span(1)))
                    temp_spans.append((match.span(1)))

                else:
                    continue

        except:
            continue
            
            
    
    #print(temp_spans, 'temp_spans')    
    
    
    for word_num in bag_of_words_fre:
        #print(word_num)
        
        try:
            trial = re.finditer(word_num, doc.text, re.IGNORECASE)
            for match in trial:
                #print(match, match.span(1), match.group(1))
                if check_ranges(track_spans, match.span(1)):
                    match_span = match.span(1)
                    #print(match_span, 'match_span')
                    number_spans.append((match.group(1), match_span))
                    track_spans.append(match_span)

                else:
                    continue

        except:
            #print('error')
            continue


    #print('temp', temp_spans)
    if len(number_spans) > 0:
        for span in temp_spans:
            #print(span)
            matches = get_closest_number(number_spans, (span[0] + span[1])/2)

            if check_ranges(track_matches, matches[0][1]):
                final_matches.append(matches[0][1])
                track_matches.append(matches[0][1])
                #print(matches)
                #indexes.append(matches[0][1])
                #print(indexes)
                
    
    
    final_matches.sort(key=lambda x : x[1])
    
    for ent in final_matches:
        start, end = ent
        #print(start, end, 'sn')
        other_span = doc.char_span(start, end, alignment_mode = 'expand')
        #print('other_span', other_span)
        if span is not None:
            per_ent = Span(doc, other_span.start, other_span.end, label = 'Bristol')
            indexes.append((other_span.start, other_span.end))
            original_ents.append(per_ent)
        
        
    #print(original_ents)
        
    filtered = filter_spans(original_ents)
    doc.ents = filtered
    #print('index', indexes)

    #count = 0
    if len(indexes) != 0:
        count = 0
        for start, end in indexes:
            #print(start, end)
            if end - start > 1:
                start -= count
                end -= count
                
                with doc.retokenize() as retokenizer:
                    attrs = {'POS': 'NOUN', 'ENT_TYPE': 'Bristol'}
                    retokenizer.merge(doc[start: end], attrs = attrs)
                    count += end - start - 1
            else:
                pass
                #count += 1 
        
            
            
    
    
    return doc

def add_pipes(nlp_spacy):
    nlp_spacy.add_pipe("Bristol_Score_NER", before = 'negex')
    nlp_spacy.add_pipe("Stool_Range_NER", before = 'negex')
    nlp_spacy.add_pipe("Multi_Word_NER", before = 'negex')
    return nlp_spacy

@Language.component("Stool_Range_NER")
def Stool_Range_NER(doc):
    
    bag_of_words_bm = ['(bm[a-zA-Z]*)', '(bowel[ ]{1,}movement[a-zA-Z]*)', '(stool[a-zA-Z]*)', '(move.{,10}bowel[a-z]*)']
    
    bag_of_words_fre = [r'([0-9]+[ ]{0,}\-[ ]{0,}[0-9]+)[^.]{,10}per[ ]{1,}day', 
                            r'([0-9]+[ ]{0,}\-?[ ]{0,}[0-9]+)', 
                            r'([0-9]+[ ]{1,}[0-9]+)', 
                            r'~[ ]{0,}([0-9]+)',
                            r'(?<!\d|/|\.|&|%)([0-9]{1,2})(?=[^0-9\/])(?![ ]{0,}(week|day|year|month|hour|%|&))', ' (one) ', ' (two) ', 
                            ' (three) ', ' (four) ', ' (five) ', ' (six) ', ' (seven) ', ' (eight) ', ' (nine) ', ' (a) ']
    
    negative_identifiers = [' test[a-zA-Z]*', 'study', 'studie[a-zA-Z]*', 'o&p', 'gram']
    
   
    bm_spans = []
    final_matches = []
    track = []
    number_spans = []
    track_spans = []
    temp_spans = []
    indexes = []
    temp = []
    #count = 0 

    track_matches = []
    
    original_ents = list(doc.ents)
    
    for word_bm in bag_of_words_bm:
            
        try:
            trial = re.finditer(word_bm, doc.text, re.IGNORECASE)
            
            for match in trial:
                #print(match)
                if check_ranges(temp_spans, match.span(1)):
                    temp_spans.append((match.span(1)))

                else:
                    continue

        except:
            continue
            
            
    
    #print(temp_spans, 'temp_spans')    
    
    
    for word_num in bag_of_words_fre:
        #print(word_num)
        
        try:
            trial = re.finditer(word_num, doc.text, re.IGNORECASE)
            for match in trial:
                #print(match, match.span(1), match.group(1))
                if check_ranges(track_spans, match.span(1)):
                    match_span = match.span(1)
                    #print(match_span, 'match_span')
                    number_spans.append((match.group(1), match_span))
                    track_spans.append(match_span)

                else:
                    continue

        except:
            #print('error')
            continue
            
    #print('number_spans', number_spans)
    #print('temp_spans', temp_spans)



    if len(number_spans) > 0:
        for span in temp_spans:
            #print((span[0] + span[1])/2)
            matches = get_closest_number(number_spans, (span[0] + span[1])/2)

            if check_ranges(track_matches, matches[0][1]):
                final_matches.append(matches[0][1])
                track_matches.append(matches[0][1])
               
                
    final_matches.sort(key=lambda x : x[1])
    
    #print(final_matches, 'final_matches')
    if len(final_matches) != 0:
        
        for ent in final_matches:
            start, end = ent
            #print(start, end, 'sn')
            other_span = doc.char_span(start, end, alignment_mode = 'expand')
            #print('other_span', other_span.start, other_span.end)
            if other_span is not None:
                per_ent = Span(doc, other_span.start, other_span.end, label = 'range')
                indexes.append((other_span.start, other_span.end))
                original_ents.append(per_ent)
    
        
        
    
    
   
        
    
    #print(original_ents)
        
    filtered = filter_spans(original_ents)
    doc.ents = filtered
    #print('index', indexes)

    #count += end - start - 1
    
    if len(indexes) != 0:
    
        count = 0
        
        for start, end in indexes:
            #print(start, end)
            if end - start > 1:
                start -= count
                end -= count

                #print(start, end)

                with doc.retokenize() as retokenizer:
                    attrs = {'POS': 'NOUN', 'ENT_TYPE': 'range'}
                    retokenizer.merge(doc[start:end], attrs = attrs)
                    count += (end - start - 1) 

            else:
                pass


            
    
    
    return doc
    


### ###

def get_note_id(note):
    return re.search(r'(.*?),', note).group(1)
    
def remove_edge(note):
    return re.search(r',(.+),\d', note).group(1)
    
def get_line(note):
    try:
        return int(re.search(r',(\d+)$', note).group(1))
    except:
        return np.nan

def collapse_function(set_of_ids, df_grouped):
    
    table = pd.DataFrame(columns = ['note_id', 'note_ur'])


    for id_ in set_of_ids:
        note_text = ''
        id_rows = df_grouped[df_grouped['note_id'] == id_]['note_ur'].values
        id_rows.sort_values
        for value in id_rows:
            note_text += value

        table = table.append({'note_id': id_, 'note_ur': note_text}, ignore_index=True)


    return table

def find_diarrhea(note):
    
    bag_of_words = [r'diarrhea', 
                'water[a-zA-Z]*[ ]{1,}(stool[a-zA-Z]*|bm[a-zA-Z]*|bowel[ ]{1,}movement[a-zA-Z]*)',
                'normal[a-zA-Z]*[ ]{1,}(stool[a-zA-Z]*|bm[a-zA-Z]*|bowel[ ]{1,}movement[a-zA-Z]*)',
                'liquid[a-zA-Z]*[ ]{1,}(stoo[a-zA-Z]*l|bm[a-zA-Z]*|bowel[a-zA-Z]*[ ]{1,}movement[a-zA-Z]*)', 
                'loose[a-zA-Z]*[ ]{1,}(stool[a-zA-Z]*|bm[a-zA-Z]*|bowel[ ]{1,}movement[a-zA-Z]*)', 
                'frequent[a-zA-Z]*[ ]{1,}(stool[a-zA-Z]*|bm[a-zA-Z]*|bowel[ ]{1,}movement[a-zA-Z]*)', 
                'semi[ ]{1,}form[a-zA-Z]*[ ]{1,}(stool[a-zA-Z]*|bm[a-zA-Z]*|bowel[ ]{1,}movement[a-zA-Z]*)',
                'partial[a-zA-Z]*[ ]{1,}form[a-zA-Z]*[ ]{1,}(stool[a-zA-Z]*|bm[a-zA-Z]*|bowel[ ]{1,}movement[a-zA-Z]*)',
                'somewhat[a-zA-Z]*[ ]{1,}form[a-zA-Z]*[ ]{1,}(stool[a-zA-Z]*|bm[a-zA-Z]*|bowel[ ]{1,}movement[a-zA-Z]*)',
                'poor[a-zA-Z]*[ ]{1,}form[a-zA-Z]*[ ]{1,}(stool[a-zA-Z]*|bm[a-zA-Z]*|bowel[ ]{1,}movement[a-zA-Z]*)', 
                ' form[a-zA-Z]*[ ]{1,}(stool[a-zA-Z]*|bm[a-zA-Z]*|bowel[ ]{1,}movement[a-zA-Z]*)',
                'unform[a-zA-Z]*[ ]{1,}(stool[a-zA-Z]*|bm[a-zA-Z]*|bowel[ ]{1,}movement[a-zA-Z]*)', 'constipa[a-zA-Z]*'
                  
                   ]
    
    bag_of_words_gen = ['bm[a-zA-Z]*', ' bowel[ ]{1,}movement[a-zA-Z]*', 'stool[a-zA-Z]*']
    
    
    
    matches = {}
    keywords = []
    
    spans = []
    
    spans_gen = []
    
    note = preprocessing(note)
    
    
    for word in bag_of_words:
        
        regex = '[^.]*' + word + r'[^.]*'
        
        try:
            trial = re.finditer(regex, note, re.IGNORECASE)
        
            for match in trial:
                #print(match)
                span = match.span()
                matches[span[1]] = match.group(0)
                keywords.append(word)
                spans.append(span)
                
        except:
            continue
            
    for gen_word in bag_of_words_gen:
        
        regex = r'[^.]*' + gen_word + r'[^.]*'
        #print(regex)
        try:
            trial = re.finditer(regex, note, re.IGNORECASE)
            
            for match in trial:
                #print(match.group(0))
                #print([match.group(0) for match in trial])
                span = match.span()
                #print(span)
                if span not in spans:
                    #print('hererere')
                    result, adj = get_adj(match.group(0), nlp_spacy)
                    #print('result', result)
                    if result == True:
                        matches[span[1]] = match.group(0)
                        keywords.append((gen_word, adj))
                        spans.append(span)
                        spans_gen.append((match.group(0), match.group(1), match.spans(1)))
        except:
            pass
    
    #print(spans_gen)
    ordered_matches, spans = re_order(matches)
    
    if len(ordered_matches) == 0:
        return np.nan, np.nan, np.nan
    
  
 
    return ordered_matches, keywords, spans

def get_adj(sentence, nlp_spacy):
    
    positive_indicators = [r'loose[a-zA-Z]*', r'water[a-zA-Z]*', r'frequent[a-zA-Z]*', r'liquid[a-zA-Z]*', 'semi[ ]{1,}form[a-zA-Z]*',  r'unform[a-zA-Z]*']
    
    adj_negation = [r'^form[a-zA-Z]*', r'normal', r' hard']

    ### remove / and other special characters
    #sentence = preprocessing(sentence)
    
    doc = nlp_spacy(sentence)
    
    #doc = retokenize_ent(diarrhea_bag_of_words, sentence, doc, 'Diarrhea')
    #doc = retokenize_ent(bag_of_words_gen, sentence, doc, 'Stool_Gen')
    #doc = retokenize_ent(adj_bag, sentence, doc, 'ADJ')
    
    #print(doc.ents)
    
    sentence_words = [word.text for word in doc]
            
            
    #print(sentence_words)
       
    #print([doc.label_ for doc in doc.ents])
    result, indicator_word_1 = check_word(positive_indicators, sentence_words)
    neg_result, indicator_word_2 = check_word(adj_negation, sentence_words)
        
    if result == True:
        return True, indicator_word_1
    elif neg_result == True:
        return True, indicator_word_2
    else:
        return False, None

def get_adjective(sentence,  nlp_spacy):
    
    positive_indicators = [r'loose[a-zA-Z]*', r'water[a-zA-Z]*', r'frequent[a-zA-Z]*', r'liquid[a-zA-Z]*', r'unform[a-zA-Z]*']
    
    adj_negation = [r'form[a-zA-Z]*']
   
    
    
    
    ### remove / and other special characters
    sentence = preprocessing(sentence)
    
    doc = nlp_spacy(sentence)
    
    doc = retokenize_ent(diarrhea_bag_of_words, sentence, doc, 'Diarrhea')
    doc = retokenize_ent(bag_of_words_gen, sentence, doc, 'Stool_Gen')
    
    #print(doc)
    children_left_pos = []
    children_left_text = []
    
    sen_length = len([token.text for token in doc])
    
    
    for token in doc:
        if token.ent_type_ == 'Stool_Gen':
            print(token.text, token.pos_, token.dep_, token.ent_type_, token.head)
            #print([[child.text, child.dep_] for child in token.children])
            if token.pos_ == 'NOUN':
                
                children_left_pos +=  [child.pos_ for child in token.subtree]
                children_left_text += [child.text for child in token.subtree]
                
                for child in token.subtree:
                    if len([sub_child.text for sub_child in child.children]) < 0:
                        children_left_pos +=  [sub_child.pos_ for sub_child in child.children]
                        children_left_text += [sub_child.text for sub_child in child.children]
                        
                
                if token.dep_ == 'compound':
                    head_text = token.head.text
                    head_pos = token.head.pos_
                    children_left_text.append(head_text)
                    children_left_pos.append(head_pos)
                    
                index = token.i
                
                
                #children_pos = [child.text for child in token.subtree]
                
                #print(token.head)
                
                
                if not token.is_sent_end:
                    #print(token.pos_, 'yes', token.text)
                    if doc[index + 1].pos_ == 'ADP':
                        children_left_text += [child.text for child in doc[index + 1].children]
                        children_left_pos += [child.pos_ for child in doc[index + 1].children]
                #print(children_left_text)
                #print([child.text for child in token.ancestors])
            
                #print(children_left_text)
                #if 'ADP' in children_pos:
                    #for child in token.children:
                        #if child.pos_ == 'ADP':
                            #print(yes)
            
            
                if token.dep_ == 'conj':
                    head_children_pos = [child.pos_ for child in token.head.rights]
                    head_children_text = [child.text for child in token.head.rights]
                    #print(head_children_text)
                    
                    children_left_pos += head_children_pos
                    children_left_text += head_children_text
                    
                    #print(children_left_text)
                    #print(children_left_pos)
                    
                    
            #elif token.pos_ == 'VERB':
                #return True
            
            
    
    
    if len(children_left_text) > 0:
        result, indicator_word = check_word(positive_indicators, children_left_text)
        neg_result, indicator_word = check_word(adj_negation, children_left_text)
        
        if result == True:
            return True, indicator_word
        elif neg_result == True:
            return True, indicator_word
        else:
            return False

def check_sentence(identifiers, sentence):
    for ident in identifiers:
        if re.search(ident, sentence, re.IGNORECASE) is not None:
            return True
        else:
            continue
    return False

def find_stool_frequency(note):
    
    
    bag_of_words_bm = ['bm[a-zA-Z]*', 'bowel[ ]{1,}movement[a-zA-Z]*', 'stool[a-zA-Z]*']
    
    bag_of_words_fre = [r'([0-9]+[ ]{0,}\-[ ]{0,}[0-9]+)[^.]{,10}per[ ]{1,}day', r'([0-9]+[ ]{0,}\-[ ]{0,}[0-9]+)', r'([0-9]+[ ]{1,}[0-9]+)',r'~[ ]{0,}([0-9]+)', r'( [0-9]+ )', '(one)', '(two)', '(three)', '(four)', '(five)', '(six)', '(seven)', '(eight)', '(nine)']
    
    negative_identifiers = [' test[a-zA-Z]*', 'study', 'studie[a-zA-Z]*', 'o&p']
    
    matches = []
    keywords = []
    spans = {}
    ent_spans = []
    
    
    ###try:
    ##    note = w2n.word_to_num(note)
    ##except:
        ##pass
      
    for word_bm in bag_of_words_bm:
        for word_fre in bag_of_words_fre:
            regex = '[^.]* ' + '(?:' + word_fre + '[^.]{,35}' + word_bm + '|' + word_bm + '[^.]{,35}' + word_fre  +  ')' + '[^.]*'
            
            #print(regex)
            
            try:
                trial = re.finditer(regex, note, re.IGNORECASE)
                for match in trial:
                    print(match)
                    if not check_sentence(negative_identifiers, match.group(0)):
                        span = match.span(0)

                        if span in spans:
                            if len(spans[span]) >= len(match.group(0)):
                                continue

                            else:

                                start, end = match.span(0)  
                                matches.append((match.group(0), [group for group in match.groups() if group != None], (start, end)))#sentence matches
                                #keywords.append(word)
                                spans[span] = match.group(1)
                        else:
                            start, end = match.span(0)
                            matches.append((match.group(0), [group for group in match.groups() if group != None], (start, end)))#sentence matches
                                #keywords.append(word)
                            spans[span] = match.group(0)

            
            except:
                continue
            
            
    return matches

def find_stool_freq(note):
    bag_of_words_bm = ['(bm[a-zA-Z]*)', '(bowel[ ]{1,}movement[a-zA-Z]*)', '(stool[a-zA-Z]*)', '(bristol)|(bss)', '(move.{,10}bowel[a-z]*)']
    
    bag_of_words_fre = [r'([0-9]+[ ]{0,}\-[ ]{0,}[0-9]+)[^.]{,10}per[ ]{1,}day', 
                            r'([0-9]+[ ]{0,}\-[ ]{0,}[0-9]+)', 
                            r'([0-9]+[ ]{1,}[0-9]+)',r'~[ ]{0,}([0-9]+)',
                            r'(?<!\d|/|\.|&|%)([0-9]{1,2})(?=[^0-9\/])(?![ ]{0,}(week|day|year|month|hour|%|&))', ' (one) ', ' (two) ', 
                            ' (three) ', ' (four) ', ' (five) ', ' (six) ', ' (seven) ', ' (eight) ', ' (nine) ', ' (a) ']
    
    negative_identifiers = [' test[a-zA-Z]*', ' study', ' studie[a-zA-Z]*', ' o&p', ' gram']
    
    bm_sentence = []
    bm_spans = []
    final_matches = []
    track = []
    
    note = preprocessing(note)
    

    
    
    
    for word_bm in bag_of_words_bm:
        
        regex = '[^.]*' + word_bm + '[^.]*'
        
        try:
            trial = re.finditer(regex, note, re.IGNORECASE)

            for match in trial:
                #print(match)
                if not check_spans_overlap(match.span(0), track):
                    if not check_sentence(negative_identifiers, match.group(0)):
                        bm_sentence.append(match.group(0))
                        track.append(match.span(0))
        except:
            continue
    
    #print(bm_sentence)
    for sentence in bm_sentence:
        #print('sentence',sentence)
        temp = []
        temp_spans = []
        
        for word_bm in bag_of_words_bm:
            
            try:
                trial = re.finditer(word_bm, sentence, re.IGNORECASE)
        #print([match.group(0) for match in trial])
                        
                
        

            except:
                continue
            
            for match in trial:
                #print(match)
                if not check_spans_overlap(match.span(1), temp_spans):
                    temp.append((match.group(1), match.span(1)))
                    temp_spans.append((match.span(1)))
                        
        bm_spans.append((sentence, temp))
        
    #print('bm_spans', bm_spans)
    if len(bm_spans) > 0:
    
        for sentence, bm_span in bm_spans:
            
            sentence_matches = []
            number_spans = []
            track_spans = []
            track_matches = []
            
            for word_num in bag_of_words_fre:
                #print(word_num)
                try:
                    trial = re.finditer(word_num, sentence, re.IGNORECASE)
                    
    
                        
                        
                except:
                    #print('error')
                    continue
                
                for match_1 in trial:
                        #print(match)
                    if not check_spans_overlap(match_1.span(1), track_spans):
                            #print(match_1)
                        match_span = match_1.span(1)
                        number_spans.append((match_1.group(1), match_span))
                        track_spans.append(match_1.span(1))
                            
                        
            #print('number_spans', number_spans)   
            
            if len(number_spans) > 0:
                for span in bm_span:
                    #print('span', span)
                    #print(span[0], (span[1][0] + span[1][1])/2 )
                    matches = get_closest_number(number_spans, (span[1][0] + span[1][1])/2 )
                    if matches not in track_matches:
                        final_matches.append((sentence, matches[0][0], matches[0][1]))
                        track_matches.append(matches)
                 
            else:
                continue

        return final_matches
                    
    else:
        return np.nan

def check_spans_overlap(current_span, list_of_spans):
    for span in list_of_spans:
        start, end = span
        if not current_span[0] > end and not current_span[1] < start:
            return True

def get_closest_number(num_matches, bm_match):
    
    num_matches_span = [num[1][0] for num in num_matches]
    
    #print('num_match_span', num_matches_span)
    
    final_matches = []
    
    
    match_loc_diff = 1000
    current_favorite = np.nan
        
        
    for index, num_match in enumerate(num_matches_span):
        #print('index', index)
        #print('num_match', num_match)
        #print('bm_match', bm_match)

        match_loc_diff_test = abs(float(num_match) - float(bm_match)) 
        #print('loc_diff', match_loc_diff_test)

        if match_loc_diff_test < match_loc_diff:
            match_loc_diff = match_loc_diff_test
            current_favorite = index

    final_matches.append(num_matches[current_favorite])
    #print('final_matches', final_matches)

    
    return final_matches

def find_clinical_remission(note):
    
    if not isinstance(note, str):
        return np.nan, np.nan, np.nan
    
    bag_of_words = ['clin[a-zA-Z]*[ ]{1,}rem[a-zA-Z]*', 'quiescent', 'symptomatic[ ]{1,}remission'] 
    matches = {}
    keywords = []
    
    note = preprocessing(note)
    
    for word in bag_of_words:
        
        regex = r'[^.]*' + word + r'[^.]*\.'
        try:
            trial = re.finditer(regex, note, re.IGNORECASE)
        
            for match in trial:
                span = match.span()
                matches[span[1]] = match.group(0)
                keywords.append(word)
                
        except:
            continue
            
    
    ordered_matches, spans = re_order(matches)
    
    
        
    if len(ordered_matches) == 0:
        return np.nan, np.nan, np.nan
    
  
 
    return ordered_matches, keywords, spans

def find_no_symptoms(note):
    
    if not isinstance(note, str):
        return np.nan, np.nan, np.nan
    
    bag_of_words = ['doing[ ]{1,}well', 
                    'asymptomatic[a-zA-Z]*', 
                    '(no|w/o)[ ]{1,}sx[a-zA-Z]*', 
                    'no[ ]{1,}complaint' ,
                    'gi[ ]{1,}(issue[a-zA-Z]*|complaint[a-zA-Z]*|symptom[a-zA-Z]*)',
                    'normal[ ]{1,}(bm[a-zA-Z]*|bowel[a-zA-Z]*)', 
                    'feel[a-zA-Z]*[ ]{1,}well']
    matches = {}
    keywords = []
    
    note = preprocessing(note)
    
    for word in bag_of_words:
        
        regex = r'[^.]*' + word + r'[^.]*\.'
        try:
            trial = re.finditer(regex, note, re.IGNORECASE)
        
            for match in trial:
                span = match.span()
                matches[span[1]] = match.group(0)
                keywords.append(word)
                
        except:
            continue
            
    
    ordered_matches, spans = re_order(matches)
    
    
        
    if len(ordered_matches) == 0:
        return np.nan, np.nan, np.nan
    
  
 
    return ordered_matches, keywords, spans

def check_ranges(range_list, current_range):
    #print('range', range_)
    if len(range_list) == 0:
        return True
    for range_ in range_list:
        #print('range', range_)
        if range_[0] <= current_range[1] and current_range[0] <= range_[1]:
            return False
    
    
    return True

def get_closest_number_2(num_matches, bm_match):
    
    match_loc_diff = 1000 
    final_matches = []
    current_favorite = np.nan
    
    for index, num_match in enumerate(num_matches):
        
        match_loc_diff_test = abs(num_match[1][0] - bm_match) 

        if match_loc_diff_test < match_loc_diff:
            match_loc_diff = match_loc_diff_test
            current_favorite = index
            
    final_matches.append(num_matches[current_favorite])

    
    return final_matches

def get_closest_number(num_matches, bm_match):
    
    num_matches_span = [num[1][0] for num in num_matches]
    
    final_matches = []
    
    
    match_loc_diff = 1000
    current_favorite = np.nan
        
        
    for index, num_match in enumerate(num_matches_span):

        match_loc_diff_test = abs(num_match - bm_match) 

        if match_loc_diff_test < match_loc_diff:
            match_loc_diff = match_loc_diff_test
            current_favorite = index

    final_matches.append(num_matches[current_favorite])

    
    return final_matches

def present_tense(nlp_spacy, sentence, ent_label, ent_start):
    
    present_tense_verbs = [ 'VBP', 'VBZ']
    present_participle = ['VBG']
    past_tense = ['VBD']
    past_participle = ['VBN']
    
    All_present_tense_verbs = ['VB', 'VBP', 'VBZ', 'VBG']
    past_tense_all = ['VBD', 'VBN']

    verbs = ['VBP', 'VBZ', 'VBD', 'VBN', 'VBG']
    
    sentence = re.sub(r'\*', '', sentence)
    sentence = re.sub(r'nb', 'n b', sentence, re.IGNORECASE)
    
    #span_text = kwargs.get('span_text', None)
    
    if ent_label == 'Diarrhea':
        ent_label = ['Diarrhea', 'Diarrhea_neg', 'Diarrhea_imp']
    
    elif ent_label == 'WELL':
        ent_label = ['WELL', 'WELL_neg']
        
    else:
        ent_label = [ent_label]
    

    doc = nlp_spacy(sentence)
    
    #doc = retokenize_ent(diarrhea_bag_of_words, sentence, doc, 'Diarrhea')
    
    #doc = retokenize_ent(diarrhea_negation, sentence, doc, 'Diarrhea_neg')
    
    #if ent_label == 'range':
        #doc = tokenize_range(doc, sentence, span_text, 'range')
    
    

    
    
    false_negatives = ['left']
    #print(doc.ents, [d.label_ for d in doc.ents])

        
    pos_tags = [ token.tag_ for token in doc]
    ents_types = [ent.label_ for ent in doc.ents]
    pos_ = [token.pos_ for token in doc]
    
    #print(pos_tags, pos_)
    
    if len(set(pos_tags).intersection(verbs)) == 0:
        #print('no_verb')                           #checks if there are verbs in sentence
        return True, "SIMPLE_PRESENT"
    
    if 'CURRENT' in ents_types:
        #print('current_label')
        return True, 'CURRENT'
    
    if len(set(pos_tags).intersection(All_present_tense_verbs)) + len(set(pos_tags).intersection(past_tense_all)) == 1:
        #print('yes')
        if len(set(pos_tags).intersection(All_present_tense_verbs)) == 1:
            return True, "SIMPLE_PRESENT"
        elif len(set(pos_tags).intersection(past_tense_all)) == 1 and 'FORMED' not in ents_types:
            return False, 'SIMPLE_PAST'
        else:
            return True, "SIMPLE_PRESENT"
    
    
    
    for token in doc:

        if token.ent_type_ in ent_label or (ent_label == 'Diarrhea' and (token.ent_type_ == 'Diarrhea_neg' or token.ent_type_ == 'Diarrhea_imp')) and token.i == ent_start:
            #print(token.text, token.tag_, token.pos_, token.dep_, [token.text for token in token.ancestors],[token.tag_ for token in token.ancestors], [token.pos_ for token in token.ancestors], [token.dep_ for token in token.ancestors]  )
            
            
            if token.tag_ == 'VBG':
                #print(token.text)
                #print('heretwo')
                for child in token.children:
                    if child.pos_ == 'AUX' and child.tag_ in present_tense_verbs:
                        return True, "SIMPLE_PRESENT"
                    elif child.pos_ == 'AUX' and child.tag_ in past_tense:
                        return False, 'SIMPLE_PAST'
                
            if token.dep_ == 'ROOT' and token.pos_ == 'VERB':
                #print(token.text, token.pos_, 'here')
                #print([(child.text, child.pos_) for child in token.children])
        
                for child in token.children:
                    #print([(child.text, children.pos_) for child in token.children])
                    if child.tag_ in present_tense_verbs:
                        return True, 'SIMPLE_PRESENT'
                    elif child.tag_ in past_participle:
                        for child_2 in child.children:
                            if child_2.pos_ == 'AUX' and child_2.tag_ in present_tense_verbs:
                                return True, 'PAST_PART'
                    elif child.tag in past_tense:
                        return False, 'PAST_TENSE'
                    else:
                        return True, 'SIMPLE_PRESENT'
        
                
            for ancestor in token.ancestors:
                #print(ancestor.text, ancestor.pos, ancestor.tag_, ancestor.morph, ancestor.dep_, ancestor.head, 'here')
                
                #print([children.tag_ for children in ancestor.children])
            
                if ancestor.tag_ in present_tense_verbs:
                    #print('ancestor Present')
                    if ancestor.dep_ != 'ccomp' and ancestor.dep_ !=  'xcomp' and ancestor.dep_ != 'pcomp':
                        #print('here erorr')
                        #if ancestor.dep_ == 'ROOT':
                            #return True, 'SIMPLE_PRESENT'
                        #print('here')
                        ancestor_tags = [children.tag_ for children in ancestor.lefts]
                        for tag in ancestor_tags:
                            #print('here')
                            if tag in past_tense_all:
                                return False, 'SIMPLE_PAST'
                            else:
                                continue
                        return True, 'SIMPLE_PRESENT'
                    
                    else:
                        #print('printhere')
                        ancestor_pos_ = [children.pos_ for children in ancestor.ancestors]
                        ancestor_text = [children.text for children in ancestor.ancestors]
                        #print(ancestor_pos_, ancestor_text,'ancestor_pos_')
                        if not 'VERB'in ancestor_pos_ or 'AUX' in ancestor_pos_:
                            #print('here')
                            return True, 'SIMPLE_PRESENT'
                    #print('simple_present')
                
                elif ancestor.tag_ in past_tense:
                    #print('herere')
                    if ancestor.dep_ != 'ccomp' and ancestor.dep_ !=  'xcomp' and ancestor.dep_ != 'pcomp':
                        if ancestor.ent_type_ != 'FORMED' and ancestor.text not in false_negatives:
                            return False, 'SIMPLE_PAST'

                            #print('simple_past')

                        elif ancestor.dep_ == 'ROOT':
                            #print([ancestor.text for ancestor in ancestor.ancestors])
                            return True, 'SIMPLE_PRESENT'
                    
                    else:
                        ancestor_pos_ = [children.pos_ for children in ancestor.ancestors]
                        ancestor_tag_ = [children.tag_ for children in ancestor.ancestors]
                        
                        #print(ancestor_pos_)
                        if not 'VERB'in ancestor_pos_ and not 'AUX' in ancestor_pos_:
                            #print(99)
                            return True, 'SIMPLE_PRESENT'
                        else:
                            if len(set(ancestor_tag_).intersection(past_tense_all)) == 1:
                                return False, 'SIMPLE_PAST'
                            elif len(set(ancestor_tag_).intersection(All_present_tense_verbs)) == 1:
                                return True, 'SIMPLE_PRESENT'
                            else:
                                return True, 'SIMPLE_PRESENT'
                        
                            
                            
                    
                
                elif ancestor.tag_ in past_participle:
                    for child in ancestor.children:
                        #print(child.text, child.tag_, child.pos_)
                        if child.pos_ == 'AUX' and child.tag_ in present_tense_verbs:
                            #print('present_perfect_continuous')
                            return True, 'PRES_PERF_CON'
                        elif child.pos_ == 'AUX' and child.tag_ in past_tense:
                            #print('past_tense')
                            return False, 'PAST_TENSE'
                        elif not 'AUX' in [child.pos_ for child in ancestor.children]:
                            return True, 'PRES_PERF_CON'
                
                
                elif ancestor.tag_ == 'VBG' and ancestor.dep_ == 'xcomp':
                    #print('present_perfect_continuous')
                    return True, 'PRES_PERF_CON'
                    
                elif ancestor.tag_ == 'VBG' or ancestor.tag_ == 'VB':
                    for child in ancestor.children:
                        #print(child.text, child.tag_, child.pos_)
                        if child.pos_ == 'AUX' and child.tag_ in past_tense:
                            #print(child.text, 'past_tense')
                            return False, 'PAST_TENSE'
                        if child.pos_ == 'AUX' and child.tag_ in present_tense_verbs:
                            #print(child.text, 'present_tense')
                            return True, 'PRESENT_PART'
                else:
                    continue
            
            for token in doc:
                #print(token.)
                if token.dep_ == 'ROOT':
                    if token.tag_ in past_tense_all:
                        return False, 'PAST_TENSE'
                    elif token.tag_ in present_tense_verbs:
                        return True, 'SIMPLE_PRESENT'
                    elif token.tag_ in present_participle:
                        for child in token.children:
                            #print([(child.text, child.pos_) for child in token.children])
                            if child.tag_ in present_tense_verbs:
                                return True, 'SIMPLE_PRESENT'
                            elif child.tag_ in past_participle:
                                for child_2 in child.children:
                                    if child_2.pos_ == 'AUX' and child_2.tag_ in present_tense_verbs:
                                        return True, 'PAST_PART'
                            elif child.tag in past_tense:
                                return False, 'PAST_TENSE'
                            else:
                                return True, 'SIMPLE_PRESENT'
                    
                    
            return True, 'SIMPLE_PRESENT'

def binary_negation(value):
    if int(value) == 0:
        return 1
    elif int(value) == 1:
        return 0

def get_tense_negation(mentions, Label, nlp_spacy):
    
    if not isinstance(mentions, list):
        return np.nan
    
    if Label == 'range':
        
        span_text = [item[1] for item in mentions]
        mentions = [item[0] for item in mentions] 
        
    
    
    negation = []
    
    
    present_tense_true_neg = np.nan
    Tense_2 = np.nan

    for index, mention in enumerate(mentions):
        
        try:
            #print(mention)
            mention = re.sub(r'\*', '', mention)
            mention = re.sub(r'nb', 'n b', mention, re.IGNORECASE)
            #print(mention)
            temp = []
            #if keyword == 'PAIN_GEN' or keyword == 'cramp':
                #print(True)
                #Label = 'PAIN_GEN'
            if Label == 'range':
                #span_text = kwargs.get('span_text', None)
                present_tense_true, Tense = present_tense(nlp_spacy, mention, 'range')
                #print(present_tense_true)
            elif Label =='Diarrhea':
                #print()
                present_tense_true, Tense = present_tense(nlp_spacy, mention, Label)
                
                present_tense_true_neg, Tense_2 = present_tense(nlp_spacy, mention, 'Diarrhea_neg')
                
                #print(present_tense_true_neg)
            #print(present_tense_true, Tense)
            #print(mention)
            
        


            if present_tense_true == True or (Label == 'Diarrhea' and present_tense_true_neg == True):
                #print(Tense)
                doc = nlp_spacy(mention.lower())
                
                #if Label == 'Diarrhea':
                    #doc = retokenize_ent(diarrhea_bag_of_words, mention, doc, Label)
                    #print(doc.ents, [ent._.negex for ent in doc.ents])
                    #doc = retokenize_ent(diarrhea_negation, mention, doc, 'Diarrhea_neg')
                    #print(doc.ents, [ent.label_ for ent in doc.ents])
                
                    
                    #doc = retokenize_ent(bag_of_words_gen, mention, doc, 'Diarrhea')
                    
                    #if all(x not in [ent.label_ for ent in doc.ents] for x in ['Diarrhea', 'Diarrhea_neg']):
                         #doc = retokenize_ent(bag_of_words_gen, mention, doc, 'Diarrhea')
                        
            
                #elif Label == 'range':
                    #doc = tokenize_range(doc, mention, span_text[index], 'range')
                
                #print(doc.ents)
                    
                    
                    
                for ent in doc.ents:
                    
                    #print(ent.text, ent.label_)
                    if ent.label_ == Label or (Label == 'Diarrhea' and (ent.label_ == 'Diarrhea_neg' or ent.label_ == 'Diarrhea_imp')):
                        
                        if ent.label_ == 'Diarrhea':
                            #print('jer')
                            x = 1
                        elif ent.label_ == 'Diarrhea_neg':
                            #print('herr')
                            x = 0
                          
                        if ent._.negex == False:
                            if Label == 'range':
                                #print('here')
                                number = encode_stool_freq(ent.text, 3)
                                temp.append((number, Tense, mention))
                            elif Label == "Bristol":
                                number = encode_stool_freq(ent.text, 5)
                                temp.append((number, Tense, mention))
                            elif Label == 'Diarrhea'and ent.label_ == 'Diarrhea_imp':
                                temp.append((0, ))
                                
                            else:
                                temp.append((x, Tense, mention))
                        else:
                            #print('herererererer')
                            if Label == 'range':
                                number = encode_stool_freq(ent.text, 3)
                                temp.append((number, Tense, mention))
                            
                            elif Label == "Bristol":
                                number = encode_stool_freq(ent.text, 5)
                                temp.append((number, Tense, mention))
                                
                                 
                            else:
                                #print('herrr')
                                temp.append((binary_negation(x), Tense, mention))
                            
                   

                negation += temp

                        
        except:
            continue



    if len(negation) > 0:
        return negation
    else:
        return np.nan

def get_tense_negation(mentions, Label, nlp_spacy, note_id):
    
    #print(note_id)
     
    if not isinstance(mentions, list):
        return np.nan
    
    if Label == 'range' or Label == 'Bristol':
        span_text = [item[1] for item in mentions]
        mentions = [item[0] for item in mentions] 
        
    #print(mentions)    
    
    if Label == 'Diarrhea':
        #print('hereere')
        Label_1 = ['Diarrhea', 'Diarrhea_neg', 'Diarrhea_imp']
        
    elif Label == 'WELL':
        Label_1 = ['WELL', 'WELL_neg']
        
    else:
        Label_1 = [Label]
        
    
    negation = []
    
    for index, mention in enumerate(mentions):
        
        #print(mention)
        mention = re.sub(r'\*', '', mention)
        mention = re.sub(r'nb', 'n b', mention, re.IGNORECASE)
        mention = preprocessing(mention)

        temp = []

        doc = nlp_spacy(mention.lower())
        #print(doc.ents)

        for index, ent in enumerate(doc.ents):
            
            if ent.label_ in Label_1:
                #print(ent.text, ent.label_)
                
                present_tense_true, Tense = present_tense(nlp_spacy, mention, ent.label_, ent.start)
                #print(present_tense_true, Tense, ent._.negex)
                if present_tense_true == True:
                    
                    if ent.label_ == 'Diarrhea':
                            x = 1
                    elif ent.label_ == 'Diarrhea_neg':
                            x = 0
                          
                    if ent._.negex == False:
                        #print('here')
                       
                        if ent.label_ == 'range':
                            number = encode_stool_freq(ent.text, 3)
                            negation.append((number, Tense, mention))

                        elif ent.label_ == "Bristol":
                            number = encode_stool_freq(ent.text, 4)
                            negation.append((number, Tense, mention))

                        elif ent.label_ == 'Diarrhea_imp':
                            negation.append((0, Tense, mention))

                        elif ent.label_ == 'Diarrhea' or ent.label_ == 'Diarrhea_neg':
                            negation.append((x, Tense, mention))
                        elif ent.label_ != 'WELL_neg':
                            negation.append((1, Tense, mention))
                        elif ent.label_ == 'Clin_Rem':
                            negation.append((1, Tense, mention))
                        else:
                            pass

                    else:
                        #print('here')
                        if ent.label_ == 'range':
                            number = encode_stool_freq(ent.text, 3)
                            negation.append((number, Tense, mention))

                        elif ent.label_ == "Bristol":
                            number = encode_stool_freq(ent.text, 4)
                            negation.append((number, Tense, mention))
                            
                        elif ent.label_ == 'Diarrhea_imp':
                            pass


                        elif ent.label_ == 'Diarrhea' or ent.label_ == 'Diarrhea_neg':
                            negation.append((binary_negation(x), Tense, mention))
                            
                        elif ent.label_ == 'WELL_neg':
                            negation.append((1, Tense, mention))
                        else:
                            pass
                            
                    
            #negation += temp
                            
                else:
                    continue
            else:
                continue



                
            
            
            
    if len(negation) > 0:
        return negation
    else:
        return np.nan

def preprocessing(note):
    
    if not isinstance(note, str):
        return note
        
    
    processed_note = remove_ROS(note)

    processed_note = re.sub(r'\'', '', processed_note) #removes apostrohpes
    processed_note = re.sub(r'\(', '', processed_note)
    processed_note = re.sub(r'\)', '', processed_note)
    processed_note = re.sub('([a-zA-Z])/([a-zA-Z])', r'\1, \2', processed_note) #removes / and replaces with comma
    processed_note = re.sub(r'-', ' ', processed_note) #removes emdash
    #processed_note = re.sub(r'(?<=\w)\s{3,}', '.    ', processed_note) # replaces 
    processed_note = re.sub(r'([a-z0-9])\s{2,}([A-Z])', r'\1. \2', processed_note)
    processed_note = re.sub(r'([0-9])\.([0-9])', r'\1\2', processed_note)
    processed_note = re.sub(r';', '.', processed_note)
    #processed_note = remove_ROS(processed_note)



    return processed_note

def remove_duplicates(range_list):
    total_list = []
    lst = []
    for range_ in range_list:
        start = range_[0]
        end = range_[1] 
        total_list += list(range(start, end))
        
    total_list_filtered = list(set(total_list))
    total_list_filtered = sorted(total_list_filtered)
    
    lst = modify_ranges(total_list_filtered)
  
    
    return lst

def takeSecond(elem):
    return elem[1]

def get_adj(sentence,  nlp_spacy):
    
    positive_indicators = [r'loose[a-zA-Z]*', r'water[a-zA-Z]*', r'frequent[a-zA-Z]*', r'liquid[a-zA-Z]*', 'semi[ ]{1,}form[a-zA-Z]*',  r'unform[a-zA-Z]*']
    
    adj_negation = [r' form[a-zA-Z]*', r'normal']

    ### remove / and other special characters
    #sentence = preprocessing(sentence)
    
    doc = nlp_spacy(sentence)
    
    doc = retokenize_ent(diarrhea_bag_of_words, sentence, doc, 'Diarrhea')
    doc = retokenize_ent(bag_of_words_gen, sentence, doc, 'Stool_Gen')
    doc = retokenize_ent(adj_bag, sentence, doc, 'ADJ')
    
    #print(doc.ents)
    
    for token in doc:
        if token.ent_type_ == 'Stool_Gen' or token.ent_type_ == 'BM':
            sentence_words = [word.text for word in doc]
            
            
            #print(sentence_words)
       
    #print([doc.label_ for doc in doc.ents])
    result, indicator_word_1 = check_word(positive_indicators, sentence_words)
    neg_result, indicator_word_2 = check_word(adj_negation, sentence_words)
        
    if result == True:
        return True, indicator_word_1
    elif neg_result == True:
        return True, indicator_word_2
    else:
        return False, None

def check_sentence(identifiers, sentence):
    for ident in identifiers:
        #print(ident)
        if re.search(ident, sentence, re.IGNORECASE) is not None:
            return True
        else:
            continue
    return False

def check_spans(current_span, spans):
    if len(spans) == 0:
        return True
    for entry in spans:
        if entry[0] <= current_span[0] <= entry[1]:
            return False
        else:
            return True

def check_word(identifiers, words, noun_span):
    #print('noun_span', noun_span)
    for positive in identifiers:
        #print(positive)
        for index, word in enumerate(words):
            #print(word, 'word', index)
            if re.search(positive, word, re.IGNORECASE) is not None:
                if abs(noun_span - index) < 6:
                    return True, positive
                else:
                    continue
            else:
                continue
    
    return False, None

def check_span(span, span_range):
    for i in range(span[0], span[1]):
        if i in span_range:
            return False
    
    return True

def remove_ROS(note):
    try:
        if re.search(r'HPI\/ROS', note) is not None:
            return re.search(r'(.*?)HPI\/ROS(.*)\: (Yes|No)(.*?)', note).group(1) + re.search(r'(.*?)HPI\/ROS(.*)\: (Yes|No)(.*?)', note).group(4)
        elif re.search(r'question\s*\d.\/\d.\/\d*', note, re.IGNORECASE) is not None:
            return re.search(r'(.*?)question\s*\d.\/\d.\/\d*(.*)\: (Yes|No)(.*?)', note, re.IGNORECASE).group(1) + re.search(r'(.*?)question\s*\d.\/\d.\/\d*(.*)\: (Yes|No)(.*?)', note, re.IGNORECASE).group(4)
        else:
            return note
    except:
        return note

def re_order(dictionary):
    keys = list(dictionary.keys())
    keys = sorted(keys)
    
    sorted_list = []
    spans = []
    
    for key in keys:
        #print('key', key)
        sorted_list.append(dictionary[key])
        spans.append(key)
        
    return sorted_list, spans

def find_stool_freq(note):
    bag_of_words_bm = ['(bm[a-zA-Z]*)', '(bowel[ ]{1,}movement[a-zA-Z]*)', '(stool[a-zA-Z]*)', '(bristol)|(bss)', '(move.{,10}bowel[a-z]*)']
    
    bag_of_words_fre = [r'([0-9]+[ ]{0,}\-[ ]{0,}[0-9]+)[^.]{,10}per[ ]{1,}day', 
                            r'([0-9]+[ ]{0,}\-[ ]{0,}[0-9]+)', 
                            r'([0-9]+[ ]{1,}[0-9]+)',r'~[ ]{0,}([0-9]+)',
                            r'(?<!\d|/|\.|&|%)([0-9]{1,2})(?=[^0-9\/])(?![ ]{0,}(week|day|year|month|hour|%|&))', ' (one) ', ' (two) ', 
                            ' (three) ', ' (four) ', ' (five) ', ' (six) ', ' (seven) ', ' (eight) ', ' (nine) ', ' (a) ']
    
    negative_identifiers = [' test[a-zA-Z]*', ' study', ' studie[a-zA-Z]*', ' o&p', ' gram']
    
    bm_sentence = []
    bm_spans = []
    final_matches = []
    track = []
    
    note = preprocessing(note)
    

    
    
    
    for word_bm in bag_of_words_bm:
        
        regex = '[^.]*' + word_bm + '[^.]*'
        
        try:
            trial = re.finditer(regex, note, re.IGNORECASE)

            for match in trial:
                #print(match)
                if not check_spans_overlap(match.span(0), track):
                    if not check_sentence(negative_identifiers, match.group(0)):
                        bm_sentence.append(match.group(0))
                        track.append(match.span(0))
        except:
            continue
    
    #print(bm_sentence)
    for sentence in bm_sentence:
        #print('sentence',sentence)
        temp = []
        temp_spans = []
        
        for word_bm in bag_of_words_bm:
            
            try:
                trial = re.finditer(word_bm, sentence, re.IGNORECASE)
        #print([match.group(0) for match in trial])
                        
                
        

            except:
                continue
            
            for match in trial:
                #print(match)
                if not check_spans_overlap(match.span(1), temp_spans):
                    temp.append((match.group(1), match.span(1)))
                    temp_spans.append((match.span(1)))
                        
        bm_spans.append((sentence, temp))
        
    #print('bm_spans', bm_spans)
    if len(bm_spans) > 0:
    
        for sentence, bm_span in bm_spans:
            
            sentence_matches = []
            number_spans = []
            track_spans = []
            track_matches = []
            
            for word_num in bag_of_words_fre:
                #print(word_num)
                try:
                    trial = re.finditer(word_num, sentence, re.IGNORECASE)
                    
    
                        
                        
                except:
                    #print('error')
                    continue
                
                for match_1 in trial:
                        #print(match)
                    if not check_spans_overlap(match_1.span(1), track_spans):
                            #print(match_1)
                        match_span = match_1.span(1)
                        number_spans.append((match_1.group(1), match_span))
                        track_spans.append(match_1.span(1))
                            
                        
            #print('number_spans', number_spans)   
            
            if len(number_spans) > 0:
                for span in bm_span:
                    #print('span', span)
                    #print(span[0], (span[1][0] + span[1][1])/2 )
                    matches = get_closest_number(number_spans, (span[1][0] + span[1][1])/2)
                    if matches not in track_matches:
                        final_matches.append((sentence, matches[0][0], matches[0][1]))
                        track_matches.append(matches)
                 
            else:
                continue

        return final_matches
                    
    else:
        return np.nan

def find_previous_note(note_id, df):

    row = df[df['deid_note_id'] == note_id]
    patient_id = row['deid_PatientDurableKey']
    date = row['deid_service_date_cdw'].values
    
    
    patient_rows = df[df['deid_PatientDurableKey'] == patient_id.values[0]]
    note_ids = patient_rows['deid_note_id'].values
    dates = patient_rows['deid_service_date_cdw'].values
    
    number_dates = len(dates) 
    
    # creates a dictionary with dates as keys for note ids
    
    dates_dic = dict(zip(dates, note_ids))
    
    # if there are more than one notes with patient id, finds all dates which are less than input date 
    
    if len(dates) > 1:
        previous_dates =[]
        for x in dates:
            if x < date:
                previous_dates.append(x)
        
    # sorts dates in ascending order
        previous_dates = sorted(previous_dates)
                     
    # returns input note id if there are no ther dates corresponding to patient ID
    else:
        return np.nan
    
    # checks to see if there are any dates which were less than date of input note id, returns note_id if none               
    if len(previous_dates) > 0:
        previous_date = previous_dates[-1]
    
    else:
        return np.nan
                    
    #returns note id coressponding to the highest date of those which fall behind input date
    return dates_dic[previous_date]

def encode_column(df1, df2, column_to_encode, encoded_column_name, df_same):
    
    values_list = df2[df2[column_to_encode].notnull()][column_to_encode]
    values_index = values_list.index
    
    encoded_values = []
    
    for index, values in zip(values_index, values_list):
        #print(values)
        Tenses = [value[1] for value in values]
        #print(Tenses)
        #negations = [value[1] for value in values]
        #print(Blood_Values, Blood_Values[0])
        
        values = [value[0] for value in values]
        #print(values)
        
        
        if 'CURRENT' in Tenses:
            True_index = [index for index, tense in enumerate(Tenses) if tense == 'CURRENT']
            True_Values = [value for index, value in enumerate(values) if index in True_index]
            #True_negation = [neg for index, negation in enumerate(negations) if index in True_index]
            
            if len(True_index) == 1:
                encoded_values.append(True_Values[0])
                
    
            else:
                if len(set(True_Values)) == 1:
                    encoded_values.append(list(set(True_Values))[0])
                    
                    
                else:
                    encoded_values.append(int(1))
        
        elif 'NO_VERB' in Tenses or 'SIMPLE_PRESENT' in Tenses:
            True_index = [index for index, tense in enumerate(Tenses) if tense == 'NO_VERB' or tense == 'SIMPLE_PRESENT']
            
            True_Values = [value for index, value in enumerate(values) if index in True_index]
            #True_negation = [neg for index, negation in enumerate(negations) if index in True_index]
            
            if len(True_index) == 1:
                encoded_values.append(True_Values[0])
                
    
            else:
                if len(set(True_Values)) == 1:
                    encoded_values.append(list(set(True_Values))[0])
                    
                    
                else:
                    encoded_values.append(int(1))
                        
                            
                    
                    
        elif 'PRESENT_PART' in Tenses:
            True_index = [index for index, tense in enumerate(Tenses) if tense == 'PRESENT_PART']
            True_Values = [value for index, value in enumerate(values) if index in True_index]
            #True_negation = [neg for index, negation in enumerate(negations) if index in True_index]
            
            if len(True_index) == 1:
                encoded_values.append(True_Values[0])
                
    
            else:
                if len(set(True_Values)) == 1:
                    encoded_values.append(list(set(True_Values))[0])
                    
                    
                else:
                    encoded_values.append(int(1))
        
        elif 'PRES_PERF_CON' in Tenses:

            True_index = [index for index, tense in enumerate(Tenses) if tense == 'PRES_PERF_CON']
            True_Values = [value for index, value in enumerate(values) if index in True_index]
            #True_negation = [neg for index, negation in enumerate(negations) if index in True_index]
            
            if len(True_index) == 1:
                encoded_values.append(True_Values[0])
                
    
            else:
                if len(set(True_Values)) == 1:
                    encoded_values.append(list(set(True_Values))[0])
                    
                    
                else:
                    encoded_values.append(int(1))
    
    if df_same == False:
        id_list = []
        for index in values_index:
            id_list.append(df2['deid_note_id'][index])
            
    
        values_index = get_index(id_list, df1)
    
    
    for index, value in zip(values_index, encoded_values):
        df1.at[index, encoded_column_name] = value

def encode_stool_freq(stool_number, threshold):
    #print(stool_number)
    stripped = stool_number.strip()
    #print(stripped, 'stri')
    if re.search(r'[0-9]+', stripped) is not None:
        stripped = re.sub(r'[^0-9 ]+', ' ', stripped)
        stripped = stripped.strip()
  
    try:
        number = int(stripped[-2:])
        #print(number)
    except:
        try:
            number = w2n.word_to_num(stripped)
            number = int(number)
            #print(number)
        except:
            try:
                if re.search(r'a', stripped) is not None:
                    number = int(1)
            except:
                return 'error'

    
    if number > threshold:
        return 1
    else:
        return 0

def encode_diarrhea(df1, df2, column_to_encode, encoded_column_name, df_same):
    
    values_list = df2[df2[column_to_encode].notnull()][column_to_encode]
    values_index = values_list.index
    
    encoded_values = []
    
    for index, values in zip(values_index, values_list):
        
        Tenses = [value[2] for value in values]
        #print(Tenses)
        stool_values = [value[1] for value in values]
        #print(Blood_Values, Blood_Values[0])
        
        keywords = [value[0] for value in values]

def get_range(HPI_offset, HPI):
    lst = []
    lst.append((int(HPI_offset), int(HPI_offset + len(HPI))))
    return lst

def get_index(lst, df):
    lst_ = []
    for l in lst:
        lst_.append(df[df['deid_note_id'] == l].index.values[0])
        
    
    return lst_

# ### Remaining Notes ###

### Function drops entity mentions that fall within the first 40% character range of the HPI 

def get_mention_location(HPI, mentions, spans, keywords):
    
    count = 0
    
    if isinstance(mentions, list) and len(mentions) is not 0:
        character_length = float(len(HPI))
        #print(character_length)

        for index, mention in enumerate(mentions):
            #print(mention)
            doc_location = float(spans[index])/character_length 
            #print(doc_location)
            if doc_location < .3:
                del mentions[index - count]
                del keywords[index - count]
                count += 1 

            
    return mentions, keywords


def consolidate_columns(row_names, master_column, df):
    conflicting_rows = []
    
    for index, row in df.iterrows():
        #print(row[row_names[0]])
        if not math.isnan(row[row_names[0]]): 
            #print(row[row_names[0]])
            df.at[index, master_column] = row[row_names[0]]
            continue
        elif not math.isnan(row[row_names[1]]): #pain_mention_Interval_History
            if not math.isnan(row[row_names[2]]): #'pain_mention_Previous_note'
                if row[row_names[1]] == row[row_names[2]]:
                    df.at[index, master_column] = row[row_names[1]] #pain_mention_Interval_History
                    continue
                else:
                    df.at[index, master_column] = row[row_names[1]] #pain_mention_Interval_History
                    conflicting_rows.append(index)
                    continue
            else:
                df.at[index, master_column] = row[row_names[1]] #pain_mention_Interval_History
                continue
                
        elif not math.isnan(row[row_names[2]]) and not type(row['Interval_History']) == str: #'pain_mention_Previous_note'
                df.at[index, master_column] = row[row_names[2]] #'pain_mention_Previous_note'
                continue
            
        
        else:
            if not math.isnan(row[row_names[3]]):
                df.at[index, master_column] = row[row_names[3]]
                continue
        
    
    return conflicting_rows

