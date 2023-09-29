import math
import os
import re
import warnings
from datetime import date, datetime, time
from xml.dom import minidom

import matplotlib.pyplot as plt
# import medspacy
import numpy as np
import pandas as pd
import spacy
from negspacy.negation import Negex
from negspacy.termsets import termset
from spacy import displacy
from spacy.language import Language
from spacy.matcher import Matcher, PhraseMatcher
from spacy.pipeline import EntityRuler
from spacy.tokens import Span
from spacy.util import filter_spans
# from tqdm import tqdm

row_names = [
    "Pain_ROS",
    "pain_mention_Interval_History",
    "pain_mention_Previous_note",
    "pain_mention_HPI",
]
row_names_blood = [
    "Fecal_Blood_Value",
    "blood_mention_Interval_History",
    "blood_mention_Previous_note",
    "blood_mention_HPI",
]
rows_names_cr = [
    "CR_ROS",
    "cr_mention_Interval_History",
    "cr_mention_Previous_note",
    "cr_mention_HPI",
]
row_names_well = [
    "Well_ROS",
    "well_mention_Interval_History",
    "well_mention_Previous_note",
    "well_mention_HPI",
]


def return_scispacy():
    # nlp = spacy.load('en_core_sci_lg-0.4.0/en_core_sci_lg/en_core_sci_lg-0.4.0/')
    nlp = spacy.load("en_core_sci_lg")
    return nlp


sci_spacy_model = return_scispacy()


def reorder_ctakes(list_of_dics):
    return sorted(list_of_dics, key=lambda d: d["text_range"][0])


def add_column_todf(df1, df2, column_name, note_id_name):

    for index, row in df1.iterrows():
        value = row[column_name]
        # value  = convert_list_string(value)
        # print(value)
        id_ = row[note_id_name]
        index_one = df2[df2[note_id_name] == id_].index[0]
        # print(index)

        df2.loc[index_one, column_name] = value


def re_order(dictionary):
    keys = list(dictionary.keys())
    keys = sorted(keys)

    sorted_list = []
    spans = []

    for key in keys:
        # print('key', key)
        sorted_list.append(dictionary[key])
        spans.append(key)

    return sorted_list, spans


def find_blood(note):

    if not isinstance(note, str):
        return np.nan, np.nan, np.nan

    bag_of_words = [
        "blood",
        "bleed",
        "hematochezia",
        "BRB[a-zA-Z]*?",
        "maroon",
        "melena",
        "bright[ ]{1,red[ ]{1,}blood",
        "red(?:.*)stool",
    ]
    matches = {}
    keywords = []

    note = preprocessing(note)

    for word in bag_of_words:

        regex = r"[^.]*" + word + r"[^.]*\."
        try:
            trial = re.finditer(regex, note, re.IGNORECASE)

            for match in trial:
                # print(match)
                span = match.span()
                matches[span[1]] = match.group(0)
                keywords.append(word)

        except:
            continue

    ordered_matches, spans = re_order(matches)

    # print(ordered_matches)
    # count = 0
    indices = []
    for index, match in enumerate(ordered_matches):
        # print(match)
        if (
            re.search(r" test ", match, re.IGNORECASE) is not None
            or re.search(r"blood\s*work", match, re.IGNORECASE) is not None
            or re.search(r"blood[ ]{0,}sugar", match, re.IGNORECASE) is not None
            or re.search(r" clot ", match, re.IGNORECASE) is not None
            or re.search(r"transfusion", match, re.IGNORECASE) is not None
            or re.search(r"abscess", match, re.IGNORECASE) is not None
            or re.search(r"blood\s*count", match, re.IGNORECASE) is not None
            or re.search(r"blood\s*culture", match, re.IGNORECASE) is not None
            or re.search(r"blood\s*loss", match, re.IGNORECASE) is not None
        ):  # or re.search(r' stoma ', match, re.IGNORECASE) is not None #or re.search(r'anastomotic', match, re.IGNORECASE) is not None: #or re.search(r'output', match, re.IGNORECASE) is not None or re.search(r'fistula', match, re.IGNORECASE) is not None:
            indices.append(index)

    count = 0
    for index in indices:
        del ordered_matches[index - count]
        del keywords[index - count]
        del spans[index - count]
        count += 1

    if len(ordered_matches) == 0:
        return np.nan, np.nan, np.nan
    else:
        return ordered_matches, keywords, spans


def map_pain(note, nlp_spacy):

    note = preprocessing(note)
    doc = nlp_spacy(note)

    pain_map = []
    previous_pain = True

    for ent in doc.ents:
        if ent.label_ == "PAIN":
            pain_map.append(1)
            previous_pain = True

        elif ent.label_ == "PAIN_GEN":
            pain_token = doc[ent.start]
            pain_sentence = pain_token.sent
            pain_adj = get_pain_adjective(pain_sentence.text, doc, pain_token.i)

            if pain_adj == True and previous_pain == True:
                pain_map.append(1)
                previous_pain = True
            else:
                pain_map.append(0)
                previous_pain = False

    return pain_map


def find_pain(note, nlp_spacy):
    bag_of_words = [
        "abd[a-zA-Z]*[ ]{1,}(pain|cramp[a-zA-Z]*|discomfort|ache)",
        "rlq[ ]{1,}(pain|cramp[a-zA-Z]*|discomfort|ache)",
        "epigastric[ ]{1,}(pain|cramp[a-zA-Z]*|discomfort|ache)",
        "llq[ ]{1,}(pain|cramp[a-zA-Z]*|discomfort|ache)",
        "pelvic[ ]{1,}(pain|cramp[a-zA-Z]*|discomfort|ache)",
        "stomach[ ]{1,}(pain|cramp[a-zA-Z]*|discomfort|ache)",
        "luq[ ]{1,}(pain|cramp[a-zA-Z]*|discomfort|ache)",
        "ruq[ ]{1,}(pain|cramp[a-zA-Z]*|discomfort|ache)",
        "right[ ]{1,}lower[ ]{1,}quadrant[ ]{1,}(pain|cramp[a-zA-Z]*|discomfort|ache)",
        "left[ ]{1,}lower[ ]{1,}quadrant[ ]{1,}(pain|cramp[a-zA-Z]*|discomfort|ache)",
        "left[ ]{1,}upper[ ]{1,}quadrant[ ]{1,}(pain|cramp[a-zA-Z]*|discomfort|ache)"
        "right[ ]{1,}upper[ ]{1,}quadrant[ ]{1,}(pain|cramp[a-zA-Z]*|discomfort|ache)",
    ]

    gen_bag_of_words = ["(pain )", "(discomfort)", "(cramp[a-zA-Z]*)", "(ache)"]

    matches = {}
    keywords = []
    spans_1 = []
    nlp_en = spacy.blank("en")

    if not isinstance(note, str):
        return np.nan, np.nan, np.nan

    note = preprocessing(note)

    for word in bag_of_words:

        regex = "[^.]*" + word + "[^.]*"

        try:
            trial = re.finditer(regex, note, re.IGNORECASE)

            for match in trial:
                # print(match)
                span = match.span()
                spans_1.append(span)
                matches[span[1]] = match.group(0)
                keywords.append(word)
        except:
            continue

    for word in gen_bag_of_words:

        try:
            regex = r"[^.]*" + word + r"[^.]*(?:\.|$)"
            trial = re.finditer(regex, note, re.IGNORECASE)

            for match in trial:
                # print(match)
                start, end = match.span(1)
                # print(start, end)
                doc = nlp_en(match.group(0))
                location = doc.char_span(start, end, alignment_mode="expand")

                if match.span() not in spans_1:
                    if get_pain_adjective(match.group(0), doc, location) == True:
                        # print('True')
                        span = match.span()
                        matches[span[1]] = match.group(0)
                        keywords.append("PAIN_GEN")
                    else:
                        continue
                else:
                    continue
        except:
            continue

    ordered_matches, spans = re_order(matches)

    count = 0

    if "PAIN_GEN" not in keywords and len(ordered_matches) != 0:
        return ordered_matches, keywords, spans

    if len(matches) != 0:
        pain_map = map_pain(note, nlp_spacy)
        # print(pain_map)

        for (index, match), pain in zip(enumerate(ordered_matches), pain_map):

            if pain == 0:
                del ordered_matches[index - count]
                del keywords[index - count]
                del spans[index - count]
                count += 1
                continue

            # if re.search(r'fistula', match, re.IGNORECASE) is not None:
            # del ordered_matches[index - count]
            # del keywords[index - count]
            # del spans[index - count]
            # count += 1
            # continue

            if not with_bm(match, nlp_spacy):
                del ordered_matches[index - count]
                del keywords[index - count]
                del spans[index - count]
                count += 1

    else:
        return np.nan, np.nan, np.nan

    if len(ordered_matches) == 0:
        return np.nan, np.nan, np.nan

    return ordered_matches, keywords, spans


def remove_ROS(note):
    try:
        if re.search(r"HPI\/ROS", note) is not None:
            return re.search(
                r"(.*?)Answers for HPI\/ROS(.*)\: (Yes|No)(.*?)", note
            ).group(1) + re.search(
                r"(.*?)Answers for HPI\/ROS(.*)\: (Yes|No)(.*?)", note
            ).group(
                4
            )
        elif re.search(r"question\s*\d.\/\d.\/\d*", note, re.IGNORECASE) is not None:
            return re.search(
                r"(.*?)question\s*\d.\/\d.\/\d*(.*)\: (Yes|No)(.*?)",
                note,
                re.IGNORECASE,
            ).group(1) + re.search(
                r"(.*?)question\s*\d.\/\d.\/\d*(.*)\: (Yes|No)(.*?)",
                note,
                re.IGNORECASE,
            ).group(
                4
            )
        else:
            return note
    except:
        return note


def preprocessing(note):

    if not isinstance(note, str):
        return note

    processed_note = remove_ROS(note)

    processed_note = re.sub(r"\'", "", processed_note)  # removes apostrohpes
    processed_note = re.sub(r"\(", "", processed_note)
    processed_note = re.sub(r"\)", "", processed_note)
    processed_note = re.sub(
        "([a-zA-Z])/([a-zA-Z])", r"\1, \2", processed_note
    )  # removes / and replaces with comma
    processed_note = re.sub(r"\-", " ", processed_note)  # removes emdash
    # processed_note = re.sub(r'(?<=\w)\s{3,}', '.    ', processed_note) # replaces
    processed_note = re.sub(r"([a-z0-9])\s{2,}([A-Z])", r"\1. \2", processed_note)
    processed_note = re.sub(r"([0-9])\.([0-9])", r"\1\2", processed_note)
    processed_note = re.sub(r";", ".", processed_note)
    # processed_note = remove_ROS(processed_note)

    return processed_note


def find_clinical_remission(note):

    if not isinstance(note, str):
        return np.nan, np.nan, np.nan

    bag_of_words = [
        "clin[a-zA-Z]*[ ]{1,}rem[a-zA-Z]*",
        "quiescent",
        "symptomatic[ ]{1,}remission",
    ]
    matches = {}
    keywords = []

    note = preprocessing(note)

    for word in bag_of_words:

        regex = r"[^.]*" + word + r"[^.]*\."
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

    bag_of_words = [
        "doing[ ]{1,}well",
        "asymptomatic[a-zA-Z]*",
        "(no|w/o)[ ]{1,}sx[a-zA-Z]*",
        "no[ ]{1,}complaint",
        "gi[ ]{1,}(issue[a-zA-Z]*|complaint[a-zA-Z]*|symptom[a-zA-Z]*)",
        "normal[ ]{1,}(bm[a-zA-Z]*|bowel[a-zA-Z]*)",
        "feel[a-zA-Z]*[ ]{1,}well",
    ]
    matches = {}
    keywords = []

    note = preprocessing(note)

    for word in bag_of_words:

        regex = r"[^.]*" + word + r"[^.]*\."
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


def return_abdominal_spacy():
    ts = termset("en_clinical")
    nlp_spacy = spacy.load("en_core_web_sm", disable=["ner"])
    ruler = nlp_spacy.add_pipe("entity_ruler")
    sentencizer = nlp_spacy.add_pipe("sentencizer")
    ruler.add_patterns(
        [
            {"label": "BLOOD", "pattern": [{"TEXT": {"REGEX": "(b|B)lood[a-zA-Z]*"}}]},
            {"label": "BLOOD", "pattern": [{"TEXT": {"REGEX": "(b|B)leed[a-zA-Z]*"}}]},
            {
                "label": "BLOOD",
                "pattern": [{"TEXT": {"REGEX": "(h|H)ematochezia[a-zA-Z]*"}}],
            },
            {
                "label": "BLOOD",
                "pattern": [{"TEXT": {"REGEX": "(BRBPR[a-zA-Z]*|brbpr[a-zA-Z]*)"}}],
            },
            {"label": "BLOOD", "pattern": [{"TEXT": {"REGEX": "(m|M)aroon[a-zA-Z]*"}}]},
            {"label": "BLOOD", "pattern": [{"TEXT": {"REGEX": "(m|M)elena[a-zA-Z]*"}}]},
            {"label": "BLOOD", "pattern": [{"TEXT": {"REGEX": "(B|b)rown"}}]},
            {
                "label": "PAIN_GEN",
                "pattern": [{"TEXT": {"REGEX": "(c|C)ramp[a-zA-Z]*"}}],
            },
            # {"label": 'PAIN', "pattern": [{"TEXT": {"REGEX": r"(a|A)bdominal"}}, {"TEXT": {"REGEX": r"(p|P)ain"}}]},
            {
                "label": "PAIN",
                "pattern": [
                    {"TEXT": {"REGEX": r"(a|A)bd[a-zA-Z]*"}},
                    {"TEXT": {"REGEX": r"(p|P)ain"}},
                ],
            },
            {
                "label": "PAIN",
                "pattern": [
                    {"TEXT": {"REGEX": r"(R|r)lq"}},
                    {"TEXT": {"REGEX": r"(p|P)ain"}},
                ],
            },
            {
                "label": "PAIN",
                "pattern": [
                    {"TEXT": {"REGEX": r"(R|r)uq"}},
                    {"TEXT": {"REGEX": r"(p|P)ain"}},
                ],
            },
            {
                "label": "PAIN",
                "pattern": [
                    {"TEXT": {"REGEX": r"llq"}},
                    {"TEXT": {"REGEX": r"(p|P)ain"}},
                ],
            },
            {
                "label": "PAIN",
                "pattern": [
                    {"TEXT": {"REGEX": r"LLQ"}},
                    {"TEXT": {"REGEX": r"(p|P)ain"}},
                ],
            },
            {
                "label": "PAIN",
                "pattern": [
                    {"TEXT": {"REGEX": r"(P|p)elvic"}},
                    {"TEXT": {"REGEX": r"(p|P)ain"}},
                ],
            },
            {
                "label": "PAIN",
                "pattern": [
                    {"TEXT": {"REGEX": r"(A|a)bdominal"}},
                    {"TEXT": {"REGEX": r"(D|d)iscomfort"}},
                ],
            },
            {
                "label": "PAIN",
                "pattern": [
                    {"TEXT": {"REGEX": r"(O|o)bstruct[a-zA-Z]*"}},
                    {"TEXT": {"REGEX": r"(S|s)ymptom[a-zA-Z]*"}},
                ],
            },
            {
                "label": "WELL",
                "pattern": [
                    {"TEXT": {"REGEX": r"(F|f)eel[a-zA-Z]*"}},
                    {"TEXT": {"REGEX": r"(W|w)ell[a-zA-Z]*"}},
                ],
            },
            {
                "label": "WELL",
                "pattern": [
                    {"TEXT": {"REGEX": r"(C|c)linical"}},
                    {"TEXT": {"REGEX": r"(R|r)emission"}},
                ],
            },
            {"label": "BM", "pattern": [{"TEXT": {"REGEX": "urgency"}}]},
            {"label": "BM", "pattern": [{"TEXT": {"REGEX": "(B|b)(M|m)[a-zA-Z]*"}}]},
            {"label": "BM", "pattern": [{"TEXT": {"REGEX": r"(B|b)\.(M|m)\."}}]},
            {
                "label": "PAIN_GEN",
                "pattern": [{"TEXT": {"REGEX": "(p|P)ain[a-zA-Z]*"}}],
            },
            {"label": "PAIN_GEN", "pattern": [{"TEXT": {"REGEX": "(d|D)iscomfort"}}]},
            {
                "label": "CURRENT",
                "pattern": [{"TEXT": {"REGEX": "(c|C)urrent[a-zA-Z]*"}}],
            },
            {
                "label": "CURRENT",
                "pattern": [{"TEXT": {"REGEX": "(t|T)oday[a-zA-Z]*"}}],
            },
            {
                "label": "CURRENT",
                "pattern": [{"TEXT": {"REGEX": "(y|Y)esterday[a-zA-Z]*"}}],
            },
            {"label": "CURRENT", "pattern": [{"TEXT": {"REGEX": "(p|P)resent( |,)"}}]},
            {
                "label": "CURRENT",
                "pattern": [
                    {"TEXT": {"REGEX": "(l|L)ast"}},
                    {"TEXT": {"REGEX": "(w|W)eek"}},
                ],
            },
            ##{"label": 'CURRENT', "pattern": [{"TEXT": {"REGEX": "(n|N)ow"}}]},
            {"label": "FORMED", "pattern": [{"TEXT": {"REGEX": "(F|f)ormed"}}]},
            {
                "label": "Clin_Rem",
                "pattern": [
                    {"TEXT": {"REGEX": "clin[a-zA-Z]*"}},
                    {"TEXT": {"REGEX": "rem[a-zA-Z]*"}},
                ],
            },
            {"label": "Clin_Rem", "pattern": [{"TEXT": {"REGEX": "remission"}}]},
            {
                "label": "Clin_Rem",
                "pattern": [{"TEXT": {"REGEX": "asymptomatic[a-zA-Z]*"}}],
            },
            {
                "label": "Clin_Rem",
                "pattern": [
                    {"TEXT": {"REGEX": "(n|N)o"}},
                    {"TEXT": {"REGEX": "sx[a-zA-Z]*"}},
                ],
            },
        ]
    )

    ts.remove_patterns(
        {
            "pseudo_negations": [
                "no further",
                "not able to be",
                "not certain if",
                "not certain whether",
                "not necessarily",
                "without any further",
                "without difficulty",
                "without further",
                "might not",
                "not only",
                "no increase",
                "no significant change",
                "no change",
                "no definite change",
                "not extend",
                "not cause",
            ]
        }
    )

    ### additional term sets can be added to negspacy-- for full documentation see: https://pypi.org/project/negspacy/

    ts.add_patterns(
        {
            "preceding_negations": [
                "hasn't noticed",
                "non",
                "non-",
                "no",
                "none",
                "no further",
                "deny",
                "hasnt had any",
                "hasn't had",
                "hasn't",
                "hasnt",
                "resolution",
                "resolution of",
            ],
            "following_negations": ["resolved", "subsided", "none now"],
            "pseudo_negations": [
                "does not typically",
                "almost resolved",
                "no significant",
            ],
            "termination": [
                "endorses",
                "and only",
                "scant",
                "intermittent",
                "with",
                "then",
                ";",
                "severe",
                "infrequent",
            ],
        }
    )

    nlp_spacy.add_pipe("negex", config={"neg_termset": ts.get_patterns()})
    return nlp_spacy


@Language.component("Multi_Word_NER")
def Multi_Word_NER(doc):

    # text = doc.text
    # print(doc)

    # doc = nlp_spacy(doc.text)
    bag_of_words_bm = [
        "(bm[a-zA-Z]*)",
        "(bowel[ ]{1,}movement[a-zA-Z]*)",
        "(stool[a-zA-Z]*)",
        "(move.{,10}bowel[a-z]*)",
    ]

    Well_bag_of_words = [
        "(doing[ ]{1,}well)",
        "(asymptomatic[a-zA-Z]*)",
        "(no|w/o)[ ]{1,}sx[a-zA-Z]*",
        "(no[ ]{1,}complaint)",
        "(gi[ ]{1,}(issue[a-zA-Z]*|complaint[a-zA-Z]*|symptom[a-zA-Z]*))",
        "(normal[ ]{1,}(bm[a-zA-Z]*|bowel[a-zA-Z]*))",
        "(feel[a-zA-Z]*[ ]{1,}well)",
    ]

    blood_bag_of_words = ["(bright[ ]{1,red[ ]{1,}blood)", "(red(?:.*)stool)"]

    cr_bag_of_words = [
        "(clin[a-zA-Z]*[ ]{1,}rem[a-zA-Z]*)",
        "(quiescent)",
        "(symptomatic[ ]{1,}remission)",
    ]

    Well_bag_of_words_neg = [
        "(gi[ ]{1,}(issue[a-zA-Z]*|complaint[a-zA-Z]*|symptom[a-zA-Z]*))"
    ]

    pain_bag_of_words = [
        "(abd[a-zA-Z]*[ ]{1,}(pain|cramp[a-zA-Z]*|discomfort|ache))",
        "(rlq[ ]{1,}(pain|cramp[a-zA-Z]*|discomfort|ache))",
        "(epigastric[ ]{1,}(pain|cramp[a-zA-Z]*|discomfort|ache))",
        "(llq[ ]{1,}(pain|cramp[a-zA-Z]*|discomfort|ache))",
        "(pelvic[ ]{1,}(pain|cramp[a-zA-Z]*|discomfort|ache))",
        "(stomach[ ]{1,}(pain|cramp[a-zA-Z]*|discomfort|ache))",
        "(luq[ ]{1,}(pain|cramp[a-zA-Z]*|discomfort|ache))",
        "(ruq[ ]{1,}(pain|cramp[a-zA-Z]*|discomfort|ache))"
        "(right[ ]{1,}lower[ ]{1,}quadrant[ ]{1,}(pain|cramp[a-zA-Z]*|discomfort|ache))",
        "(left[ ]{1,}lower[ ]{1,}quadrant[ ]{1,}(pain|cramp[a-zA-Z]*|discomfort|ache))",
        "(left[ ]{1,}upper[ ]{1,}quadrant[ ]{1,}(pain|cramp[a-zA-Z]*|discomfort|ache))"
        "(right[ ]{1,}upper[ ]{1,}quadrant[ ]{1,}(pain|cramp[a-zA-Z]*|discomfort|ache))",
    ]

    bag_of_words_gen = [
        "(bm[a-zA-Z]*)",
        "(bowel[ ]{1,}movement[a-zA-Z]*)",
        "(stool[a-zA-Z]*)",
    ]

    current_bag_of_words = [
        "(past[ ]{1,}week)",
        "(last[ ]{1,}week)",
        "(this[ ]{1,}week)",
    ]

    for ent_category, Label in zip(
        [
            blood_bag_of_words,
            pain_bag_of_words,
            bag_of_words_gen,
            Well_bag_of_words,
            Well_bag_of_words_neg,
            cr_bag_of_words,
            current_bag_of_words,
        ],
        ["BlOOD", "PAIN", "PAIN_gen", "WELL", "WELL_neg", "Clin_Rem", "CURRENT"],
    ):

        # print([token.text for token in doc])
        original_ents = list(doc.ents)

        mwt_ents = []
        indexes = []
        # track_ents = []
        Label_gen = []

        # print('orriginalents', original_ents)

        for regex in ent_category:
            for match in re.finditer(regex, doc.text, re.IGNORECASE):

                start, end = match.span(1)
                # print('start', start, end)
                span = doc.char_span(start, end, alignment_mode="expand")

                if Label == "PAIN_gen":

                    if not get_pain_adjective(doc.text, doc, span.start):
                        continue

                if span is not None:
                    mwt_ents.append((span.start, span.end, span.text))
                    # print(mwt_ents)
                    token_locations = (span.start, span.end)
                    indexes.append(token_locations)
                    # track_ents.append((start, end))

        mwt_ents.sort(key=lambda x: x[1])

        indexes.sort(key=lambda x: x[1])

        if len(mwt_ents) != 0:
            if Label == "PAIN_gen":
                Label == "PAIN"
            for ent in mwt_ents:
                start, end, name = ent
                if end - start != 0:
                    per_ent = Span(doc, start, end, label=Label)
                    original_ents.append(per_ent)
                else:
                    continue

            filtered = filter_spans(original_ents)
            doc.ents = filtered

            count = 0

            for start, end in indexes:
                # print(start, end)
                with doc.retokenize() as retokenizer:
                    if end - start > 1:
                        start -= count
                        end -= count
                        attrs = {"POS": "NOUN", "ENT_TYPE": Label}
                        retokenizer.merge(doc[start:end], attrs=attrs)
                        count += end - start - 1
                    else:
                        pass
                # count += 1

        else:
            continue

    return doc


def add_pipe(nlp_spacy):
    nlp_spacy.add_pipe("Multi_Word_NER", before="negex")
    return nlp_spacy


def word_vector_comparison(keywords, threshold, word, vectors_model=sci_spacy_model):
    warnings.filterwarnings("ignore")

    doc1 = vectors_model(word)

    for keyword in keywords:
        # print(word, keyword)
        # keyword = remove_stopwords(keyword)
        doc2 = vectors_model(keyword)
        # print(doc1[0], doc2[0])
        similarity = doc1[0].similarity(doc2[0])

        # print(word, similarity, keyword)
        if similarity > threshold:
            return True

    return False


def get_pain_adjective(sentence, doc, location, vectors_model=sci_spacy_model):

    positive_indicators = [
        "epigastric",
        "flank",
        "abdominal",
        "abdomen",
        "stomach",
        "abd",
        "pelvic",
        "pelvis",
        "periumbilical",
        "quadrant",
        "RLQ",
        "rlq",
        "LLQ",
        "llq",
        "RUQ",
        "ruq",
        "luq",
        "LUQ",
    ]
    anatomy_terms = [
        "joint",
        "eye",
        "abdominal",
        "wrist",
        "back",
        "neck",
        "ankle",
        "rectal",
        "msk",
        "medication",
    ]

    ### remove / and other special characters
    # sentence = preprocessing(sentence)

    # doc = nlp_spacy(sentence)
    # print(doc)
    children_left_pos = []
    children_left_text = []

    sen_length = len([token.text for token in doc])

    for token in doc:
        if token.i == location:
            # print(token.text, token.pos_, token.dep_, token.ent_type_, token.head)
            # print([[child.text, child.dep_] for child in token.children])
            if token.pos_ == "NOUN":

                children_left_pos += [child.pos_ for child in token.lefts]
                children_left_text += [child.text for child in token.lefts]

                for child in token.lefts:
                    if len([sub_child.text for sub_child in child.children]) < 0:
                        children_left_pos += [
                            sub_child.pos_ for sub_child in child.children
                        ]
                        children_left_text += [
                            sub_child.text for sub_child in child.children
                        ]

                if token.dep_ == "compound":
                    head_text = token.head.text
                    head_pos = token.head.pos_
                    children_left_text.append(head_text)
                    children_left_pos.append(head_pos)

                index = token.i

                # children_pos = [child.text for child in token.subtree]

                # print(token.head)

                if not token.is_sent_end:
                    # print(token.pos_, 'yes', token.text)
                    if doc[index + 1].pos_ == "ADP":
                        children_left_text += [
                            child.text for child in doc[index + 1].children
                        ]
                        children_left_pos += [
                            child.pos_ for child in doc[index + 1].children
                        ]
                # print(children_left_text)
                # print([child.text for child in token.ancestors])

                # print(children_left_text)
                # if 'ADP' in children_pos:
                # for child in token.children:
                # if child.pos_ == 'ADP':
                # print(yes)

                if token.dep_ == "conj":
                    head_children_pos = [child.pos_ for child in token.head.lefts]
                    head_children_text = [child.text for child in token.head.lefts]
                    # print(head_children_text)

                    children_left_pos += head_children_pos
                    children_left_text += head_children_text

                    # print(children_left_text)
                    # print(children_left_pos)

            elif token.pos_ == "VERB":
                return True

    # print(children_left_text)
    if len(children_left_text) > 0:

        if "ADJ" in children_left_pos or "NOUN" in children_left_pos:
            if len(set(children_left_text).intersection(positive_indicators)) == 0:
                if len(set(children_left_text).intersection(anatomy_terms)) == 0:

                    # print(True)
                    for word in children_left_text:
                        # print(word)
                        if word_vector_comparison(anatomy_terms, 0.30, word):
                            return False

                    return True

                else:
                    return False

            else:
                return True
        else:
            return True

    else:
        return True

    """ elif token.pos_ == 'ADJ':
        ancestor_pos =  [ancestor.pos_ for ancestor in token.ancestors]
        ancestor_text = [ancestor.text for ancestor in token.ancestors]
        ancestor_ents = [ancestor.ent_type_ for ancestor in token.ancestors]


        if len(set(ancestor_text).intersection(positive_indicators)) == 0:
            for word in ancestor_text:
                    if word_vector_comparison(nlp, positive_indicators, .30, word):
                        return True
                    else:
                        return False

        else:
            return True
    """


def convert_list_string(list_strings):
    try:
        string = ""
        for lst in list_strings:
            string += lst
        return string
    except:
        return list_strings


def get_text_from_character_range(character_range, note):
    text = ""
    try:
        for range_ in character_range:
            start = int(range_[0])
            end = int(range_[1])
            sentences = note[start:end]
            text += sentences
        return text
    except:
        return 1


def takeSecond(elem):
    return elem[1]


def with_bm(sentence, nlp_spacy):
    doc = nlp_spacy(sentence)
    # doc = retokenize_ent(bm_bag_of_words, sentence, doc, 'BM')

    pain_location = []
    bm_location = np.nan

    for token in doc:
        # print(token.text)
        if token.ent_type_ == "BM":
            bm_location = token.i
        elif token.ent_type_ == "PAIN":
            pain_location.append(token.i)

        elif token.ent_type_ == "PAIN_GEN":
            pain_location.append(token.i)

    if math.isnan(bm_location):
        return True
    else:

        for token_index in pain_location:
            if abs(token_index - bm_location) < 4:
                return False

        return True
        # previous_word = doc[token.i - 1]
        # if previous_word.pos_ == 'ADP':
        # return False


def present_tense(nlp_spacy, sentence, ent_label, ent_start):

    present_tense_verbs = ["VBP", "VBZ"]
    present_participle = ["VBG"]
    past_tense = ["VBD"]
    past_participle = ["VBN"]

    All_present_tense_verbs = ["VB", "VBP", "VBZ", "VBG"]
    past_tense_all = ["VBD", "VBN"]

    verbs = ["VBP", "VBZ", "VBD", "VBN", "VBG"]

    sentence = re.sub(r"\*", "", sentence)
    sentence = re.sub(r"nb", "n b", sentence, re.IGNORECASE)

    # span_text = kwargs.get('span_text', None)

    if ent_label == "Diarrhea":
        ent_label = ["Diarrhea", "Diarrhea_neg", "Diarrhea_imp"]

    elif ent_label == "WELL":
        ent_label = ["WELL", "WELL_neg"]

    else:
        ent_label = [ent_label]

    doc = nlp_spacy(sentence)

    # doc = retokenize_ent(diarrhea_bag_of_words, sentence, doc, 'Diarrhea')

    # doc = retokenize_ent(diarrhea_negation, sentence, doc, 'Diarrhea_neg')

    # if ent_label == 'range':
    # doc = tokenize_range(doc, sentence, span_text, 'range')

    false_negatives = ["left"]
    # print(doc.ents, [d.label_ for d in doc.ents])

    pos_tags = [token.tag_ for token in doc]
    ents_types = [ent.label_ for ent in doc.ents]
    pos_ = [token.pos_ for token in doc]

    # print(pos_tags, pos_)

    if len(set(pos_tags).intersection(verbs)) == 0:
        # print('no_verb')                           #checks if there are verbs in sentence
        return True, "SIMPLE_PRESENT"

    if "CURRENT" in ents_types:
        # print('current_label')
        return True, "CURRENT"

    if (
        len(set(pos_tags).intersection(All_present_tense_verbs))
        + len(set(pos_tags).intersection(past_tense_all))
        == 1
    ):
        # print('yes')
        if len(set(pos_tags).intersection(All_present_tense_verbs)) == 1:
            return True, "SIMPLE_PRESENT"
        elif (
            len(set(pos_tags).intersection(past_tense_all)) == 1
            and "FORMED" not in ents_types
        ):
            return False, "SIMPLE_PAST"
        else:
            return True, "SIMPLE_PRESENT"

    for token in doc:

        if token.ent_type_ in ent_label and token.i == ent_start:
            # print(token.text, token.tag_, token.pos_, token.dep_, [token.text for token in token.ancestors],[token.tag_ for token in token.ancestors], [token.pos_ for token in token.ancestors], [token.dep_ for token in token.ancestors]  )

            if token.tag_ == "VBG":
                # print(token.text)
                # print('heretwo')
                for child in token.children:
                    if child.pos_ == "AUX" and child.tag_ in present_tense_verbs:
                        return True, "SIMPLE_PRESENT"
                    elif child.pos_ == "AUX" and child.tag_ in past_tense:
                        return False, "SIMPLE_PAST"

            if token.dep_ == "ROOT" and token.pos_ == "VERB":
                # print(token.text, token.pos_, 'here')
                # print([(child.text, child.pos_) for child in token.children])

                for child in token.children:
                    # print([(child.text, children.pos_) for child in token.children])
                    if child.tag_ in present_tense_verbs:
                        return True, "SIMPLE_PRESENT"
                    elif child.tag_ in past_participle:
                        for child_2 in child.children:
                            if (
                                child_2.pos_ == "AUX"
                                and child_2.tag_ in present_tense_verbs
                            ):
                                return True, "PAST_PART"
                    elif child.tag in past_tense:
                        return False, "PAST_TENSE"
                    else:
                        return True, "SIMPLE_PRESENT"

            for ancestor in token.ancestors:
                # print(ancestor.text, ancestor.pos, ancestor.tag_, ancestor.morph, ancestor.dep_, ancestor.head, 'here')

                # print([children.tag_ for children in ancestor.children])

                if ancestor.tag_ in present_tense_verbs:
                    # print('ancestor Present')
                    if (
                        ancestor.dep_ != "ccomp"
                        and ancestor.dep_ != "xcomp"
                        and ancestor.dep_ != "pcomp"
                    ):
                        # print('here erorr')
                        # if ancestor.dep_ == 'ROOT':
                        # return True, 'SIMPLE_PRESENT'
                        # print('here')
                        ancestor_tags = [children.tag_ for children in ancestor.lefts]
                        for tag in ancestor_tags:
                            # print('here')
                            if tag in past_tense_all:
                                return False, "SIMPLE_PAST"

                            else:
                                continue

                        return True, "SIMPLE_PRESENT"

                    else:
                        # print('printhere')
                        ancestor_pos_ = [
                            children.pos_ for children in ancestor.ancestors
                        ]
                        ancestor_text = [
                            children.text for children in ancestor.ancestors
                        ]
                        # print(ancestor_pos_, ancestor_text,'ancestor_pos_')
                        if not "VERB" in ancestor_pos_ or "AUX" in ancestor_pos_:
                            # print('here')
                            return True, "SIMPLE_PRESENT"
                    # print('simple_present')

                elif ancestor.tag_ in past_tense:
                    # print('herere')
                    if (
                        ancestor.dep_ != "ccomp"
                        and ancestor.dep_ != "xcomp"
                        and ancestor.dep_ != "pcomp"
                    ):
                        if (
                            ancestor.ent_type_ != "FORMED"
                            and ancestor.text not in false_negatives
                        ):
                            return False, "SIMPLE_PAST"

                        elif ancestor.dep_ == "ROOT":
                            # print([ancestor.text for ancestor in ancestor.ancestors])
                            return True, "SIMPLE_PRESENT"

                        else:
                            continue

                    else:
                        ancestor_pos_ = [
                            children.pos_ for children in ancestor.ancestors
                        ]
                        ancestor_tag_ = [
                            children.tag_ for children in ancestor.ancestors
                        ]

                        # print(ancestor_pos_)
                        if not "VERB" in ancestor_pos_ and not "AUX" in ancestor_pos_:
                            # print(99)
                            return True, "SIMPLE_PRESENT"
                        else:
                            if (
                                len(set(ancestor_tag_).intersection(past_tense_all))
                                == 1
                            ):
                                return False, "SIMPLE_PAST"
                            elif (
                                len(
                                    set(ancestor_tag_).intersection(
                                        All_present_tense_verbs
                                    )
                                )
                                == 1
                            ):
                                return True, "SIMPLE_PRESENT"
                            else:
                                return True, "SIMPLE_PRESENT"

                elif ancestor.tag_ in past_participle:
                    for child in ancestor.children:
                        # print(child.text, child.tag_, child.pos_)
                        if child.pos_ == "AUX" and child.tag_ in present_tense_verbs:
                            # print('present_perfect_continuous')
                            return True, "SIMPLE_PRESENT"  #'PRES_PERF_CON'
                        elif child.pos_ == "AUX" and child.tag_ in past_tense:
                            # print('past_tense')
                            return False, "PAST_TENSE"
                        elif not "AUX" in [child.pos_ for child in ancestor.children]:
                            return True, "SIMPLE_PRESENT"  #'PRES_PERF_CON'

                elif ancestor.tag_ == "VBG" and ancestor.dep_ == "xcomp":
                    # print('present_perfect_continuous')
                    return True, "SIMPLE_PRESENT"  #'PRES_PERF_CON'

                elif ancestor.tag_ == "VBG" or ancestor.tag_ == "VB":
                    for child in ancestor.children:
                        # print(child.text, child.tag_, child.pos_)
                        if child.pos_ == "AUX" and child.tag_ in past_tense:
                            # print(child.text, 'past_tense')
                            return False, "PAST_TENSE"
                        if child.pos_ == "AUX" and child.tag_ in present_tense_verbs:
                            # print(child.text, 'present_tense')
                            return True, "SIMPLE_PRESENT"  #'PRESENT_PART'
                else:
                    continue

            for token in doc:
                # print(token.)
                if token.dep_ == "ROOT":
                    if token.tag_ in past_tense_all:
                        return False, "PAST_TENSE"
                    elif token.tag_ in present_tense_verbs:
                        return True, "SIMPLE_PRESENT"
                    elif token.tag_ in present_participle:
                        for child in token.children:
                            # print([(child.text, child.pos_) for child in token.children])
                            if child.tag_ in present_tense_verbs:
                                return True, "SIMPLE_PRESENT"
                            elif child.tag_ in past_participle:
                                for child_2 in child.children:
                                    if (
                                        child_2.pos_ == "AUX"
                                        and child_2.tag_ in present_tense_verbs
                                    ):
                                        return True, "PAST_PART"
                            elif child.tag in past_tense:
                                return False, "PAST_TENSE"
                            else:
                                return True, "SIMPLE_PRESENT"

            return True, "SIMPLE_PRESENT"


def get_tense_negation(mentions, Label, nlp_spacy, note_id):

    # print(note_id)

    if not isinstance(mentions, list):
        return np.nan

    if Label == "WELL":
        Label_1 = ["WELL", "WELL_neg"]

    else:
        Label_1 = [Label]

    negation = []

    for index, mention in enumerate(mentions):

        # print(mention)
        mention = re.sub(r"\*", "", mention)
        mention = re.sub(r"nb", "n b", mention, re.IGNORECASE)
        mention = preprocessing(mention)
        # print(mention)

        temp = []

        doc = nlp_spacy(mention.lower())
        # print(doc.ents)

        for index, ent in enumerate(doc.ents):

            if ent.label_ in Label_1:
                # print(ent.text, ent.label_, mention, ent.start)

                present_tense_true, Tense = present_tense(
                    nlp_spacy, doc.text, ent.label_, ent.start
                )
                # print(present_tense_true, Tense)
                if present_tense_true == True:

                    if ent._.negex == False:
                        # print('here')

                        if ent.label_ == "BLOOD":
                            negation.append((1, Tense, doc.text))

                        elif ent.label_ == "PAIN":
                            negation.append((1, Tense, doc.text))

                        elif ent.label_ == "WELL":
                            negation.append((1, Tense, doc.text))
                        elif ent.label_ == "Clin_Rem":
                            negation.append((1, Tense, doc.text))
                        else:
                            pass

                    else:
                        # print('here')
                        if ent.label_ == "BLOOD":
                            negation.append((0, Tense, doc.text))

                        elif ent.label_ == "PAIN":
                            negation.append((0, Tense, doc.text))

                        elif ent.label_ == "WELL_neg":
                            negation.append((1, Tense, doc.text))
                        else:
                            pass

                # negation += temp

                else:
                    continue
            else:
                continue

    if len(negation) > 0:
        return negation
    else:
        return np.nan


def check_entity_keyword(entity, keywords):
    for keyword in keywords:
        regex = r"" + keyword
        if re.search(keyword, entity) is not None:
            return keyword

    return False


# ### Assembling Blood Mention to Single Column ###


def encode(text):
    if text == "No":
        return int(0)
    if text == "Yes":
        return int(1)

    else:
        return np.nan


def encode_values(list_of_indices, value, df, column_name):

    for index in list_of_indices:
        df.at[index, column_name] = value


def encode_column(df1, df2, column_to_encode, encoded_column_name, df_same):

    values_list = df2[df2[column_to_encode].notnull()][column_to_encode]
    values_index = values_list.index

    encoded_values = []

    for index, values in zip(values_index, values_list):
        # print(values)
        Tenses = [value[2] for value in values]
        # print(Tenses)
        Blood_Values = [value[1] for value in values]
        # print(Blood_Values, Blood_Values[0])

        keywords = [value[0] for value in values]

        if "CURRENT" in Tenses:
            True_index = [
                index for index, tense in enumerate(Tenses) if tense == "CURRENT"
            ]
            True_Blood_Values = [
                value for index, value in enumerate(Blood_Values) if index in True_index
            ]

            if len(True_index) == 1:
                if True_Blood_Values[0] == True:
                    encoded_values.append(int(1))
                else:
                    encoded_values.append(int(0))
            else:
                if len(set(True_Blood_Values)) == 1:
                    if list(set(True_Blood_Values))[0] == True:
                        encoded_values.append(int(1))
                    else:
                        encoded_values.append(int(0))
                else:
                    encoded_values.append(int(1))

        elif "NO_VERB" in Tenses:
            True_index = [
                index for index, tense in enumerate(Tenses) if tense == "NO_VERB"
            ]
            True_Blood_Values = [
                value for index, value in enumerate(Blood_Values) if index in True_index
            ]
            # print(True_Blood_Values[0])
            if len(True_index) == 1:
                # print('True_index', True_index)
                if True_Blood_Values[0] == True:
                    encoded_values.append(int(1))
                else:
                    encoded_values.append(int(0))
            else:
                if len(set(True_Blood_Values)) == 1:
                    if list(set(True_Blood_Values))[0] == True:
                        encoded_values.append(int(1))
                    else:
                        encoded_values.append(int(0))
                else:
                    encoded_values.append(int(1))

        elif "SIMPLE_PRESENT" in Tenses:
            True_index = [
                index for index, tense in enumerate(Tenses) if tense == "SIMPLE_PRESENT"
            ]
            True_Blood_Values = [
                value for index, value in enumerate(Blood_Values) if index in True_index
            ]

            if len(True_index) == 1:
                if True_Blood_Values[0] == True:
                    encoded_values.append(int(1))
                else:
                    encoded_values.append(int(0))
            else:
                if len(set(True_Blood_Values)) == 1:
                    if list(set(True_Blood_Values))[0] == True:
                        encoded_values.append(int(1))

                    else:
                        encoded_values.append(int(0))
                else:
                    encoded_values.append(int(1))

        elif "PRESENT_PART" in Tenses:
            True_index = [
                index for index, tense in enumerate(Tenses) if tense == "PRESENT_PART"
            ]
            True_Blood_Values = [
                value for index, value in enumerate(Blood_Values) if index in True_index
            ]
            # print(True_Blood_Values)
            if len(True_index) == 1:
                if True_Blood_Values[0] == True:
                    encoded_values.append(int(1))
                else:
                    encoded_values.append(int(0))
            else:
                if len(set(True_Blood_Values)) == 1:
                    if list(set(True_Blood_Values))[0] == True:
                        encoded_values.append(int(1))
                    else:
                        encoded_values.append(int(0))
                else:
                    encoded_values.append(int(1))

        elif "PRES_PERF_CON" in Tenses:

            True_index = [
                index for index, tense in enumerate(Tenses) if tense == "PRES_PERF_CON"
            ]
            True_Blood_Values = [
                value for index, value in enumerate(Blood_Values) if index in True_index
            ]
            # print(True_Blood_Values[0])

            if len(True_index) == 1:
                if True_Blood_Values[0] == True:
                    # print(True_Blood_Values[0])
                    encoded_values.append(int(1))
                else:
                    encoded_values.append(int(0))
            else:
                if len(set(True_Blood_Values)) == 1:
                    if list(set(True_Blood_Values))[0] == True:
                        encoded_values.append(int(1))

                    else:
                        encoded_values.append(int(0))
                else:
                    encoded_values.append(int(1))

    if df_same == False:
        id_list = []
        for index in values_index:
            id_list.append(df2["deid_note_id"][index])

        values_index = get_index(id_list, df1)

    for index, value in zip(values_index, encoded_values):
        df1.at[index, encoded_column_name] = value


def encode_column(df1, df2, column_to_encode, encoded_column_name, df_same):

    values_list = df2[df2[column_to_encode].notnull()][column_to_encode]
    values_index = values_list.index

    encoded_values = []

    for index, values in zip(values_index, values_list):
        # print(values)
        Tenses = [value[1] for value in values]
        # print(Tenses)
        # negations = [value[1] for value in values]
        # print(Blood_Values, Blood_Values[0])

        values = [value[0] for value in values]
        # print(values)

        if "CURRENT" in Tenses:
            True_index = [
                index for index, tense in enumerate(Tenses) if tense == "CURRENT"
            ]
            True_Values = [
                value for index, value in enumerate(values) if index in True_index
            ]
            # True_negation = [neg for index, negation in enumerate(negations) if index in True_index]

            if len(True_index) == 1:
                encoded_values.append(True_Values[0])

            else:
                if len(set(True_Values)) == 1:
                    encoded_values.append(list(set(True_Values))[0])

                else:
                    encoded_values.append(int(1))

        elif "NO_VERB" in Tenses or "SIMPLE_PRESENT" in Tenses:
            True_index = [
                index
                for index, tense in enumerate(Tenses)
                if tense == "NO_VERB" or tense == "SIMPLE_PRESENT"
            ]

            True_Values = [
                value for index, value in enumerate(values) if index in True_index
            ]
            # True_negation = [neg for index, negation in enumerate(negations) if index in True_index]

            if len(True_index) == 1:
                encoded_values.append(True_Values[0])

            else:
                if len(set(True_Values)) == 1:
                    encoded_values.append(list(set(True_Values))[0])

                else:
                    encoded_values.append(int(1))

        elif "PRESENT_PART" in Tenses:
            True_index = [
                index for index, tense in enumerate(Tenses) if tense == "PRESENT_PART"
            ]
            True_Values = [
                value for index, value in enumerate(values) if index in True_index
            ]
            # True_negation = [neg for index, negation in enumerate(negations) if index in True_index]

            if len(True_index) == 1:
                encoded_values.append(True_Values[0])

            else:
                if len(set(True_Values)) == 1:
                    encoded_values.append(list(set(True_Values))[0])

                else:
                    encoded_values.append(int(1))

        elif "PRES_PERF_CON" in Tenses:

            True_index = [
                index for index, tense in enumerate(Tenses) if tense == "PRES_PERF_CON"
            ]
            True_Values = [
                value for index, value in enumerate(values) if index in True_index
            ]
            # True_negation = [neg for index, negation in enumerate(negations) if index in True_index]

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
            id_list.append(df2["deid_note_id"][index])

        values_index = get_index(id_list, df1)

    for index, value in zip(values_index, encoded_values):
        df1.at[index, encoded_column_name] = value


def check_values(lst):
    for value in lst:
        if value[1] == True:
            return True

    return False


def encode_column2(df1, df2, column_to_encode, encoded_column_name, df_same):

    values = df2[df2[column_to_encode].notnull()][column_to_encode]
    values_index = values.index

    encoded_values = []

    for index, values in zip(values_index, values):
        if check_values(values) == True:
            encoded_values.append(int(1))

        elif check_values(values) == False:
            encoded_values.append(int(0))
        else:
            raise ValueError("values take more than 2 types.")

    if df_same == False:
        id_list = []
        for index in values_index:
            id_list.append(df2["deid_note_id"][index])

        values_index = get_index(id_list, df1)

    for index, value in zip(values_index, encoded_values):
        df1.at[index, encoded_column_name] = value


def encode_column_ctakes(df, encoded_column_name):

    negative_blood2 = []
    positive_blood2 = []

    for index, row in df.iterrows():

        if row["c_takes_fecalblood"] != 1:
            dic = reorder_ctakes(row["c_takes_fecalblood"])
            dic = dic[-1]

            if dic["negated"] == "t":
                negative_blood2.append(index)
            if dic["negated"] == "f":
                positive_blood2.append(index)

    for indices in positive_blood2:
        df.at[indices, encoded_column_name] = 1

    for indices in negative_blood2:
        df.at[indices, encoded_column_name] = 0


def find_previous_note(note_id, df):

    row = df[df["deid_note_id"] == note_id]
    patient_id = row["deid_PatientDurableKey"]
    date = row["deid_service_date_cdw"].values

    patient_rows = df[df["deid_PatientDurableKey"] == patient_id.values[0]]
    note_ids = patient_rows["deid_note_id"].values
    dates = patient_rows["deid_service_date_cdw"].values

    number_dates = len(dates)

    # creates a dictionary with dates as keys for note ids

    dates_dic = dict(zip(dates, note_ids))

    # if there are more than one notes with patient id, finds all dates which are less than input date

    if len(dates) > 1:
        previous_dates = []
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

    # returns note id coressponding to the highest date of those which fall behind input date
    return dates_dic[previous_date]


def get_range(HPI_offset, HPI):
    lst = []
    lst.append((int(HPI_offset), int(HPI_offset + len(HPI))))
    return lst


def get_index(lst, df):
    lst_ = []
    for l in lst:
        lst_.append(df[df["deid_note_id"] == l].index.values[0])

    return lst_


### Function drops entity mentions that fall within the first 40% character range of the HPI


def get_mention_location(HPI, mentions, spans, keywords):

    count = 0

    if isinstance(mentions, list) and len(mentions) is not 0:
        character_length = float(len(HPI))
        # print(character_length)

        for index, mention in enumerate(mentions):
            # print(mention)
            doc_location = float(spans[index]) / character_length
            # print(doc_location)
            if doc_location < 0.3:
                del mentions[index - count]
                del keywords[index - count]
                count += 1

    return mentions, keywords


def consolidate_columns(row_names, master_column, df):
    conflicting_rows = []

    for index, row in df.iterrows():
        # print(row[row_names[0]])
        if not math.isnan(row[row_names[0]]):
            # print(row[row_names[0]])
            df.at[index, master_column] = row[row_names[0]]
            continue
        elif not math.isnan(row[row_names[1]]):  # pain_mention_Interval_History
            if not math.isnan(row[row_names[2]]):  #'pain_mention_Previous_note'
                if row[row_names[1]] == row[row_names[2]]:
                    df.at[index, master_column] = row[
                        row_names[1]
                    ]  # pain_mention_Interval_History
                    continue
                else:
                    df.at[index, master_column] = row[
                        row_names[1]
                    ]  # pain_mention_Interval_History
                    conflicting_rows.append(index)
                    continue
            else:
                df.at[index, master_column] = row[
                    row_names[1]
                ]  # pain_mention_Interval_History
                continue

        elif (
            not math.isnan(row[row_names[2]])
            and not type(row["Interval_History"]) == str
        ):  #'pain_mention_Previous_note'
            df.at[index, master_column] = row[
                row_names[2]
            ]  #'pain_mention_Previous_note'
            continue

        else:
            if not math.isnan(row[row_names[3]]):
                df.at[index, master_column] = row[row_names[3]]
                continue

    return conflicting_rows
