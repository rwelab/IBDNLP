# import collections
import difflib
# import json
from itertools import groupby

import numpy as np
import pandas as pd
import spacy
from more_itertools import consecutive_groups
# from scipy.stats import beta
# from tqdm import tqdm


def get_note(_id, df):
    row = df[df["deid_note_id"] == _id]
    return row["note"].values[0]


# simple function which returns the HPI text from its note ID
def get_HPI(IBD_Fecal_Blood, ID):
    try:
        return IBD_Fecal_Blood[IBD_Fecal_Blood["deid_note_id"] == ID]["HPI"].values[0]
    except:
        return 1


# ### Previous Note Comparison Pipeline Functions


def get_text_from_range_list(doc1, ranges):
    text = ""
    if len(ranges) > 1:
        for range_ in ranges:
            x = range_[0]
            y = range_[1]
            temporary = str(doc1[x:y]) + " "
            text += temporary
    elif len(ranges) == 0:
        return 1
    else:
        x = ranges[0][0]
        y = ranges[0][1]

        text = str(doc1[x:y])
    return text


""" 
this function takes a list of character ranges as input and modifies the range 
"""


def modify_range(range_list):
    ranges = []
    gb = groupby(enumerate(range_list), key=lambda x: x[0] - x[1])
    all_groups = ([i[1] for i in g] for _, g in gb)
    x = list(filter(lambda x: len(x) > 1, all_groups))

    for lists in x:
        ranges.append([lists[0], lists[-1]])

    return ranges


def modify_ranges(set_list):
    ranges_list = []
    for group in consecutive_groups(set_list):
        temp = list(group)
        ranges = (temp[0], temp[-1] + 1)
        ranges_list.append(ranges)

    return ranges_list


""" 
function takes difflib 'seqmatcher' object as input
"""


def get_changes(seqmatcher, doc1):

    additions = []
    addition_range = []

    for opcode in seqmatcher.get_opcodes():
        last_two = opcode[-2:]
        if opcode[0] == "replace":

            character_range = last_two

            if abs(character_range[0] - character_range[1]) >= 2:
                addition_lower, addition_upper = get_period_range(doc1, character_range)
                addition_range += list(range(addition_lower, addition_upper + 1))

        if opcode[0] == "insert":

            character_range = last_two

            if abs(character_range[0] - character_range[1]) >= 2:
                addition_lower, addition_upper = get_period_range(doc1, character_range)

                addition_range += list(range(addition_lower, addition_upper + 1))

    addition_range = modify_ranges(addition_range)
    addition_range = list(set(addition_range))

    return addition_range


def get_period_range(doc, character_range):

    lower_index = character_range[0]
    upper_index = character_range[1] - 1

    limit = len(doc) - 1

    while str(doc[lower_index]) != ".":
        if lower_index == 0:
            break
        lower_index -= 1

    while str(doc[upper_index]) != ".":
        if upper_index == limit:
            break
        upper_index += 1

    return lower_index, upper_index


# ### Previous Note Comparison Pipline ###

# [difflib documentation](https://docs.python.org/3/library/difflib.html)


"""
for more information on difflib sequencematcher see documentation above
"""


def find_additions(nlp, current_HPI, previous_HPI, character_offset, note):

    if previous_HPI == 1:
        return [np.nan, current_HPI]

    doc1 = nlp(current_HPI)  # creates Spacy doc for current HPI
    doc2 = nlp(previous_HPI)  # creates Spacy doc for previous HPI

    doc_tokens1 = [token.text for token in doc1]  # creates a list of tokens for doc1
    doc_tokens2 = [token.text for token in doc2]  # creates a list of tokens for doc2

    seqmatcher = difflib.SequenceMatcher(
        a=doc_tokens2, b=doc_tokens1, autojunk=False
    )  # creates a SequenceMatcher object

    addition_ranges = get_changes(
        seqmatcher, doc1
    )  # returns an array of token ranges corresponding to note additions
    addition_ranges = reorder(addition_ranges)  # reorders array chronologically
    addition_ranges = remove_duplicates(addition_ranges)

    addition_ranges = reorder(addition_ranges)
    try:

        additions = doc1[addition_ranges[0][0] :].text
    except:
        return doc1.text
    # additions =  get_text_from_range_list(doc1, addition_ranges)
    # character_ranges = convert_offset(doc1, addition_ranges, character_offset)
    try:
        # character_ranges = convert_offset(doc1, addition_ranges, character_offset)
        # character_ranges = remove_duplicates(character_ranges)
        # additions = get_text_from_character_range(character_ranges, note)
        # character_ranges = add_period(character_ranges, note)
        # addition_ranges = present_tense(additions, doc1)
        # current_information_ranges = present_tense(additions, character_ranges)

        # additions = de_tokenize(additions)

        return [addition_ranges, additions]
        # return addition_ranges, additions, character_ranges, current

    except:
        return [np.nan, current_HPI]


def convert_offset(doc, ranges, HPI_offset):

    character_offsets = []
    offset = int(HPI_offset)

    if len(ranges) > 1:
        for range_ in ranges:

            x = range_[1] - 1
            y = range_[0]
            length = len(doc[x])
            start = doc[y].idx + offset
            end = doc[x].idx + offset + length
            character_offsets.append([start, end])

    elif len(ranges) == 0:
        return 1

    else:
        y = ranges[0][0]
        x = ranges[0][1] - 1

        length = len(doc[x])

        start = doc[y].idx + offset
        end = doc[x].idx + offset + length
        character_offsets.append([start, end])

    return character_offsets


def remove_duplicates(range_list):
    set_list = []

    if len(range_list) == 1:
        return range_list

    for range_ in range_list:
        set_list += range(range_[0], range_[1])

    set_list = list(set(set_list))

    set_list = modify_ranges(set_list)

    return set_list


def tail_test(ranges, tokens):

    if len(ranges) == 0:
        return 0

    # doc = nlp(HPI)
    # doc_tokens = [token.text for token in doc]
    last_element = len(tokens) - 1

    # print('initial_lastelement:', ranges)
    # print('last-_element:',  last_element)
    if abs(last_element - ranges[-1][1] < 10):
        ranges[-1][1] = last_element

    # print('final_lastelement', ranges)

    return ranges


def reorder(list_ranges):
    from operator import itemgetter

    return sorted(list_ranges, key=itemgetter(0))


def get_text_from_character_range(character_range, note):
    text = []
    first_instance = character_range[0][0]
    try:
        for range_ in character_range:
            start = range_[0]
            end = range_[1]
            sentences = note[start:end]
            text.append(sentences)
        return text
    except:
        return 1


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


def add_period(character_ranges, note):
    new_ranges = []
    for range_ in character_ranges:
        start = range_[0]
        end = range_[1]
        if end != ".":
            end = end + 1
        new_ranges.append((start, end))
    return new_ranges