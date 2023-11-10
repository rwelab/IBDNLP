# import os
import sys
import re
from datetime import date, datetime, time
# from xml.dom import minidom

# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# import scispacy
# import spacy

sys.path.append('../user_definition.py')
from user_definition import *

# an exploratory function that searches for a an HPI/ROS questionnaire


def search_ROS(note):
    search = re.search(r"HPI/ROS", note, re.IGNORECASE)
    if search is not None:
        return 1
    else:
        search = re.search(r"question\s*(\d./\d.\/\d*)", note, re.IGNORECASE)
        if search is not None:
            return 1
        else:
            return 2


# gets the date of the HPI/ROS questionaire submission by the patient


def get_ROS_date(note):
    try:
        return re.search(r"HPI/ROS.*?(\d./\d./\d*)", note).group(1)
    except:
        try:
            return re.search(r"question\s*(\d./\d.\/\d*)", note, re.IGNORECASE).group(1)
        except:

            return None


def check_index(df):
    for i in range(len(df)):
        try:
            df.iloc[i]
        except:
            print(i)


def clean_tail(HPI):
    try:
        return re.search(r"(.*?) past $", HPI, re.IGNORECASE).group(1)
    except:
        return HPI


# finds blood in your stool ROS question and returns the response


def get_fecal_answer(note):
    try:
        answer1 = re.search(
            r".*? blood in your stool\:\s(.*?)\s", note, re.IGNORECASE
        ).group(1)
    except:
        try:
            answer2 = re.search(
                r".*? or sticky bowel movements:\s(.*?)\s", note, re.IGNORECASE
            ).group(1)
            return answer2
        except:
            return None

    try:
        answer2 = re.search(
            r".*? or sticky bowel movements:\s(.*?)\s", note, re.IGNORECASE
        ).group(1)
    except:
        return answer1

    if answer1.lower() == answer2.lower():
        return answer1
    else:
        return "Yes"


def extract_HPI(note):

    # checks for the presence of an HPI header
    if (
        re.search(r"(History\s*of\s*Present\s*Illness\s*|\s*HPI)", note, re.IGNORECASE)
        is not None
    ):
        try:
            r = re.search(
                r"(?:History\s*of\s*Present\s*Illness|HPI)(.*?)(Medical\s*(:?H|h)istory|Surgical\s*(:?H|h)istory)",
                note,
            )
            return [r.group(1), r.start(1)]
        except:
            # checks for the presence of a PMS or PSH Header
            if (
                re.search(
                    r"((Past)?Medical(\/Surgical)?|Surgical)\s*(:?H|h)istory", note
                )
                is not None
            ):
                try:
                    r = re.search(
                        r"(.*?)(:?(Past)?Medical(\/Surgical)?|Surgical)\s*(:?H|h)istory",
                        note,
                    )
                    return [r.group(1), r.start(1)]
                except:  # checks for the presence of a Physical exam or review of systems Header
                    if (
                        re.search(r"Physical\s*Exam|Review\s*of\s*Systems", note)
                        is not None
                    ):
                        try:
                            r = re.search(
                                r"(.*?)(Physical\s*Exam\s*|\s*Review\s*of\s*Systems)",
                                note,
                            )
                            return [r.group(1), r.start(1)]
                        except:
                            return [1, 1]
            else:
                if (
                    re.search(
                        r"Physical\s*Exam|Review\s*of\s*Systems", note, re.IGNORECASE
                    )
                    is not None
                ):
                    try:
                        r = re.search(
                            r"(.*?)(Physical\s*Exam\s*|\s*Review\s*of\s*Systems)", note
                        )
                        return [r.group(1), r.start(1)]
                    except:
                        return [1, 1]

        # regex for notes which are missing HPI header, but contain a Past Medical History
    elif (
        re.search(r"(:?(Past)?Medical(/Surgical)?|Surgical)\s*(:?H|h)istory", note)
        is not None
    ):
        try:  # returns everything that comes before instance of past medical history header
            r = re.search(
                r"(.*?)(:?(Past)?Medical(/Surgical)?|Surgical)\s*(:?H|h)istory", note
            )
            return [r.group(1), r.start(1)]
        except:
            return [1, 1]

    # assumes that None of the previous is present
    elif (
        re.search(r"(Physical\s*Exam|Review\s*of\s*Systems)", note, re.IGNORECASE)
        is not None
    ):
        try:
            r = re.search(r"(.*?)(Physical\s*Exam|Review\s*of\s*Systems)", note)
            return [r.group(1), r.start(1)]
        except:
            return [1, 1]

    else:
        return [1, 1]


# an exploratory function that searches for a Nutrition Services Header
def Nutrition_services(note):
    search = re.search(r"Nutrition Services", note, re.IGNORECASE)
    if search is not None:
        return 1
    else:
        return 2


# ROS Questionnaires with delta between service date less than or equal to 14 are marked as valid
def valid_ROS(delta):
    if delta <= 14:
        return True
    else:
        return False


def removekey(d, key):
    d = dict(d)
    del d[key]
    return d


def get_index(note_id, df):
    return df[df[id_column] == note_id].index.values[0]


# ### Exploring Previous Note Comparison for Extraction ###


# def find_previous_note(note_id, df):

#     row = df[df[note_column] == note_id]
#     patient_id = row[patient_durable_key]
#     date = row[service_date].values

#     patient_rows = df[df[patient_durable_key] == patient_id.values[0]]
#     note_ids = patient_rows[note_column].values
#     dates = patient_rows[service_date].values

#     number_dates = len(dates)

#     # creates a dictionary with dates as keys for note ids

#     dates_dic = dict(zip(dates, note_ids))

#     # if there are more than one notes with patient id, finds all dates which are less than input date

#     if len(dates) > 1:
#         previous_dates = []
#         for x in dates:
#             if x < date:
#                 previous_dates.append(x)

#         # sorts dates in ascending order
#         previous_dates = sorted(previous_dates)

#     # returns input note id if there are no ther dates corresponding to patient ID
#     else:
#         return 1

#     # checks to see if there are any dates which were less than date of input note id, returns note_id if none
#     if len(previous_dates) > 0:
#         previous_date = previous_dates[-1]

#     else:
#         return 1

#     # returns note id coressponding to the highest date of those which fall behind input date
#     return dates_dic[previous_date]


# ### Applying Interval History Parser      ###


# extracts the last instance of Interval History header and gets the subsequent text---
def find_IntervalHistory(text, HPI_offset):
    if isinstance(text, str):

        count = 0
        start = int(HPI_offset)

        while True:
            try:
                r = re.search(
                    r"(?:Interval\s*(?:History|Event|hx)|Interim(?: History|Event|hx)?|20\d\d:)(.*?)\Z",
                    text,
                    re.IGNORECASE,
                )

                match = r.group(1)

                start += int(r.start(1))
                end = int(len(match) + start)

                count += 1

                text = match

            except:
                break

        if count > 0:
            return [match, [start, end]]
        else:
            return [1, 1]
    else:
        return [1, 1]


def get_noteid_fromkey(note_key, df):
    row = df[df[key_column] == note_key]
    note_id = row[note_column].values[0]
    return note_id


# # Extract Blood Mentions #


def blood_re_order(dictionary):
    keys = list(dictionary.keys())
    keys = sorted(keys)

    sorted_list = []

    for key in keys:
        sorted_list.append(dictionary[key])

    return sorted_list


def find_blood_two(note):

    bag_of_words = ["blood", "bleed", "hematochezia", "BRBPR"]

    matches = {}

    for word in bag_of_words:

        regex = r"[^.]*" + word + r"[^.]*\."
        try:
            trial = re.finditer(regex, note, re.IGNORECASE)

            for match in trial:
                span = match.span()
                matches[span[1]] = match.group(0)
        except:
            continue

    ordered_matches = blood_re_order(matches)

    return ordered_matches


# ### Blood Mention Functions ###


def find_Today(HPI):
    if re.search(r"Today", HPI, re.IGNORECASE) is not None:
        try:
            return re.search(r".*\.(.*?today.*?)\.", HPI, re.IGNORECASE).group(1)
        except:
            return 0


def find_blood(note):
    try:
        trial = re.finditer(r"[^.]*blood[^.]*\.", note, re.IGNORECASE)
        if trial:
            matches = [match.group() for match in trial]  # Extract matched strings
            return matches

        else:
            return 0
    except:
        return 0


def find_bleed(note):
    try:
        trial = re.finditer(r"[^.]*bleed[^.]*\.", note, re.IGNORECASE)
        if trial:
            matches = [match.group() for match in trial]  # Extract matched strings
            return matches

        else:
            return 0
    except:
        return 0


def find_hema(note):
    try:
        trial = re.finditer(r"[^.]*hematochezia[^.]*\.", note, re.IGNORECASE)
        if trial:
            matches = [match.group() for match in trial]  # Extract matched strings
            return matches

        else:
            return 0
    except:
        return 0


def find_BRBPR(note):
    try:
        trial = re.findall(r"[^.]*BRBPR[^.]*\.", note, re.IGNORECASE)
        if trial:
            matches = [match.group() for match in trial]  # Extract matched strings
            return matches

        else:
            return 0
    except:
        return 0


def find_red(note):
    try:
        trial = re.findall(r"[^.]*\sred\s[^.]*\.", note, re.IGNORECASE)
        if trial:
            matches = [match.group() for match in trial]  # Extract matched strings
            return matches

        else:
            return 0
    except:
        return 0


def find_Blood(note):
    text = []
    bleed = find_bleed(note)
    try:
        if bleed is not None:
            text += bleed
    except:
        pass
    #     if bleed != 0 and bleed != None:
    #         try: text += bleed
    #         except: pass
    blood = find_blood(note)
    try:
        if blood is not None:
            text += blood
    except:
        pass
    #     if blood != 0 and blood != None:
    #         try: text += blood
    #         except: pass
    hema = find_hema(note)
    try:
        if hema is not None:
            text += hema
    except:
        pass
    #     if hema != 0 and hema != None:
    #         try: text += hema
    #         except: pass
    BRBPR = find_BRBPR(note)
    try:
        if BRBPR is not None:
            text += BRBPR
    except:
        pass
    #     if BRBPR != 0 and BRBPR != None:
    #         try: text += BRBPR
    #         except: pass
    red = find_red(note)
    try:
        if red is not None:
            text += red
    except:
        pass
    #     if red != 0 and red != None:
    #         try: text += red
    #         except: pass

    if len(text) >= 1:
        return list(text)

    else:
        return 0


def find_bowel(note):
    re.search(r".*\.(.*?bowel.*?)\.", note, re.IGNORECASE).group(1)
    re.search(r".*\.(.*?bm.*?)\.", note, re.IGNORECASE).group(1)

