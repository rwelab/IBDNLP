import os
import sys

from functions.blood_pipeline import *
from functions.diarrhea_pipeline import *
from functions.prev_note_pipeline import *
from functions.primary_pipeline import *
from user_definition import *

from tqdm import tqdm


tqdm.pandas()


def read_file(filepath):
    # grabs the extension
    file_extension = os.path.splitext(filepath)[1]

    # determines correct method of reading file
    if file_extension == ".json":
        data = pd.read_json(filepath)
        return data
    elif file_extension in [".csv", ".tsv"]:
        if file_extension == ".tsv":
            delimiter = "\t" 
        else:
            delimiter = ","
        data = pd.read_csv(filepath, delimiter=delimiter)
        return data
    else:
        print("Unsupported file format. Please try .tsv, .csv, or .json.")
        sys.exit(1)


data_file = sys.argv[1]

# ### Import IBD Notes Dataframe ###

IBD_notes = read_file(data_file)

# searches through IBD_notes df to detect if note is a Nutrition Services Note

NS_search = IBD_notes[note_column].progress_apply(lambda x: Nutrition_services(x))
# creates an updated df w/o Nutrition Services notes
IBD_Fecal_Blood = IBD_notes.drop(NS_search[NS_search == 1].index).reset_index(drop=True)

# ### Detect HPI/ROS Questionnaire Presence ###


# creates column of HPI/ROS questionnaire Presence
IBD_Fecal_Blood["HPI/ROS Presence"] = IBD_Fecal_Blood[note_column].progress_apply(
    lambda x: search_ROS(x)
)


# creates column of HPI/ROS questionnaire date
IBD_Fecal_Blood["ROS_date"] = IBD_Fecal_Blood[note_column].progress_apply(
    lambda x: get_ROS_date(x)
)

# ### Detect HPI/ROS Questionnaire Presence ###


# creates column of HPI/ROS questionnaire Presence
IBD_Fecal_Blood["HPI/ROS Presence"] = IBD_Fecal_Blood[note_column].progress_apply(
    lambda x: search_ROS(x)
)


# creates column of HPI/ROS questionnaire date
IBD_Fecal_Blood["ROS_date"] = IBD_Fecal_Blood[note_column].progress_apply(
    lambda x: get_ROS_date(x)
)

# converts ROS_dates and deid service dates to datetime format:
IBD_Fecal_Blood["ROS_date"] = pd.to_datetime(
    IBD_Fecal_Blood["ROS_date"], format="%m/%d/%Y", errors="coerce"
)
IBD_Fecal_Blood[service_date] = pd.to_datetime(IBD_Fecal_Blood[service_date])

ROS_days_between = []
# calculates the difference between ROS date and service date, NA rows appended to value 999
for i in tqdm(range(len(IBD_Fecal_Blood["ROS_date"]))):
    if IBD_Fecal_Blood["ROS_date"].iloc[i] is not pd.NaT:
        days_between = (
            IBD_Fecal_Blood[service_date].iloc[i] - IBD_Fecal_Blood["ROS_date"].iloc[i]
        )
        ROS_days_between.append(days_between.days)
    else:
        ROS_days_between.append(999)

# ### Create List of Valid ROS questionaire ###

# creates array of True/False values corresponding to whether or not ROS is valid and present
ROS_valid = []
for delta in tqdm(ROS_days_between):
    ROS_valid.append(valid_ROS(delta))


IBD_Fecal_Blood["ROS_Valid"] = ROS_valid

# ### Grabbing Fecal Blood Input from Valid ROS Questionaires ###


# grab all note id's which correspond to valid ROS
Valid_note_ids = IBD_Fecal_Blood[IBD_Fecal_Blood["ROS_Valid"] == True][id_column]

ROS_answers = {}
count = 0
for ids in tqdm(Valid_note_ids):
    count += 1
    # print(count)
    index = IBD_Fecal_Blood[IBD_Fecal_Blood[id_column] == ids].index.values[0]
    ROS_answers[ids] = get_fecal_answer(IBD_Fecal_Blood[note_column][index])

# create a list of indexes for ROS where the parser did not find an answer to the blood in stool question
None_list = []
ROS_keys = ROS_answers.keys()

for ids in tqdm(ROS_keys):
    if ROS_answers[ids] == None:
        # deletes dictionary none value
        ROS_answers = removekey(ROS_answers, ids)

# ### Making Fecal_Blood_Value Column ####


shape = (len(IBD_Fecal_Blood), 1)
IBD_Fecal_Blood["Fecal_Blood_Value"] = np.ones(shape)

# ### Adding ROS/HPI Answers to Fecal_Blood_Value Column ###

for key in ROS_answers.keys():
    index = get_index(key, IBD_Fecal_Blood)
    IBD_Fecal_Blood.loc[index, "Fecal_Blood_Value"] = ROS_answers[key]

# ### Create HPI Column ###


# apply HPI extraction to each note
IBD_Fecal_Blood[["extracted_HPI", "HPI_extraction_offset"]] = IBD_Fecal_Blood.progress_apply(
    lambda x: extract_HPI(x[note_column]), axis=1, result_type="expand"
)

IBD_Fecal_Blood["HPI"] = IBD_Fecal_Blood["extracted_HPI"].progress_apply(lambda x: clean_tail(x))

# ### Remove Rows which Failed HPI Extraction ###


# converts every None value in HPI column to 1 for the purposes of removal
for index, row in enumerate(IBD_Fecal_Blood["HPI"]):
    if row is None:
        IBD_Fecal_Blood.loc[index, "HPI"] = 1

# ### Find Interval History ###

IBD_Fecal_Blood["Interval_History"] = IBD_Fecal_Blood.progress_apply(
    lambda x: find_IntervalHistory(x["HPI"], x["HPI_extraction_offset"]), axis=1
)

# ### Create Previous Note ID Column ###


# create a previous note id column for all remaining rows
IBD_Fecal_Blood["Previous_note_id"] = IBD_Fecal_Blood.progress_apply(
    lambda x: find_previous_note(x[id_column], IBD_Fecal_Blood), axis=1
)

IBD_Fecal_Blood.to_json(os.path.join(data_folder, "ibd_notes.json"))

# ### Preparing Dataframes for Previous Note Comparison ###


Previous_Note_df = IBD_Fecal_Blood[
    ["HPI", "HPI_extraction_offset", patient_durable_key, id_column, "Previous_note_id"]
]


Previous_Note_df = Previous_Note_df[Previous_Note_df["Previous_note_id"] != 1]

nlp = spacy.load("en_core_web_sm")

Previous_note = Previous_Note_df[
    [id_column, "Previous_note_id", "HPI_extraction_offset"]
].copy()

# ### Get HPI from each note

# creates two columns current_HPI and previous_HPI
Previous_note["current_HPI"] = Previous_note[id_column].progress_apply(
    lambda x: get_HPI(IBD_Fecal_Blood, x)
)
Previous_note["previous_HPI"] = Previous_note["Previous_note_id"].progress_apply(
    lambda x: get_HPI(IBD_Fecal_Blood, x)
)

Previous_note[note_column] = Previous_note[id_column].progress_apply(
    lambda x: get_note(x, IBD_Fecal_Blood)
)

# ### Applying Previous Note Extraction to each Note ###

Previous_note[["addition_ranges", "additions"]] = Previous_note.progress_apply(
    lambda x: find_additions(
        nlp,
        x["current_HPI"],
        x["previous_HPI"],
        x["HPI_extraction_offset"],
        x[note_column]
    )
    if x["current_HPI"] != 1
    else np.nan,
    axis=1,
    result_type="expand",
)

# Previous_note.to_json(os.path.join(data_folder, "previous_note.json"))

# # ### Compile Dataframes ###
# IBD_Fecal_Blood = pd.read_json("data/ibd_notes.json")
# Previous_note = pd.read_json("data/previous_note.json")

# create an empty additions column
shape = (len(IBD_Fecal_Blood), 1)
IBD_Fecal_Blood["additions"] = np.ones(shape)


note_ids = Previous_note[id_column]
notes = Previous_note["additions"]

# gets a list of indices from note ids
note_ids_index = note_ids.progress_apply(lambda x: get_index(x, IBD_Fecal_Blood))

for index, note_index in enumerate(note_ids_index):
    IBD_Fecal_Blood.loc[note_index, "additions"] = notes.iloc[index]

# ### Apply 'Current' Extraction to IBD_Fecal_Blood ###


IBD_Fecal_Blood["Current_Extract"] = IBD_Fecal_Blood["HPI"].progress_apply(
    lambda x: find_Today(x) if type(x) == str else 1
)

# ### Applying Find_Blood Function ###


# applies find_Blood function to Interval History sections
Blood_Indicator_Interval_History = IBD_Fecal_Blood["Interval_History"].progress_apply(
    lambda x: find_Blood(x) if x != 1 else 1
)

# applies find_Blood function to HPI sections
Blood_Indicator_HPI = IBD_Fecal_Blood["HPI"].progress_apply(
    lambda x: find_Blood(x) if x != 1 else 1
)


# applies find_Blood function to Addition section
Blood_Indicator_Additions = IBD_Fecal_Blood["additions"].progress_apply(
    lambda x: find_Blood(x) if x != 1 else 1
)

# creates list of note Ids
Note_Ids = IBD_Fecal_Blood[id_column]

# ### Make Blood_Indicator Dataframe ###


Blood_Indicator_zip = list(
    zip(
        Note_Ids,
        IBD_Fecal_Blood["HPI"],
        IBD_Fecal_Blood["Interval_History"],
        IBD_Fecal_Blood["additions"],
        Blood_Indicator_Interval_History,
        Blood_Indicator_Additions,
        IBD_Fecal_Blood["Fecal_Blood_Value"],
        Blood_Indicator_HPI
    )
)


Blood_Indicator_df = pd.DataFrame(
    Blood_Indicator_zip,
    columns=[
        id_column,
        "HPI",
        "Interval_History",
        "Additions",
        "Blood_Indicator_Interval",
        "Blood_Indicator_Addition",
        "Fecal_Blood_Value",
        "Blood_Indicator_HPI"
    ],
)

# generate models for abdominal pain
nlp_abdominal = return_abdominal_spacy()
nlp_abdominal = add_pipe(nlp_abdominal)

IBD_Fecal_Blood[
    ["blood_mentions", "blood_mention_keyword", "blood_spans"]
] = IBD_Fecal_Blood.progress_apply(
    lambda x: find_blood_return_matches(x["Interval_History"]), axis=1, result_type="expand"
)

IBD_Fecal_Blood[
    ["pain_mentions", "pain_mention_keyword", "pain_spans"]
] = IBD_Fecal_Blood.progress_apply(
    lambda x: find_pain(x["Interval_History"], nlp_abdominal),
    axis=1,
    result_type="expand",
)

IBD_Fecal_Blood[
    ["cr_mentions", "cr_mention_keyword", "cr_spans"]
] = IBD_Fecal_Blood.progress_apply(
    lambda x: find_clinical_remission(x["Interval_History"]),
    axis=1,
    result_type="expand",
)

IBD_Fecal_Blood[
    ["well_mentions", "well_mention_keyword", "well_spans"]
] = IBD_Fecal_Blood.progress_apply(
    lambda x: find_no_symptoms(x["Interval_History"]), axis=1, result_type="expand"
)

IBD_Fecal_Blood["Fecal_Blood_Present"] = IBD_Fecal_Blood.progress_apply(
    lambda x: get_abdominal_tense_negation(
        x["blood_mentions"], "BLOOD", nlp_abdominal, x[id_column]
    ),
    axis=1,
)

IBD_Fecal_Blood["Pain_Present"] = IBD_Fecal_Blood.progress_apply(
    lambda x: get_abdominal_tense_negation(
        x["pain_mentions"], "PAIN", nlp_abdominal, x[id_column]
    ),
    axis=1,
)

IBD_Fecal_Blood["CR_Present"] = IBD_Fecal_Blood.progress_apply(
    lambda x: get_abdominal_tense_negation(
        x["cr_mentions"], "Clin_Rem", nlp_abdominal, x[id_column]
    ),
    axis=1,
)

IBD_Fecal_Blood["Well_Present"] = IBD_Fecal_Blood.progress_apply(
    lambda x: get_abdominal_tense_negation(
        x["well_mentions"], "WELL", nlp_abdominal, x[id_column]
    ),
    axis=1,
)

# ### Apply Blood Mention Extraction to Previous_Note Extraction ###

Previous_note["additions"] = Previous_note["additions"].progress_apply(
    lambda x: convert_list_string(x)
)

Previous_note[
    ["blood_mentions", "blood_mention_keyword", "blood_spans"]
] = Previous_note.progress_apply(
    lambda x: find_blood_return_matches(x["additions"]), axis=1, result_type="expand"
)

Previous_note[
    ["pain_mentions", "pain_mention_keyword", "pain_spans"]
] = Previous_note.progress_apply(
    lambda x: find_pain(x["additions"], nlp_abdominal), axis=1, result_type="expand"
)

# Previous_note[
#     ["cr_mentions", "cr_mention_keyword", "cr_spans"]
# ] = Previous_note.progress_apply(
#     lambda x: find_clinical_remission(x["additions"]), axis=1, result_type="expand"
# )

Previous_note[
    ["cr_mentions", "cr_mention_keyword", "cr_spans"]
] = Previous_note.progress_apply(
    lambda x: find_clinical_remission(x["additions"])
    if x["additions"] != 1
    else find_clinical_remission(x["HPI"]),
    axis=1,
    result_type="expand",
)

# Previous_note[
#     ["well_mentions", "well_mention_keyword", "well_spans"]
# ] = Previous_note.progress_apply(
#     lambda x: find_no_symptoms(x["additions"]), axis=1, result_type="expand"
# )

Previous_note[
    ["well_mentions", "well_mention_keyword", "well_spans"]
] = Previous_note.progress_apply(
    lambda x: find_no_symptoms(x["additions"])
    if x["additions"] != 1
    else find_no_symptoms(x["HPI"]),
    axis=1,
    result_type="expand",
)

Previous_note["Fecal_Blood_Present"] = Previous_note.progress_apply(
    lambda x: get_abdominal_tense_negation(
        x["blood_mentions"], "BLOOD", nlp_abdominal, x[id_column]
    ),
    axis=1,
)

Previous_note["Pain_Present"] = Previous_note.progress_apply(
    lambda x: get_abdominal_tense_negation(
        x["pain_mentions"], "PAIN", nlp_abdominal, x[id_column]
    ),
    axis=1,
)

Previous_note["CR_Present"] = Previous_note.progress_apply(
    lambda x: get_abdominal_tense_negation(
        x["cr_mentions"], "Clin_Rem", nlp_abdominal, x[id_column]
    ),
    axis=1,
)

Previous_note["Well_Present"] = Previous_note.progress_apply(
    lambda x: get_abdominal_tense_negation(
        x["well_mentions"], "WELL", nlp_abdominal, x[id_column]
    ),
    axis=1,
)

IBD_Fecal_Blood["Fecal_Blood_Value"] = IBD_Fecal_Blood["Fecal_Blood_Value"].progress_apply(
    lambda x: encode(x)
)

IBD_Fecal_Blood["Previous_note_id"] = IBD_Fecal_Blood.progress_apply(
    lambda x: find_previous_note(x[id_column], IBD_Fecal_Blood), axis=1
)

# ### Make Blank Columns ###

# IBD_Fecal_Blood['blood_mention_c_takes'] = np.nan
IBD_Fecal_Blood["blood_mention_Previous_note"] = np.nan
IBD_Fecal_Blood["blood_mention_Interval_History"] = np.nan
IBD_Fecal_Blood["pain_mention_Previous_note"] = np.nan
IBD_Fecal_Blood["pain_mention_Interval_History"] = np.nan
IBD_Fecal_Blood["cr_mention_Previous_note"] = np.nan
IBD_Fecal_Blood["cr_mention_Interval_History"] = np.nan
IBD_Fecal_Blood["well_mention_Previous_note"] = np.nan
IBD_Fecal_Blood["well_mention_Interval_History"] = np.nan

# ### Encode Each Column ###
encode_column(
    IBD_Fecal_Blood,
    Previous_note,
    "Fecal_Blood_Present",
    "blood_mention_Previous_note",
    False,
)
encode_column(
    IBD_Fecal_Blood, Previous_note, "Pain_Present", "pain_mention_Previous_note", False
)
encode_column(
    IBD_Fecal_Blood, Previous_note, "CR_Present", "cr_mention_Previous_note", False
)
encode_column(
    IBD_Fecal_Blood, Previous_note, "Well_Present", "well_mention_Previous_note", False
)
encode_column(
    IBD_Fecal_Blood,
    IBD_Fecal_Blood,
    "Fecal_Blood_Present",
    "blood_mention_Interval_History",
    True,
)
encode_column(
    IBD_Fecal_Blood,
    IBD_Fecal_Blood,
    "Pain_Present",
    "pain_mention_Interval_History",
    True,
)
encode_column(
    IBD_Fecal_Blood, IBD_Fecal_Blood, "CR_Present", "cr_mention_Interval_History", True
)
encode_column(
    IBD_Fecal_Blood,
    IBD_Fecal_Blood,
    "Well_Present",
    "well_mention_Interval_History",
    True,
)

# #### Blood Mention Extraction Remaining Notes #####

IBD_Fecal_Blood[
    ["blood_mentions_HPI", "blood_mentions_remaining_keywords", "blood_spans"]
] = IBD_Fecal_Blood.progress_apply(lambda x: find_blood_return_matches(x["HPI"]), result_type="expand", axis=1)

IBD_Fecal_Blood[
    ["blood_mentions_HPI", "blood_mentions_remaining_keywords"]
] = IBD_Fecal_Blood.progress_apply(
    lambda x: get_mention_location(
        x["HPI"],
        x["blood_mentions_HPI"],
        x["blood_spans"],
        x["blood_mentions_remaining_keywords"],
    ),
    result_type="expand",
    axis=1,
)

IBD_Fecal_Blood[
    ["pain_mentions_HPI", "pain_mentions_remaining_keywords", "pain_spans"]
] = IBD_Fecal_Blood.progress_apply(
    lambda x: find_pain(x["HPI"], nlp_abdominal), result_type="expand", axis=1
)

IBD_Fecal_Blood[
    ["pain_mentions_HPI_location", "pain_mentions_remaining_keywords"]
] = IBD_Fecal_Blood.progress_apply(
    lambda x: get_mention_location(
        x["HPI"],
        x["pain_mentions_HPI"],
        x["pain_spans"],
        x["pain_mentions_remaining_keywords"],
    ),
    result_type="expand",
    axis=1,
)

# IBD_Fecal_Blood[
#     ["cr_mentions_HPI", "cr_mention_keyword", "cr_spans"]
# ] = IBD_Fecal_Blood.progress_apply(
#     lambda x: find_clinical_remission(x["HPI"]), axis=1, result_type="expand"
# )

IBD_Fecal_Blood[
    ["cr_mentions_HPI_location", "cr_mention_keyword"]
] = IBD_Fecal_Blood.progress_apply(
    lambda x: get_mention_location(
        x["HPI"], x["cr_mentions"], x["cr_spans"], x["cr_mention_keyword"]
    ),
    axis=1,
    result_type="expand",
)

# IBD_Fecal_Blood[
#     ["well_mentions_HPI", "well_mention_keyword", "well_spans"]
# ] = IBD_Fecal_Blood.progress_apply(
#     lambda x: find_no_symptoms(x["HPI"]), axis=1, result_type="expand"
# )

IBD_Fecal_Blood[
    ["well_mentions_HPI_location", "well_mention_keyword"]
] = IBD_Fecal_Blood.progress_apply(
    lambda x: get_mention_location(
        x["HPI"], x["well_mentions"], x["well_spans"], x["well_mention_keyword"]
    ),
    axis=1,
    result_type="expand",
)

IBD_Fecal_Blood["Pain_Present_Remaining"] = IBD_Fecal_Blood.progress_apply(
    lambda x: get_abdominal_tense_negation(
        x["pain_mentions_HPI"], "PAIN", nlp_abdominal, x[id_column]
    ),
    axis=1,
)

IBD_Fecal_Blood["Fecal_Blood_Present_HPI"] = IBD_Fecal_Blood.progress_apply(
    lambda x: get_abdominal_tense_negation(
        x["blood_mentions_HPI"], "BLOOD", nlp_abdominal, x[id_column]
    )
    if x["blood_mentions_HPI"] != 1
    else np.nan,
    axis=1,
)

IBD_Fecal_Blood["Pain_Present_HPI"] = IBD_Fecal_Blood.progress_apply(
    lambda x: get_abdominal_tense_negation(
        x["pain_mentions_HPI_location"], "PAIN", nlp_abdominal, x[id_column]
    ),
    axis=1,
)

IBD_Fecal_Blood["CR_Present_HPI"] = IBD_Fecal_Blood.progress_apply(
    lambda x: get_abdominal_tense_negation(
        x["cr_mentions_HPI_location"], "Clin_Rem", nlp_abdominal, x[id_column]
    ),
    axis=1,
)

IBD_Fecal_Blood["Well_Present_HPI"] = IBD_Fecal_Blood.progress_apply(
    lambda x: get_abdominal_tense_negation(
        x["well_mentions_HPI_location"], "WELL", nlp_abdominal, x[id_column]
    ),
    axis=1,
)

# #### Encode Results ####
IBD_Fecal_Blood["pain_mention_remaining_notes"] = np.nan
IBD_Fecal_Blood["blood_mention_HPI"] = np.nan
IBD_Fecal_Blood["pain_mention_HPI"] = np.nan
IBD_Fecal_Blood["cr_mention_HPI"] = np.nan
IBD_Fecal_Blood["well_mention_HPI"] = np.nan
encode_column(
    IBD_Fecal_Blood,
    IBD_Fecal_Blood,
    "Pain_Present_Remaining",
    "pain_mention_remaining_notes",
    True,
)
encode_column(
    IBD_Fecal_Blood,
    IBD_Fecal_Blood,
    "Fecal_Blood_Present_HPI",
    "blood_mention_HPI",
    True,
)
encode_column(
    IBD_Fecal_Blood, IBD_Fecal_Blood, "Pain_Present_HPI", "pain_mention_HPI", True
)
encode_column(
    IBD_Fecal_Blood, IBD_Fecal_Blood, "CR_Present_HPI", "cr_mention_HPI", True
)
encode_column(
    IBD_Fecal_Blood, IBD_Fecal_Blood, "Well_Present_HPI", "well_mention_HPI", True
)

# ### Consolidate Each Blood Mention into a single Column ###
IBD_Fecal_Blood["Fecal_Blood_Value_Master"] = np.nan
IBD_Fecal_Blood["Pain_Value_Master"] = np.nan
# IBD_Fecal_Blood["Pain_ROS"] = np.nan
IBD_Fecal_Blood["CR_Master"] = np.nan
# IBD_Fecal_Blood["CR_ROS"] = np.nan
IBD_Fecal_Blood["Well_Master"] = np.nan
# IBD_Fecal_Blood["Well_ROS"] = np.nan


consolidate_columns(row_names_blood, "Fecal_Blood_Value_Master", IBD_Fecal_Blood)
consolidate_columns(row_names_pain, "Pain_Value_Master", IBD_Fecal_Blood)
consolidate_columns(rows_names_cr, "CR_Master", IBD_Fecal_Blood)
consolidate_columns(row_names_well, "Well_Master", IBD_Fecal_Blood)

# IBD_Fecal_Blood.to_json("data/abdominal_results.json")

# Diarrhea Portion of the Pipeline

nlp_diarrhea = return_diarrhea_spacy()
nlp_diarrhea = add_pipes(nlp_diarrhea)

IBD_Fecal_Blood[
    ["dia_mentions", "dia_mention_keyword", "dia_spans"]
] = IBD_Fecal_Blood.progress_apply(
    lambda x: find_diarrhea(x["Interval_History"]), axis=1, result_type="expand"
)

IBD_Fecal_Blood["stool_fre_mentions"] = IBD_Fecal_Blood.progress_apply(
    lambda x: find_stool_freq(x["Interval_History"]), axis=1
)

# IBD_Fecal_Blood[
#     ["cr_mentions", "cr_mention_keyword", "cr_spans"]
# ] = IBD_Fecal_Blood.progress_apply(
#     lambda x: find_clinical_remission(x["Interval_History"]),
#     axis=1,
#     result_type="expand",
# )

# IBD_Fecal_Blood[
#     ["well_mentions", "well_mention_keyword", "well_spans"]
# ] = IBD_Fecal_Blood.progress_apply(
#     lambda x: find_no_symptoms(x["Interval_History"]), axis=1, result_type="expand"
# )

IBD_Fecal_Blood["Diarrhea_Present"] = IBD_Fecal_Blood.progress_apply(
    lambda x: get_diarrhea_tense_negation(
        x["dia_mentions"], "Diarrhea", nlp_diarrhea, x[id_column]
    ),
    axis=1,
)

IBD_Fecal_Blood["stool_range_Present"] = IBD_Fecal_Blood.progress_apply(
    lambda x: get_diarrhea_tense_negation(
        x["stool_fre_mentions"], "range", nlp_diarrhea, x[id_column]
    ),
    axis=1,
)

IBD_Fecal_Blood["Bristol_Present"] = IBD_Fecal_Blood.progress_apply(
    lambda x: get_diarrhea_tense_negation(
        x["stool_fre_mentions"], "Bristol", nlp_diarrhea, x[id_column]
    ),
    axis=1,
)

# IBD_Fecal_Blood["CR_Present"] = IBD_Fecal_Blood.progress_apply(
#     lambda x: get_diarrhea_tense_negation(
#         x["cr_mentions"], "Clin_Rem", nlp_diarrhea, x[id_column]
#     ),
#     axis=1,
# )

# IBD_Fecal_Blood["Well_Present"] = IBD_Fecal_Blood.progress_apply(
#     lambda x: get_diarrhea_tense_negation(
#         x["well_mentions"], "WELL", nlp_diarrhea, x[id_column]
#     ),
#     axis=1,
# )

Previous_note["HPI"] = Previous_note[id_column].map(
    IBD_Fecal_Blood.set_index(id_column)["HPI"]
)

Previous_note[
    ["dia_mentions", "dia_mention_keyword", "dia_spans"]
] = Previous_note.progress_apply(
    lambda x: find_diarrhea(x["additions"])
    if x["additions"] != 1
    else find_diarrhea(x["HPI"]),
    axis=1,
    result_type="expand",
)

Previous_note["stool_fre_mentions"] = Previous_note.progress_apply(
    lambda x: find_stool_freq(x["additions"])
    if x["additions"] != 1
    else find_stool_freq(x["HPI"]),
    axis=1,
)

# Previous_note[
#     ["cr_mentions", "cr_mention_keyword", "cr_spans"]
# ] = Previous_note.progress_apply(
#     lambda x: find_clinical_remission(x["additions"])
#     if x["additions"] != 1
#     else find_clinical_remission(x["HPI"]),
#     axis=1,
#     result_type="expand",
# )

# Previous_note[
#     ["well_mentions", "well_mention_keyword", "well_spans"]
# ] = Previous_note.progress_apply(
#     lambda x: find_no_symptoms(x["additions"])
#     if x["additions"] != 1
#     else find_no_symptoms(x["HPI"]),
#     axis=1,
#     result_type="expand",
# )

Previous_note["Diarrhea_Present"] = Previous_note.progress_apply(
    lambda x: get_diarrhea_tense_negation(
        x["dia_mentions"], "Diarrhea", nlp_diarrhea, x[id_column]
    ),
    axis=1,
)

Previous_note["stool_range_Present"] = Previous_note.progress_apply(
    lambda x: get_diarrhea_tense_negation(
        x["stool_fre_mentions"], "range", nlp_diarrhea, x[id_column]
    ),
    axis=1,
)

Previous_note["Bristol_Present"] = Previous_note.progress_apply(
    lambda x: get_diarrhea_tense_negation(
        x["stool_fre_mentions"], "Bristol", nlp_diarrhea, x[id_column]
    ),
    axis=1,
)

# Previous_note["CR_Present"] = Previous_note.progress_apply(
#     lambda x: get_diarrhea_tense_negation(
#         x["cr_mentions"], "Clin_Rem", nlp_diarrhea, x[id_column]
#     ),
#     axis=1,
# )

# Previous_note["Well_Present"] = Previous_note.progress_apply(
#     lambda x: get_diarrhea_tense_negation(
#         x["well_mentions"], "WELL", nlp_diarrhea, x[id_column]
#     ),
#     axis=1,
# )

IBD_Fecal_Blood["dia_mention_Previous_note"] = np.nan
IBD_Fecal_Blood["dia_mention_Interval_History"] = np.nan

IBD_Fecal_Blood["stool_fre_mention_Previous_note"] = np.nan
IBD_Fecal_Blood["stool_fre_mention_Interval_History"] = np.nan

IBD_Fecal_Blood["bristol_mention_Previous_note"] = np.nan
IBD_Fecal_Blood["bristol_mention_Interval_History"] = np.nan

IBD_Fecal_Blood["CR_mention_Previous_note"] = np.nan
IBD_Fecal_Blood["CR_mention_Interval_History"] = np.nan

IBD_Fecal_Blood["well_mention_Previous_note"] = np.nan
IBD_Fecal_Blood["well_mention_Interval_History"] = np.nan

encode_column(
    IBD_Fecal_Blood,
    Previous_note,
    "Diarrhea_Present",
    "dia_mention_Previous_note",
    False,
)
encode_column(
    IBD_Fecal_Blood,
    IBD_Fecal_Blood,
    "Diarrhea_Present",
    "dia_mention_Interval_History",
    True,
)
encode_column(
    IBD_Fecal_Blood,
    Previous_note,
    "stool_range_Present",
    "stool_fre_mention_Previous_note",
    False,
)
encode_column(
    IBD_Fecal_Blood,
    IBD_Fecal_Blood,
    "stool_range_Present",
    "stool_fre_mention_Interval_History",
    True,
)
encode_column(
    IBD_Fecal_Blood,
    IBD_Fecal_Blood,
    "Bristol_Present",
    "bristol_mention_Interval_History",
    True,
)
encode_column(
    IBD_Fecal_Blood,
    Previous_note,
    "Bristol_Present",
    "bristol_mention_Previous_note",
    False,
)
encode_column(
    IBD_Fecal_Blood,
    IBD_Fecal_Blood,
    "Well_Present",
    "well_mention_Interval_History",
    True,
# )
# encode_column(
#     IBD_Fecal_Blood, Previous_note, "Well_Present", "well_mention_Previous_note", False
# )
# encode_column(
#     IBD_Fecal_Blood, IBD_Fecal_Blood, "CR_Present", "CR_mention_Interval_History", True
# )
# encode_column(
#     IBD_Fecal_Blood, Previous_note, "CR_Present", "CR_mention_Previous_note", False
# )

IBD_Fecal_Blood["Previous_note_id"] = IBD_Fecal_Blood.progress_apply(
    lambda x: find_previous_note(x[id_column], IBD_Fecal_Blood), axis=1
)

IBD_Fecal_Blood["Interval_History"] = IBD_Fecal_Blood["Interval_History"].progress_apply(
    lambda x: x if x != 1 else np.nan
)

IBD_Fecal_Blood["HPI_range"] = IBD_Fecal_Blood.progress_apply(
    lambda x: get_range(x["HPI_extraction_offset"], x["HPI"]) if x["HPI"] != 1 else 1,
    axis=1,
)

IBD_Fecal_Blood[
    ["dia_mentions_HPI", "dia_mentions_HPI_keywords", "dia_HPI_spans"]
] = IBD_Fecal_Blood.progress_apply(
    lambda x: find_diarrhea(x["HPI"]), result_type="expand", axis=1
)

IBD_Fecal_Blood[
    [
        "dia_mentions_HPI",
        "dia_mentions_HPI_keywords",
    ]
] = IBD_Fecal_Blood.progress_apply(
    lambda x: get_mention_location(
        x["HPI"],
        x["dia_mentions_HPI"],
        x["dia_HPI_spans"],
        x["dia_mentions_HPI_keywords"],
    ),
    result_type="expand",
    axis=1,
)

IBD_Fecal_Blood["stool_fre_mentions_HPI"] = IBD_Fecal_Blood.progress_apply(
    lambda x: find_stool_freq(x["HPI"]), axis=1
)

IBD_Fecal_Blood[
    ["cr_mentions_HPI", "cr_mention_HPI_keyword", "cr_HPI_spans"]
] = IBD_Fecal_Blood.progress_apply(
    lambda x: find_clinical_remission(x["HPI"]), axis=1, result_type="expand"
)

IBD_Fecal_Blood[
    ["well_mentions_HPI", "well_mention_HPI_keyword", "well_HPI_spans"]
] = IBD_Fecal_Blood.progress_apply(
    lambda x: find_no_symptoms(x["HPI"]), axis=1, result_type="expand"
)

IBD_Fecal_Blood["Dia_Present_Remaining"] = IBD_Fecal_Blood.progress_apply(
    lambda x: get_diarrhea_tense_negation(
        x["dia_mentions_HPI"], "Diarrhea", nlp_diarrhea, x[id_column]
    ),
    axis=1,
)

IBD_Fecal_Blood["stool_range_Present"] = IBD_Fecal_Blood.progress_apply(
    lambda x: get_diarrhea_tense_negation(
        x["stool_fre_mentions_HPI"], "range", nlp_diarrhea, x[id_column]
    ),
    axis=1,
)

IBD_Fecal_Blood["Bristol_Present"] = IBD_Fecal_Blood.progress_apply(
    lambda x: get_diarrhea_tense_negation(
        x["stool_fre_mentions_HPI"], "Bristol", nlp_diarrhea, x[id_column]
    ),
    axis=1,
)

# IBD_Fecal_Blood["CR_Present"] = IBD_Fecal_Blood.progress_apply(
#     lambda x: get_diarrhea_tense_negation(
#         x["cr_mentions_HPI"], "Clin_Rem", nlp_diarrhea, x[id_column]
#     ),
#     axis=1,
# )

# IBD_Fecal_Blood["Well_Present"] = IBD_Fecal_Blood.progress_apply(
#     lambda x: get_diarrhea_tense_negation(
#         x["well_mentions_HPI"], "WELL", nlp_diarrhea, x[id_column]
#     ),
#     axis=1,
# )

IBD_Fecal_Blood["dia_mention_remaining_notes"] = np.nan
IBD_Fecal_Blood["stool_fre_mention_remaining_notes"] = np.nan
IBD_Fecal_Blood["bristol_mention_remaining_notes"] = np.nan

# IBD_Fecal_Blood["CR_mention_remaining_notes"] = np.nan
# IBD_Fecal_Blood["well_mention_remaining_notes"] = np.nan

encode_column(
    IBD_Fecal_Blood,
    IBD_Fecal_Blood,
    "Dia_Present_Remaining",
    "dia_mention_remaining_notes",
    True,
)
encode_column(
    IBD_Fecal_Blood,
    IBD_Fecal_Blood,
    "stool_range_Present",
    "stool_fre_mention_remaining_notes",
    True,
)
encode_column(
    IBD_Fecal_Blood,
    IBD_Fecal_Blood,
    "Bristol_Present",
    "bristol_mention_remaining_notes",
    True,
)
# encode_column(
#     IBD_Fecal_Blood,
#     IBD_Fecal_Blood,
#     "Well_Present",
#     "well_mention_remaining_notes",
#     True,
# )
# encode_column(
#     IBD_Fecal_Blood, IBD_Fecal_Blood, "CR_Present", "CR_mention_remaining_notes", True
# )

# IBD_Fecal_Blood["Dia_ROS"] = np.nan
IBD_Fecal_Blood["Dia_Master"] = np.nan
# IBD_Fecal_Blood["stool_fre_ROS"] = np.nan
IBD_Fecal_Blood["Stool_fre_Master"] = np.nan
# IBD_Fecal_Blood["Bristol_ROS"] = np.nan
IBD_Fecal_Blood["Bristol_Master"] = np.nan
# IBD_Fecal_Blood["CR_ROS"] = np.nan
# IBD_Fecal_Blood["CR_Master"] = np.nan
# IBD_Fecal_Blood["Well_ROS"] = np.nan
# IBD_Fecal_Blood["Well_Master"] = np.nan

consolidate_columns(row_names_dia, "Dia_Master", IBD_Fecal_Blood)
consolidate_columns(row_names_stool_fre, "Stool_fre_Master", IBD_Fecal_Blood)
consolidate_columns(row_names_bristol, "Bristol_Master", IBD_Fecal_Blood)
# consolidate_columns(row_names_well, "Well_Master", IBD_Fecal_Blood)
# consolidate_columns(row_names_cr, "CR_Master", IBD_Fecal_Blood)

# IBD_Fecal_Blood.to_json("data/diarrhea_results.json")
IBD_Fecal_Blood.to_json(os.path.join(data_folder, "extracted_data.json"))