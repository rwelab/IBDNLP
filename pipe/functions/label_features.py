abd_features_1 = ['pain_mention_Previous_note', 
                  'pain_mention_HPI', 
                  'Pain_Value_Master', 
                  'pain_mentions', 
                  'pain_mention_Interval_History', 
                  'well_mention_HPI_keyword', 
                  'stool_fre_mentions', 
                  'pain_mentions_HPI', 
                  'Fecal_Blood_Present', 
                  'dia_mentions_HPI', 
                  'well_mentions_HPI', 
                  'blood_mention_keyword', 
                  'stool_fre_mentions_HPI', 
                  'cr_mention_keyword', 
                  'Stool_fre_Master', 
                  'dia_mention_Interval_History', 
                  'HPI/ROS Presence']

abd_features_2 = ['Pain_Present_HPI', 
                  'Pain_Present', 
                  'pain_mention_Previous_note', 
                  'pain_mention_HPI', 
                  'Pain_Value_Master', 
                  'pain_mention_Interval_History', 
                  'pain_mention_keyword', 
                  'CR_Master', 
                  'well_mentions', 
                  'blood_mentions', 
                  'pain_mentions', 
                  'stool_fre_mentions', 
                  'dia_mention_keyword', 
                  'Well_Present_HPI', 
                  'stool_fre_mention_Interval_History', 
                  'pain_mentions_HPI', 
                  'dia_mentions_HPI', 
                  'Fecal_Blood_Present', 
                  'Well_Present', 
                  'well_mention_HPI', 
                  'stool_fre_mentions_HPI', 
                  'Stool_fre_Master', 
                  'blood_mention_keyword', 
                  'Well_Master', 
                  'dia_mentions', 
                  'Fecal_Blood_Present_HPI', 
                  'well_mention_Interval_History', 
                  'cr_mentions_HPI', 
                  'well_mention_HPI_keyword', 
                  'pain_mentions_remaining_keywords', 
                  'Dia_Present_Remaining', 
                  'stool_range_Present', 
                  'dia_mentions_HPI_keywords', 
                  'blood_mention_Previous_note']

diarrhea_features_1 = ['stool_fre_mention_remaining_notes', 
                       'Dia_Master', 
                       'Well_Present_HPI', 
                       'Fecal_Blood_Value_Master', 
                       'Dia_Present_Remaining', 
                       'dia_mention_remaining_notes', 
                       'pain_mentions_HPI', 
                       'blood_mention_Previous_note', 
                       'stool_fre_mentions_HPI', 
                       'well_mentions', 
                       'blood_mention_keyword']


diarrhea_features_2 = ['well_mentions', 
                       'stool_range_Present', 
                       'stool_fre_mentions', 
                       'dia_mentions', 
                       'Dia_Master', 
                       'dia_mentions_HPI_keywords', 
                       'dia_mentions_HPI', 
                       'blood_mentions_remaining_keywords', 
                       'well_mentions_HPI', 
                       'stool_fre_mention_remaining_notes', 
                       'pain_mentions_remaining_keywords', 
                       'stool_fre_mention_Previous_note', 
                       'Well_Present', 
                       'Well_Present_HPI', 
                       'Stool_fre_Master', 
                       'Pain_Present_HPI', 
                       'ROS_Valid', 
                       'Fecal_Blood_Present_HPI', 
                       'stool_fre_mentions_HPI', 
                       'dia_mention_remaining_notes']


def map2int(x):
    if type(x) == str:
        return 1
    if type(x) == list:
        return len(x)
    if type(x) == bool:
        return int(x)
    else:
        return x