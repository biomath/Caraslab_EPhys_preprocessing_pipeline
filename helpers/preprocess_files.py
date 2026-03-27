from re import split, search
from glob import glob
from datetime import datetime
from platform import system
from os.path import sep
# Tweak the regex file separator for cross-platform compatibility
if system() == 'Windows':
    REGEX_SEP = sep * 2
else:
    REGEX_SEP = sep
import json

from pandas import DataFrame, read_csv


def match_spike_times_with_keys(input_list):
    # Match spike_times with appropriate key_files

    memory_path, all_json, SETTINGS_DICT = input_list
    # Split path name to get subject, session and unit ID for prettier output
    split_memory_path = split(REGEX_SEP, memory_path)  # split path
    unit_id = split_memory_path[-1][:-4]  # Example unit id: SUBJ-ID-26_200711_concat_cluster41
    split_timestamps_name = split('_*_', unit_id)
    cur_date = split_timestamps_name[1]
    subject_id = split_timestamps_name[0]
    recording_type = SETTINGS_DICT['RECORDING_TYPE_DICT'][subject_id]
    sampling_rate = SETTINGS_DICT['SAMPLING_RATE_DICT'][recording_type]

    # Use subj-session identifier to grab appropriate key
    # Stimulus info is in trialInfo

    # These are in alphabetical order. Must sort by date_trial or match with filev
    # Match by name for now for breakpoints
    key_paths_info = glob(SETTINGS_DICT['KEYS_PATH'] + sep + subject_id + '*' +
                          cur_date + '*_trialInfo.csv')
    key_paths_spoutTTL = glob(SETTINGS_DICT['KEYS_PATH'] + sep + subject_id + '*' +
                           cur_date + '*spoutTimestamps.csv')
    key_paths_optoTTL = glob(SETTINGS_DICT['KEYS_PATH'] + sep + subject_id + '*' +
                           cur_date + '*optoTimestamps.csv')

    if len(key_paths_info) == 0:
        # Maybe the key file wasn't found because date is in Intan format
        # Convert date to ePsych format
        modified_date = datetime.strptime(cur_date, '%y%m%d')
        modified_date = datetime.strftime(modified_date, '%y-%m-%d')
        key_paths_info = glob(SETTINGS_DICT['KEYS_PATH'] + sep + subject_id + '*' +
                              modified_date + '*_trialInfo.csv')
        key_paths_spoutTTL = glob(SETTINGS_DICT['KEYS_PATH'] + sep + subject_id + '*' +
                               modified_date + '*_spoutTimestamps.csv')
        key_paths_optoTTL = glob(SETTINGS_DICT['KEYS_PATH'] + sep + subject_id + '*' +
                               modified_date + '*_optoTimestamps.csv')
    if len(key_paths_info) == 0:
        print('Key not found for ' + unit_id)
        return
    try:  # First date format
        cur_breakpoint_file = glob(SETTINGS_DICT['BREAKPOINT_PATH'] + sep + subject_id + '*' +
                           cur_date + '*' + '_breakpoints.csv')[0]
        cur_breakpoint_df = read_csv(cur_breakpoint_file)
    except IndexError: # Second date format
        try:
            modified_date = datetime.strptime(cur_date, '%y-%m-%d')
            modified_date = datetime.strftime(modified_date, '%y%m%d')
            cur_breakpoint_file = glob(SETTINGS_DICT['BREAKPOINT_PATH'] + sep + subject_id + '*' +
                                       modified_date + '*' + '_breakpoints.csv')[0]
            cur_breakpoint_df = read_csv(cur_breakpoint_file)
        except IndexError:
            print('Breakpoint file not found for ' + unit_id + '. Assuming non-concatenated...')
            cur_breakpoint_df = DataFrame()

    # If no JSON for that unit exists, create UnitData
    try:
        cur_unitData_name = all_json[
            all_json.index(SETTINGS_DICT['OUTPUT_PATH'] + sep + 'JSON files' + sep + unit_id + '_unitData.json')]
        with open(cur_unitData_name, 'r') as json_file:
            cur_unitData = json.load(json_file)
    except ValueError:
        cur_unitData = {'Unit': unit_id, 'Subject': subject_id,
                        'Recording_type': recording_type,
                        'Sampling_rate': sampling_rate,
                        'Session': {}}

    return memory_path, key_paths_info, key_paths_spoutTTL, key_paths_optoTTL, cur_unitData, cur_breakpoint_df


def process_metadata(subject_id, key_path_info, key_paths_spoutTTL, key_paths_optoTTL,
                     key_finder_index_dict, cur_breakpoint_df, recording_type, sampling_rate):
    '''
    Grab some files and specifics about each behavioral file
    '''
    key_finder_index = key_finder_index_dict[recording_type]

    if recording_type == 'synapse':
        key_finder = split(REGEX_SEP, key_path_info)[-1]
        key_finder = split('_*_', key_finder)[key_finder_index]
    else:
        key_finder = split(REGEX_SEP, key_path_info)[-1]
        key_finder = split('_*_', key_finder)
        key_finder = '_'.join([key_finder[x] for x in key_finder_index])

        # This is able to handle the extra SUBJ field before the key identifier in some intan recordings.
        if ('passive' not in key_finder.lower() and 'active' not in key_finder.lower() and
                'aversive' not in key_finder.lower() and 'extinction' not in key_finder.lower()):
            key_finder = split(REGEX_SEP, key_path_info)[-1]
            key_finder = split('_*_', key_finder)
            key_finder = '_'.join([key_finder[x+1] for x in key_finder_index])

    try:
        spoutTTL_path_finder = [search(key_finder, file_name) for file_name in key_paths_spoutTTL]
        spoutTTL_path_finder = [i for i, x in enumerate(spoutTTL_path_finder) if x is not None][0]
        key_path_spoutTTL = key_paths_spoutTTL[spoutTTL_path_finder]
    except IndexError:
        print('Spout TTL file not found for ' + subject_id + ', file ' + key_path_info+ '. Ignore if this is a passive recording.')
        key_path_spoutTTL = None

    try:
        optoTTL_path_finder = [search(key_finder, file_name) for file_name in key_paths_optoTTL]
        optoTTL_path_finder = [i for i, x in enumerate(optoTTL_path_finder) if x is not None][0]
        key_path_optoTTL = key_paths_optoTTL[optoTTL_path_finder]
    except IndexError:
        print('Opto TTL file not found for ' + subject_id + ', file ' + key_path_info+ '. Ignore if no opto was done.')
        key_path_optoTTL = None

    # Find appropriate breakpoint for file if it exists
    try:
        breakpoint_offset_idx = cur_breakpoint_df.index[
            cur_breakpoint_df['Session_file'].str.contains(key_finder)]
    except KeyError:
        print('Breakpoint file not found for ' + subject_id + ', file ' + key_path_info + '. Ignore if this is a non-cocatenated recording.')
        breakpoint_offset_idx = 0

    # also grab previous session's breakpoint if it exists
    try:
        breakpoint_offset = cur_breakpoint_df.loc[
            breakpoint_offset_idx - 1, 'Break_point_seconds'].values[0]
    # Older recordings do not have Break_point_seconds but Break_point. Need to divide by sampling rate
    except KeyError:
        try:
            breakpoint_offset = cur_breakpoint_df.loc[
                breakpoint_offset_idx - 1, 'Break_point'].values[0]  # grab previous session's breakpoint
            breakpoint_offset = breakpoint_offset / sampling_rate
        except IndexError as e:
            print('Something off with ' + subject_id + ', file ' + key_finder)
            raise e
        except KeyError:
            breakpoint_offset = 0  # first file; no breakpoint offset needed
        except Exception as e:
            raise e
    return key_path_spoutTTL, key_path_optoTTL, breakpoint_offset, key_finder