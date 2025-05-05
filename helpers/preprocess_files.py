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


def preprocess_files(input_list):
    # Match spike_times with appropriate key_files

    memory_path, all_json, SETTINGS_DICT = input_list
    # Split path name to get subject, session and unit ID for prettier output
    split_memory_path = split(REGEX_SEP, memory_path)  # split path
    unit_id = split_memory_path[-1][:-4]  # Example unit id: SUBJ-ID-26_200711_concat_cluster41
    split_timestamps_name = split("_*_", unit_id)
    cur_date = split_timestamps_name[1]
    subject_id = split_timestamps_name[0]
    recording_type = SETTINGS_DICT['RECORDING_TYPE_DICT'][subject_id]
    sampling_rate = SETTINGS_DICT['SAMPLING_RATE_DICT'][recording_type]

    # Use subj-session identifier to grab appropriate key
    # Stimulus info is in trialInfo

    # These are in alphabetical order. Must sort by date_trial or match with filev
    # Match by name for now for breakpoints
    key_paths_info = glob(SETTINGS_DICT['KEYS_PATH'] + sep + subject_id + '*' +
                          cur_date + "*_trialInfo.csv")
    key_paths_spout = glob(SETTINGS_DICT['KEYS_PATH'] + sep + subject_id + '*' +
                           cur_date + "*spoutTimestamps.csv")

    if len(key_paths_info) == 0:
        # Maybe the key file wasn't found because date is in Intan format
        # Convert date to ePsych format
        cur_date = datetime.strptime(cur_date, '%y%m%d')
        cur_date = datetime.strftime(cur_date, '%y-%m-%d')
        key_paths_info = glob(SETTINGS_DICT['KEYS_PATH'] + sep + subject_id + '*' +
                              cur_date + "*_trialInfo.csv")
        key_paths_spout = glob(SETTINGS_DICT['KEYS_PATH'] + sep + subject_id + '*' +
                               cur_date + "*_spoutTimestamps.csv")

    if len(key_paths_info) == 0:
        print("Key not found for " + unit_id)
        return
    try:
        cur_breakpoint_file = glob(SETTINGS_DICT['BREAKPOINT_PATH'] + sep +
                                   "_".join(split_timestamps_name[0:3]) + "_breakpoints.csv")[0]
        cur_breakpoint_df = read_csv(cur_breakpoint_file)
    except IndexError:
        print("Breakpoint file not found for " + unit_id + ". Assuming non-concatenated...")
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

    return memory_path, key_paths_info, key_paths_spout, cur_unitData, cur_breakpoint_df


def find_spoutfile_and_breakpoint(subject_id, key_path_info, key_paths_spout, cur_breakpoint_df, recording_type, sampling_rate):
    """
    Grab some files and specifics about each behavioral file
    Only works with this format. Modify indices below if you need to modify
    - Synapse:
        - Behavior file: SUBJ-ID-154_MML-Aversive-AM-210501-112033_trialInfo.csv
    - Intan:
        - Behavior file: SUBJ-ID-231_2021-07-17_15-19-28_Active_trialInfo.csv
    """
    synapse_key_finder_index = 1  # MML-Aversive-AM-210501-112033 after splitting at "_"
    intan_key_finder_index = [1, 2, 3]  # 2021-07-17, 15-19-28, Active after splitting at "_"

    if recording_type == 'synapse':
        key_finder = split(REGEX_SEP, key_path_info)[-1]
        key_finder = split("_*_", key_finder)[synapse_key_finder_index]
    else:
        key_finder = split(REGEX_SEP, key_path_info)[-1]
        key_finder = split("_*_", key_finder)
        key_finder = '_'.join([key_finder[x] for x in intan_key_finder_index])

        # This is able to handle the extra SUBJ field before the key identifier in some intan recordings.
        if 'Passive' not in key_finder and 'Active' not in key_finder and 'Aversive' not in key_finder and 'Extinction' not in key_finder:
            key_finder = split(REGEX_SEP, key_path_info)[-1]
            key_finder = split("_*_", key_finder)
            key_finder = '_'.join([key_finder[x+1] for x in intan_key_finder_index])

    key_path_spout_finder = [search(key_finder, file_name) for file_name in key_paths_spout]

    try:
        key_path_spout_finder = [i for i, x in enumerate(key_path_spout_finder) if x is not None][0]
        key_path_spout = key_paths_spout[key_path_spout_finder]
    except IndexError:
        key_path_spout = None

    # Find appropriate breakpoint for file if it exists
    try:
        breakpoint_offset_idx = cur_breakpoint_df.index[
            cur_breakpoint_df['Session_file'].str.contains(key_finder)]
    except KeyError:
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
    return key_path_spout, breakpoint_offset, key_finder