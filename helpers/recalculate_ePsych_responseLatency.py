from glob import glob
from datetime import datetime
import numpy as np
from re import split
from os.path import sep
from os import makedirs
import platform
from helpers.preprocess_files import preprocess_files
from pandas import read_csv

# Tweak the regex file separator for cross-platform compatibility
if platform.system() == 'Windows':
    REGEX_SEP = sep * 2
else:
    REGEX_SEP = sep


def recalculate_ePsych_responseLatency(input_list):
    """
    Recalculates response latency based on spout offset responses for the AM detection task
    Older RPvds circuits did not properly calculate response latencies

    :param input_list: list of inputs used by this function. Convoluted because
                       I just copied the structure from the multiprocessing functions
    :return: None; alters the trialInfo.csv files
    """

    (session_date_paths, SETTINGS_DICT) = input_list

    # Load globals
    shock_start_end = SETTINGS_DICT['SHOCK_START_END']

    output_path = SETTINGS_DICT['KEYS_PATH']
    save_dir = output_path

    # save_dir = output_path + sep + 'new_respLatencies'
    # makedirs(save_dir, exist_ok=True)

    for recording_path in session_date_paths:
        # Automatically skip passive files here for obvious reasons :)
        if 'Passive' in recording_path:
            continue

        # Split path name to get subject, session and unit ID for prettier output
        split_filename = split('_*_', split(REGEX_SEP, recording_path)[3])  # split path
        cur_date = split('-*-', split_filename[1])[3]
        subject_id = split_filename[0]

        # Use subj-session identifier to grab appropriate key
        # Stimulus info is in trialInfo

        # These are in alphabetical order. Must sort by date_trial or match with file
        # Match by name for now for breakpoints
        key_paths_spout = glob(SETTINGS_DICT['KEYS_PATH'] + sep + subject_id + '*Aversive*' +
                               cur_date + "*spoutTimestamps.csv")

        if len(key_paths_spout) == 0:
            # Maybe the key file wasn't found because date is in Intan format
            # Convert date to ePsych format
            cur_date = datetime.strptime(cur_date, '%y%m%d')
            cur_date = datetime.strftime(cur_date, '%y-%m-%d')
            key_paths_spout = glob(SETTINGS_DICT['KEYS_PATH'] + sep + subject_id + '*' +
                                   cur_date + "*_spoutTimestamps.csv")

        info_key_times = read_csv(recording_path)
        spout_key_times = read_csv(key_paths_spout[0])

        try:
            spout_offsets = spout_key_times['Spout_offset'].values
        except TypeError:
            print('Something weird with: ' + recording_path + '. Could not gather spout offset times\n\n')
            continue

        new_latencies = np.zeros(len(info_key_times))

        for row_idx, row_slice in info_key_times.iterrows():
            if (row_slice['Hit'] == 1) | (row_slice['FA'] == 1):
                cur_onset = row_slice['Trial_onset']
                cur_offset = row_slice['Trial_offset']
                cur_spout_offsets = spout_offsets[(spout_offsets >= cur_onset) & (spout_offsets < cur_offset)]
                if len(cur_spout_offsets) == 0:  # Sometimes this is not registered properly in RZ6
                    print('Spout offset not registered properly in: ' + recording_path +
                          '\nTrialID: ' + str(row_slice['TrialID']) + '\n\n')
                    new_latencies[row_idx] = np.nan
                else:
                    last_offset = cur_spout_offsets[-1]  # Last offset probably triggered the outcome
                    new_latencies[row_idx] = last_offset - cur_onset
            elif row_slice['Miss'] == 1:
                # If miss trial:
                # 1. Look for spout offsets during the trial. These trials can be handled separately since the animal
                #   might have detected the AM sound but failed to stay off spout for some reason
                # 2. If no spout offsets during the trial were found, look for offsets during the shock period. If
                #   none are found, return NaN
                cur_onset = row_slice['Trial_onset']
                cur_offset = row_slice['Trial_offset']

                cur_spout_offsets = spout_offsets[(spout_offsets >= cur_onset) & (spout_offsets < cur_offset)]
                if len(cur_spout_offsets) == 0:
                    cur_spout_offsets = spout_offsets[(spout_offsets >= (cur_onset + shock_start_end[0])) &
                                                  (spout_offsets < (cur_offset + shock_start_end[1]))]

                if len(cur_spout_offsets) == 0:  # Either animal did not withdraw with shock or this was a non-shocked miss
                    new_latencies[row_idx] = np.nan
                else:
                    last_offset = cur_spout_offsets[-1]  # Get the last offset
                    new_latencies[row_idx] = last_offset - cur_onset

            else:
                new_latencies[row_idx] = np.nan

        # Replace dummy latencies
        info_key_times['RespLatency'] = new_latencies

        # Save new file
        info_key_times.to_csv(save_dir + sep + split(REGEX_SEP, recording_path)[-1], index=False)
