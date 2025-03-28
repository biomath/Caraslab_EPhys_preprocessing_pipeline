from glob import glob
import numpy as np
from re import split, search
from os.path import sep
import platform
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
    key_paths_spout = glob(SETTINGS_DICT['KEYS_PATH'] + sep + "*spoutTimestamps.csv")
    save_dir = output_path

    # save_dir = output_path + sep + 'new_respLatencies'
    # makedirs(save_dir, exist_ok=True)

    for recording_path in session_date_paths:
        # Automatically skip passive files here for obvious reasons :)
        if 'Passive' in recording_path:
            continue

        split_key_path = split(REGEX_SEP, recording_path)[-1]  # split path
        subject_id = split('_*_', split_key_path)[0]
        recording_type = SETTINGS_DICT['RECORDING_TYPE_DICT'][subject_id]

        # Split path name to get subject, session and unit ID for prettier output
        synapse_key_finder_index = 1  # MML-Aversive-AM-210501-112033 after splitting at "_"
        intan_key_finder_index = [1, 2, 3]  # 2021-07-17, 15-19-28, Active after splitting at "_"

        if recording_type == 'synapse':
            key_finder = split(REGEX_SEP, recording_path)[-1]
            key_finder = split("_*_", key_finder)[synapse_key_finder_index]
        else:
            key_finder = split(REGEX_SEP, recording_path)[-1]
            key_finder = split("_*_", key_finder)
            key_finder = '_'.join([key_finder[x] for x in intan_key_finder_index])

            # This is able to handle the extra SUBJ field before the key identifier in some intan recordings.
            if 'Passive' not in key_finder and 'Active' not in key_finder and 'Aversive' not in key_finder and 'Exctinction' not in key_finder:
                key_finder = split(REGEX_SEP, recording_path)[-1]
                key_finder = split("_*_", key_finder)
                key_finder = '_'.join([key_finder[x + 1] for x in intan_key_finder_index])

        key_path_spout_finder = [search(key_finder, file_name) for file_name in key_paths_spout]

        key_path_spout_finder = [i for i, x in enumerate(key_path_spout_finder) if x is not None][0]
        key_path_spout = key_paths_spout[key_path_spout_finder]

        info_key_times = read_csv(recording_path)
        spout_key_times = read_csv(key_path_spout)

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
