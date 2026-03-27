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
    key_finder_index_dict = SETTINGS_DICT['KEY_FINDER_INDEX']

    save_dir = output_path

    # save_dir = output_path + sep + 'new_respLatencies'
    # makedirs(save_dir, exist_ok=True)

    for recording_path in session_date_paths:
        # Automatically skip passive files here for obvious reasons :)
        if 'passive' in recording_path.lower():
            continue

        split_key_path = split(REGEX_SEP, recording_path)[-1]  # split path
        subject_id = split('_*_', split_key_path)[0]
        recording_type = SETTINGS_DICT['RECORDING_TYPE_DICT'][subject_id]
        key_finder_index = key_finder_index_dict[recording_type]

        if recording_type == 'synapse':
            key_finder = split(REGEX_SEP, recording_path)[-1]
            key_finder = split("_*_", key_finder)[key_finder_index]
        else:
            key_finder = split(REGEX_SEP, recording_path)[-1]
            key_finder = split("_*_", key_finder)
            key_finder = '_'.join([key_finder[x] for x in key_finder_index])

            # This is able to handle the extra SUBJ field before the key identifier in some intan recordings.
            if ('passive' not in key_finder.lower() and 'active' not in key_finder.lower() and
                    'aversive' not in key_finder.lower() and 'extinction' not in key_finder.lower()):
                key_finder = split(REGEX_SEP, recording_path)[-1]
                key_finder = split("_*_", key_finder)
                key_finder = '_'.join([key_finder[x + 1] for x in key_finder_index])

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
            cur_onset = row_slice['Trial_onset']
            cur_offset = row_slice['Trial_offset']
            cur_spout_offsets = spout_offsets[(spout_offsets >= cur_onset) & (spout_offsets < cur_offset)]
            if (row_slice['Hit'] == 1) | (row_slice['FA'] == 1):
                if len(cur_spout_offsets) == 0:  # Sometimes this is not registered properly in RZ6
                    print('Spout offset not registered properly in: ' + recording_path +
                          '\nTrialID: ' + str(row_slice['TrialID']) + '\n\n')
                    new_latencies[row_idx] = np.nan
                else:
                    last_offset = cur_spout_offsets[-1]  # Last offset probably triggered the outcome
                    new_latencies[row_idx] = last_offset - cur_onset
            elif row_slice['Miss'] == 1:
                # If miss trial:
                # 1. Look for spout offsets during the trial (above). These trials can be handled separately since the animal
                #   might have detected the AM sound but failed to stay off spout for some reason
                # 2. If no spout offsets during the trial were found, look for offsets during the shock period. If
                #   none are found, return NaN
                if len(cur_spout_offsets) == 0:  # No spout offsets during trial, look for offsets during shock period plus 0.5 s
                    cur_spout_offsets = spout_offsets[(spout_offsets >= (cur_onset + shock_start_end[0])) &
                                                  (spout_offsets < (cur_onset + shock_start_end[1] + 0.5))]

                if len(cur_spout_offsets) == 0:  # Either animal did not withdraw with shock or this was a non-shocked miss without any spout withdrawals
                    new_latencies[row_idx] = np.nan
                else:
                    last_offset = cur_spout_offsets[-1]  # Get the last offset
                    new_latencies[row_idx] = last_offset - cur_onset
            else:  # CR trials: ther might be special cases where the animal quickly left the spout and came back; handle these trials with caution
                if len(cur_spout_offsets) == 0:
                    new_latencies[row_idx] = np.nan
                else:
                    last_offset = cur_spout_offsets[-1]  # Get the last offset
                    new_latencies[row_idx] = last_offset - cur_onset

        # Replace dummy latencies
        info_key_times['RespLatency'] = new_latencies

        # Save new file
        info_key_times.to_csv(save_dir + sep + split(REGEX_SEP, recording_path)[-1], index=False)
