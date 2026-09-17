from glob import glob
import numpy as np
from re import split, search
from os.path import sep
import platform
from pandas import read_csv

from helpers.preprocess_files import extract_session_key

# Tweak the regex file separator for cross-platform compatibility
if platform.system() == 'Windows':
    REGEX_SEP = sep * 2
else:
    REGEX_SEP = sep

# Window (in seconds, relative to trial onset) over which spout onsets/offsets
# are collected for the SpoutOnset_times / SpoutOffset_times output columns
SAVE_WINDOW_AFTER_ONSET = 2.0


def _spoutreturn_latency(trigger_offset, spout_onsets, max_lag):
    """Lag (s) from the spout offset that triggered the trial outcome to the next spout onset.

    ``trigger_offset`` is the spout offset assumed to have triggered the trial outcome
    (the last offset within the window used to compute RespLatency). ``spout_onsets`` is
    the full array of spout onset timestamps for the session. ``max_lag`` bounds the
    search: only spout onsets occurring within ``max_lag`` seconds after ``trigger_offset``
    are considered. Returns the gap to the first such spout onset (the animal returning to
    the spout), or NaN if the animal does not return to the spout within that window.
    """
    later_onsets = spout_onsets[(spout_onsets > trigger_offset) &
                                (spout_onsets <= trigger_offset + max_lag)]
    if len(later_onsets) == 0:
        return np.nan
    return later_onsets[0] - trigger_offset


def recalculate_ePsych_responseLatency(input_list):
    """
    Recalculates response latency based on spout offset responses for the AM detection task
    Older RPvds circuits did not properly calculate response latencies

    Also (re)writes these columns on each active-session trialInfo.csv:
      - RespLatency: latency from trial onset to the outcome-triggering spout offset
      - SpoutReturn_latency: lag from that outcome-triggering spout offset to the next
        spout onset (the animal returning to the spout), searched only within
        SETTINGS_DICT['SPOUTRETURN_LATENCY_MAX_LAG'] seconds (default 5); NaN if the
        animal does not return to the spout within that window
      - SpoutOffset_times / SpoutOnset_times: semicolon-joined spout event timestamps
        within SAVE_WINDOW_AFTER_ONSET seconds of trial onset

    :param input_list: list of inputs used by this function. Convoluted because
                       I just copied the structure from the multiprocessing functions
    :return: None; alters the trialInfo.csv files
    """

    (session_date_paths, SETTINGS_DICT) = input_list

    # Load globals
    shock_start_end = SETTINGS_DICT['SHOCK_START_END']
    # Upper bound (s) on how far after the outcome-triggering spout offset to look for the
    # next spout onset when computing SpoutReturn_latency. Optional in SETTINGS_DICT.
    spoutreturn_max_lag = SETTINGS_DICT.get('SPOUTRETURN_LATENCY_MAX_LAG', 5.0)
    output_path = SETTINGS_DICT['KEYS_PATH']
    key_paths_spout = glob(SETTINGS_DICT['KEYS_PATH'] + sep + "*spoutTimestamps.csv")

    save_dir = output_path

    # save_dir = output_path + sep + 'new_respLatencies'
    # makedirs(save_dir, exist_ok=True)

    for recording_path in session_date_paths:
        # Automatically skip passive files here for obvious reasons :)
        if 'passive' in recording_path.lower():
            continue

        split_key_path = split(REGEX_SEP, recording_path)[-1]  # split path
        subject_id = split('_*_', split_key_path)[0]
        key_finder = extract_session_key(split_key_path)

        key_path_spout_finder = [search(key_finder, file_name) for file_name in key_paths_spout]

        key_path_spout_finder = [i for i, x in enumerate(key_path_spout_finder) if x is not None][0]
        key_path_spout = key_paths_spout[key_path_spout_finder]

        info_key_times = read_csv(recording_path)
        spout_key_times = read_csv(key_path_spout)

        try:
            spout_offsets = spout_key_times['Spout_offset'].values
            spout_onsets = spout_key_times['Spout_onset'].values
        except TypeError:
            print('Something weird with: ' + recording_path + '. Could not gather spout offset times\n\n')
            continue

        new_latencies = np.zeros(len(info_key_times))
        spoutreturn_latencies = np.full(len(info_key_times), np.nan)
        spout_offset_times_list = [None] * len(info_key_times)
        spout_onset_times_list = [None] * len(info_key_times)

        for row_idx, row_slice in info_key_times.iterrows():
            cur_onset = row_slice['Trial_onset']
            cur_offset = row_slice['Trial_offset']
            cur_spout_offsets = spout_offsets[(spout_offsets >= cur_onset) & (spout_offsets < cur_offset)]
            cur_spout_onsets = spout_onsets[(spout_onsets >= cur_onset) & (spout_onsets < cur_offset)]
            if (row_slice['Hit'] == 1) | (row_slice['FA'] == 1):
                if len(cur_spout_offsets) == 0:  # Sometimes this is not registered properly in RZ6
                    print('Spout offset not registered properly in: ' + recording_path +
                          '\nTrialID: ' + str(row_slice['TrialID']) + '\n\n')
                    new_latencies[row_idx] = np.nan
                else:
                    last_offset = cur_spout_offsets[-1]  # Last offset probably triggered the outcome
                    new_latencies[row_idx] = last_offset - cur_onset
                    spoutreturn_latencies[row_idx] = _spoutreturn_latency(last_offset, spout_onsets, spoutreturn_max_lag)
            elif row_slice['Miss'] == 1:
                # If miss trial:
                # 1. Look for spout offsets during the trial (above). These trials can be handled separately since the animal
                #   might have detected the AM sound but failed to stay off spout for some reason
                # 2. If no spout offsets during the trial were found, look for offsets during the shock period. If
                #   none are found, return NaN
                if len(cur_spout_offsets) == 0:  # No spout offsets during trial, look for offsets during shock period
                    cur_spout_offsets = spout_offsets[(spout_offsets >= (cur_onset + shock_start_end[0])) &
                                                      (spout_offsets < (cur_onset + shock_start_end[1]))]
                    cur_spout_onsets = spout_onsets[(spout_onsets >= (cur_onset + shock_start_end[0])) &
                                                    (spout_onsets < (cur_onset + shock_start_end[1]))]

                if len(cur_spout_offsets) == 0:  # Either animal did not withdraw with shock or this was a non-shocked miss without any spout withdrawals
                    new_latencies[row_idx] = np.nan
                else:
                    last_offset = cur_spout_offsets[-1]  # Get the last offset
                    new_latencies[row_idx] = last_offset - cur_onset
                    spoutreturn_latencies[row_idx] = _spoutreturn_latency(last_offset, spout_onsets, spoutreturn_max_lag)
            else:  # CR trials: there might be special cases where the animal quickly left the spout and came back; handle these trials with caution
                if len(cur_spout_offsets) == 0:
                    new_latencies[row_idx] = np.nan
                else:
                    last_offset = cur_spout_offsets[-1]  # Get the last offset
                    new_latencies[row_idx] = last_offset - cur_onset
                    spoutreturn_latencies[row_idx] = _spoutreturn_latency(last_offset, spout_onsets, spoutreturn_max_lag)

            # For the saved output columns, collect every spout onset/offset from
            # trial onset up to SAVE_WINDOW_AFTER_ONSET seconds afterwards,
            # independent of the (narrower) windows used for latency above
            save_window_end = cur_onset + SAVE_WINDOW_AFTER_ONSET
            save_spout_offsets = spout_offsets[(spout_offsets >= cur_onset) & (spout_offsets < save_window_end)]
            save_spout_onsets = spout_onsets[(spout_onsets >= cur_onset) & (spout_onsets < save_window_end)]

            spout_offset_times_list[row_idx] = ';'.join(map(str, save_spout_offsets))
            spout_onset_times_list[row_idx] = ';'.join(map(str, save_spout_onsets))

        # Replace dummy latencies
        info_key_times['RespLatency'] = new_latencies

        # Lag between the outcome-triggering spout offset and the next spout return (onset)
        info_key_times['SpoutReturn_latency'] = spoutreturn_latencies

        info_key_times['SpoutOffset_times'] = spout_offset_times_list
        info_key_times['SpoutOnset_times'] = spout_onset_times_list

        # Save new file
        info_key_times.to_csv(save_dir + sep + split(REGEX_SEP, recording_path)[-1], index=False)
