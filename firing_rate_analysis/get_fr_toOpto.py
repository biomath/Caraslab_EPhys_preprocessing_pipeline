import pandas as pd

import numpy as np
from pandas import read_csv
import csv
from re import split
from os.path import sep
import platform

# Tweak the regex file separator for cross-platform compatibility
if platform.system() == 'Windows':
    REGEX_SEP = sep * 2
else:
    REGEX_SEP = sep


def get_fr_toOpto(memory_path,
                    key_path_info,
                    key_path_optoTTL,
                    unit_id,
                    output_path,
                    cur_unitData,
                    experiment_tag=None,
                    first_cell_flag=True,
                    breakpoint_offset=0,
                    baseline_duration_for_fr: dict | float = 0.5,
                    resp_duration_for_fr: dict | float = 0.5,
                    pre_stim_raster: dict | float = 1.,
                    post_stim_raster: dict | float = 1.
                  ):
    """
    Process spike data around opto onset and offset TTLs
    Takes key files and writes two files:
    1. Firing rates within window (_opto_firing_rate.csv)
    2. All timestamped spikes during that window for timeseries analyses (_opto_spikes.json)
    """

    # Load opto key files
    if key_path_optoTTL is not None:
        opto_key_times = read_csv(key_path_optoTTL)
    else:
        print('No opto TTL key file found. Skipping this step for: ' + memory_path)
        return cur_unitData

    # Load spike times
    spike_times = np.genfromtxt(memory_path)

    led_trial_id = list()

    # "Baseline" will be period immediately preceding the opto onset or offset
    led_on_baseline_FR_list = list()
    led_on_resp_FR_list = list()
    led_on_zerocentered_spikes = list()

    led_off_baseline_FR_list = list()
    led_off_resp_FR_list = list()
    led_off_zerocentered_spikes = list()

    for dummy_idx, cur_trial in opto_key_times.iterrows():
        led_trial_id.append(int(dummy_idx)+1)
        # LED onset
        timestamp_breakpointed = cur_trial['LED_onset'] + breakpoint_offset
        spikes_around_trial = spike_times[
            (spike_times >= (timestamp_breakpointed - pre_stim_raster)) &
            (spike_times < (timestamp_breakpointed + post_stim_raster))]

        # Zero center around trial onset
        led_on_zerocentered_spikes.append(spikes_around_trial - timestamp_breakpointed)

        baseline_spikes = spike_times[
            (spike_times >= timestamp_breakpointed - baseline_duration_for_fr) &
            (spike_times < timestamp_breakpointed)]

        resp_spikes = spike_times[
            (spike_times >= timestamp_breakpointed) &
            (spike_times < timestamp_breakpointed + resp_duration_for_fr)]

        # Save to lists
        led_on_baseline_FR_list.append(len(baseline_spikes) / baseline_duration_for_fr)
        led_on_resp_FR_list.append(len(resp_spikes) / resp_duration_for_fr)

        # LED offset
        timestamp_breakpointed = cur_trial['LED_offset'] + breakpoint_offset
        spikes_around_trial = spike_times[
            (spike_times >= (timestamp_breakpointed - pre_stim_raster)) &
            (spike_times < (timestamp_breakpointed + post_stim_raster))]

        # Zero center around trial onset
        led_off_zerocentered_spikes.append(spikes_around_trial - timestamp_breakpointed)

        baseline_spikes = spike_times[
            (spike_times >= timestamp_breakpointed - baseline_duration_for_fr) &
            (spike_times < timestamp_breakpointed)]

        resp_spikes = spike_times[
            (spike_times >= timestamp_breakpointed) &
            (spike_times < timestamp_breakpointed + resp_duration_for_fr)]

        # Save to lists
        led_off_baseline_FR_list.append(len(baseline_spikes) / baseline_duration_for_fr)
        led_off_resp_FR_list.append(len(resp_spikes) / resp_duration_for_fr)

    if experiment_tag is None:
        csv_name = ''
    else:
        csv_name = experiment_tag + '_opto_firing_rate.csv'

    trialInfo_filename = split(REGEX_SEP, key_path_info)[-1][:-4]

    # write or append
    if first_cell_flag:
        write_or_append_flag = 'w'
    else:
        write_or_append_flag = 'a'
    with open(output_path + sep + csv_name, write_or_append_flag, newline='') as file:
        writer = csv.writer(file, delimiter=',')
        # Write header if first cell
        if write_or_append_flag == 'w':
            writer.writerow(['Unit'] +
                            ['Key_file'] +
                            ['TrialID'] +
                            ['LED_onset'] +
                            ['LED_offset'] +
                            ['Period'] +
                            ['FR_LED_onset'] +
                            ['FR_LED_offset']
                            )
        for dummy_idx in range(0, len(led_on_baseline_FR_list)):
            cur_row = opto_key_times.iloc[dummy_idx, :]

            for (period, led_on_FR_list, led_off_FR_list) in \
                    zip(('baseline', 'response'),
                        (led_on_baseline_FR_list, led_on_resp_FR_list),
                        (led_off_baseline_FR_list, led_off_resp_FR_list)
                        ):
                try:
                    writer.writerow([unit_id] +
                                    [trialInfo_filename] +
                                    [led_trial_id[dummy_idx]] +
                                    [cur_row['LED_onset']] +
                                    [cur_row['LED_offset']] +
                                    [period] +
                                    [led_on_FR_list[dummy_idx]] +
                                    [led_off_FR_list[dummy_idx]])
                except KeyError:
                    pass

    # Add all info to unitData
    for key_name in ('LED_onset', 'LED_offset'):
        cur_unitData["Session"][trialInfo_filename][key_name] = np.round(opto_key_times[key_name].values, 4)

    cur_unitData["Session"][trialInfo_filename]['LED_on_trialSpikes'] = np.round(led_on_zerocentered_spikes, 4)
    cur_unitData["Session"][trialInfo_filename]['LED_off_trialSpikes'] = np.round(led_off_zerocentered_spikes, 4)

    cur_unitData["Session"][trialInfo_filename]['LED_on_baseline_FR'] = np.round(led_on_baseline_FR_list, 4)
    cur_unitData["Session"][trialInfo_filename]['LED_off_baseline_FR'] = np.round(led_off_baseline_FR_list, 4)

    cur_unitData["Session"][trialInfo_filename]['LED_on_response_FR'] = np.round(led_on_resp_FR_list, 4)
    cur_unitData["Session"][trialInfo_filename]['LED_off_response_FR'] = np.round(led_off_resp_FR_list, 4)

    return cur_unitData
