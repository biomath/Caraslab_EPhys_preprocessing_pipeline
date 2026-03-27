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


def get_fr_toTrials(memory_path,
                    key_path_info,
                    unit_id,
                    output_path,
                    cur_unitData,
                    experiment_tag=None,
                    first_cell_flag=True,
                    breakpoint_offset=0,
                    nonAM_duration_for_fr: dict | int = 0.5,
                    trial_duration_for_fr: dict | int = 0.5,
                    trialOnset_duration_for_fr: dict | int = 0.4,
                    pre_stim_raster=2.,  # For timestamped spikeTimes
                    post_stim_raster=4.,  # For timestamped spikeTimes
                    aftertrial_FR_start: dict | int = 1.3,
                    # For calculating after-stimulus firing rate; useful for Misses
                    aftertrial_FR_end: dict | int = 2,
                    resptime_FR_start: dict | int = 0.3,  # For calculating after-response firing rate
                    resptime_FR_end: dict | int = 1.8,
                    beforeresp_FR_start: dict | int = 0.5,
                    # For calculating pre-response firing rate (will be converted to negative)
                    beforeresp_FR_end: dict | int = 0
                    ):
    """
    Process spike data around trials
    Takes key files and writes two files:
    1. Firing rates within window (_AMsound_firing_rate.csv)
    2. Number of spout events during that window (_AMsound_firing_rate.csv)
    3. All timestamped spikes during that window for timeseries analyses (_AMsound_spikes.json)
    """
    # Load key files
    info_key_times = read_csv(key_path_info)

    # Load spike times
    spike_times = np.genfromtxt(memory_path)

    # The default is 0, so this will only make a difference if set at function call
    breakpoint_offset_time = breakpoint_offset

    # Check for opto tags. Add dummy tags if non-existent
    if 'JitOnset' not in info_key_times.columns:
        info_key_times['JitOnset'] = 0
    if 'LED_TTL' not in info_key_times.columns:
        info_key_times['LED_TTL'] = 0

    # Grab all trials
    # These can be Hit, Miss or FA
    # CR trials are not included to save space, but you can remove this filter if you wish
    relevant_key_times = info_key_times[(info_key_times['Reminder'] == 0) & (info_key_times['CR'] == 0)].copy()

    # Now grab spike times
    # Baseline will be the CR trial immediately preceding the current trial
    nonAM_FR_list = list()
    trial_FR_list = list()
    trialOnset_FR_list = list()
    aftertrial_FR_list = list()
    resptime_FR_list = list()
    beforeresp_FR_list = list()

    # Actual spike timestamps around trial and spout offset
    zerocentered_trial_spikes = list()
    zerocentered_response_spikes = list()

    for dummy_index, cur_trial in relevant_key_times.iterrows():
        # Get current trial's response time (i.e., latency)
        # Only relevant for active sessions
        # IMPORTANT NOTE: In miss trials, if latency value is during the AM period, this indicates that the animal
        #   attempted to withdraw from spout before the shock, but returned for some reason. Handle these trials as you wish
        #   For true undetected misses, respLatency (if it exists) will always be after the AM period during the shock period
        if 'passive' not in key_path_info.lower():
            # Get spike times around the current stimulus onset
            # For baseline, go to the previous trial that resulted in a correct rejection (NO-GO) which
            # may not be the immediately preceding trial
            # Also ensure the RespLatency is NaN, indicating that the animal was stable at the spout
            try:
                previous_cr = info_key_times[(info_key_times['CR'] == 1) &
                                             (info_key_times['TrialID'] < cur_trial['TrialID']) &
                                             np.isnan(info_key_times['RespLatency'])].iloc[-1]
                previous_cr_onset = previous_cr['Trial_onset'] + breakpoint_offset_time
                valid_baseline = True
            except IndexError:  # In case there is no valid CR before current trial, make baseline firing = NaN
                valid_baseline = False

            cur_resptime = cur_trial['RespLatency']  # Either NA or >0

            if cur_trial['Hit'] == 1:
                cur_trial_type = 'Hit'
            elif cur_trial['Miss'] == 1:
                cur_trial_type = 'Miss'
            elif cur_trial['FA'] == 1:
                cur_trial_type = 'FA'  # RespLatency is a different interpretation for FAs, since the start of the trial is arbitrary, but compute anyways
            else:
                cur_trial_type = 'CR'

        else:
            # The only difference between active and passive sessions is that the respLatency is not considered for passive sessions,
            # when gathering the baseline trial
            try:
                previous_cr = info_key_times[(info_key_times['CR'] == 1) &
                                             (info_key_times['TrialID'] < cur_trial['TrialID'])].iloc[-1]
                previous_cr_onset = previous_cr['Trial_onset'] + breakpoint_offset_time
                valid_baseline = True
            except IndexError:  # In case there is no valid CR before current trial, make baseline firing = NaN
                valid_baseline = False
            cur_resptime = 0
            cur_trial_type = 'Passive'
        
        # Add breakpoint to trial onset and offset
        cur_trial_onset = cur_trial['Trial_onset'] + breakpoint_offset_time
        cur_trial_offset = cur_trial['Trial_offset'] + breakpoint_offset_time
        
        # Use different onsets depending on trial type
        if type(trial_duration_for_fr) is dict:
            cur_trial_duration_for_fr_s = trial_duration_for_fr[cur_trial_type]
        else:
            cur_trial_duration_for_fr_s = trial_duration_for_fr

        if type(trialOnset_duration_for_fr) is dict:
            cur_trialOnset_duration_for_fr_s = trialOnset_duration_for_fr[cur_trial_type]
        else:
            cur_trialOnset_duration_for_fr_s = trialOnset_duration_for_fr

        if type(aftertrial_FR_start) is dict:
            cur_aftertrial_start = aftertrial_FR_start[cur_trial_type]
            cur_aftertrial_end = aftertrial_FR_end[cur_trial_type]
        else:
            cur_aftertrial_start = aftertrial_FR_start
            cur_aftertrial_end = aftertrial_FR_end

        if type(resptime_FR_start) is dict:
            cur_resptime_start = resptime_FR_start[cur_trial_type]
            cur_resptime_end = resptime_FR_end[cur_trial_type]
        else:
            cur_resptime_start = resptime_FR_start
            cur_resptime_end = resptime_FR_end

        if type(beforeresp_FR_start) is dict:
            cur_beforeresp_start = beforeresp_FR_start[cur_trial_type]
            cur_beforeresp_end = beforeresp_FR_end[cur_trial_type]
        else:
            cur_beforeresp_start = beforeresp_FR_start
            cur_beforeresp_end = beforeresp_FR_end

        # SPIKE TIMES
        # Get spikes in the interval [trial_onset - pre_stim_raster; trial_onset + post_stim_raster]
        spikes_around_trial = spike_times[
            (spike_times >= (cur_trial_onset - pre_stim_raster)) &
            (spike_times < (cur_trial_onset + post_stim_raster))]

        # Zero center around trial onset
        zerocentered_trial_spikes.append(spikes_around_trial - cur_trial_onset)

        # Zero center around response time
        if np.isnan(cur_resptime):
            zerocentered_response_spikes.append([None, ])
        else:
            zerocentered_response_spikes.append(
                spikes_around_trial - (cur_trial_onset + cur_resptime))

        if valid_baseline:
            nonAM_spikes = spike_times[
                (previous_cr_onset < spike_times) &
                (spike_times < previous_cr_onset + nonAM_duration_for_fr)]
        else:
            nonAM_spikes = None

        trial_spikes = spike_times[(spike_times >= cur_trial_onset) &
                                   (spike_times < (cur_trial_onset + cur_trial_duration_for_fr_s))]
        trialOnset_spikes = spike_times[(spike_times >= cur_trial_onset) &
                                        (spike_times < (cur_trial_onset + cur_trialOnset_duration_for_fr_s))]
        aftertrial_spikes = spike_times[
            (spike_times >= (cur_trial_onset + cur_aftertrial_start)) &
            (spike_times < (cur_trial_onset + cur_aftertrial_end))]

        resptime_spikes = spike_times[
            (spike_times >= (cur_trial_onset + cur_resptime + cur_resptime_start)) &
            (spike_times < (cur_trial_onset + cur_resptime + cur_resptime_end))]

        beforeresp_spikes = spike_times[
            (spike_times >= (cur_trial_onset + cur_resptime - cur_beforeresp_start)) &
            (spike_times < (cur_trial_onset + cur_resptime - cur_beforeresp_end))]

        # FR calculations
        if valid_baseline:
            cur_nonAM_FR = len(nonAM_spikes) / nonAM_duration_for_fr
        else:
            cur_nonAM_FR = None
        cur_trial_FR = len(trial_spikes) / cur_trial_duration_for_fr_s
        cur_trialOnset_FR = len(trialOnset_spikes) / cur_trialOnset_duration_for_fr_s
        cur_aftertrial_fr = len(aftertrial_spikes) / (cur_aftertrial_end - cur_aftertrial_start)
        cur_resptime_fr = len(resptime_spikes) / (cur_resptime_end - cur_resptime_start)
        cur_beforeresp_fr = len(beforeresp_spikes) / (cur_beforeresp_start - cur_beforeresp_end)

        nonAM_FR_list.append(cur_nonAM_FR)
        trial_FR_list.append(cur_trial_FR)
        trialOnset_FR_list.append(cur_trialOnset_FR)
        aftertrial_FR_list.append(cur_aftertrial_fr)
        resptime_FR_list.append(cur_resptime_fr)
        beforeresp_FR_list.append(cur_beforeresp_fr)

    # Cap floating point precision for optimized memory storage
    nonAM_FR_list = np.round(nonAM_FR_list, 4)
    trial_FR_list = np.round(trial_FR_list, 4)
    trialOnset_FR_list = np.round(trialOnset_FR_list, 4)
    aftertrial_FR_list = np.round(aftertrial_FR_list, 4)
    resptime_FR_list = np.round(resptime_FR_list, 4)
    beforeresp_FR_list = np.round(beforeresp_FR_list, 4)
    zerocentered_trial_spikes = [np.round(_spikes, 4) for _spikes in zerocentered_trial_spikes]
    zerocentered_response_spikes = [np.round(_spikes, 4) for _spikes in zerocentered_response_spikes]
    relevant_key_times.loc[:, ['Trial_onset', 'Trial_offset', 'RespLatency']] = (
        relevant_key_times.loc[:, ['Trial_onset', 'Trial_offset', 'RespLatency']].round(4)
    )
    relevant_key_times.loc[:, 'AMdepth'] = relevant_key_times['AMdepth'].round(2)

    if experiment_tag is None:
        csv_name = ''
    else:
        csv_name = experiment_tag + '_AMsound_firing_rate.csv'

    # write or append
    if first_cell_flag:
        write_or_append_flag = 'w'
    else:
        write_or_append_flag = 'a'
    with open(output_path + sep + csv_name, write_or_append_flag, newline='') as file:
        writer = csv.writer(file, delimiter=',')
        # Write header if first cell
        if write_or_append_flag == 'w':
            writer.writerow(['Unit'] + ['Key_file'] + ['TrialID'] + ['AMdepth'] + ['Reminder'] + ['ShockFlag'] +
                            ['Hit'] + ['Miss'] + ['CR'] + ['FA'] + ['Period'] + ['Trial_onset'] + ['Trial_offset'] +
                            ['JitOnset'] + ['LED_TTL'] + ['RespLatency'] + ['FR_Hz']
                            )
        for dummy_idx in range(0, len(nonAM_FR_list)):
            cur_row = relevant_key_times.iloc[dummy_idx, :]

            for (trial_period, FR_list) in  zip(
                # Trial periods
                ('Baseline', 'Trial', 'TrialOnset',
                 'Aftertrial', 'RespTime', 'BeforeResp'),

                # FR lists
                (nonAM_FR_list, trial_FR_list, trialOnset_FR_list,
                 aftertrial_FR_list, resptime_FR_list, beforeresp_FR_list)
                ):
                try:
                    writer.writerow([unit_id] + [split(REGEX_SEP, key_path_info)[-1][:-4]] +
                                [cur_row['TrialID']] + [cur_row['AMdepth']] + [cur_row['Reminder']] +
                                [cur_row['ShockFlag']] +
                                [cur_row['Hit']] + [cur_row['Miss']] +
                                [cur_row['CR']] + [cur_row['FA']] +
                                [trial_period] + [cur_row['Trial_onset']] + [cur_row['Trial_offset']] +
                                [cur_row['JitOnset']] +
                                [cur_row['LED_TTL']] +
                                [cur_row['RespLatency']] +
                                [FR_list[dummy_idx]])
                except KeyError:
                    pass

    # Add all info to unitData
    trialInfo_filename = split(REGEX_SEP, key_path_info)[-1][:-4]
    for key_name in ('TrialID', 'Reminder', 'ShockFlag', 'JitOnset', 'LED_TTL', 'Hit', 'Miss', 'CR', 'FA',
                     'Trial_onset', 'Trial_offset', 'RespLatency'):
        cur_unitData["Session"][trialInfo_filename][key_name] = relevant_key_times[key_name].values

    cur_unitData["Session"][trialInfo_filename]['AMdepth'] = relevant_key_times['AMdepth'].values
    cur_unitData["Session"][trialInfo_filename]['Trial_spikes'] = zerocentered_trial_spikes
    cur_unitData["Session"][trialInfo_filename]['Response_spikes'] = zerocentered_response_spikes
    cur_unitData["Session"][trialInfo_filename]['Baseline_FR'] = nonAM_FR_list
    cur_unitData["Session"][trialInfo_filename]['Trial_FR'] = trial_FR_list
    cur_unitData["Session"][trialInfo_filename]['TrialOnset_FR'] = trialOnset_FR_list
    cur_unitData["Session"][trialInfo_filename]['Aftertrial_FR'] = aftertrial_FR_list
    cur_unitData["Session"][trialInfo_filename]['ResponseTime_FR'] = resptime_FR_list
    cur_unitData["Session"][trialInfo_filename]['BeforeResponse_FR'] = beforeresp_FR_list

    return cur_unitData
