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


def get_fr_toTrials(memory_name,
                    key_path_info,
                    key_path_spout,
                    unit_name,
                    output_path,
                    cur_unitData,
                    experiment_tag=None,
                    first_cell_flag=True,
                    breakpoint_offset=0,
                    nonAM_duration_for_fr: dict or int=0.5,
                    trial_duration_for_fr: dict or int=0.5,
                    pre_stim_raster=2.,  # For timestamped spikeTimes
                    post_stim_raster=4.,  # For timestamped spikeTimes
                    aftertrial_FR_start: dict or int=1.3,  # For calculating after-stimulus firing rate; useful for Misses
                    aftertrial_FR_end: dict or int=2,
                    resptime_FR_start: dict or int=0.3,  # For calculating after-response firing rate
                    resptime_FR_end: dict or int=1.8,
                    beforeresp_FR_start: dict or int=0.5,  # For calculating pre-response firing rate (will be converted to negative)
                    beforeresp_FR_end: dict or int=0,
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

    # Load spout file if it exists
    try:
        spout_key_times = read_csv(key_path_spout)
    except ValueError:
        spout_key_times = {'Spout_onset': [], 'Spout_offset': []}
        spout_key_times = pd.DataFrame(spout_key_times)

    # Load spike times
    spike_times = np.genfromtxt(memory_name)

    # The default is 0, so this will only make a difference if set at function call
    breakpoint_offset_time = breakpoint_offset

    # Grab GO trials and FA trials for stim response then walk back to get the immediately preceding CR trial
    # Also ignore reminder trials here
    relevant_key_times = info_key_times[
        ((info_key_times['TrialType'] == 0) | (info_key_times['FA'] == 1)) & (info_key_times['Reminder'] == 0)]

    # Now grab spike times
    # Baseline will be the CR trial immediately preceding a GO or a FA
    nonAM_FR_list = list()
    trial_FR_list = list()
    aftertrial_FR_list = list()
    resptime_FR_list = list()
    beforeresp_FR_list = list()

    # Raw spike counts
    nonAM_spikeCount_list = list()
    trial_spikeCount_list = list()
    aftertrial_spikeCount_list = list()
    resptime_spikeCount_list = list()
    beforeresp_spikeCount_list = list()

    # Actual spike and spout timestamps around trial
    zerocentered_trial_spikes = list()
    zerocentered_response_spikes = list()

    for dummy_index, cur_trial in relevant_key_times.iterrows():
        # Get spike times around the current stimulus onset
        # For baseline, go to the previous trial that resulted in a correct rejection (NO-GO) which
        # may not be the immediately preceding trial
        try:
            previous_cr = info_key_times[(info_key_times['CR'] == 1) &
                                         (info_key_times['TrialID'] < cur_trial['TrialID'])].iloc[-1]
        except IndexError:  # In case there is no CR before a hit, skip trial
            relevant_key_times.drop(dummy_index, inplace=True)
            continue

        # Get current trial's response time (i.e., latency)
        # Only relevant for active sessions
        # IMPORTANT NOTE: In miss trials, if latency value is during the AM period, this indicates that the animal
        #   attempted to withdraw from spout before the shock, but returned for some reason. Handle these trials as you wish
        #   For true undetected misses, respLatency will be after the AM period during the shock period
        if 'Passive' not in key_path_info:
            cur_resptime = cur_trial['RespLatency']
        else:
            cur_resptime = 0

        if cur_trial['Hit'] == 1:
            cur_trial_type = 'Hit'
        elif cur_trial['Miss'] == 1:
            cur_trial_type = 'Miss'
        else:
            cur_trial_type = 'FA'

        # Use different onsets depending on trial type
        if type(trial_duration_for_fr) is dict:
            cur_stim_duration_for_fr_s = trial_duration_for_fr[cur_trial_type]
        else:
            cur_stim_duration_for_fr_s = trial_duration_for_fr

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
            (spike_times >= cur_trial['Trial_onset'] + breakpoint_offset_time - pre_stim_raster) &
            (spike_times < (cur_trial['Trial_onset'] + breakpoint_offset_time + post_stim_raster))]

        # Zero center around trial onset
        zerocentered_trial_spikes.append(spikes_around_trial - (cur_trial['Trial_onset'] + breakpoint_offset_time))

        # Zero center around response time
        if np.isnan(cur_resptime):
            zerocentered_response_spikes.append([np.nan,])
        else:
            zerocentered_response_spikes.append(spikes_around_trial - (cur_trial['Trial_onset'] + cur_resptime + breakpoint_offset_time))

        nonAM_spikes = spike_times[
            (previous_cr['Trial_onset'] + breakpoint_offset_time < spike_times) &
            (spike_times < previous_cr['Trial_onset'] + breakpoint_offset_time + nonAM_duration_for_fr)]

        trial_spikes = spike_times[(spike_times > cur_trial['Trial_onset'] + breakpoint_offset_time) &
                                   (spike_times < (cur_trial[
                                                       'Trial_onset'] + breakpoint_offset_time + cur_stim_duration_for_fr_s))]

        aftertrial_spikes = spike_times[
            (spike_times > (cur_trial['Trial_onset'] + breakpoint_offset_time + cur_aftertrial_start)) &
            (spike_times < (cur_trial['Trial_onset'] + breakpoint_offset_time + cur_aftertrial_end))]

        resptime_spikes = spike_times[
            (spike_times > (cur_trial['Trial_onset'] + breakpoint_offset_time + cur_resptime + cur_resptime_start)) &
            (spike_times < (cur_trial['Trial_onset'] + breakpoint_offset_time + cur_resptime + cur_resptime_end))]

        beforeresp_spikes = spike_times[
            (spike_times > (cur_trial['Trial_onset'] + breakpoint_offset_time + cur_resptime - cur_beforeresp_start)) &
            (spike_times < (cur_trial['Trial_onset'] + breakpoint_offset_time + cur_resptime - cur_beforeresp_end))]

        # FR calculations
        cur_nonAM_FR = len(nonAM_spikes) / nonAM_duration_for_fr
        cur_trial_FR = len(trial_spikes) / cur_stim_duration_for_fr_s
        cur_aftertrial_fr = len(aftertrial_spikes) / (cur_aftertrial_end - cur_aftertrial_start)
        cur_resptime_fr = len(resptime_spikes) / (cur_resptime_end - cur_resptime_start)
        cur_beforeresp_fr = len(beforeresp_spikes) / (cur_beforeresp_start - cur_beforeresp_end)

        nonAM_FR_list.append(cur_nonAM_FR)
        trial_FR_list.append(cur_trial_FR)
        aftertrial_FR_list.append(cur_aftertrial_fr)
        resptime_FR_list.append(cur_resptime_fr)
        beforeresp_FR_list.append(cur_beforeresp_fr)

        # Spike counts for eventual binomial glms
        nonAM_spikeCount_list.append(len(nonAM_spikes))
        trial_spikeCount_list.append(len(trial_spikes))
        aftertrial_spikeCount_list.append(len(aftertrial_spikes))
        resptime_spikeCount_list.append(len(resptime_spikes))
        beforeresp_spikeCount_list.append(len(beforeresp_spikes))

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
                            ['RespLatency'] + ['FR_Hz'] + ['Spike_count']
                            )
        for dummy_idx in range(0, len(nonAM_FR_list)):
            cur_row = relevant_key_times.iloc[dummy_idx, :]

            for (trial_period, FR_list, spikeCount_list) in \
                zip(('Baseline', 'Trial', 'Aftertrial',
                     'RespTime', 'BeforeResp'),
                    (nonAM_FR_list, trial_FR_list, aftertrial_FR_list,
                     resptime_FR_list, beforeresp_FR_list),
                    (nonAM_spikeCount_list, trial_spikeCount_list, aftertrial_spikeCount_list,
                     resptime_spikeCount_list, beforeresp_spikeCount_list)
                    ):

                writer.writerow([unit_name] + [split(REGEX_SEP, key_path_info)[-1][:-4]] +
                                [cur_row['TrialID']] + [round(cur_row['AMdepth'], 2)] + [cur_row['Reminder']] +
                                [cur_row['ShockFlag']] +
                                [cur_row['Hit']] + [cur_row['Miss']] +
                                [cur_row['CR']] + [cur_row['FA']] +
                                [trial_period] + [cur_row['Trial_onset']] + [cur_row['Trial_offset']] +
                                [cur_row['RespLatency']] +
                                [FR_list[dummy_idx]] +
                                [spikeCount_list[dummy_idx]])

    # Add all info to unitData
    trialInfo_filename = split(REGEX_SEP, key_path_info)[-1][:-4]
    for key_name in ('TrialID', 'Reminder', 'ShockFlag', 'Hit', 'Miss', 'CR', 'FA',
                     'Trial_onset', 'Trial_offset', 'RespLatency'):
        cur_unitData["Session"][trialInfo_filename][key_name] = relevant_key_times[key_name].values

    cur_unitData["Session"][trialInfo_filename]['AMdepth'] = np.round(relevant_key_times['AMdepth'].values, 2)
    cur_unitData["Session"][trialInfo_filename]['Trial_spikes'] = zerocentered_trial_spikes
    cur_unitData["Session"][trialInfo_filename]['Response_spikes'] = zerocentered_response_spikes
    cur_unitData["Session"][trialInfo_filename]['Baseline_spikeCount'] = nonAM_spikeCount_list
    cur_unitData["Session"][trialInfo_filename]['Baseline_FR'] = nonAM_FR_list
    cur_unitData["Session"][trialInfo_filename]['Trial_spikeCount'] = trial_spikeCount_list
    cur_unitData["Session"][trialInfo_filename]['Trial_FR'] = trial_FR_list
    cur_unitData["Session"][trialInfo_filename]['Aftertrial_spikeCount'] = aftertrial_spikeCount_list
    cur_unitData["Session"][trialInfo_filename]['Aftertrial_FR'] = aftertrial_FR_list
    cur_unitData["Session"][trialInfo_filename]['ResponseTime_spikeCount'] = resptime_spikeCount_list
    cur_unitData["Session"][trialInfo_filename]['ResponseTime_FR'] = resptime_FR_list
    cur_unitData["Session"][trialInfo_filename]['BeforeResponse_spikeCount'] = beforeresp_spikeCount_list
    cur_unitData["Session"][trialInfo_filename]['BeforeResponse_FR'] = beforeresp_FR_list

    return cur_unitData