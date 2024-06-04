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
                    nonAM_duration_for_fr=0.5,
                    trial_duration_for_fr=0.5,
                    pre_stim_raster=2.,  # For timestamped spikeTimes
                    post_stim_raster=4.,  # For timestamped spikeTimes
                    afterTrial_FR_start=1.3,  # For calculating after-stimulus firing rate; useful for Misses
                    afterTrial_FR_end=2):
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
    afterTrial_FR_list = list()

    # Raw spike counts
    nonAM_spikeCount_list = list()
    trial_spikeCount_list = list()
    afterTrial_spikeCount_list = list()

    # Amount of spout onset timestamps around trial
    nonAM_spoutOn_frequency_list = list()
    trial_spoutOn_frequency_list = list()
    afterTrial_spoutOn_frequency_list = list()

    # Amount of spout offset timestamps around trial
    nonAM_spoutOff_frequency_list = list()
    trial_spoutOff_frequency_list = list()
    afterTrial_spoutOff_frequency_list = list()

    # Actual spike and spout timestamps around trial
    timestamped_trial_spikes = list()
    spoutOn_timestamps_list = list()
    spoutOff_timestamps_list = list()
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

        # Use different onsets depending on trial type
        if type(afterTrial_FR_start) is dict:
            cur_trial_type = ''
            if cur_trial['Hit'] == 1:
                cur_trial_type = 'Hit'
            elif cur_trial['Miss'] == 1:
                cur_trial_type = 'Miss'
            else:
                cur_trial_type = 'FA'

            cur_aftertrial_start = afterTrial_FR_start[cur_trial_type]
            cur_aftertrial_end = afterTrial_FR_end[cur_trial_type]
        else:
            cur_aftertrial_start = afterTrial_FR_start
            cur_aftertrial_end = afterTrial_FR_end

        if type(trial_duration_for_fr) is dict:
            if cur_trial['Hit'] == 1:
                cur_trial_type = 'Hit'
            elif cur_trial['Miss'] == 1:
                cur_trial_type = 'Miss'
            else:
                cur_trial_type = 'FA'

            cur_stim_duration_for_fr_s = trial_duration_for_fr[cur_trial_type]
        else:
            cur_stim_duration_for_fr_s = trial_duration_for_fr

        # Count the number of spout onsets and offsets during the current trials; could be interesting...
        # Also transform to Hz in case I decide to change the window in the future
        nonAM_spoutOn = \
            spout_key_times[
                (spout_key_times['Spout_onset'].values >
                 previous_cr['Trial_onset']) &
                (spout_key_times['Spout_onset'].values <
                 previous_cr['Trial_onset'] + nonAM_duration_for_fr)]['Spout_onset'].values
        nonAM_spoutOff = \
            spout_key_times[
                (spout_key_times['Spout_offset'].values >
                 previous_cr['Trial_onset']) &
                (spout_key_times['Spout_offset'].values <
                 previous_cr['Trial_onset'] + nonAM_duration_for_fr)]['Spout_offset'].values

        trial_spoutOn = \
            spout_key_times[
                (spout_key_times['Spout_onset'].values >
                 cur_trial['Trial_onset']) &
                (spout_key_times['Spout_onset'].values <
                 cur_trial['Trial_onset'] + cur_stim_duration_for_fr_s)]['Spout_onset'].values
        trial_spoutOff = \
            spout_key_times[
                (spout_key_times['Spout_offset'].values >
                 cur_trial['Trial_onset']) &
                (spout_key_times['Spout_offset'].values <
                 cur_trial['Trial_onset'] + cur_stim_duration_for_fr_s)]['Spout_offset'].values

        afterTrial_spoutOn = \
            spout_key_times[
                (spout_key_times['Spout_onset'].values >
                 cur_trial['Trial_onset'] + cur_aftertrial_start) &
                (spout_key_times['Spout_onset'].values <
                 cur_trial['Trial_onset'] + cur_aftertrial_end)]['Spout_onset'].values
        afterTrial_spoutOff = \
            spout_key_times[
                (spout_key_times['Spout_offset'].values >
                 cur_trial['Trial_onset'] + cur_aftertrial_start) &
                (spout_key_times['Spout_offset'].values <
                 cur_trial['Trial_onset'] + cur_aftertrial_end)]['Spout_offset'].values

        # Append to lists
        nonAM_spoutOn_frequency_list.append(len(nonAM_spoutOn) / nonAM_duration_for_fr)
        nonAM_spoutOff_frequency_list.append(len(nonAM_spoutOff) / nonAM_duration_for_fr)
        trial_spoutOn_frequency_list.append(len(trial_spoutOn) / cur_stim_duration_for_fr_s)
        trial_spoutOff_frequency_list.append(len(trial_spoutOff) / cur_stim_duration_for_fr_s)
        afterTrial_spoutOn_frequency_list.append(len(afterTrial_spoutOn) / (cur_aftertrial_end - cur_aftertrial_start))
        afterTrial_spoutOff_frequency_list.append(len(afterTrial_spoutOff) / (cur_aftertrial_end - cur_aftertrial_start))

        # Spout events around AM trial [trial_onset - baseline_duration; trial_onset + stim_duration]
        spoutOn_around_trial = \
            spout_key_times[
                (spout_key_times['Spout_onset'].values >
                 cur_trial['Trial_onset'] - pre_stim_raster) &
                (spout_key_times['Spout_onset'].values <
                 cur_trial['Trial_onset'] + post_stim_raster)]['Spout_onset'].values
        spoutOff_around_trial = \
            spout_key_times[
                (spout_key_times['Spout_offset'].values >
                 cur_trial['Trial_onset'] - pre_stim_raster) &
                (spout_key_times['Spout_offset'].values <
                 cur_trial['Trial_onset'] + post_stim_raster)]['Spout_offset'].values
        # zero-align to trial onset
        spoutOn_around_trial -= (cur_trial['Trial_onset'])
        spoutOff_around_trial -= (cur_trial['Trial_onset'])

        spoutOn_timestamps_list.append(spoutOn_around_trial)
        spoutOff_timestamps_list.append(spoutOff_around_trial)

        # Now get the spikes
        nonAM_spikes = spike_times[
            (previous_cr['Trial_onset'] + breakpoint_offset_time < spike_times) &
            (spike_times < previous_cr['Trial_onset'] + breakpoint_offset_time + nonAM_duration_for_fr)]

        trial_spikes = spike_times[(spike_times > cur_trial['Trial_onset'] + breakpoint_offset_time) &
                                   (spike_times < (cur_trial[
                                                       'Trial_onset'] + breakpoint_offset_time + cur_stim_duration_for_fr_s))]

        afterTrial_spikes = spike_times[
            (spike_times > cur_trial['Trial_onset'] + breakpoint_offset_time + cur_aftertrial_start) &
            (spike_times < (cur_trial['Trial_onset'] + breakpoint_offset_time + cur_aftertrial_end))]

        # Get spikes in the interval [trial_onset - pre_stim_raster; trial_onset + post_stim_raster]
        spikes_around_trial = spike_times[
            (spike_times >= cur_trial['Trial_onset'] + breakpoint_offset_time - pre_stim_raster) &
            (spike_times < (cur_trial['Trial_onset'] + breakpoint_offset_time + post_stim_raster))]

        # Zero center around trial onset
        spikes_around_trial -= (cur_trial['Trial_onset'] + breakpoint_offset_time)

        timestamped_trial_spikes.append(spikes_around_trial)

        # FR calculations
        cur_nonAM_FR = len(nonAM_spikes) / nonAM_duration_for_fr
        cur_trial_FR = len(trial_spikes) / cur_stim_duration_for_fr_s
        cur_afterTrial_FR = len(afterTrial_spikes) / (cur_aftertrial_end - cur_aftertrial_start)

        nonAM_FR_list.append(cur_nonAM_FR)
        trial_FR_list.append(cur_trial_FR)
        afterTrial_FR_list.append(cur_afterTrial_FR)

        # Spike counts for eventual binomial glms
        nonAM_spikeCount_list.append(len(nonAM_spikes))
        trial_spikeCount_list.append(len(trial_spikes))
        afterTrial_spikeCount_list.append(len(afterTrial_spikes))

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
                            ['FR_Hz'] + ['Spike_count'] + ['Spout_onsets_Hz'] + ['Spout_offsets_Hz']
                            )
        for dummy_idx in range(0, len(nonAM_FR_list)):
            cur_row = relevant_key_times.iloc[dummy_idx, :]

            for (trial_period, FR_list, spikeCount_list, spoutOn_frequency_list, spoutOff_frequency_list) in \
                zip(('Baseline', 'Trial', 'Aftertrial',
                     'previous_Baseline', 'previous_Trial', 'previous_Aftertrial'),
                    (nonAM_FR_list, trial_FR_list, afterTrial_FR_list),
                    (nonAM_spikeCount_list, trial_spikeCount_list, afterTrial_spikeCount_list),
                    (nonAM_spoutOn_frequency_list, trial_spoutOn_frequency_list, afterTrial_spoutOn_frequency_list,
                     np.repeat(np.NaN, len(nonAM_FR_list)), np.repeat(np.NaN, len(nonAM_FR_list)), np.repeat(np.NaN, len(nonAM_FR_list))),
                    (nonAM_spoutOff_frequency_list, trial_spoutOff_frequency_list, afterTrial_spoutOff_frequency_list,
                     np.repeat(np.NaN, len(nonAM_FR_list)), np.repeat(np.NaN, len(nonAM_FR_list)), np.repeat(np.NaN, len(nonAM_FR_list)))):

                writer.writerow([unit_name] + [split(REGEX_SEP, key_path_info)[-1][:-4]] +
                                [cur_row['TrialID']] + [round(cur_row['AMdepth'], 2)] + [cur_row['Reminder']] +
                                [cur_row['ShockFlag']] +
                                [cur_row['Hit']] + [cur_row['Miss']] +
                                [cur_row['CR']] + [cur_row['FA']] +
                                [trial_period] + [cur_row['Trial_onset']] + [cur_row['Trial_offset']] +
                                [FR_list[dummy_idx]] +
                                [spikeCount_list[dummy_idx]] +
                                [spoutOn_frequency_list[dummy_idx]] +
                                [spoutOff_frequency_list[dummy_idx]])

    # Add all info to unitData
    trialInfo_filename = split(REGEX_SEP, key_path_info)[-1][:-4]
    for key_name in ('TrialID', 'Reminder', 'ShockFlag', 'Hit', 'Miss', 'CR', 'FA', 'Trial_onset', 'Trial_offset'):
        cur_unitData["Session"][trialInfo_filename][key_name] = relevant_key_times[key_name].values

    cur_unitData["Session"][trialInfo_filename]['AMdepth'] = np.round(relevant_key_times['AMdepth'].values, 2)

    cur_unitData["Session"][trialInfo_filename]['Trial_spikes'] = timestamped_trial_spikes
    cur_unitData["Session"][trialInfo_filename]['Baseline_spikeCount'] = nonAM_spikeCount_list
    cur_unitData["Session"][trialInfo_filename]['Baseline_FR'] = nonAM_FR_list
    cur_unitData["Session"][trialInfo_filename]['Trial_spikeCount'] = trial_spikeCount_list
    cur_unitData["Session"][trialInfo_filename]['Trial_FR'] = trial_FR_list

    cur_unitData["Session"][trialInfo_filename]['Baseline_spoutOn_frequency'] = nonAM_spoutOn_frequency_list
    cur_unitData["Session"][trialInfo_filename]['Trial_spoutOn_frequency'] = trial_spoutOn_frequency_list
    cur_unitData["Session"][trialInfo_filename]['Baseline_spoutOff_frequency'] = nonAM_spoutOff_frequency_list
    cur_unitData["Session"][trialInfo_filename]['Trial_spoutOff_frequency'] = trial_spoutOff_frequency_list

    cur_unitData["Session"][trialInfo_filename]['SpoutOn_times_during_trial'] = spoutOn_timestamps_list
    cur_unitData["Session"][trialInfo_filename]['SpoutOff_times_during_trial'] = spoutOff_timestamps_list

    return cur_unitData