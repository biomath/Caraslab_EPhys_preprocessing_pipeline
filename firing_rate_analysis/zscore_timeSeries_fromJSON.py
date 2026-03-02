from os import remove, makedirs
from os.path import sep
from re import split
import platform
from time import time
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rcParams
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib

import csv
from helpers.write_json import write_json

# Tweak the regex file separator for cross-platform compatibility
if platform.system() == 'Windows':
    REGEX_SEP = sep * 2
else:
    REGEX_SEP = sep

def tic():
    return time()

def __get_trialID_timeSeries(baseline_spikes, all_spikes, bin_size, signal_start_end, baseline_start_end, zscore_or_not):
    trial_count = 1
    if any([isinstance(i, np.ndarray) for i in baseline_spikes]):  # If baseline_spikes is a list of arrays, gather trial_count and flatten them
        trial_count = len(baseline_spikes)
        baseline_spikes = sorted([item for sublist in baseline_spikes for item in sublist])

    bin_cuts = np.arange(baseline_start_end[0], baseline_start_end[1] + bin_size, bin_size)
    baseline_raster, _ = np.histogram(baseline_spikes, bins=bin_cuts)
    baseline_raster = baseline_raster / trial_count # Only matters for global_baseline=True

    bin_cuts = np.arange(signal_start_end[0], signal_start_end[1] + bin_size, bin_size)
    binned_raster, _ = np.histogram(all_spikes, bins=bin_cuts)

    # z-score it
    if zscore_or_not:
        baseline_mean = np.nanmean(baseline_raster)
        baseline_std = np.nanstd(baseline_raster, ddof=1)

        if baseline_std == 0:
            binned_raster = np.full(np.size(binned_raster), np.nan)
        else:
            binned_raster = (binned_raster - baseline_mean) / baseline_std

    return binned_raster

def output_timeSeries_to_csv(data_list, output_path, do_zscore, global_baseline):
    # Output CSV
    _columns_prefix = 'TP.'
    for zscore_or_not in do_zscore:
        first_run_flag = True
        file_name = 'FR_timeSeries_data'
        file_name += '_zscore' if zscore_or_not else '_FR'
        with open(sep.join([output_path, file_name + '.csv']), 'w', newline='', encoding='utf-8') as file:
            for data_dict in data_list:
                for session in data_dict['Session'].keys():
                    cur_session_data = data_dict['Session'][session]

                    for t_or_r_align in ('trial_aligned', 'response_aligned'):
                        if t_or_r_align == 'trial_aligned':
                            column_name = 'Trial_timeSeries'
                        else:
                            column_name = 'Response_timeSeries'

                        column_name += '_FR' if not zscore_or_not else '_zscore'

                        column_name += '_globalBaseline' if global_baseline else ''

                        cur_sigs = np.round(cur_session_data[column_name], 4)  # No need to go crazy with floating point precision

                        hitFlag_list = cur_session_data['Hit']
                        missFlag_list = cur_session_data['Miss']
                        FAFlag_list = cur_session_data['FA']
                        shockFlag_list = cur_session_data['ShockFlag']
                        amdepth_list = np.round(cur_session_data['AMdepth'], 2)
                        trialID_list = cur_session_data['TrialID']
                        trialOnset_list = np.round(cur_session_data['Trial_onset'], 4)
                        respLatency_list = np.round(cur_session_data['RespLatency'], 4)

                        writer = csv.writer(file, delimiter=',')
                        if first_run_flag:
                            csv_header = ['Unit', 'Session', 'Alignment', 'Hit', 'Miss', 'FA', 'ShockFlag',
                                          'AMDepth', 'TrialID', 'Trial_onset', 'RespLatency']
                            csv_header.extend([_columns_prefix + str(dummy_idx + 1) for dummy_idx in range(len(cur_sigs[1]))])

                            writer.writerow(csv_header)
                            first_run_flag = False

                        # print('Adding ' + data_dict['Unit'] + ' ' + session)
                        for trial_idx in range(len(trialID_list)):
                            csv_row = [data_dict['Unit'], session, t_or_r_align, hitFlag_list[trial_idx],
                                       missFlag_list[trial_idx], FAFlag_list[trial_idx], shockFlag_list[trial_idx],
                                       amdepth_list[trial_idx], trialID_list[trial_idx], trialOnset_list[trial_idx],
                                       respLatency_list[trial_idx], *cur_sigs[trial_idx]]

                            writer.writerow(csv_row)

        # Output txt with info
        file_name = 'Zscore_timeSeries_info'
        with open(sep.join([output_path, file_name + '.txt']), 'w') as file:
            file.write('Zscore bin size: ' + str(data_list[0]['Zscore_timeSeries_bin_size']) + '\n')
            file.write('Signal start/end: ' + str(data_list[0]['Zscore_timeSeries_start_end']) + '\n')
            file.write('Baseline start/end: ' + str(data_list[0]['Zscore_timeSeries_baseline_start_end']) + '\n')


def extract_fr_timeSeries_fromJSON(input_list):
    data_dict, SETTINGS_DICT = input_list

    output_path = SETTINGS_DICT['OUTPUT_PATH']

    bin_size = SETTINGS_DICT['ZSCORE_BIN_SIZE']
    signal_start_end = SETTINGS_DICT['ZSCORE_START_END']
    baseline_start_end = SETTINGS_DICT['ZSCORE_BASELINE_START_END']
    use_nonAM_baseline = SETTINGS_DICT['ZSCORE_USE_NONAM_BASELINE']
    global_baseline = SETTINGS_DICT['ZSCORE_GLOBAL_BASELINE']

    if type(SETTINGS_DICT['ZSCORE_DO_ZSCORE']) == str:
        do_zscore = [SETTINGS_DICT['ZSCORE_DO_ZSCORE'], ]
    else:  # must be list
        assert type(SETTINGS_DICT['ZSCORE_DO_ZSCORE']) == list, \
            'ZSCORE_DO_ZSCORE must be a string or a list'
        do_zscore = SETTINGS_DICT['ZSCORE_DO_ZSCORE']

    if type(SETTINGS_DICT['ZSCORE_TRIAL_OR_RESPONSE_ALIGNED']) == str:
        trial_or_response_aligned = [SETTINGS_DICT['ZSCORE_TRIAL_OR_RESPONSE_ALIGNED'], ]
    else:  # must be list
        assert type(SETTINGS_DICT['ZSCORE_TRIAL_OR_RESPONSE_ALIGNED']) == list, \
            'ZSCORE_TRIAL_OR_RESPONSE_ALIGNED must be a string or a list'
        trial_or_response_aligned = SETTINGS_DICT['ZSCORE_TRIAL_OR_RESPONSE_ALIGNED']

    for session in data_dict['Session'].keys():
        cur_session_data = data_dict['Session'][session]
        for t_or_r_align in trial_or_response_aligned:
            for zscore_or_not in do_zscore:
                cur_session_rasters = list()
                if t_or_r_align == 'trial_aligned':
                    cur_session_spikes = cur_session_data['Trial_spikes']
                    output_column_name = 'Trial_timeSeries'
                else:
                    cur_session_spikes = cur_session_data['Response_spikes']
                    output_column_name = 'Response_timeSeries'

                if zscore_or_not:
                    output_column_name += '_zscore'
                else:
                    output_column_name += '_FR'

                if len(cur_session_spikes) == 0:
                    print('Spikes were not found for session ' + session + ' in JSON file for' + + t_or_r_align + ' data. Skipping.' )
                    continue
                if global_baseline:
                    output_column_name += '_globalBaseline'
                    baseline_spikes = []
                    # Cycle through trials one time first to get all baseline
                    for trial_n in range(0, len(cur_session_data['TrialID'])):
                        if use_nonAM_baseline and (t_or_r_align == 'response_aligned'):
                            cur_trial_baseline = np.array(cur_session_data['Trial_spikes'][trial_n])
                            cur_trial_baseline = cur_trial_baseline[(cur_trial_baseline >= baseline_start_end[0]) &
                                                              (cur_trial_baseline < baseline_start_end[1])]
                            baseline_spikes.append(cur_trial_baseline)
                        else:
                            cur_trial_baseline = np.array(cur_session_data['Response_spikes'][trial_n])
                            cur_trial_baseline = cur_trial_baseline[(cur_trial_baseline >= baseline_start_end[0]) &
                                                              (cur_trial_baseline < baseline_start_end[1])]
                            baseline_spikes.append(cur_trial_baseline)

                    for trial_n in range(0, len(cur_session_data['TrialID'])):
                        cur_spikes = np.array(cur_session_spikes[trial_n])
                        raster = __get_trialID_timeSeries(baseline_spikes, cur_spikes, bin_size, signal_start_end,
                                                          baseline_start_end, zscore_or_not=zscore_or_not)

                        cur_session_rasters.append(raster)
                else:
                    for trial_n in range(0, len(cur_session_data['TrialID'])):
                        cur_spikes = np.array(cur_session_spikes[trial_n])
                        if use_nonAM_baseline:
                            baseline_spikes = np.array(cur_session_data['Trial_spikes'][trial_n])
                            baseline_spikes = baseline_spikes[(baseline_spikes >= baseline_start_end[0]) & (baseline_spikes < baseline_start_end[1])]
                        else:
                            baseline_spikes = cur_spikes[(cur_spikes >= baseline_start_end[0]) & (cur_spikes < baseline_start_end[1])]

                        raster = __get_trialID_timeSeries(baseline_spikes, cur_spikes, bin_size, signal_start_end,
                                                                 baseline_start_end, zscore_or_not=zscore_or_not)

                        cur_session_rasters.append(raster)


                data_dict['Session'][session][output_column_name] = cur_session_rasters

    # Info params
    data_dict['Zscore_timeSeries_bin_size'] = bin_size
    data_dict['Zscore_timeSeries_start_end'] = signal_start_end
    data_dict['Zscore_timeSeries_baseline_start_end'] = baseline_start_end

    # Update JSON
    write_json(data_dict, output_path + sep + 'JSON files', data_dict['Unit'] + '_unitData.json')