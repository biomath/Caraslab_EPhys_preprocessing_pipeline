from platform import system
from os.path import sep

# Tweak the regex file separator for cross-platform compatibility
if system() == 'Windows':
    REGEX_SEP = sep * 2
else:
    REGEX_SEP = sep

from multiprocessing import current_process
from re import split
from glob import glob

from helpers.write_json import write_json
from helpers.preprocess_files import match_spike_times_with_keys, process_metadata

from firing_rate_analysis.get_fr_toTrials import get_fr_toTrials
from firing_rate_analysis.get_fr_toOpto import get_fr_toOpto
from auROC_analysis.calculate_auROC import *


def run_pipeline(input_list):
    # Gather settings
    _, _, SETTINGS_DICT = input_list
    output_path = SETTINGS_DICT['OUTPUT_PATH']

    experiment_tag = SETTINGS_DICT['EXPERIMENT_TAG']

    # FR calculation windows
    nonAM_duration_for_fr = SETTINGS_DICT['NONAM_DURATION_FOR_FR']
    trial_duration_for_fr = SETTINGS_DICT['TRIAL_DURATION_FOR_FR']
    trialOnset_duration_for_fr = SETTINGS_DICT['TRIALONSET_DURATION_FOR_FR']
    aftertrial_FR_start = SETTINGS_DICT['AFTERTRIAL_FR_START']
    aftertrial_FR_end = SETTINGS_DICT['AFTERTRIAL_FR_END']
    resptime_FR_start = SETTINGS_DICT['RESPTIME_FR_START']
    resptime_FR_end = SETTINGS_DICT['RESPTIME_FR_END']
    beforeresp_FR_start = SETTINGS_DICT['BEFORERESP_FR_START']
    beforeresp_FR_end = SETTINGS_DICT['BEFORERESP_FR_END']

    pretrial_duration_for_spiketimes = SETTINGS_DICT['PRETRIAL_DURATION_FOR_SPIKETIMES']
    posttrial_duration_for_spiketimes = SETTINGS_DICT['POSTTRIAL_DURATION_FOR_SPIKETIMES']

    memory_path, key_paths_info, key_paths_spoutTTL, key_paths_optoTTL, cur_unitData, cur_breakpoint_df = (
        match_spike_times_with_keys(input_list))

    subject_id = cur_unitData['Subject']
    unit_id = cur_unitData['Unit']
    sampling_rate = cur_unitData['Sampling_rate']
    recording_type = cur_unitData['Recording_type']
    key_finder_index_dict = SETTINGS_DICT['KEY_FINDER_INDEX']

    for key_path_info in key_paths_info:
        key_path_spoutTTL, key_path_optoTTL, breakpoint_offset, key_finder = process_metadata(subject_id,
                                                                                              key_path_info=key_path_info, key_paths_spoutTTL=key_paths_spoutTTL,
                                                                                              key_paths_optoTTL=key_paths_optoTTL, key_finder_index_dict=key_finder_index_dict,
                                                                                              cur_breakpoint_df=cur_breakpoint_df, recording_type=recording_type,
                                                                                              sampling_rate=sampling_rate)

        # Add keys to JSON structure if they don't already exist
        cur_unitData["Session"].update({split(REGEX_SEP, key_path_info)[-1][:-4]: {}})

        # Flag to indicate this is the first entry to the CSV file so headers will be printed
        # Check if worker file already exists then turn flag to false
        if len(glob(output_path + sep + current_process().name + "_tempfile_" + experiment_tag + '*.csv')) > 0:
            first_entry_flag = False
        else:
            first_entry_flag = True

        # Add all info to unitData
        # Start with settings parameters
        if SETTINGS_DICT is not None:
            cur_unitData['Pipeline_settings'] = SETTINGS_DICT

        cur_unitData = get_fr_toTrials(memory_path=memory_path, key_path_info=key_path_info, unit_id=unit_id,
                                       output_path=output_path, cur_unitData=cur_unitData,
                                       experiment_tag=current_process().name + "_tempfile_" + experiment_tag,
                                       first_cell_flag=first_entry_flag, breakpoint_offset=breakpoint_offset,
                                       nonAM_duration_for_fr=nonAM_duration_for_fr,
                                       trial_duration_for_fr=trial_duration_for_fr,
                                       trialOnset_duration_for_fr=trialOnset_duration_for_fr,
                                       pre_stim_raster=pretrial_duration_for_spiketimes,
                                       post_stim_raster=posttrial_duration_for_spiketimes,
                                       aftertrial_FR_start=aftertrial_FR_start, aftertrial_FR_end=aftertrial_FR_end,
                                       resptime_FR_start=resptime_FR_start, resptime_FR_end=resptime_FR_end,
                                       beforeresp_FR_start=beforeresp_FR_start, beforeresp_FR_end=beforeresp_FR_end)
        write_json(cur_unitData, output_path + sep + 'JSON files', cur_unitData['Unit'] + '_unitData.json')

        if key_path_optoTTL is not None:
            cur_unitData = get_fr_toOpto(memory_path=memory_path,
                                         key_path_info=key_path_info,
                                         key_path_optoTTL=key_path_optoTTL,
                                         unit_id=unit_id,
                                         output_path=output_path,
                                         cur_unitData=cur_unitData,
                                         experiment_tag=current_process().name + "_tempfile_" + experiment_tag,
                                         first_cell_flag=first_entry_flag, breakpoint_offset=breakpoint_offset,
                                         baseline_duration_for_fr=0.5,
                                         resp_duration_for_fr=0.5,
                                         pre_stim_raster=1.,
                                         post_stim_raster=1.
                                         )
            write_json(cur_unitData, output_path + sep + 'JSON files', cur_unitData['Unit'] + '_unitData.json')