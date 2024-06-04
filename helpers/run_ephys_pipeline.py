from os import remove, makedirs
import warnings
from platform import system
from os.path import sep

# Tweak the regex file separator for cross-platform compatibility
if system() == 'Windows':
    REGEX_SEP = sep * 2
else:
    REGEX_SEP = sep

from datetime import datetime
from multiprocessing import Pool, cpu_count, current_process
from re import search, split
from glob import glob

from helpers.compile_fr_result_csv import compile_fr_result_csv
from helpers.write_json import write_json
from helpers.preprocess_files import preprocess_files, find_spoutfile_and_breakpoint

from firing_rate_analysis.get_fr_toTrials import get_fr_toTrials
from auROC_analysis.calculate_auROC import *

def run_pipeline(input_list):
    # Gather settings
    _, _, SETTINGS_DICT = input_list
    pipeline_switchboard = SETTINGS_DICT['PIPELINE_SWITCHBOARD']
    output_path = SETTINGS_DICT['OUTPUT_PATH']

    experiment_tag = SETTINGS_DICT['EXPERIMENT_TAG']
    nonAM_duration_for_fr = SETTINGS_DICT['NONAM_DURATION_FOR_FR']
    trial_duration_for_fr = SETTINGS_DICT['TRIAL_DURATION_FOR_FR']
    pretrial_duration_for_spiketimes = SETTINGS_DICT['PRETRIAL_DURATION_FOR_SPIKETIMES']
    posttrial_duration_for_spiketimes = SETTINGS_DICT['POSTTRIAL_DURATION_FOR_SPIKETIMES']
    afterTrial_FR_start = SETTINGS_DICT['AFTERTRIAL_FR_START']
    afterTrial_FR_end = SETTINGS_DICT['AFTERTRIAL_FR_END']

    memory_path, key_paths_info, key_paths_spout, cur_unitData, cur_breakpoint_df = (
        preprocess_files(input_list))

    subject_id = cur_unitData['Subject']
    unit_id = cur_unitData['Unit']
    sampling_rate = cur_unitData['Sampling_rate']
    recording_type = cur_unitData['Recording_type']
    for key_path_info in key_paths_info:
        key_path_spout, breakpoint_offset, key_finder = find_spoutfile_and_breakpoint(subject_id,
                                                                          key_path_info, key_paths_spout,
                                                                          cur_breakpoint_df, recording_type,
                                                                          sampling_rate)
        # Add keys to JSON structure if they don't already exist
        try:
            if split(REGEX_SEP, key_path_info)[-1][:-4] not in cur_unitData["Session"]:
                cur_unitData["Session"].update({split(REGEX_SEP, key_path_info)[-1][:-4]: {}})
        except KeyError:
            cur_unitData["Session"].update({split(REGEX_SEP, key_path_info)[-1][:-4]: {}})

        # Flag to indicate this is the first entry to the CSV file so headers will be printed
        # Check if worker file already exists then turn flag to false
        if len(glob(output_path + sep + current_process().name + "_tempfile_" + experiment_tag + '*.csv')) > 0:
            first_entry_flag = False
        else:
            first_entry_flag = True

        if pipeline_switchboard['firing_rate_to_trials']:
            cur_unitData = get_fr_toTrials(memory_path, key_path_info, key_path_spout, unit_name=unit_id,
                                           output_path=output_path, cur_unitData=cur_unitData,
                                           experiment_tag=current_process().name + "_tempfile_" + experiment_tag,
                                           first_cell_flag=first_entry_flag, breakpoint_offset=breakpoint_offset,
                                           nonAM_duration_for_fr=nonAM_duration_for_fr,
                                           trial_duration_for_fr=trial_duration_for_fr,
                                           pre_stim_raster=pretrial_duration_for_spiketimes,
                                           post_stim_raster=posttrial_duration_for_spiketimes,
                                           afterTrial_FR_start=afterTrial_FR_start, afterTrial_FR_end=afterTrial_FR_end)
            write_json(cur_unitData, output_path)

        '''
        THE FOLLOWING BLOCK OF FUNCTIONS IS ONLY RELEVANT FOR ACTIVE SESSIONS
        '''
        if 'Passive' not in key_finder:
            if pipeline_switchboard['auROC_spoutOffHit']:
                cur_unitData = calculate_auROC_spoutOffHit(cur_unitData=cur_unitData,
                                                           session_name=split(REGEX_SEP, key_path_info)[-1][:-4],
                                                           pre_stimulus_baseline_start=pretrial_duration_for_spiketimes,
                                                           pre_stimulus_baseline_end=pretrial_duration_for_spiketimes - 1,
                                                           pre_stimulus_raster=pretrial_duration_for_spiketimes,
                                                           post_stimulus_raster=posttrial_duration_for_spiketimes
                                                           )
                write_json(cur_unitData, output_path)

            if pipeline_switchboard['auROC_spoutOffFA']:
                cur_unitData = calculate_auROC_spoutOffFA(cur_unitData=cur_unitData,
                                                          session_name=split(REGEX_SEP, key_path_info)[-1][:-4],
                                                          pre_stimulus_baseline_start=pretrial_duration_for_spiketimes,
                                                          pre_stimulus_baseline_end=pretrial_duration_for_spiketimes - 1,
                                                          pre_stimulus_raster=pretrial_duration_for_spiketimes,
                                                          post_stimulus_raster=posttrial_duration_for_spiketimes
                                                          )
                write_json(cur_unitData, output_path)

            if pipeline_switchboard['auROC_spoutOffMiss']:
                cur_unitData = calculate_auROC_spoutOffMiss(cur_unitData=cur_unitData,
                                                            session_name=split(REGEX_SEP, key_path_info)[-1][:-4],
                                                            pre_stimulus_baseline_start=pretrial_duration_for_spiketimes,
                                                            pre_stimulus_baseline_end=pretrial_duration_for_spiketimes - 1,
                                                            pre_stimulus_raster=pretrial_duration_for_spiketimes,
                                                            post_stimulus_raster=posttrial_duration_for_spiketimes
                                                            )
                write_json(cur_unitData, output_path)

            if pipeline_switchboard['auROC_hit']:
                cur_unitData = calculate_auROC_hit(cur_unitData=cur_unitData,
                                                   session_name=split(REGEX_SEP, key_path_info)[-1][:-4],
                                                   pre_stimulus_baseline_start=pretrial_duration_for_spiketimes,
                                                   pre_stimulus_baseline_end=pretrial_duration_for_spiketimes - 1,
                                                   pre_stimulus_raster=pretrial_duration_for_spiketimes,
                                                   post_stimulus_raster=posttrial_duration_for_spiketimes
                                                   )
                write_json(cur_unitData, output_path)

            if pipeline_switchboard['auROC_hitByShock']:
                # For fair comparison with shocked misses
                cur_unitData = calculate_auROC_hitByShock(cur_unitData=cur_unitData,
                                                          session_name=split(REGEX_SEP, key_path_info)[-1][:-4],
                                                          pre_stimulus_baseline_start=pretrial_duration_for_spiketimes,
                                                          pre_stimulus_baseline_end=pretrial_duration_for_spiketimes - 1,
                                                          pre_stimulus_raster=pretrial_duration_for_spiketimes,
                                                          post_stimulus_raster=posttrial_duration_for_spiketimes
                                                          )
                write_json(cur_unitData, output_path)
            if pipeline_switchboard['auROC_FA']:
                cur_unitData = calculate_auROC_FA(cur_unitData=cur_unitData,
                                                  session_name=split(REGEX_SEP, key_path_info)[-1][:-4],
                                                  pre_stimulus_baseline_start=pretrial_duration_for_spiketimes,
                                                  pre_stimulus_baseline_end=pretrial_duration_for_spiketimes - 1,
                                                  pre_stimulus_raster=pretrial_duration_for_spiketimes,
                                                  post_stimulus_raster=posttrial_duration_for_spiketimes
                                                  )
                write_json(cur_unitData, output_path)
            if pipeline_switchboard['auROC_missByShock']:
                cur_unitData = calculate_auROC_missByShock(cur_unitData=cur_unitData,
                                                           session_name=split(REGEX_SEP, key_path_info)[-1][:-4],
                                                           pre_stimulus_baseline_start=pretrial_duration_for_spiketimes,
                                                           pre_stimulus_baseline_end=pretrial_duration_for_spiketimes - 1,
                                                           pre_stimulus_raster=pretrial_duration_for_spiketimes,
                                                           post_stimulus_raster=posttrial_duration_for_spiketimes
                                                           )
                write_json(cur_unitData, output_path)
            if pipeline_switchboard['auROC_miss']:
                cur_unitData = calculate_auROC_miss(cur_unitData=cur_unitData,
                                                    session_name=split(REGEX_SEP, key_path_info)[-1][:-4],
                                                    pre_stimulus_baseline_start=pretrial_duration_for_spiketimes,
                                                    pre_stimulus_baseline_end=pretrial_duration_for_spiketimes - 1,
                                                    pre_stimulus_raster=pretrial_duration_for_spiketimes,
                                                    post_stimulus_raster=posttrial_duration_for_spiketimes
                                                    )
                write_json(cur_unitData, output_path)

        '''
        COMPUTATIONS RELEVANT TO BOTH PASSIVE AND ACTIVE SESSIONS
        '''
        if pipeline_switchboard['auROC_AMTrial']:
            cur_unitData = calculate_auROC_AMTrial(cur_unitData=cur_unitData,
                                                   session_name=split(REGEX_SEP, key_path_info)[-1][:-4],
                                                   pre_stimulus_baseline_start=pretrial_duration_for_spiketimes,
                                                   pre_stimulus_baseline_end=pretrial_duration_for_spiketimes - 1,
                                                   pre_stimulus_raster=pretrial_duration_for_spiketimes,
                                                   post_stimulus_raster=posttrial_duration_for_spiketimes
                                                   )
            write_json(cur_unitData, output_path)

        if pipeline_switchboard['auROC_AMdepthByAMdepth']:
            cur_unitData = calculate_auROC_AMdepthByAMdepth(cur_unitData=cur_unitData,
                                                            session_name=split(REGEX_SEP, key_path_info)[-1][:-4],
                                                            pre_stimulus_baseline_start=pretrial_duration_for_spiketimes,
                                                            pre_stimulus_baseline_end=pretrial_duration_for_spiketimes - 1,
                                                            pre_stimulus_raster=pretrial_duration_for_spiketimes,
                                                            post_stimulus_raster=posttrial_duration_for_spiketimes
                                                            )
            write_json(cur_unitData, output_path)
