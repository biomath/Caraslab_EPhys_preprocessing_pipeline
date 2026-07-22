from platform import system
from os.path import sep

# Tweak the regex file separator for cross-platform compatibility
if system() == 'Windows':
    REGEX_SEP = sep * 2
else:
    REGEX_SEP = sep

from multiprocessing import current_process, Pool
from os import remove, makedirs
from re import split
from glob import glob

from helpers.write_json import write_json
from helpers.preprocess_files import match_spike_times_with_keys, process_metadata
from helpers.compile_fr_result_csv import compile_fr_result_csv

from firing_rate_analysis.get_fr_toTrials import get_fr_toTrials
from firing_rate_analysis.get_fr_toOpto import get_fr_toOpto
from auROC_analysis.calculate_auROC import *


def run_pipeline(input_list):
    """Run the full per-unit firing-rate pipeline for every session of one unit.

    Designed to be mapped over units (e.g. via multiprocessing), where each
    worker process appends its results to its own ``*_tempfile_*`` CSV
    (named after the process) so concurrent workers never write to the same
    file; these are later stitched together by
    ``helpers.compile_fr_result_csv.compile_fr_result_csv``.

    For each session found for this unit: locates the matching key/TTL files
    and breakpoint offset, computes trial-aligned firing rates via
    ``get_fr_toTrials``, and — if an opto TTL file exists — also computes
    opto-aligned firing rates via ``get_fr_toOpto``. Results are accumulated
    into ``cur_unitData`` and written out to a per-unit JSON file after each step.

    Args:
        input_list (tuple): ``(memory_path, all_json, SETTINGS_DICT)`` — see
            ``helpers.preprocess_files.match_spike_times_with_keys``, which
            this passes through as-is. ``SETTINGS_DICT`` supplies output
            paths, FR calculation window sizes, and the experiment tag used
            to name temp files.

    Returns:
        None. Writes/updates ``<unit_id>_unitData.json`` under
        ``<output_path>/JSON files`` and appends rows to this worker's temp CSV.
    """
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

    for key_path_info in key_paths_info:
        key_path_spoutTTL, key_path_optoTTL, breakpoint_offset, key_finder = process_metadata(subject_id,
                                                                                              key_path_info=key_path_info, key_paths_spoutTTL=key_paths_spoutTTL,
                                                                                              key_paths_optoTTL=key_paths_optoTTL,
                                                                                              cur_breakpoint_df=cur_breakpoint_df,
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


def run_full_pipeline(SETTINGS_DICT):
    """Discover units, match them to key files, and dispatch ``run_pipeline`` over all of them.

    This is the notebook-level driver for the main FR-calculation stage: it
    creates the JSON output directory, clears stale per-worker temp CSVs
    from a previous run, globs every spike-times file under
    ``SETTINGS_DICT['SPIKES_PATH']`` (filtered by ``SESSIONS_TO_RUN``/
    ``SESSIONS_TO_EXCLUDE``), runs ``run_pipeline`` over each — either
    directly or via a multiprocessing pool per ``SETTINGS_DICT['MULTIPROCESS']``
    — and finally stitches every worker's temp CSV into the master firing-rate
    CSV via ``compile_fr_result_csv``.

    Args:
        SETTINGS_DICT (dict): Pipeline settings. Uses ``OUTPUT_PATH``,
            ``SPIKES_PATH``, ``KEYS_PATH``, ``SESSIONS_TO_RUN``,
            ``SESSIONS_TO_EXCLUDE``, ``MULTIPROCESS``, ``NUMBER_OF_CORES``,
            ``EXPERIMENT_TAG``, and ``OVERWRITE_PREVIOUS_CSV``.

    Returns:
        None. Writes/updates one JSON per unit and the compiled
        ``<EXPERIMENT_TAG>_AMsound_firing_rate.csv`` master CSV.
    """
    makedirs(SETTINGS_DICT['OUTPUT_PATH'] + sep + 'JSON files', exist_ok=True)

    # Load existing JSONs; will be empty if this is the first time running
    json_filenames = glob(SETTINGS_DICT['OUTPUT_PATH'] + sep + 'JSON files' + sep + '*json')

    # Clear older temp files if they exist
    process_tempfiles = glob(SETTINGS_DICT['OUTPUT_PATH'] + sep + '*_tempfile_*.csv')
    [remove(f) for f in process_tempfiles]

    # Generate a list of inputs to be passed to each worker
    input_lists = list()
    memory_paths = glob(SETTINGS_DICT['SPIKES_PATH'] + sep + '*cluster*.txt')
    keys_paths = glob(SETTINGS_DICT['KEYS_PATH'] + sep + '*trialInfo.csv')

    for dummy_idx, memory_path in enumerate(memory_paths):
        if SETTINGS_DICT['SESSIONS_TO_RUN'] is not None:
            if any([chosen for chosen in SETTINGS_DICT['SESSIONS_TO_RUN'] if chosen in memory_path]):
                pass
            else:
                continue

        if SETTINGS_DICT['SESSIONS_TO_EXCLUDE'] is not None:
            if any([chosen for chosen in SETTINGS_DICT['SESSIONS_TO_EXCLUDE'] if chosen in memory_path]):
                continue
            else:
                pass

        if SETTINGS_DICT['MULTIPROCESS']:
            input_lists.append((memory_path, json_filenames, SETTINGS_DICT))
        else:
            run_pipeline((memory_path, json_filenames, SETTINGS_DICT))

    if SETTINGS_DICT['MULTIPROCESS']:
        pool = Pool(SETTINGS_DICT['NUMBER_OF_CORES'])

        # Feed each worker with all memory paths from one unit
        pool_map_result = pool.map_async(run_pipeline, input_lists)

        pool.close()

        pool.join()

    compile_fr_result_csv(SETTINGS_DICT['EXPERIMENT_TAG'] + '_AMsound_firing_rate.csv',
                          SETTINGS_DICT['OUTPUT_PATH'], SETTINGS_DICT['OVERWRITE_PREVIOUS_CSV'])