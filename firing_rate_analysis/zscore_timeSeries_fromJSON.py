from os import remove, makedirs
from os.path import sep
from re import split
import platform
from time import time
from itertools import product
from multiprocessing import Pool
import numpy as np
from astropy.convolution import convolve_fft, Gaussian1DKernel

import csv
from helpers.write_json import write_json
from helpers import get_JSON_data

# Tweak the regex file separator for cross-platform compatibility
if platform.system() == 'Windows':
    REGEX_SEP = sep * 2
else:
    REGEX_SEP = sep

def tic():
    """Return the current time (seconds); pair with a second ``tic()`` call to time a block."""
    return time()

def __get_trialID_timeSeries(baseline_spikes, all_spikes, bin_size, signal_start_end, baseline_start_end, zscore_or_not,
                             baseline_bin_cuts=None, signal_bin_cuts=None, gaussian_width=None):
    """Bin one trial's spikes into a firing-rate/z-score time series.

    Histograms ``all_spikes`` into ``bin_size``-wide bins spanning
    ``signal_start_end``. If ``zscore_or_not``, the bins are z-scored against
    a baseline distribution built by histogramming ``baseline_spikes`` (which
    may be per-trial baselines, for a trial-by-trial baseline, or a single
    global baseline) over ``baseline_start_end``. Optionally smooths the
    result with a Gaussian kernel.

    Args:
        baseline_spikes (Sequence[np.ndarray]): One or more spike-time arrays
            used to estimate the baseline mean/std for z-scoring.
        all_spikes (np.ndarray): Spike timestamps for the trial being binned.
        bin_size (float): Histogram bin width, same units as the spike times.
        signal_start_end (tuple[float, float]): (start, end) window to bin ``all_spikes`` over.
        baseline_start_end (tuple[float, float]): (start, end) window used for the baseline bins.
        zscore_or_not (bool): If True, z-score the binned raster against the
            baseline; if False, return raw binned counts.
        baseline_bin_cuts (np.ndarray, optional): Precomputed baseline bin
            edges; computed from ``baseline_start_end``/``bin_size`` if omitted.
        signal_bin_cuts (np.ndarray, optional): Precomputed signal bin edges;
            computed from ``signal_start_end``/``bin_size`` if omitted.
        gaussian_width (float, optional): Std-dev (in the same time units) of
            a Gaussian kernel to convolve the result with; skipped if None or <= 0.

    Returns:
        np.ndarray: Binned (optionally z-scored and smoothed) firing-rate
        time series for this trial. All-NaN if the baseline std is zero
        (z-scoring only).
    """
    if signal_bin_cuts is None:
        signal_bin_cuts = np.arange(signal_start_end[0], signal_start_end[1] + bin_size, bin_size)
    binned_raster, _ = np.histogram(all_spikes, bins=signal_bin_cuts)

    # z-score it
    if zscore_or_not:
        if baseline_bin_cuts is None:
            baseline_bin_cuts = np.arange(baseline_start_end[0], baseline_start_end[1] + bin_size, bin_size)

        per_trial_rasters = np.array([
            np.histogram(trial_spikes, bins=baseline_bin_cuts)[0]
            for trial_spikes in baseline_spikes
        ])  # shape: (n_trials, n_bins)

        baseline_mean = np.nanmean(per_trial_rasters)
        baseline_std = np.nanstd(per_trial_rasters, ddof=1)

        if baseline_std == 0:
            binned_raster = np.full(np.size(binned_raster), np.nan)
        else:
            binned_raster = (binned_raster - baseline_mean) / baseline_std

    if gaussian_width is not None and gaussian_width > 0:
        sigma_bins = gaussian_width / bin_size
        binned_raster = convolve_fft(binned_raster, Gaussian1DKernel(sigma_bins))

    return binned_raster

def extract_fr_timeSeries_fromJSON(input_list):
    """Compute per-trial firing-rate/z-score time series for one unit's JSON file.

    Loads a unit's JSON data, and for every session and every trial builds a
    binned time series (via ``__get_trialID_timeSeries``) aligned either to
    trial onset or to response, optionally z-scored against a baseline
    window. The baseline can be per-trial (default), drawn from the
    non-AM/trial period even when response-aligned (``use_nonAM_baseline``),
    or pooled across all trials in the session (``global_baseline``). Results
    are stored back onto ``data_dict`` and also returned as flat CSV rows.

    Args:
        input_list (tuple): ``(file_name, SETTINGS_DICT, zscore_or_not,
            t_or_r_align, use_nonAM_baseline, global_baseline)`` where
            ``file_name`` is the unit JSON path, ``SETTINGS_DICT`` supplies
            ``ZSCORE_BIN_SIZE``/``ZSCORE_START_END``/
            ``ZSCORE_BASELINE_START_END``/``ZSCORE_GAUSSIAN_KERNEL_WIDTH``,
            ``t_or_r_align`` is ``'trial_aligned'`` or ``'response_aligned'``,
            and the two flags select the baseline strategy described above.

    Returns:
        tuple: ``({csv_file_name: rows}, data_dict)`` — see
        ``_collect_timeSeries_rows``. ``data_dict`` also carries the
        z-score bin/window settings used, for later reference in
        ``output_timeSeries_to_csv``.
    """
    file_name, SETTINGS_DICT, zscore_or_not, t_or_r_align, use_nonAM_baseline, global_baseline = input_list

    data_dict = get_JSON_data._load_one_json(file_name)

    bin_size = SETTINGS_DICT['ZSCORE_BIN_SIZE']
    signal_start_end = SETTINGS_DICT['ZSCORE_START_END']
    baseline_start_end = SETTINGS_DICT['ZSCORE_BASELINE_START_END']
    gaussian_kernel_width = SETTINGS_DICT['ZSCORE_GAUSSIAN_KERNEL_WIDTH']

    baseline_bin_cuts = np.arange(baseline_start_end[0], baseline_start_end[1] + bin_size, bin_size)
    signal_bin_cuts = np.arange(signal_start_end[0], signal_start_end[1] + bin_size, bin_size)

    if t_or_r_align == 'trial_aligned':
        input_column_name = 'Trial_spikes'
        output_column_name = 'Trial_timeSeries'
    else:
        input_column_name = 'Response_spikes'
        output_column_name = 'Response_timeSeries'

    output_column_name += '_zscore' if zscore_or_not else '_FR'
    if global_baseline:
        output_column_name += '_globalBaseline'

    for session in data_dict['Session'].keys():
        cur_session_data = data_dict['Session'][session]
        cur_session_spikes = cur_session_data[input_column_name]
        cur_session_rasters = []

        if len(cur_session_spikes) == 0:
            print('Spikes were not found for session ' + session + ' in JSON file for ' + t_or_r_align + ' data. Skipping.')
            continue

        if global_baseline:
            baseline_spikes = []
            for trial_n in range(len(cur_session_data['TrialID'])):
                if use_nonAM_baseline and (t_or_r_align == 'response_aligned'):
                    cur_trial_baseline = np.array(cur_session_data['Trial_spikes'][trial_n])
                else:
                    cur_trial_baseline = np.array(cur_session_data[input_column_name][trial_n])
                cur_trial_baseline = cur_trial_baseline[
                    (cur_trial_baseline >= baseline_start_end[0]) &
                    (cur_trial_baseline < baseline_start_end[1])
                ]
                baseline_spikes.append(cur_trial_baseline)

            for trial_n in range(len(cur_session_data['TrialID'])):
                cur_spikes = np.array(cur_session_spikes[trial_n])
                raster = __get_trialID_timeSeries(baseline_spikes, cur_spikes, bin_size, signal_start_end,
                                                  baseline_start_end, zscore_or_not=zscore_or_not,
                                                  baseline_bin_cuts=baseline_bin_cuts,
                                                  signal_bin_cuts=signal_bin_cuts,
                                                  gaussian_width=gaussian_kernel_width)
                cur_session_rasters.append(raster)
        else:
            for trial_n in range(len(cur_session_data['TrialID'])):
                cur_spikes = np.array(cur_session_spikes[trial_n])
                if use_nonAM_baseline:
                    baseline_spikes = np.array(cur_session_data['Trial_spikes'][trial_n])
                    baseline_spikes = baseline_spikes[
                        (baseline_spikes >= baseline_start_end[0]) &
                        (baseline_spikes < baseline_start_end[1])
                    ]
                else:
                    baseline_spikes = [cur_spikes[
                        (cur_spikes >= baseline_start_end[0]) &
                        (cur_spikes < baseline_start_end[1])
                    ]]
                raster = __get_trialID_timeSeries(baseline_spikes, cur_spikes, bin_size, signal_start_end,
                                                  baseline_start_end, zscore_or_not=zscore_or_not,
                                                  baseline_bin_cuts=baseline_bin_cuts,
                                                  signal_bin_cuts=signal_bin_cuts,
                                                  gaussian_width=gaussian_kernel_width)
                cur_session_rasters.append(raster)

        data_dict['Session'][session][output_column_name] = cur_session_rasters

    # Info params
    data_dict['Zscore_timeSeries_bin_size'] = bin_size
    data_dict['Zscore_timeSeries_start_end'] = signal_start_end
    data_dict['Zscore_timeSeries_baseline_start_end'] = baseline_start_end

    # Update JSON
    # write_json(data_dict, output_path + sep + 'JSON files', data_dict['Unit'] + '_unitData.json')

    return _collect_timeSeries_rows(data_dict, zscore_or_not, t_or_r_align, global_baseline)


def _collect_timeSeries_rows(data_dict, zscore_or_not, t_or_r_align, global_baseline):
    """Extract CSV rows from a single processed data_dict.

    Args:
        data_dict (dict): Unit JSON data already populated with the
            time-series column by ``extract_fr_timeSeries_fromJSON``.
        zscore_or_not (bool): Whether z-scored or raw-FR columns were computed
            (selects the source column and output filename).
        t_or_r_align (str): 'trial_aligned' or 'response_aligned' (selects the source column).
        global_baseline (bool): Whether a global (session-pooled) baseline was
            used (selects the source column).

    Returns:
        tuple: ``({file_name: rows}, data_dict)`` where ``rows`` is a list of
        flat per-trial rows (trial metadata columns followed by the binned
        time-series values), keyed by the CSV filename they belong in.
    """
    _columns_prefix = 'TP.'

    file_name = 'FR_timeSeries_data' + ('_zscore' if zscore_or_not else '_FR') + '.csv'
    column_name = ('Trial_timeSeries' if t_or_r_align == 'trial_aligned' else 'Response_timeSeries')
    column_name += '_zscore' if zscore_or_not else '_FR'
    column_name += '_globalBaseline' if global_baseline else ''

    rows = []
    for session in data_dict['Session'].keys():
        cur_session_data = data_dict['Session'][session]
        cur_sigs = np.round(cur_session_data[column_name], 4)

        hitFlag_list = cur_session_data['Hit']
        missFlag_list = cur_session_data['Miss']
        FAFlag_list = cur_session_data['FA']
        shockFlag_list = cur_session_data['ShockFlag']
        amdepth_list = np.round(cur_session_data['AMdepth'], 2)
        trialID_list = cur_session_data['TrialID']
        trialOnset_list = np.round(cur_session_data['Trial_onset'], 4)
        respLatency_list = np.round(cur_session_data['RespLatency'], 4)

        for trial_idx in range(len(trialID_list)):
            rows.append([
                data_dict['Unit'], session, t_or_r_align, column_name,
                hitFlag_list[trial_idx], missFlag_list[trial_idx],
                FAFlag_list[trial_idx], shockFlag_list[trial_idx],
                amdepth_list[trial_idx], trialID_list[trial_idx],
                trialOnset_list[trial_idx], respLatency_list[trial_idx],
                *cur_sigs[trial_idx]
            ])

    return {file_name: rows}, data_dict


def output_timeSeries_to_csv(all_rows, first_data_dict, SETTINGS_DICT):
    """Write all collected rows to CSV. Call once after all files are processed.

    Args:
        all_rows (dict): Mapping of output CSV filename to list of row lists,
            typically merged from multiple ``_collect_timeSeries_rows`` calls
            (one per unit).
        first_data_dict (dict): Any one processed unit's data_dict, used only
            to pull the shared z-score bin-size/window settings for the info file.
        SETTINGS_DICT (dict): Pipeline settings; only ``OUTPUT_PATH`` is used here.

    Returns:
        None. Writes one CSV per key in ``all_rows`` plus a
        ``Zscore_timeSeries_info.txt`` summary, all under ``SETTINGS_DICT['OUTPUT_PATH']``.
    """
    _columns_prefix = 'TP.'
    output_path = SETTINGS_DICT['OUTPUT_PATH']

    for file_name, rows in all_rows.items():
        info_columns = ['Unit', 'Session', 'Alignment', 'AnalysisID', 'Hit', 'Miss', 'FA', 'ShockFlag',
                      'AMDepth', 'TrialID', 'Trial_onset', 'RespLatency']
        n_timepoints = len(rows[0]) - len(info_columns)  # subtract non-TP columns
        csv_header = info_columns + [_columns_prefix + str(i + 1) for i in range(n_timepoints)]

        with open(sep.join([output_path, file_name]), 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(csv_header)
            writer.writerows(rows)

    # Write info txt
    with open(sep.join([output_path, 'Zscore_timeSeries_info.txt']), 'w') as f:
        f.write('Zscore bin size: ' + str(first_data_dict['Zscore_timeSeries_bin_size']) + '\n')
        f.write('Signal start/end: ' + str(first_data_dict['Zscore_timeSeries_start_end']) + '\n')
        f.write('Baseline start/end: ' + str(first_data_dict['Zscore_timeSeries_baseline_start_end']) + '\n')


def _to_list(val):
    """Wrap a scalar setting in a single-item list so it can be iterated like a multi-value one."""
    return val if isinstance(val, list) else [val]


def run_full_zscore_pipeline(filtered_files, SETTINGS_DICT):
    """Expand the ZSCORE_* settings into every combination and compute/write z-score time series for all of them.

    This is the notebook-level driver: builds one
    ``extract_fr_timeSeries_fromJSON`` input tuple per (unit, settings
    combination) pair, where "settings combination" is the Cartesian product
    of ``ZSCORE_DO_ZSCORE`` x ``ZSCORE_TRIAL_OR_RESPONSE_ALIGNED`` x
    ``ZSCORE_USE_NONAM_BASELINE`` x ``ZSCORE_GLOBAL_BASELINE`` (each may be a
    single value or a list, via ``_to_list``), runs them serially or via a
    process pool per ``SETTINGS_DICT['MULTIPROCESS']``, accumulates the
    returned rows by output CSV filename, and writes them all out via
    ``output_timeSeries_to_csv``.

    Args:
        filtered_files (Iterable[str]): Unit JSON paths to process.
        SETTINGS_DICT (dict): Pipeline settings. Uses ``ZSCORE_DO_ZSCORE``,
            ``ZSCORE_TRIAL_OR_RESPONSE_ALIGNED``, ``ZSCORE_USE_NONAM_BASELINE``,
            ``ZSCORE_GLOBAL_BASELINE``, ``MULTIPROCESS``, ``NUMBER_OF_CORES``.

    Returns:
        None. Writes one CSV per (do_zscore) combination plus
        ``Zscore_timeSeries_info.txt`` under ``SETTINGS_DICT['OUTPUT_PATH']``.
    """
    do_zscore_opts = _to_list(SETTINGS_DICT['ZSCORE_DO_ZSCORE'])
    t_or_r_align_opts = _to_list(SETTINGS_DICT['ZSCORE_TRIAL_OR_RESPONSE_ALIGNED'])
    use_nonAM_opts = _to_list(SETTINGS_DICT['ZSCORE_USE_NONAM_BASELINE'])
    global_baseline_opts = _to_list(SETTINGS_DICT['ZSCORE_GLOBAL_BASELINE'])

    input_list = [
        (fn, SETTINGS_DICT, zscore_or_not, t_or_r_align, use_nonAM, global_bl)
        for fn in filtered_files
        for zscore_or_not, t_or_r_align, use_nonAM, global_bl
        in product(do_zscore_opts, t_or_r_align_opts, use_nonAM_opts, global_baseline_opts)
    ]

    all_rows = {}
    first_data_dict = None

    if not SETTINGS_DICT['MULTIPROCESS']:
        for item in input_list:
            rows_by_file, data_dict = extract_fr_timeSeries_fromJSON(item)
            if first_data_dict is None:
                first_data_dict = data_dict
            for file_name, rows in rows_by_file.items():
                all_rows.setdefault(file_name, []).extend(rows)
    else:
        with Pool(SETTINGS_DICT['NUMBER_OF_CORES']) as pool:
            for rows_by_file, data_dict in pool.map(extract_fr_timeSeries_fromJSON, input_list, chunksize=1):
                if first_data_dict is None:
                    first_data_dict = data_dict
                for file_name, rows in rows_by_file.items():
                    all_rows.setdefault(file_name, []).extend(rows)

    output_timeSeries_to_csv(all_rows, first_data_dict, SETTINGS_DICT)