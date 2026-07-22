"""
outcome_decoding_timeSeries.py

Time-resolved trial-outcome decoding from population spike rasters.
Decodes at every time bin, optionally using lagged history features.
Designed to be called from an external pipeline after JSON preprocessing,
in the same style as outcome_decoding_atemporal.

External pipeline usage
-----------------------
    import outcome_decoding_timeSeries

    outcome_decoding_timeSeries.run(filtered_files, SETTINGS_DICT)

All classification settings are read from SETTINGS_DICT (see SETTINGS keys below).

Standalone usage
----------------
    python outcome_decoding_timeSeries.py
"""

from os import sep, makedirs
import platform
import json
from glob import glob

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import patches
import scipy.stats as st
from scipy.stats import gaussian_kde
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import balanced_accuracy_score
from sklearn.multiclass import OneVsOneClassifier
from astropy.convolution import convolve_fft, Gaussian1DKernel
from sklearn.svm import LinearSVC, SVC
from matplotlib.backends.backend_pdf import PdfPages
from helpers.format_axes import format_ax

import pandas as pd
from re import split
import csv
import warnings

warnings.filterwarnings("ignore")

if platform.system() == 'Windows':
    REGEX_SEP = sep * 2
else:
    REGEX_SEP = sep


# ── SETTINGS_DICT keys used by this module ────────────────────────────────────
#
#   DECODER_TS_METHOD            str   'lda' | 'lda_svc' | 'ovo' | 'svc'
#   DECODER_TS_SESSION           str   'active' | 'pre' | 'post' | 'post1h'
#   DECODER_TS_SPIKES_FIELDNAME  str   e.g. 'Response_spikes'
#   DECODER_TS_SPIKE_TIME_FORMAT bool  True if field contains spike times, False for pre-binned
#   DECODER_TS_START_TIME        float signal window start (s)
#   DECODER_TS_END_TIME          float signal window end   (s)
#   DECODER_TS_BIN_SIZE          float bin width (s)
#   DECODER_TS_PSTH_STEP_SIZE    float sliding-window step (s), usually == DECODER_TS_BIN_SIZE
#   DECODER_TS_GAUSSIAN_SIGMA    float smoothing kernel SD (s); 0 = no smoothing
#   DECODER_TS_SHOCK_ARTIFACT    list | None  [start, end] in seconds to blank, or None
#   DECODER_TS_DECODE_TARGET     str   what to decode at each time bin:
#                                'outcome'        Hit vs Miss (default; labels: 0=Hit, 1=Miss)
#                                'amdepth'        AM depth class from all Hit+Miss trials
#                                'amdepth_hits'   AM depth class from Hit trials only
#                                'amdepth_misses' AM depth class from Miss trials only
#   DECODER_TS_SHOCK_FLAG_FILTER int | None   ShockFlag value applied to the selected trial subset:
#                                             1 = ShockFlag==1 only (default)
#                                             0 = ShockFlag==0 only
#                                             None = all trials regardless of ShockFlag
#                                             (FA trials are never included regardless of this setting)
#   DECODER_TS_FEATURE_MODE      str   how to build the feature matrix for each time bin:
#                                'full_raster' (default) — flatten (units × history_lags)
#                                'mean_rate'             — average over history lags → (units,)
#                                'isi'                   — KDE of log-ISI per unit over the bin
#                                                          window (extended by history bins) →
#                                                          (units × n_grid,)
#                                                          requires DECODER_TS_SPIKE_TIME_FORMAT=True
#   DECODER_TS_ISI_GRID_MIN      float  shortest ISI included in KDE grid (s); default 0.001 (1 ms)
#   DECODER_TS_ISI_GRID_MAX      float  longest  ISI included in KDE grid (s); default 2.0
#   DECODER_TS_ISI_N_GRID        int    number of log-spaced grid evaluation points; default 20
#   DECODER_TS_N_HISTORY_BINS    int   lagged bins appended to feature vector; 0 = current bin only
#   DECODER_TS_MIN_TRIALS_PER_CLASS int  drop any class with fewer than this many trials before CV;
#                                        0 = disabled (default). Useful for amdepth targets where
#                                        rare depths would otherwise force CV fold reduction.
#   DECODER_TS_CSV_FILE            str | None  path to a pre-computed z-score CSV.  Required when
#                                              DECODER_TS_FEATURE_MODE='csv'; ignored otherwise.
#                                              Same format as described for the atemporal decoder.
#   DECODER_TS_CSV_SAMPLING_RATE   float       sampling rate (Hz) of the TP columns.
#                                              TP.1 = DECODER_TS_START_TIME;
#                                              each subsequent TP is 1/SAMPLING_RATE seconds later.
#   DECODER_TS_CSV_ANALYSIS_ID     str | None  filter CSV by AnalysisID; None = all (default).
#   DECODER_TS_LOG1P_TRANSFORM   bool  apply np.log1p() to the feature matrix at each time bin
#                                      before classification; default False.  Same rationale as
#                                      DECODER_N_LOG1P_TRANSFORM.  Ignored for 'isi' and 'csv' modes.
#   DECODER_TS_N_CV_SPLITS       int   number of stratified CV folds (default 5)
#   DECODER_TS_RANDOM_STATE      int   RNG seed (default 0)
#   DECODER_TS_SUBJECTS          list | None  subject IDs to include; None = all
#   DECODER_TS_SESSIONS_TO_EXCLUDE list | None  subject_date strings to drop; None = none
#   DECODER_TS_UNITS_CSV         str | None   path to SU_list.csv; None = all units
#   DECODER_TS_THRESHOLD_CSV     str          path to OFCPL_threshold_df.csv
#   DECODER_TS_OUTPUT_FOLDER     str          folder for CSV + PDF output
#   DECODER_TS_FILE_NAME_TAG     str          suffix for output file names
#
# ─────────────────────────────────────────────────────────────────────────────


# ── Standalone defaults ───────────────────────────────────────────────────────

_STANDALONE_SETTINGS = {
    'DECODER_TS_METHOD':            'lda',
    'DECODER_TS_SESSION':           'active',
    'DECODER_TS_SPIKES_FIELDNAME':  'Response_spikes',
    'DECODER_TS_SPIKE_TIME_FORMAT': True,
    'DECODER_TS_START_TIME':        -2.0,
    'DECODER_TS_END_TIME':           3.0,
    'DECODER_TS_BIN_SIZE':           0.1,
    'DECODER_TS_PSTH_STEP_SIZE':     0.1,
    'DECODER_TS_GAUSSIAN_SIGMA':     0,
    'DECODER_TS_SHOCK_ARTIFACT':    [-0.3, 0.3],
    'DECODER_TS_DECODE_TARGET':      'outcome',   # 'outcome' | 'amdepth' | 'amdepth_hits' | 'amdepth_misses'
    'DECODER_TS_SHOCK_FLAG_FILTER':  1,           # 1 | 0 | None (all trials in selected subset)
    'DECODER_TS_FEATURE_MODE':       'full_raster',  # 'full_raster' | 'mean_rate' | 'isi' | 'csv'
    'DECODER_TS_ISI_GRID_MIN':       0.001,           # seconds (1 ms)
    'DECODER_TS_ISI_GRID_MAX':       2.0,             # seconds
    'DECODER_TS_ISI_N_GRID':         20,              # log-spaced evaluation points
    'DECODER_TS_MIN_TRIALS_PER_CLASS': 0,              # drop classes with fewer trials; 0 = disabled
    'DECODER_TS_CSV_FILE':           None,            # path to z-score CSV; required for 'csv' mode
    'DECODER_TS_CSV_SAMPLING_RATE':  10,             # Hz — TP column sampling rate
    'DECODER_TS_CSV_ANALYSIS_ID':    None,            # filter CSV by AnalysisID; None = all
    'DECODER_TS_LOG1P_TRANSFORM':    False,           # apply log1p() to features; ignored for 'isi'/'csv'
    'DECODER_TS_N_HISTORY_BINS':     0,
    'DECODER_TS_N_CV_SPLITS':        5,
    'DECODER_TS_RANDOM_STATE':       0,
    'DECODER_TS_SUBJECTS':          ['SUBJ-ID-154', 'SUBJ-ID-389', 'SUBJ-ID-390',
                                     'SUBJ-ID-1036', 'SUBJ-ID-1037', 'SUBJ-ID-1038'],
    'DECODER_TS_SESSIONS_TO_EXCLUDE': None,
    'DECODER_TS_UNITS_CSV':         '.' + sep + sep.join(['Data', 'Output', 'SU_list.csv']),
    'DECODER_TS_THRESHOLD_CSV':     '.' + sep + sep.join(['Data', 'Output', 'OFCPL_threshold_df.csv']),
    'DECODER_TS_OUTPUT_FOLDER':     '.' + sep + sep.join(['Data', 'Output']),
    'DECODER_TS_FILE_NAME_TAG':     'TrialType_HitVsMiss_shockFlag1_lda_timeSeries',
}


# ── Spike helpers ─────────────────────────────────────────────────────────────

def _sliding_hist(trial_spikes, bin_cuts, step_size):
    """Count spikes in a fixed-width window sliding over ``bin_cuts`` start times.

    Args:
        trial_spikes (np.ndarray): Spike timestamps for one trial.
        bin_cuts (np.ndarray): Window start times.
        step_size (float): Width of each counting window.

    Returns:
        np.ndarray: Spike count per window, same length as ``bin_cuts``.
    """
    ret = np.zeros(len(bin_cuts))
    for i, start in enumerate(bin_cuts):
        ret[i] = np.sum((trial_spikes >= start) & (trial_spikes < (start + step_size)))
    return ret


def _gaussian_smooth(spikes, sigma_seconds, bin_size_seconds):
    """Smooth a binned spike-count/rate array with a Gaussian kernel via FFT convolution.

    Args:
        spikes (np.ndarray): Binned spike counts/rates to smooth.
        sigma_seconds (float): Gaussian kernel standard deviation, in seconds.
        bin_size_seconds (float): Duration of one bin, used to convert
            ``sigma_seconds`` into bins for the kernel.

    Returns:
        np.ndarray: Smoothed array, same length as ``spikes``.
    """
    sigma_bins = sigma_seconds / bin_size_seconds
    return convolve_fft(spikes, Gaussian1DKernel(stddev=sigma_bins))


def _resolve_session_name(session_names, session_type):
    """Find the session name matching a session-type keyword ('active'/'pre'/'post'/'post1h').

    Args:
        session_names (Iterable[str]): Candidate session names to search.
        session_type (str): One of 'active', 'pre', 'post', 'post1h'.

    Returns:
        str or None: The first matching session name, or None if
        ``session_type`` is unrecognized or no session matches (a message is
        printed in the unrecognized case).
    """
    matchers = {
        'active': lambda s: 'aversive' in s.lower() or 'active' in s.lower(),
        'pre':    lambda s: 'pre'      in s.lower(),
        'post':   lambda s: 'post_'    in s.lower() or 'post-' in s.lower(),
        'post1h': lambda s: 'post1h'   in s.lower(),
    }
    if session_type not in matchers:
        print(f'Session type "{session_type}" undefined')
        return None
    matches = [s for s in session_names if matchers[session_type](s)]
    return matches[0] if matches else None


# ── ISI feature extractor ─────────────────────────────────────────────────────

def _isi_kde_features(spike_times_per_trial, t_start, t_end, log_grid, exclude_range=None):
    """
    For each trial, fit a KDE to the log-ISI distribution and evaluate it at a
    fixed log-spaced grid.  Uses Scott's rule for bandwidth — adapts automatically
    to the number of observed spikes, avoiding parametric assumptions about the
    ISI distribution (Poisson, etc.).

    Parameters
    ----------
    spike_times_per_trial : list of array-like
        Spike times (seconds) for each trial.
    t_start, t_end : float
        Time window to extract spikes from.
    log_grid : np.ndarray (n_grid,)
        Log-space evaluation points, e.g. np.linspace(log(0.001), log(2.0), 20).
    exclude_range : [start, end] | None
        Optional window to blank before computing ISIs (e.g. shock artifact).

    Returns
    -------
    features : np.ndarray (n_trials, n_grid)
        KDE density at each grid point.
        Trials with fewer than 2 ISIs (< 3 spikes) get a row of zeros.
    """
    n_trials = len(spike_times_per_trial)
    n_grid   = len(log_grid)
    features = np.zeros((n_trials, n_grid))   # 0 = no information, not NaN

    for i, spikes in enumerate(spike_times_per_trial):
        spikes_in = np.sort(np.asarray(spikes, dtype=float))
        spikes_in = spikes_in[(spikes_in >= t_start) & (spikes_in < t_end)]
        if exclude_range is not None:
            spikes_in = spikes_in[~((spikes_in >= exclude_range[0]) &
                                    (spikes_in <  exclude_range[1]))]
        if len(spikes_in) < 3:          # need ≥ 2 ISIs → ≥ 3 spikes
            continue
        isis     = np.diff(spikes_in)
        isis     = isis[isis > 0]
        if len(isis) < 2:
            continue
        log_isis = np.log(isis)
        try:
            kde         = gaussian_kde(log_isis, bw_method='scott')
            features[i] = kde(log_grid)
        except Exception:
            pass

    return features


# ── CSV helpers ───────────────────────────────────────────────────────────────

def _unit_to_subject_date(unit_str):
    """
    Extract subject_date key from a Unit string.

    e.g. 'SUBJ-ID-1036_250830_concat_cluster200' → 'SUBJ-ID-1036_250830'
    Matches the subject_date format produced by _parse_subject_dates.
    """
    parts = split('_*_', unit_str)
    return f'{parts[0]}_{parts[1]}'


def _parse_subject_dates_from_csv(csv_df, subjects_to_run, sessions_to_exclude):
    """Return sorted unique subject_date strings found in the CSV Unit column."""
    sds = csv_df['Unit'].apply(_unit_to_subject_date).unique()
    result = []
    for sd in sds:
        subject = sd.split('_')[0]
        if subjects_to_run and subject not in subjects_to_run:
            continue
        if sessions_to_exclude and sd in sessions_to_exclude:
            continue
        result.append(sd)
    return sorted(result)


def _load_csv_session(subject_date, bin_cuts, csv_df, shock_flag_filter, settings,
                      prefix, hit_only=False, miss_only=False):
    """
    Load a session from a pre-computed z-score CSV, returning an array aligned to bin_cuts.

    Each element of bin_cuts maps to the TP column at the nearest sample time:
        tp_idx = round((bin_cuts[i] - START_TIME) * CSV_SAMPLING_RATE)
        column  = 'TP.{tp_idx + 1}'

    Returns (n_units, n_trials, n_bins) — same shape as the spike raster from the JSON path,
    so _decode_session_timeseries works without modification.  Bins whose corresponding TP
    column does not exist in the CSV receive NaN (filtered by the existing valid-column check).

    Parameters
    ----------
    subject_date      : str
    bin_cuts          : np.ndarray  bin start times (s)
    csv_df            : pd.DataFrame  pre-loaded CSV with '_subject_date' column
    shock_flag_filter : int | None
    settings          : dict
    prefix            : str   'DECODER_TS'
    hit_only / miss_only : bool  restrict to Hit or Miss trials only

    Returns
    -------
    session_arr : np.ndarray (n_units, n_trials, n_bins) or None
    base_meta   : pd.DataFrame or None  columns: Hit, Miss, AMdepth
    """
    start_time    = settings[f'{prefix}_START_TIME']
    sampling_rate = settings.get(f'{prefix}_CSV_SAMPLING_RATE', 100)
    analysis_id   = settings.get(f'{prefix}_CSV_ANALYSIS_ID')
    units_df      = settings.get('_UNITS_DF')

    tp_col_for_bin = [
        f'TP.{int(round((float(t) - start_time) * sampling_rate)) + 1}'
        for t in bin_cuts
    ]
    avail_tp = set(csv_df.columns)

    sess_df = csv_df[csv_df['_subject_date'] == subject_date].copy()
    if analysis_id:
        sess_df = sess_df[sess_df['AnalysisID'] == analysis_id]
    if len(sess_df) == 0:
        return None, None

    if miss_only:
        trial_mask = sess_df['Miss'] == 1
    elif hit_only:
        trial_mask = sess_df['Hit'] == 1
    else:
        trial_mask = (sess_df['Hit'] == 1) | (sess_df['Miss'] == 1)
    if shock_flag_filter is not None:
        trial_mask = trial_mask & (sess_df['ShockFlag'] == shock_flag_filter)
    sess_df = sess_df[trial_mask].copy()
    if len(sess_df) == 0:
        return None, None

    if units_df is not None:
        sess_df = sess_df[sess_df['Unit'].isin(units_df['Unit'].values)].copy()
    if len(sess_df) == 0:
        return None, None

    units   = sorted(sess_df['Unit'].unique())
    ref_df  = (sess_df[sess_df['Unit'] == units[0]]
               .sort_values('TrialID')
               .reset_index(drop=True))
    trial_ids = ref_df['TrialID'].values
    amt_col   = 'AMDepth' if 'AMDepth' in ref_df.columns else 'AMdepth'
    base_meta = (ref_df[['Hit', 'Miss', amt_col]]
                 .rename(columns={amt_col: 'AMdepth'})
                 .reset_index(drop=True))

    unit_arrays = []
    for unit in units:
        unit_df = (sess_df[sess_df['Unit'] == unit]
                   .sort_values('TrialID')
                   .set_index('TrialID')
                   .reindex(trial_ids)
                   .reset_index())
        arr = np.full((len(trial_ids), len(bin_cuts)), np.nan)
        for b_idx, tp_col in enumerate(tp_col_for_bin):
            if tp_col in avail_tp:
                arr[:, b_idx] = pd.to_numeric(unit_df[tp_col], errors='coerce').values
        unit_arrays.append(arr)

    if not unit_arrays:
        return None, None

    return np.stack(unit_arrays, axis=0), base_meta  # (n_units, n_trials, n_bins)


# ── Session preprocessor ──────────────────────────────────────────────────────

def _preprocess_session(subject_date, filtered_files, bin_cuts, settings):
    """
    Load and bin all units belonging to subject_date.

    Trial selection is driven by DECODER_TS_DECODE_TARGET:
      'outcome'        → Hit + Miss trials; labels 0=Hit, 1=Miss
      'amdepth'        → Hit + Miss trials; labels = AMdepth value
      'amdepth_hits'   → Hit trials only;   labels = AMdepth value
      'amdepth_misses' → Miss trials only;  labels = AMdepth value
    ShockFlag is filtered within the selected subset per DECODER_TS_SHOCK_FLAG_FILTER.
    FA and Reminder trials are never included.

    Returns
    -------
    session_spikes  : np.ndarray (n_units, n_trials, n_bins)  or None
    spike_times_all : list[units] of list[trials] of np.ndarray | None
        Raw spike times retained when DECODER_TS_FEATURE_MODE=='isi'; None otherwise.
    outcomes        : np.ndarray (n_trials,)  or None
        0/1 for 'outcome'; AMdepth float for 'amdepth*' targets.
    """
    spikes_fieldname  = settings['DECODER_TS_SPIKES_FIELDNAME']
    session_type      = settings['DECODER_TS_SESSION']
    spikeTime_format  = settings['DECODER_TS_SPIKE_TIME_FORMAT']
    BIN_SIZE          = settings['DECODER_TS_BIN_SIZE']
    psth_step_size    = settings['DECODER_TS_PSTH_STEP_SIZE']
    sigma             = settings['DECODER_TS_GAUSSIAN_SIGMA']
    shock             = settings['DECODER_TS_SHOCK_ARTIFACT']
    shock_flag_filter = settings.get('DECODER_TS_SHOCK_FLAG_FILTER', 1)
    decode_target     = settings.get('DECODER_TS_DECODE_TARGET', 'outcome')
    units_df          = settings.get('_UNITS_DF')
    feature_mode      = settings.get('DECODER_TS_FEATURE_MODE', 'full_raster')

    # ── CSV mode ──────────────────────────────────────────────────────────────
    if feature_mode == 'csv':
        csv_df = settings.get('_CSV_DF')
        if csv_df is None:
            raise ValueError(
                'DECODER_TS_FEATURE_MODE="csv" requires DECODER_TS_CSV_FILE to be set.')
        hit_only  = (decode_target == 'amdepth_hits')
        miss_only = (decode_target == 'amdepth_misses')
        session_arr, base_meta = _load_csv_session(
            subject_date, bin_cuts, csv_df, shock_flag_filter, settings,
            'DECODER_TS', hit_only=hit_only, miss_only=miss_only)
        if session_arr is None:
            return None, None, None
        # Build outcome labels consistent with the JSON path
        if decode_target == 'outcome':
            outcomes = np.zeros(len(base_meta))
            outcomes[base_meta['Miss'].values == 1] = 1
        else:
            outcomes = base_meta['AMdepth'].values.astype(float)
        # ISI spike times are not available in CSV mode
        return session_arr, None, outcomes

    to_remove_mask = (
        ((bin_cuts >= shock[0]) & (bin_cuts < shock[1])) if shock is not None
        else np.zeros(len(bin_cuts), dtype=bool)
    )

    cur_jsons        = [f for f in filtered_files if subject_date in f]
    session_spikes   = []
    spike_times_all  = []   # list[units] of list[trials] of arrays; ISI mode only
    outcomes         = []

    for file_name in cur_jsons:
        if units_df is not None:
            if not any(u in file_name for u in units_df['Unit'].values):
                continue

        with open(file_name, 'r') as fh:
            cur_dict = json.load(fh)

        session_name = _resolve_session_name(list(cur_dict['Session'].keys()), session_type)
        if session_name is None:
            continue

        cur_data = pd.DataFrame.from_dict(
            {k: cur_dict['Session'][session_name][k]
             for k in ['TrialID', 'Reminder', 'Hit', 'Miss', 'FA',
                       'ShockFlag', 'AMdepth', spikes_fieldname]}
        )

        # Build trial mask: FA and Reminder trials are always excluded
        base_mask = (cur_data['Reminder'] == 0)
        if shock_flag_filter is not None:
            base_mask = base_mask & (cur_data['ShockFlag'] == shock_flag_filter)

        if decode_target == 'amdepth_hits':
            trial_mask = base_mask & (cur_data['Hit'] == 1)
        elif decode_target == 'amdepth_misses':
            trial_mask = base_mask & (cur_data['Miss'] == 1)
        else:  # 'outcome' or 'amdepth' — both Hit and Miss
            trial_mask = base_mask & ((cur_data['Hit'] == 1) | (cur_data['Miss'] == 1))

        subset_data = cur_data[trial_mask].reset_index(drop=True)

        if decode_target == 'outcome':
            temp_outcomes = np.zeros(len(subset_data))
            temp_outcomes[subset_data['Miss'].values == 1] = 1
        else:  # amdepth, amdepth_hits, amdepth_misses
            temp_outcomes = subset_data['AMdepth'].values.astype(float)
        outcomes.append(temp_outcomes)

        allTrial_spikes   = np.zeros((len(subset_data), len(bin_cuts)))
        trial_spike_times = []
        for row_idx, raw_spikes in enumerate(subset_data[spikes_fieldname].to_numpy()):
            raw_spikes = np.array(raw_spikes)
            if feature_mode == 'isi' and spikeTime_format:
                trial_spike_times.append(raw_spikes.copy())
            binned = (_sliding_hist(raw_spikes, bin_cuts, psth_step_size)
                      if spikeTime_format else raw_spikes.copy())
            if sigma > 0:
                binned = _gaussian_smooth(binned, sigma, BIN_SIZE)
            if shock is not None:
                binned[to_remove_mask] = 0
            allTrial_spikes[row_idx] = binned

        session_spikes.append(allTrial_spikes)
        if feature_mode == 'isi' and spikeTime_format:
            spike_times_all.append(trial_spike_times)

    if not outcomes:
        return None, None, None

    reference_outcomes = np.asarray(outcomes[0])
    if not all(np.array_equal(reference_outcomes, np.asarray(o)) for o in outcomes[1:]):
        print(f'Outcome mismatch across units in {subject_date}')
        return None, None, None

    spike_times_out = spike_times_all if (feature_mode == 'isi' and spike_times_all) else None
    return np.stack(session_spikes, axis=0), spike_times_out, reference_outcomes


# ── Classifier ────────────────────────────────────────────────────────────────

def _fit_predict(method, X_train, y_train, X_test):
    """Fit one of several classifiers on a train split and predict labels for the test split.

    Args:
        method (str): One of 'lda' (LDA with shrinkage), 'lda_svc' (LDA
            dimensionality reduction followed by a linear SVC), 'ovo'
            (One-vs-One linear SVC), or 'svc' (RBF-kernel SVC). Any other
            value prints a message and returns None.
        X_train (array-like): Training features.
        y_train (array-like): Training labels.
        X_test (array-like): Test features to predict labels for.

    Returns:
        np.ndarray or None: Predicted labels for ``X_test``, or None if
        ``method`` is unrecognized or fitting/prediction raises
        ``IndexError``/``ValueError`` (e.g. too few samples/classes).
    """
    try:
        if method == 'lda':
            clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
            clf.fit(X_train, y_train)
            return clf.predict(X_test)

        elif method == 'lda_svc':
            lda = LinearDiscriminantAnalysis()
            Xr  = lda.fit_transform(X_train, y_train)
            Xte = lda.transform(X_test)
            clf = LinearSVC(C=.5, dual='auto', random_state=0)
            clf.fit(Xr, y_train)
            return clf.predict(Xte)

        elif method == 'ovo':
            clf = OneVsOneClassifier(LinearSVC(C=.5, dual='auto', random_state=0))
            clf.fit(X_train, y_train)
            return clf.predict(X_test)

        elif method == 'svc':
            clf = SVC(kernel='rbf', C=0.5, gamma='scale')
            clf.fit(X_train, y_train)
            return clf.predict(X_test)

        else:
            print(f'Method "{method}" not implemented.')
            return None
    except (IndexError, ValueError):
        return None


def _decode_session_timeseries(session_spikes, spike_times_all_units, outcomes, bin_cuts, settings):
    """
    Decode trial outcomes at every time bin across the session.

    At each bin, the feature vector is built from the current bin plus
    DECODER_TS_N_HISTORY_BINS preceding non-artifact bins according to
    DECODER_TS_FEATURE_MODE ('full_raster', 'mean_rate', or 'isi').
    A shuffled-label control is run in parallel to estimate per-bin chance.

    Parameters
    ----------
    spike_times_all_units : list[units] of list[trials] of np.ndarray | None
        Raw spike times; required when DECODER_TS_FEATURE_MODE=='isi'.

    Returns
    -------
    accuracy_arr : np.ndarray (n_bins,)  mean CV balanced accuracy per bin
    chance_arr   : np.ndarray (n_bins,)  shuffled-label chance per bin
    n_units      : int
    """
    method          = settings['DECODER_TS_METHOD']
    n_splits        = settings['DECODER_TS_N_CV_SPLITS']
    random_state    = settings['DECODER_TS_RANDOM_STATE']
    n_history_bins  = settings.get('DECODER_TS_N_HISTORY_BINS', 0)
    shock           = settings['DECODER_TS_SHOCK_ARTIFACT']
    psth_step_size  = settings['DECODER_TS_PSTH_STEP_SIZE']
    feature_mode    = settings.get('DECODER_TS_FEATURE_MODE', 'full_raster')

    if feature_mode == 'isi':
        isi_grid_min = settings.get('DECODER_TS_ISI_GRID_MIN', 0.001)
        isi_grid_max = settings.get('DECODER_TS_ISI_GRID_MAX', 2.0)
        isi_n_grid   = settings.get('DECODER_TS_ISI_N_GRID',   20)
        log_grid     = np.linspace(np.log(isi_grid_min), np.log(isi_grid_max), isi_n_grid)
    else:
        log_grid = None

    to_remove_mask = (
        ((bin_cuts >= shock[0]) & (bin_cuts < shock[1])) if shock is not None
        else np.zeros(len(bin_cuts), dtype=bool)
    )

    n_units, n_trials, n_bins = session_spikes.shape
    accuracy_arr = np.full(n_bins, np.nan)
    chance_arr   = np.full(n_bins, np.nan)

    y_data = LabelEncoder().fit_transform(outcomes)
    classes, counts = np.unique(y_data, return_counts=True)

    # Drop classes with too few trials (propagate to spikes and spike times)
    min_trials = settings.get('DECODER_TS_MIN_TRIALS_PER_CLASS', 0)
    if min_trials > 0:
        keep_classes = classes[counts >= min_trials]
        if len(keep_classes) < 2:
            print(f"  WARNING: fewer than 2 classes have ≥{min_trials} trials. Skipping session.")
            return accuracy_arr, chance_arr, n_units
        dropped = int((~np.isin(y_data, keep_classes)).sum())
        if dropped:
            print(f"  NOTE: dropping {dropped} trials from classes with <{min_trials} trials")
            keep_mask = np.isin(y_data, keep_classes)
            session_spikes = session_spikes[:, keep_mask, :]
            if spike_times_all_units is not None:
                spike_times_all_units = [
                    [t for t, k in zip(unit_times, keep_mask) if k]
                    for unit_times in spike_times_all_units
                ]
            y_data  = LabelEncoder().fit_transform(outcomes[keep_mask])
            classes, counts = np.unique(y_data, return_counts=True)
            n_trials = session_spikes.shape[1]

    if counts.min() < n_splits:
        print(f"  WARNING: min class count ({counts.min()}) < n_splits ({n_splits}). Skipping session.")
        return accuracy_arr, chance_arr, n_units

    cv  = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    rng = np.random.default_rng(random_state)

    valid_bins = [i for i in range(n_bins) if not to_remove_mask[i]]
    min_start  = valid_bins[n_history_bins] if len(valid_bins) > n_history_bins else None

    for bin_idx in range(n_bins):
        if to_remove_mask[bin_idx]:
            continue
        if min_start is not None and bin_idx < min_start:
            continue

        # Lagged feature vector: current bin + K preceding non-artifact bins
        preceding = [i for i in range(bin_idx - 1, -1, -1)
                     if not to_remove_mask[i]][:n_history_bins]
        history_idx = sorted(preceding) + [bin_idx]

        if feature_mode == 'mean_rate':
            X = session_spikes[:, :, history_idx].mean(axis=2).T   # (trials, units)
        elif feature_mode == 'isi':
            if spike_times_all_units is None:
                print('  ISI mode requires DECODER_TS_SPIKE_TIME_FORMAT=True. Falling back to mean_rate.')
                X = session_spikes[:, :, history_idx].mean(axis=2).T
            else:
                t_start = float(bin_cuts[history_idx[0]])
                t_end   = float(bin_cuts[bin_idx]) + psth_step_size
                X = np.concatenate(
                    [_isi_kde_features(unit_times, t_start, t_end,
                                       log_grid, exclude_range=shock)
                     for unit_times in spike_times_all_units],
                    axis=1
                )  # (n_trials, n_units * n_grid)
        else:  # 'full_raster', 'csv', or unknown
            # CSV mode: session_spikes contains one TP value per bin_cut, already aligned;
            # history_idx selects the same bins as in the spike-raster path.
            if feature_mode not in ('full_raster', 'csv'):
                print(f"  Unknown DECODER_TS_FEATURE_MODE '{feature_mode}'. Using full_raster.")
            X = session_spikes[:, :, history_idx].transpose(1, 0, 2).reshape(n_trials, -1)

        # Optional log1p transform — compresses right-skewed spike counts toward Gaussianity.
        # Skipped for 'isi' (already log-scaled) and 'csv' (pre-z-scored).
        if settings.get('DECODER_TS_LOG1P_TRANSFORM', False) and feature_mode not in ('isi', 'csv'):
            X = np.log1p(X)

        valid = ~np.isnan(X).any(axis=0)
        X = X[:, valid]
        if X.shape[1] == 0:
            continue

        fold_scores    = []
        fold_scores_sh = []

        try:
            for train_idx, test_idx in cv.split(X, y_data):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y_data[train_idx], y_data[test_idx]
                y_train_sh      = rng.permutation(y_train)

                sc      = StandardScaler()
                X_train = sc.fit_transform(X_train)
                X_test  = sc.transform(X_test)

                y_pred    = _fit_predict(method, X_train, y_train,    X_test)
                y_pred_sh = _fit_predict(method, X_train, y_train_sh, X_test)

                if y_pred is not None:
                    fold_scores.append(balanced_accuracy_score(y_test, y_pred))
                if y_pred_sh is not None:
                    fold_scores_sh.append(balanced_accuracy_score(y_test, y_pred_sh))

        except ValueError:
            pass

        if fold_scores:
            accuracy_arr[bin_idx] = float(np.mean(fold_scores))
        if fold_scores_sh:
            chance_arr[bin_idx] = float(np.mean(fold_scores_sh))

    return accuracy_arr, chance_arr, n_units


# ── File-list helpers ─────────────────────────────────────────────────────────

def _parse_subject_dates(filtered_files, subjects_to_run, sessions_to_exclude):
    """Extract sorted unique subject_date keys from JSON file paths, applying include/exclude filters.

    Args:
        filtered_files (Iterable[str]): JSON file paths to parse (filename
            format: '<subject>_<date>_...').
        subjects_to_run (Iterable[str] or None): If given, only keep entries
            whose subject is in this collection.
        sessions_to_exclude (Iterable[str] or None): If given, drop any
            subject_date found in this collection.

    Returns:
        list[str]: Sorted unique 'subject_date' strings.
    """
    subjects, dates = [], []
    for f in filtered_files:
        parts = split('_*_', split(REGEX_SEP, f)[-1])
        subjects.append(parts[0])
        dates.append(parts[1])

    subject_dates = np.array(['_'.join(p) for p in zip(subjects, dates)])
    subj_arr      = np.array(subjects)

    subj_mask = (np.in1d(subj_arr, subjects_to_run)
                 if subjects_to_run else np.ones(len(filtered_files), dtype=bool))
    excl_mask = (np.in1d(subject_dates, sessions_to_exclude)
                 if sessions_to_exclude else np.zeros(len(filtered_files), dtype=bool))

    return sorted(set(subject_dates[subj_mask & ~excl_mask]))


def _get_day_sessions(threshold_df):
    """Return (worst_day_sessions, late_day_sessions) as sets of subject_date strings.

    Args:
        threshold_df (pandas.DataFrame): Per-session training-day/threshold
            table with 'Day', 'Subject', 'Session' columns.

    Returns:
        tuple[set[str], set[str]]: ``(early, late)`` — subject_date strings
        for Day==1 sessions, and for the last few days (9/10, or 8/9/10 for
        the one subject with only 9 training days).
    """
    early_df = threshold_df[threshold_df['Day'] == 1]
    early    = set('_'.join([subj, split('-*-', sid)[0]])
                   for subj, sid in zip(early_df['Subject'], early_df['Session']))

    late_df  = threshold_df[threshold_df['Day'].isin([10, 9, 8])]
    late     = set('_'.join([subj, split('-*-', sid)[0]])
                   for subj, sid in zip(late_df['Subject'], late_df['Session']))

    return early, late


# ── Output helpers ────────────────────────────────────────────────────────────

def _save_csv(results, settings, bin_cuts):
    """Write one row per (session, time bin) of decoding accuracy/chance to a CSV.

    Args:
        results (dict): Maps subject_date -> per-session result dict with
            'accuracy'/'chance' arrays (per bin), 'n_units',
            'day_of_training', 'day_type'.
        settings (dict): Pipeline settings; uses ``DECODER_TS_OUTPUT_FOLDER``,
            ``DECODER_TS_FILE_NAME_TAG``, ``DECODER_TS_METHOD``.
        bin_cuts (np.ndarray): Time-bin start times (seconds), one CSV row per bin per session.

    Returns:
        None. Writes ``LDAoutput_<file_name_tag>.csv`` under
        ``DECODER_TS_OUTPUT_FOLDER`` and prints its path.
    """
    output_folder = settings['DECODER_TS_OUTPUT_FOLDER']
    file_name_tag = settings['DECODER_TS_FILE_NAME_TAG']
    method        = settings['DECODER_TS_METHOD']

    csv_path = sep.join([output_folder, 'LDAoutput_' + file_name_tag + '.csv'])
    with open(csv_path, 'w', newline='', encoding='utf-8') as fh:
        writer = csv.writer(fh)
        writer.writerow(['Session', 'Method', 'Time_s',
                         'Accuracy', 'Chance',
                         'Unit_count', 'Day_of_training', 'Training_stage'])
        for sd, r in results.items():
            for bin_idx, t in enumerate(bin_cuts):
                acc = r['accuracy'][bin_idx]
                chc = r['chance'][bin_idx]
                writer.writerow([
                    sd, method, round(float(t), 3),
                    round(acc * 100, 2) if not np.isnan(acc) else '',
                    round(chc * 100, 2) if not np.isnan(chc) else '',
                    r['n_units'], r['day_of_training'], r['day_type'],
                ])
    print(f"Saved CSV → {csv_path}")


def _ts_mean_se(arrays):
    """Mean and SE across a list of (n_bins,) arrays, NaN-safe, per bin."""
    if not arrays:
        return None, None
    mat = np.array(arrays, dtype=float)                          # (n, n_bins)
    n   = np.sum(~np.isnan(mat), axis=0).astype(float)
    n[n == 0] = np.nan
    mean = np.nanmean(mat, axis=0)
    se   = np.nanstd(mat, axis=0) / np.sqrt(n)
    return mean, se


def _plot_ts(ax, bin_cuts, mean, se, color, label=None, linestyle='-'):
    """Fill ±SE band and plot mean line."""
    valid = ~np.isnan(mean)
    if not np.any(valid):
        return
    ax.fill_between(bin_cuts[valid],
                    (mean - se)[valid], (mean + se)[valid],
                    alpha=0.35, color=color)
    ax.plot(bin_cuts[valid], mean[valid],
            color=color, linestyle=linestyle, label=label)


def _subject_mean_ts(results, day_type, key):
    """Return list of per-subject mean (n_bins,) arrays for a given day_type."""
    from collections import defaultdict
    by_subject = defaultdict(list)
    for sd, r in results.items():
        if r['day_type'] == day_type:
            subj = sd.split('_')[0]
            arr  = r[key]
            if not np.all(np.isnan(arr)):
                by_subject[subj].append(arr)
    return [np.nanmean(v, axis=0) for v in by_subject.values()]


def _save_pdf(results, settings, bin_cuts):
    """Plot decoding accuracy over time (all/first/best day groups, with SE bands) into a PDF.

    Args:
        results (dict): Maps subject_date -> per-session result dict, as in ``_save_csv``.
        settings (dict): Pipeline settings; uses ``DECODER_TS_OUTPUT_FOLDER``,
            ``DECODER_TS_FILE_NAME_TAG``, ``DECODER_TS_START_TIME``, ``DECODER_TS_END_TIME``.
        bin_cuts (np.ndarray): Time-bin start times (seconds), the x-axis for the plot.

    Returns:
        None. Writes ``LDAoutput_<file_name_tag>.pdf`` under ``DECODER_TS_OUTPUT_FOLDER``.
    """
    output_folder = settings['DECODER_TS_OUTPUT_FOLDER']
    file_name_tag = settings['DECODER_TS_FILE_NAME_TAG']
    start_time    = settings['DECODER_TS_START_TIME']
    end_time      = settings['DECODER_TS_END_TIME']

    # ── Collect group arrays ──────────────────────────────────────────────────
    all_acc    = [r['accuracy'] * 100 for r in results.values()]
    all_chance = [r['chance']   * 100 for r in results.values()]

    early_acc    = _subject_mean_ts(results, 'early', 'accuracy')
    early_chance = _subject_mean_ts(results, 'early', 'chance')
    late_acc     = _subject_mean_ts(results, 'late',  'accuracy')
    late_chance  = _subject_mean_ts(results, 'late',  'chance')

    # Scale subject means to %
    early_acc    = [a * 100 for a in early_acc]
    early_chance = [a * 100 for a in early_chance]
    late_acc     = [a * 100 for a in late_acc]
    late_chance  = [a * 100 for a in late_chance]

    # Above-chance: subtract per-session shuffled before averaging
    above_all   = [(a - c) for a, c in zip(all_acc,    all_chance)]
    above_early = [(a - c) for a, c in zip(early_acc,  early_chance)]
    above_late  = [(a - c) for a, c in zip(late_acc,   late_chance)]

    pdf_path = sep.join([output_folder, 'LDAoutput_' + file_name_tag + '.pdf'])
    with PdfPages(pdf_path) as pdf:

        # ── Page 1: all sessions — accuracy + chance ──────────────────────────
        fig, ax = plt.subplots()
        mean_c, se_c = _ts_mean_se(all_chance)
        mean_a, se_a = _ts_mean_se(all_acc)
        if mean_c is not None:
            _plot_ts(ax, bin_cuts, mean_c, se_c, '#D4B483', linestyle='--')
        if mean_a is not None:
            _plot_ts(ax, bin_cuts, mean_a, se_a, '#D4B483', label='All training')
        ax.set_ylabel('Decoding accuracy (%)')
        ax.set_xlabel('Time (s)')
        ax.set_title('All training sessions')
        ax.set_xlim([start_time, end_time])
        ax.legend(handles=[patches.Patch(facecolor='#D4B483', alpha=0.7, label='All training'),
                            patches.Patch(facecolor='#D4B483', alpha=0.35, label='Chance (shuffled)')],
                  frameon=False)
        format_ax(ax); pdf.savefig(); plt.close()

        # ── Page 2: early vs late — accuracy + chance ─────────────────────────
        fig, ax = plt.subplots()
        for acc_list, chance_list, color, label in [
            (early_acc, early_chance, '#C1666B', 'Early'),
            (late_acc,  late_chance,  '#4281A4', 'Late'),
        ]:
            mean_c, se_c = _ts_mean_se(chance_list)
            mean_a, se_a = _ts_mean_se(acc_list)
            if mean_c is not None:
                _plot_ts(ax, bin_cuts, mean_c, se_c, color, linestyle='--')
            if mean_a is not None:
                _plot_ts(ax, bin_cuts, mean_a, se_a, color, label=label)
        ax.set_ylabel('Decoding accuracy (%)')
        ax.set_xlabel('Time (s)')
        ax.set_title('Early vs late training')
        ax.set_xlim([start_time, end_time])
        ax.legend(handles=[patches.Patch(facecolor='#C1666B', alpha=0.7, label='Early'),
                            patches.Patch(facecolor='#4281A4', alpha=0.7, label='Late')],
                  frameon=False)
        format_ax(ax); pdf.savefig(); plt.close()

        # ── Page 3: above chance — all sessions ───────────────────────────────
        fig, ax = plt.subplots()
        mean_ab, se_ab = _ts_mean_se(above_all)
        if mean_ab is not None:
            _plot_ts(ax, bin_cuts, mean_ab, se_ab, '#D4B483', label='All training')
        ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
        ax.set_ylabel('Decoding accuracy (% above chance)')
        ax.set_xlabel('Time (s)')
        ax.set_title('All sessions — above chance')
        ax.set_xlim([start_time, end_time])
        format_ax(ax); pdf.savefig(); plt.close()

        # ── Page 4: above chance — early vs late ─────────────────────────────
        fig, ax = plt.subplots()
        for above_list, color, label in [
            (above_early, '#C1666B', 'Early'),
            (above_late,  '#4281A4', 'Late'),
        ]:
            mean_ab, se_ab = _ts_mean_se(above_list)
            if mean_ab is not None:
                _plot_ts(ax, bin_cuts, mean_ab, se_ab, color, label=label)
        ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
        ax.set_ylabel('Decoding accuracy (% above chance)')
        ax.set_xlabel('Time (s)')
        ax.set_title('Early vs late — above chance')
        ax.set_xlim([start_time, end_time])
        ax.legend(handles=[patches.Patch(facecolor='#C1666B', alpha=0.7, label='Early'),
                            patches.Patch(facecolor='#4281A4', alpha=0.7, label='Late')],
                  frameon=False)
        format_ax(ax); pdf.savefig(); plt.close()

    print(f"Saved PDF → {pdf_path}")


# ── Public entry point ────────────────────────────────────────────────────────

def run(filtered_files, SETTINGS_DICT):
    """
    Parameters
    ----------
    filtered_files : list of str
        JSON file paths (same list passed to zscore_timeSeries_fromJSON).
    SETTINGS_DICT  : dict
        Must contain the DECODER_TS_* keys documented at the top of this module.

    Returns
    -------
    results : dict keyed by subject_date:
        'accuracy'       : np.ndarray (n_bins,)  mean CV balanced accuracy per bin (0–1)
        'chance'         : np.ndarray (n_bins,)  shuffled-label chance per bin (0–1)
        'n_units'        : int
        'day_type'       : 'early' | 'late' | 'other'
        'day_of_training': int | None
    """
    settings = dict(SETTINGS_DICT)  # shallow copy — don't mutate caller's dict

    units_csv = settings.get('DECODER_TS_UNITS_CSV')
    settings['_UNITS_DF'] = pd.read_csv(units_csv) if units_csv else None

    threshold_df = pd.read_csv(settings['DECODER_TS_THRESHOLD_CSV'])
    early_day_sessions, late_day_sessions = _get_day_sessions(threshold_df)

    bin_cuts = np.arange(settings['DECODER_TS_START_TIME'],
                         settings['DECODER_TS_END_TIME'],
                         settings['DECODER_TS_BIN_SIZE'])

    # ── CSV mode: load pre-computed data file and derive sessions from it ──────
    feature_mode = settings.get('DECODER_TS_FEATURE_MODE', 'full_raster')
    if feature_mode == 'csv':
        csv_file = settings.get('DECODER_TS_CSV_FILE')
        if csv_file is None:
            raise ValueError(
                'DECODER_TS_FEATURE_MODE="csv" requires DECODER_TS_CSV_FILE to be set.')
        print(f"Loading CSV data from: {csv_file}")
        csv_df = pd.read_csv(csv_file)
        csv_df['_subject_date'] = csv_df['Unit'].apply(_unit_to_subject_date)
        settings['_CSV_DF'] = csv_df
        unique_sessions = _parse_subject_dates_from_csv(
            csv_df,
            settings.get('DECODER_TS_SUBJECTS'),
            settings.get('DECODER_TS_SESSIONS_TO_EXCLUDE'),
        )
    else:
        unique_sessions = _parse_subject_dates(
            filtered_files,
            settings.get('DECODER_TS_SUBJECTS'),
            settings.get('DECODER_TS_SESSIONS_TO_EXCLUDE'),
        )

    # Preprocess
    print(f"Preprocessing {len(unique_sessions)} sessions...")
    preprocessed = {}
    for sd in unique_sessions:
        spikes, spike_times, outcomes = _preprocess_session(sd, filtered_files, bin_cuts, settings)
        if spikes is not None:
            preprocessed[sd] = {'spikes': spikes, 'spike_times': spike_times, 'outcomes': outcomes}
    print(f"  → {len(preprocessed)} sessions retained")

    # Decode
    print(f"Running time-series classification [{settings['DECODER_TS_METHOD']}] ...")
    results = {}
    for sd in sorted(preprocessed):
        acc_arr, chance_arr, n_units = _decode_session_timeseries(
            preprocessed[sd]['spikes'],
            preprocessed[sd]['spike_times'],
            preprocessed[sd]['outcomes'],
            bin_cuts,
            settings,
        )

        day_type = ('early' if sd in early_day_sessions else
                    'late'  if sd in late_day_sessions  else 'other')

        day_row = threshold_df[threshold_df.apply(
            lambda r: '_'.join([r['Subject'], split('-*-', r['Session'])[0]]) == sd, axis=1
        )]
        day_of_training = int(day_row['Day'].values[0]) if len(day_row) else None

        peak_acc = float(np.nanmax(acc_arr) * 100)
        print(f"  {sd}: peak acc={peak_acc:.1f}%  units={n_units}  [{day_type}]")

        results[sd] = {
            'accuracy':        acc_arr,
            'chance':          chance_arr,
            'n_units':         n_units,
            'day_type':        day_type,
            'day_of_training': day_of_training,
        }

    makedirs(settings['DECODER_TS_OUTPUT_FOLDER'], exist_ok=True)

    _save_csv(results, settings, bin_cuts)
    _save_pdf(results, settings, bin_cuts)
    return results


# ── Standalone entry point ────────────────────────────────────────────────────

if __name__ == '__main__':
    INPUT_FOLDER = '.' + sep + sep.join(['Data', 'Output', 'JSON files'])
    all_json     = glob(INPUT_FOLDER + sep + '*json')
    run(all_json, _STANDALONE_SETTINGS)
