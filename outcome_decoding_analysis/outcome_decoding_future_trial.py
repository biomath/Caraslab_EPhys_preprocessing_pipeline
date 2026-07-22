"""
outcome_decoding_future_trial.py

Two-stage future-trial decoder: uses trial N neural state to predict trial N+1 outcome.
Based on outcome_decoding_atemporal.py with minimal modifications.

Trial selection includes:
  - ShockFlag==1 trials (Hit or Miss)
  - The first ShockFlag==0 trial immediately following a ShockFlag==1 trial (Hit or Miss)

Stage 1: LDA on trial N population raster → decision value + predicted label
Stage 2: LDA on (decision_value_N, predicted_label_N) → outcome_N+1

External pipeline usage
-----------------------
    import outcome_decoding_future_trial

    outcome_decoding_future_trial.run(filtered_files, SETTINGS_DICT)

All classification settings are read from SETTINGS_DICT (see SETTINGS keys below).

Standalone usage
----------------
    python outcome_decoding_future_trial.py
"""

from os import sep, makedirs
import platform
import json
from glob import glob

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import patches
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold, permutation_test_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import balanced_accuracy_score
from sklearn.decomposition import PCA
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
#   DECODER_N1_METHOD               str   'lda' | 'lda_svc' | 'ovo' | 'svc'
#   DECODER_N1_SESSION              str   'active' | 'pre' | 'post' | 'post1h'
#   DECODER_N1_SPIKES_FIELDNAME     str   e.g. 'Response_spikes'
#   DECODER_N1_SPIKE_TIME_FORMAT    bool  True if field contains spike times, False for pre-binned
#   DECODER_N1_START_TIME           float signal window start (s)
#   DECODER_N1_END_TIME             float signal window end   (s)
#   DECODER_N1_BIN_SIZE             float bin width (s)
#   DECODER_N1_PSTH_STEP_SIZE       float sliding-window step (s), usually == DECODER_N1_BIN_SIZE
#   DECODER_N1_EPOCH_START          float epoch start for decoding (s), <= DECODER_N1_START_TIME
#   DECODER_N1_EPOCH_END            float epoch end   for decoding (s), >= DECODER_N1_END_TIME
#   DECODER_N1_GAUSSIAN_SIGMA       float smoothing kernel SD (s); 0 = no smoothing
#   DECODER_N1_SHOCK_ARTIFACT       list | None  [start, end] in seconds to blank, or None
#   DECODER_N1_N_CV_SPLITS          int   number of stratified CV folds (default 5)
#   DECODER_N1_RANDOM_STATE         int   RNG seed (default 0)
#   DECODER_N1_SUBJECTS             list | None  subject IDs to include; None = all
#   DECODER_N1_SESSIONS_TO_EXCLUDE  list | None  subject_date strings to drop; None = none
#   DECODER_N1_UNITS_CSV            str | None   path to SU_list.csv; None = all units
#   DECODER_N1_THRESHOLD_CSV        str          path to OFCPL_threshold_df.csv
#   DECODER_N1_OUTPUT_FOLDER        str          folder for CSV + PDF output
#   DECODER_N1_FILE_NAME_TAG        str          suffix for output file names
#   DECODER_N1_FEATURE_MODE         str   how to build the feature matrix for each trial:
#                                   'full_raster' (default) — flatten (units × epoch_bins)
#                                   'mean_rate'             — mean spike count per unit → (units,)
#                                   'csv'                   — use pre-computed z-score CSV data;
#                                                             requires DECODER_N1_CSV_FILE.
#   DECODER_N1_CSV_FILE             str | None  path to a pre-computed z-score CSV.
#                                               Required when DECODER_N1_FEATURE_MODE='csv'.
#   DECODER_N1_CSV_SAMPLING_RATE    float       sampling rate (Hz) of the TP columns.
#                                               TP.1 = DECODER_N1_START_TIME.
#   DECODER_N1_CSV_ANALYSIS_ID      str | None  filter CSV by AnalysisID; None = all (default).
#
# ─────────────────────────────────────────────────────────────────────────────


# ── Standalone defaults ───────────────────────────────────────────────────────

_STANDALONE_SETTINGS = {
    'DECODER_N1_METHOD':              'lda',
    'DECODER_N1_SESSION':             'active',
    'DECODER_N1_SPIKES_FIELDNAME':    'Response_spikes',
    'DECODER_N1_SPIKE_TIME_FORMAT':   True,
    'DECODER_N1_START_TIME':          -2.0,
    'DECODER_N1_END_TIME':             3.0,
    'DECODER_N1_BIN_SIZE':             0.1,
    'DECODER_N1_PSTH_STEP_SIZE':       0.1,
    'DECODER_N1_EPOCH_START':         -2.0,
    'DECODER_N1_EPOCH_END':            3.0,
    'DECODER_N1_GAUSSIAN_SIGMA':       0,
    'DECODER_N1_SHOCK_ARTIFACT':       None,
    'DECODER_N1_N_CV_SPLITS':              5,
    'DECODER_N1_RANDOM_STATE':             0,
    'DECODER_N1_MIN_PAIRS_STRATIFIED':    10,   # min valid pairs per trial-N-type subset
    'DECODER_N1_N_PERMUTATIONS':        1000,   # permutation test iterations for threshold transition
    'DECODER_N1_TESTS_TO_RUN':          [      # comment out any to skip
        'atemporal',         # trial N → trial N outcome (sanity check)
        'two_stage',         # trial N neural → stage1 signal → trial N+1 outcome
        'direct',            # trial N neural → trial N+1 outcome (single stage)
        'threshold',         # ShockFlag 1→0 transitions only + permutation test
        'by_ntype',          # direct decoder split by trial N type (Hit-N / Miss-N)
        'decision_variance',     # CV decision value variance per trial type (stereotypy probe)
        'decision_correlation',  # correlation between trial-N decision value and trial-N+1 outcome
        'behavioral_baseline',   # win-stay behavioral autocorrelation (no neural data)
    ],
    'DECODER_N1_SUBJECTS':            ['SUBJ-ID-154', 'SUBJ-ID-389', 'SUBJ-ID-390',
                                'SUBJ-ID-1036', 'SUBJ-ID-1037', 'SUBJ-ID-1038'],
    'DECODER_N1_SESSIONS_TO_EXCLUDE': None,
    'DECODER_N1_UNITS_CSV':           '.' + sep + sep.join(['Data', 'Output', 'SU_list.csv']),
    'DECODER_N1_THRESHOLD_CSV':       '.' + sep + sep.join(['Data', 'Output', 'OFCPL_threshold_df.csv']),
    'DECODER_N1_OUTPUT_FOLDER':       '.' + sep + sep.join(['Data', 'Output']),
    'DECODER_N1_FILE_NAME_TAG':       'TrialType_allTTs_lda_fullRaster_futureTrial',
    'DECODER_N1_FEATURE_MODE':        'full_raster',  # 'full_raster' | 'mean_rate' | 'csv'
    'DECODER_N1_CSV_FILE':            None,            # path to z-score CSV; required for 'csv' mode
    'DECODER_N1_CSV_SAMPLING_RATE':   100,             # Hz — TP column sampling rate
    'DECODER_N1_CSV_ANALYSIS_ID':     None,            # filter CSV by AnalysisID; None = all
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


# ── CSV helpers ───────────────────────────────────────────────────────────────

def _unit_to_subject_date(unit_str):
    """
    Extract subject_date key from a Unit string.

    e.g. 'SUBJ-ID-1036_250830_concat_cluster200' → 'SUBJ-ID-1036_250830'
    Matches the subject_date format produced by _parse_subject_dates.
    """
    from re import split as re_split
    parts = re_split('_*_', unit_str)
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


def _load_csv_session_future(subject_date, bin_cuts, csv_df, settings):
    """
    Load a session from a pre-computed z-score CSV for the future-trial decoder.

    Replicates the trial selection logic of the JSON path:
      - ShockFlag==1, (Hit or Miss) trials
      - First ShockFlag==0 (Hit or Miss) trial immediately after each ShockFlag==1 trial

    Each bin_cut maps to one TP column:
        tp_idx = round((bin_cuts[i] - START_TIME) * CSV_SAMPLING_RATE)
        column  = 'TP.{tp_idx + 1}'

    Returns
    -------
    session_arr : np.ndarray (n_units, n_trials, n_bins) or None
    outcomes    : np.ndarray (n_trials,)  0=Hit, 1=Miss
    trial_ids   : np.ndarray (n_trials,)
    shock_flags : np.ndarray (n_trials,)
    """
    prefix        = 'DECODER_N1'
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
        return None, None, None, None

    if units_df is not None:
        sess_df = sess_df[sess_df['Unit'].isin(units_df['Unit'].values)].copy()
    if len(sess_df) == 0:
        return None, None, None, None

    # Build trial selection using the first unit (all units share the same trials)
    units   = sorted(sess_df['Unit'].unique())
    ref_df  = (sess_df[sess_df['Unit'] == units[0]]
               .sort_values('TrialID')
               .reset_index(drop=True))

    # ── Replicate JSON future-trial selection logic ────────────────────────
    # Include ShockFlag==1 (Hit or Miss) trials + first ShockFlag==0 trial
    # immediately following each ShockFlag==1 trial.
    shock_trial_ids   = set(ref_df.loc[ref_df['ShockFlag'] == 1, 'TrialID'])
    all_trial_ids_arr = ref_df['TrialID'].values
    sub_mask          = np.zeros(len(ref_df), dtype=bool)
    for i, tid in enumerate(all_trial_ids_arr):
        if (i > 0
                and all_trial_ids_arr[i - 1] in shock_trial_ids
                and ref_df.iloc[i]['ShockFlag'] == 0):
            sub_mask[i] = True

    keep = (
        ((ref_df['ShockFlag'] == 1) & ((ref_df['Hit'] == 1) | (ref_df['Miss'] == 1)))
        | (sub_mask & ((ref_df['Hit'] == 1) | (ref_df['Miss'] == 1)))
    )
    subset_ref = ref_df[keep].reset_index(drop=True)
    if len(subset_ref) == 0:
        return None, None, None, None

    trial_ids   = subset_ref['TrialID'].values
    shock_flags = subset_ref['ShockFlag'].values
    outcomes    = np.zeros(len(subset_ref))
    outcomes[subset_ref['Miss'].values == 1] = 1

    # Build (n_units, n_trials, n_bins)
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
        return None, None, None, None

    session_arr = np.stack(unit_arrays, axis=0)  # (n_units, n_trials, n_bins)
    return session_arr, outcomes, trial_ids, shock_flags


# ── Session preprocessor ──────────────────────────────────────────────────────

def _preprocess_session(subject_date, filtered_files, bin_cuts, settings):
    """
    Load and bin all units belonging to subject_date.

    Trial selection includes ShockFlag==1 trials (Hit or Miss) AND the first
    ShockFlag==0 trial immediately following a ShockFlag==1 trial (Hit or Miss),
    sorted by TrialID regardless of AMdepth.

    Returns
    -------
    session_spikes : np.ndarray (n_units, n_trials, n_bins)  or None
    outcomes       : np.ndarray (n_trials,)                  or None
        outcomes encoding: 0 = Hit, 1 = Miss
    trial_ids      : np.ndarray (n_trials,)                  or None
    """
    spikes_fieldname  = settings['DECODER_N1_SPIKES_FIELDNAME']
    session_type      = settings['DECODER_N1_SESSION']
    spikeTime_format  = settings['DECODER_N1_SPIKE_TIME_FORMAT']
    BIN_SIZE          = settings['DECODER_N1_BIN_SIZE']
    psth_step_size    = settings['DECODER_N1_PSTH_STEP_SIZE']
    sigma             = settings['DECODER_N1_GAUSSIAN_SIGMA']
    shock             = settings['DECODER_N1_SHOCK_ARTIFACT']
    units_df          = settings.get('_UNITS_DF')          # pre-loaded DataFrame or None
    feature_mode      = settings.get('DECODER_N1_FEATURE_MODE', 'full_raster')

    # ── CSV mode ──────────────────────────────────────────────────────────────
    if feature_mode == 'csv':
        csv_df = settings.get('_CSV_DF')
        if csv_df is None:
            raise ValueError(
                'DECODER_N1_FEATURE_MODE="csv" requires DECODER_N1_CSV_FILE to be set.')
        return _load_csv_session_future(subject_date, bin_cuts, csv_df, settings)

    to_remove_mask = (
        ((bin_cuts >= shock[0]) & (bin_cuts < shock[1])) if shock is not None
        else np.zeros(len(bin_cuts), dtype=bool)
    )

    cur_jsons = [f for f in filtered_files if subject_date in f]
    session_spikes, outcomes = [], []

    trial_ids = None  # will be set from the first unit

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

        # ── Trial selection (modified) ────────────────────────────────────────
        # Sort non-Reminder trials by TrialID regardless of AMdepth
        cur_data_sorted = cur_data[cur_data['Reminder'] == 0].sort_values('TrialID')

        # Find ShockFlag==0 trials immediately following a ShockFlag==1 trial
        shock_trial_ids = set(cur_data_sorted[cur_data_sorted['ShockFlag'] == 1]['TrialID'])
        all_trial_ids   = cur_data_sorted['TrialID'].values
        subthreshold_mask = np.zeros(len(cur_data_sorted), dtype=bool)
        for i, tid in enumerate(all_trial_ids):
            if i > 0 and all_trial_ids[i-1] in shock_trial_ids and cur_data_sorted.iloc[i]['ShockFlag'] == 0:
                subthreshold_mask[i] = True

        subset_data = cur_data_sorted[
            (
                (cur_data_sorted['ShockFlag'] == 1) &
                ((cur_data_sorted['Hit'] == 1) | (cur_data_sorted['Miss'] == 1))
            ) |
            (
                subthreshold_mask &
                ((cur_data_sorted['Hit'] == 1) | (cur_data_sorted['Miss'] == 1))
                # ShockFlag==0 Misses produce no Response_spikes (zero raster),
                # but are valid N+1 targets and meaningful N states
            )
        ].reset_index(drop=True)

        temp_outcomes = np.zeros(len(subset_data))
        temp_outcomes[subset_data['Miss'].values == 1] = 1
        outcomes.append(temp_outcomes)

        # Store TrialIDs and ShockFlags for sequential pairing (set once from first unit)
        if trial_ids is None:
            trial_ids   = subset_data['TrialID'].values
            shock_flags = subset_data['ShockFlag'].values
        # ─────────────────────────────────────────────────────────────────────

        allTrial_spikes = np.zeros((len(subset_data), len(bin_cuts)))
        for row_idx, spikes in enumerate(subset_data[spikes_fieldname].to_numpy()):
            spikes = np.array(spikes)
            if spikeTime_format:
                spikes = _sliding_hist(spikes, bin_cuts, psth_step_size)
            if sigma > 0:
                spikes = _gaussian_smooth(spikes, sigma, BIN_SIZE)
            if shock is not None:
                spikes[to_remove_mask] = 0
            allTrial_spikes[row_idx] = spikes

        session_spikes.append(allTrial_spikes)

    if not outcomes:
        return None, None, None, None

    reference_outcomes = np.asarray(outcomes[0])
    if not all(np.array_equal(reference_outcomes, np.asarray(o)) for o in outcomes[1:]):
        print(f'Outcome mismatch across units in {subject_date}')
        return None, None, None, None

    # Only need trial_ids and shock_flags from first unit; outcomes already verified to match
    return np.stack(session_spikes, axis=0), reference_outcomes, trial_ids, shock_flags  # (units, trials, bins)


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


def _decode_session(session_spikes, outcomes, bin_cuts, settings):
    """
    Decode trial outcomes from the full population raster within the
    configured epoch window.

    Returns
    -------
    accuracy : float  – mean balanced accuracy across CV folds
    n_units  : int
    """
    epoch_start  = settings['DECODER_N1_EPOCH_START']
    epoch_end    = settings['DECODER_N1_EPOCH_END']
    method       = settings['DECODER_N1_METHOD']
    n_splits     = settings['DECODER_N1_N_CV_SPLITS']
    random_state = settings['DECODER_N1_RANDOM_STATE']

    epoch_mask   = (bin_cuts >= epoch_start) & (bin_cuts < epoch_end)
    epoch_spikes = session_spikes[:, :, epoch_mask]           # (units, trials, epoch_bins)

    n_units, n_trials, _ = epoch_spikes.shape
    feature_mode = settings.get('DECODER_N1_FEATURE_MODE', 'full_raster')
    if feature_mode == 'mean_rate':
        X_data = epoch_spikes.mean(axis=2).T                  # (trials, units)
    else:
        # 'full_raster', 'csv' (CSV array is already bin-aligned so epoch_mask works),
        # or unknown → fall back to full_raster
        X_data = epoch_spikes.transpose(1, 0, 2).reshape(n_trials, -1)  # (trials, units*bins)

    valid  = ~np.isnan(X_data).any(axis=0)
    X_data = X_data[:, valid]
    if X_data.shape[1] == 0:
        return np.nan, n_units

    y_data = LabelEncoder().fit_transform(outcomes)

    # Class count check
    classes, counts = np.unique(y_data, return_counts=True)
    min_count = counts.min()
    print(f"  Class counts: { {int(c): int(n) for c, n in zip(classes, counts)} }")
    if min_count < settings['DECODER_N1_N_CV_SPLITS']:
        print(f"  WARNING: min class count ({min_count}) < n_splits "
              f"({settings['DECODER_N1_N_CV_SPLITS']}). Skipping session.")
        return np.nan, n_units

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    fold_scores = []
    try:
        for train_idx, test_idx in cv.split(X_data, y_data):
            X_train, X_test = X_data[train_idx], X_data[test_idx]
            y_train, y_test = y_data[train_idx], y_data[test_idx]

            sc      = StandardScaler()
            X_train = sc.fit_transform(X_train)
            X_test  = sc.transform(X_test)

            y_pred = _fit_predict(method, X_train, y_train, X_test)
            if y_pred is None:
                continue
            fold_scores.append(balanced_accuracy_score(y_test, y_pred))

    except ValueError:
        pass

    accuracy = float(np.mean(fold_scores)) if fold_scores else np.nan
    return accuracy, n_units


def _decode_session_decision_variance(session_spikes, outcomes, bin_cuts, settings):
    """
    Compute cross-validated LDA decision value variance per trial type.

    Uses the same atemporal trial selection (ShockFlag==1 Hit vs Miss) but
    instead of accuracy, collects decision_function values from held-out test
    folds and reports their variance split by Hit and Miss.

    Cross-validated decision values are used so the variance estimate is not
    inflated by in-sample projection. Low Hit variance = stereotyped Hit
    responses (all trials project similarly on the discrimination axis).

    Returns
    -------
    var_hit  : float   variance of test-fold decision values on Hit trials
    var_miss : float   variance of test-fold decision values on Miss trials
    var_all  : float   variance across all test-fold decision values
    n_units  : int
    """
    epoch_start  = settings['DECODER_N1_EPOCH_START']
    epoch_end    = settings['DECODER_N1_EPOCH_END']
    n_splits     = settings['DECODER_N1_N_CV_SPLITS']
    random_state = settings['DECODER_N1_RANDOM_STATE']

    epoch_mask   = (bin_cuts >= epoch_start) & (bin_cuts < epoch_end)
    epoch_spikes = session_spikes[:, :, epoch_mask]

    n_units, n_trials, _ = epoch_spikes.shape
    X_data = epoch_spikes.transpose(1, 0, 2).reshape(n_trials, -1)

    valid  = ~np.isnan(X_data).any(axis=0)
    X_data = X_data[:, valid]
    if X_data.shape[1] == 0:
        return np.nan, np.nan, np.nan, n_units

    y_data = LabelEncoder().fit_transform(outcomes)

    classes, counts = np.unique(y_data, return_counts=True)
    if counts.min() < n_splits:
        return np.nan, np.nan, np.nan, n_units

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    dec_hit, dec_miss = [], []

    try:
        for train_idx, test_idx in cv.split(X_data, y_data):
            X_train, X_test = X_data[train_idx], X_data[test_idx]
            y_train, y_test = y_data[train_idx], y_data[test_idx]

            sc      = StandardScaler()
            X_train = sc.fit_transform(X_train)
            X_test  = sc.transform(X_test)

            clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
            clf.fit(X_train, y_train)

            dec = clf.decision_function(X_test)   # shape (n_test,) for binary LDA
            dec_hit.extend(dec[y_test == 0].tolist())   # 0 = Hit
            dec_miss.extend(dec[y_test == 1].tolist())  # 1 = Miss

    except ValueError:
        pass

    if not dec_hit and not dec_miss:
        return np.nan, np.nan, np.nan, np.nan, n_units

    all_dec  = dec_hit + dec_miss
    var_hit  = float(np.var(dec_hit,  ddof=1)) if len(dec_hit)  > 1 else np.nan
    var_miss = float(np.var(dec_miss, ddof=1)) if len(dec_miss) > 1 else np.nan
    var_all  = float(np.var(all_dec,  ddof=1)) if len(all_dec)  > 1 else np.nan

    # d' on the decision axis: mean separation / pooled within-class SD
    if not np.isnan(var_hit) and not np.isnan(var_miss) and (var_hit + var_miss) > 0:
        dprime = float(abs(np.mean(dec_hit) - np.mean(dec_miss)) /
                       np.sqrt((var_hit + var_miss) / 2))
    else:
        dprime = np.nan

    return var_hit, var_miss, var_all, dprime, n_units


def _decode_session_decision_correlation(session_spikes, outcomes, trial_ids, shock_flags, bin_cuts, settings):
    """
    For each N→N+1 pair where trial N is ShockFlag==1, compute the
    point-biserial correlation between trial N's CV decision value and
    trial N+1's outcome (0=Hit, 1=Miss).

    Reported separately for all pairs, Hit-N pairs, and Miss-N pairs.

    This directly tests whether the spread in trial-N neural responses is
    *structured* with respect to the next trial — i.e., whether a stronger
    or weaker Hit response actually predicts what comes next. High variance
    with zero correlation = random noise; the N→N+1 decoder would fail.

    Decision values are cross-validated (each trial projected only when it
    was in a held-out test fold) to avoid in-sample inflation.

    Returns
    -------
    corr_all    : float   Pearson r, all N→N+1 pairs
    corr_hit_n  : float   Pearson r, Hit-N pairs only
    corr_miss_n : float   Pearson r, Miss-N pairs only
    n_units     : int
    """
    epoch_start  = settings['DECODER_N1_EPOCH_START']
    epoch_end    = settings['DECODER_N1_EPOCH_END']
    n_splits     = settings['DECODER_N1_N_CV_SPLITS']
    random_state = settings['DECODER_N1_RANDOM_STATE']

    epoch_mask   = (bin_cuts >= epoch_start) & (bin_cuts < epoch_end)
    epoch_spikes = session_spikes[:, :, epoch_mask]

    n_units, n_trials, _ = epoch_spikes.shape
    X_data = epoch_spikes.transpose(1, 0, 2).reshape(n_trials, -1)

    valid  = ~np.isnan(X_data).any(axis=0)
    X_data = X_data[:, valid]
    if X_data.shape[1] == 0:
        return np.nan, np.nan, np.nan, n_units

    y_data = LabelEncoder().fit_transform(outcomes)

    classes, counts = np.unique(y_data, return_counts=True)
    if counts.min() < n_splits:
        return np.nan, np.nan, np.nan, n_units

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    # Assign each trial its decision value from the fold where it was held out
    dec_values = np.full(n_trials, np.nan)
    try:
        for train_idx, test_idx in cv.split(X_data, y_data):
            X_train, X_test = X_data[train_idx], X_data[test_idx]
            y_train         = y_data[train_idx]

            sc      = StandardScaler()
            X_train = sc.fit_transform(X_train)
            X_test  = sc.transform(X_test)

            clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
            clf.fit(X_train, y_train)
            dec_values[test_idx] = clf.decision_function(X_test)

    except ValueError:
        pass

    if np.all(np.isnan(dec_values)):
        return np.nan, np.nan, np.nan, n_units

    # Build N→N+1 pairs sorted by TrialID
    sort_idx   = np.argsort(trial_ids)
    dec_sorted = dec_values[sort_idx]
    y_sorted   = y_data[sort_idx]
    sf_sorted  = shock_flags[sort_idx]

    valid_pair_mask = sf_sorted[:-1] == 1   # trial N must be ShockFlag==1

    dec_N = dec_sorted[:-1][valid_pair_mask]
    y_N   = y_sorted[:-1][valid_pair_mask]  # 0=Hit, 1=Miss on trial N
    y_N1  = y_sorted[1:][valid_pair_mask]   # trial N+1 outcome

    # Drop any pairs where trial N had no CV decision value
    keep   = ~np.isnan(dec_N)
    dec_N, y_N, y_N1 = dec_N[keep], y_N[keep], y_N1[keep]

    def _corr(dec, outcome):
        if len(dec) < 3 or len(np.unique(outcome)) < 2:
            return np.nan
        return float(np.corrcoef(dec, outcome)[0, 1])

    corr_all    = _corr(dec_N,              y_N1)
    corr_hit_n  = _corr(dec_N[y_N == 0],   y_N1[y_N == 0])
    corr_miss_n = _corr(dec_N[y_N == 1],   y_N1[y_N == 1])

    return corr_all, corr_hit_n, corr_miss_n, n_units


def _decode_session_behavioral_baseline(outcomes, trial_ids, shock_flags):
    """
    Behavioral autocorrelation baseline for the N→N+1 decoder.

    Uses trial N's behavioral outcome alone (no neural data) to predict
    trial N+1 via the win-stay rule: predict the same outcome as trial N.

    Balanced accuracy above 50% means behavioral serial dependence exists.
    Comparing against the neural sequential decoder isolates the neural
    contribution beyond what behavior alone predicts.

    Returns
    -------
    acc_all    : float  balanced accuracy, all N→N+1 pairs
    acc_hit_n  : float  balanced accuracy, Hit-N pairs only  (win-stay = predict Hit)
    acc_miss_n : float  balanced accuracy, Miss-N pairs only (win-stay = predict Miss)
    """
    y_data   = LabelEncoder().fit_transform(outcomes)
    sort_idx = np.argsort(trial_ids)
    y_sorted = y_data[sort_idx]
    sf_sorted = shock_flags[sort_idx]

    valid_pair_mask = sf_sorted[:-1] == 1
    y_N  = y_sorted[:-1][valid_pair_mask]
    y_N1 = y_sorted[1:][valid_pair_mask]

    if len(y_N) == 0:
        return np.nan, np.nan, np.nan

    # Win-stay: predict trial N+1 outcome = trial N outcome
    acc_all = float(balanced_accuracy_score(y_N1, y_N))

    hit_mask  = y_N == 0
    miss_mask = y_N == 1

    # Hit-N: predict Hit; all predictions are 0, so balanced accuracy =
    # P(y_N1=Hit | y_N=Hit) if there are both classes in y_N1, else nan
    def _bac(y_true, y_pred):
        if len(y_true) == 0 or len(np.unique(y_true)) < 2:
            return np.nan
        return float(balanced_accuracy_score(y_true, y_pred))

    acc_hit_n  = _bac(y_N1[hit_mask],  y_N[hit_mask])
    acc_miss_n = _bac(y_N1[miss_mask], y_N[miss_mask])

    return acc_all, acc_hit_n, acc_miss_n


def _decode_session_sequential(session_spikes, outcomes, trial_ids, shock_flags, bin_cuts, settings):
    """
    Two-stage decoder: trial N neural state → predict trial N+1 outcome.
    Stage 1: LDA on trial N raster → decision value + predicted label
    Stage 2: LDA on (decision_value_N, predicted_label_N) → outcome_N+1

    Only pairs where trial N is ShockFlag==1 are used. ShockFlag==0 trials
    may appear as N+1 targets but never as N predictors, because intervening
    trials not in the dataset exist between a subthreshold trial and the next
    shocked trial.
    """
    epoch_start  = settings['DECODER_N1_EPOCH_START']
    epoch_end    = settings['DECODER_N1_EPOCH_END']
    method       = settings['DECODER_N1_METHOD']
    n_splits     = settings['DECODER_N1_N_CV_SPLITS']
    random_state = settings['DECODER_N1_RANDOM_STATE']

    epoch_mask   = (bin_cuts >= epoch_start) & (bin_cuts < epoch_end)
    epoch_spikes = session_spikes[:, :, epoch_mask]

    n_units, n_trials, _ = epoch_spikes.shape
    X_data = epoch_spikes.transpose(1, 0, 2).reshape(n_trials, -1)

    valid  = ~np.isnan(X_data).any(axis=0)
    X_data = X_data[:, valid]
    if X_data.shape[1] == 0:
        return np.nan, n_units

    y_data = LabelEncoder().fit_transform(outcomes)

    # Build trial N → N+1 pairs using TrialID order
    sort_idx   = np.argsort(trial_ids)
    X_sorted   = X_data[sort_idx]
    y_sorted   = y_data[sort_idx]
    sf_sorted  = shock_flags[sort_idx]

    # Only allow pairs where trial N is ShockFlag==1.
    # ShockFlag==0 trials are excluded as N predictors because consecutive
    # dataset entries are not actually adjacent trials in the experiment.
    valid_pair_mask = sf_sorted[:-1] == 1

    X_N  = X_sorted[:-1][valid_pair_mask]   # trial N features
    y_N  = y_sorted[:-1][valid_pair_mask]   # trial N outcome (stage 1 training)
    y_N1 = y_sorted[1:][valid_pair_mask]    # trial N+1 outcome (stage 2 target)

    classes, counts = np.unique(y_N1, return_counts=True)
    if counts.min() < n_splits:
        print(f"  WARNING: insufficient N+1 class counts {dict(zip(classes, counts))}. Skipping.")
        return np.nan, n_units

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_scores = []

    try:
        for train_idx, test_idx in cv.split(X_N, y_N1):
            # Stage 1: fit on trial N neural data, extract decision values
            X_train_N, X_test_N = X_N[train_idx], X_N[test_idx]
            y_train_N           = y_N[train_idx]

            sc        = StandardScaler()
            X_train_N = sc.fit_transform(X_train_N)
            X_test_N  = sc.transform(X_test_N)

            clf1 = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
            clf1.fit(X_train_N, y_train_N)

            # Decision value + predicted label as stage 2 features
            dec_train = clf1.decision_function(X_train_N).reshape(-1, 1)
            lbl_train = clf1.predict(X_train_N).reshape(-1, 1)
            dec_test  = clf1.decision_function(X_test_N).reshape(-1, 1)
            lbl_test  = clf1.predict(X_test_N).reshape(-1, 1)

            X2_train = np.hstack([dec_train, lbl_train])
            X2_test  = np.hstack([dec_test,  lbl_test])

            # Stage 2: predict N+1 outcome
            y_train_N1 = y_N1[train_idx]
            y_test_N1  = y_N1[test_idx]

            clf2 = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
            clf2.fit(X2_train, y_train_N1)
            y_pred = clf2.predict(X2_test)

            fold_scores.append(balanced_accuracy_score(y_test_N1, y_pred))

    except ValueError:
        pass

    accuracy = float(np.mean(fold_scores)) if fold_scores else np.nan
    return accuracy, n_units


def _decode_session_sequential_direct(session_spikes, outcomes, trial_ids, shock_flags, bin_cuts, settings):
    """
    Single-stage decoder: trial N neural state → predict trial N+1 outcome directly.
    Simpler alternative to the two-stage decoder with no information bottleneck.
    Uses the same valid-pair mask (trial N must be ShockFlag==1).
    """
    epoch_start  = settings['DECODER_N1_EPOCH_START']
    epoch_end    = settings['DECODER_N1_EPOCH_END']
    n_splits     = settings['DECODER_N1_N_CV_SPLITS']
    random_state = settings['DECODER_N1_RANDOM_STATE']

    epoch_mask   = (bin_cuts >= epoch_start) & (bin_cuts < epoch_end)
    epoch_spikes = session_spikes[:, :, epoch_mask]

    n_units, n_trials, _ = epoch_spikes.shape
    X_data = epoch_spikes.transpose(1, 0, 2).reshape(n_trials, -1)

    valid  = ~np.isnan(X_data).any(axis=0)
    X_data = X_data[:, valid]
    if X_data.shape[1] == 0:
        return np.nan, n_units

    y_data = LabelEncoder().fit_transform(outcomes)

    sort_idx  = np.argsort(trial_ids)
    X_sorted  = X_data[sort_idx]
    y_sorted  = y_data[sort_idx]
    sf_sorted = shock_flags[sort_idx]

    valid_pair_mask = sf_sorted[:-1] == 1

    X_N  = X_sorted[:-1][valid_pair_mask]   # trial N features
    y_N1 = y_sorted[1:][valid_pair_mask]    # trial N+1 outcome (direct target)

    classes, counts = np.unique(y_N1, return_counts=True)
    if counts.min() < n_splits:
        print(f"  WARNING: insufficient N+1 class counts for direct decoder "
              f"{dict(zip(classes, counts))}. Skipping.")
        return np.nan, n_units

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_scores = []

    try:
        for train_idx, test_idx in cv.split(X_N, y_N1):
            X_train, X_test = X_N[train_idx], X_N[test_idx]
            y_train, y_test = y_N1[train_idx], y_N1[test_idx]

            sc      = StandardScaler()
            X_train = sc.fit_transform(X_train)
            X_test  = sc.transform(X_test)

            clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            fold_scores.append(balanced_accuracy_score(y_test, y_pred))

    except ValueError:
        pass

    accuracy = float(np.mean(fold_scores)) if fold_scores else np.nan
    return accuracy, n_units


def _build_balanced_pipeline(method, random_state):
    """
    Build a Pipeline-compatible estimator for the threshold transition decoder.
    All variants apply class-imbalance handling appropriate to the method:
      lda      → equal priors (priors=[0.5, 0.5])
      lda_svc  → LDA transform + LinearSVC(class_weight='balanced')
      svc      → SVC(rbf, class_weight='balanced')
      default  → LinearSVC(class_weight='balanced')
    """
    if method == 'lda':
        return Pipeline([
            ('scaler', StandardScaler()),
            ('clf',    LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto',
                                                  priors=[0.5, 0.5])),
        ])
    elif method == 'lda_svc':
        return Pipeline([
            ('scaler', StandardScaler()),
            ('lda',    LinearDiscriminantAnalysis()),
            ('clf',    LinearSVC(C=0.5, class_weight='balanced', dual='auto',
                                 random_state=random_state)),
        ])
    elif method == 'svc':
        return Pipeline([
            ('scaler', StandardScaler()),
            ('clf',    LinearSVC(C=0.5, class_weight='balanced', dual='auto',
                                 random_state=random_state)),
        ])
    else:  # 'ovo' or unknown — fall back to LinearSVC balanced
        return Pipeline([
            ('scaler', StandardScaler()),
            ('clf',    LinearSVC(C=0.5, class_weight='balanced', dual='auto',
                                 random_state=random_state)),
        ])


def _decode_session_threshold_transition(session_spikes, outcomes, trial_ids, shock_flags, bin_cuts, settings):
    """
    Threshold-transition decoder: ShockFlag==1 → ShockFlag==0 pairs only.

    Asks: does trial N neural activity predict whether the animal will detect
    the AM stimulus just below current threshold on trial N+1?

    Respects DECODER_N1_METHOD but applies class-imbalance handling appropriate to
    each method (see _build_balanced_pipeline). A permutation test
    (DECODER_N1_N_PERMUTATIONS shuffles of y_N1) provides a session-level p-value.

    Returns
    -------
    accuracy : float   balanced accuracy (0–1)
    p_value  : float   permutation test p-value
    n_units  : int
    """
    epoch_start    = settings['DECODER_N1_EPOCH_START']
    epoch_end      = settings['DECODER_N1_EPOCH_END']
    n_splits       = settings['DECODER_N1_N_CV_SPLITS']
    random_state   = settings['DECODER_N1_RANDOM_STATE']
    method         = settings.get('DECODER_N1_METHOD', 'lda')
    min_pairs      = settings.get('DECODER_N1_MIN_PAIRS_STRATIFIED', 10)
    n_permutations = settings.get('DECODER_N1_N_PERMUTATIONS', 1000)

    epoch_mask   = (bin_cuts >= epoch_start) & (bin_cuts < epoch_end)
    epoch_spikes = session_spikes[:, :, epoch_mask]

    n_units, n_trials, _ = epoch_spikes.shape
    X_data = epoch_spikes.transpose(1, 0, 2).reshape(n_trials, -1)

    valid  = ~np.isnan(X_data).any(axis=0)
    X_data = X_data[:, valid]
    if X_data.shape[1] == 0:
        return np.nan, np.nan, n_units

    y_data = LabelEncoder().fit_transform(outcomes)

    sort_idx  = np.argsort(trial_ids)
    X_sorted  = X_data[sort_idx]
    y_sorted  = y_data[sort_idx]
    sf_sorted = shock_flags[sort_idx]

    # Restrict to ShockFlag==1 → ShockFlag==0 transitions only
    transition_mask = (sf_sorted[:-1] == 1) & (sf_sorted[1:] == 0)

    X_N  = X_sorted[:-1][transition_mask]
    y_N1 = y_sorted[1:][transition_mask]

    classes, counts = np.unique(y_N1, return_counts=True)
    if len(X_N) < min_pairs or len(counts) < 2 or counts.min() < n_splits:
        print(f"  WARNING [threshold transition]: {len(X_N)} pairs, "
              f"N+1 class counts {dict(zip(classes, counts))} — skipping.")
        return np.nan, np.nan, n_units

    cv   = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    pipe = _build_balanced_pipeline(method, random_state)

    try:
        score, _, p_value = permutation_test_score(
            pipe, X_N, y_N1,
            cv=cv,
            n_permutations=n_permutations,
            scoring='balanced_accuracy',
            random_state=random_state,
        )
        accuracy = float(score)
    except (ValueError, IndexError):
        accuracy, p_value = np.nan, np.nan

    return accuracy, float(p_value), n_units


def _decode_session_sequential_by_ntype(session_spikes, outcomes, trial_ids, shock_flags, bin_cuts, settings):
    """
    Run the direct (single-stage) sequential decoder separately for
    Hit-N pairs (trial N was a Hit) and Miss-N pairs (trial N was a Miss).

    The two-stage decoder is not run here because stage 1 requires both
    Hit and Miss labels in trial N — which is undefined when the subset
    is restricted to one trial type.

    Sessions where a subset has fewer than DECODER_N1_MIN_PAIRS_STRATIFIED valid pairs
    OR fewer than DECODER_N1_N_CV_SPLITS samples in the minority N+1 class are returned
    as np.nan for that subset.

    Returns
    -------
    acc_direct_hit_n     : float
    acc_direct_miss_n    : float
    n_units              : int
    """
    epoch_start  = settings['DECODER_N1_EPOCH_START']
    epoch_end    = settings['DECODER_N1_EPOCH_END']
    n_splits     = settings['DECODER_N1_N_CV_SPLITS']
    random_state = settings['DECODER_N1_RANDOM_STATE']
    min_pairs    = settings.get('DECODER_N1_MIN_PAIRS_STRATIFIED', 10)

    epoch_mask   = (bin_cuts >= epoch_start) & (bin_cuts < epoch_end)
    epoch_spikes = session_spikes[:, :, epoch_mask]

    n_units, n_trials, _ = epoch_spikes.shape
    X_data = epoch_spikes.transpose(1, 0, 2).reshape(n_trials, -1)

    valid  = ~np.isnan(X_data).any(axis=0)
    X_data = X_data[:, valid]
    if X_data.shape[1] == 0:
        return np.nan, np.nan, n_units

    y_data = LabelEncoder().fit_transform(outcomes)

    sort_idx  = np.argsort(trial_ids)
    X_sorted  = X_data[sort_idx]
    y_sorted  = y_data[sort_idx]
    sf_sorted = shock_flags[sort_idx]

    valid_pair_mask = sf_sorted[:-1] == 1
    X_N  = X_sorted[:-1][valid_pair_mask]
    y_N  = y_sorted[:-1][valid_pair_mask]   # 0=Hit, 1=Miss on trial N
    y_N1 = y_sorted[1:][valid_pair_mask]

    def _run_direct(X_sub, y_N1_sub, label):
        classes, counts = np.unique(y_N1_sub, return_counts=True)
        if len(X_sub) < min_pairs or counts.min() < n_splits:
            print(f"    [{label}] insufficient pairs ({len(X_sub)}) or N+1 class counts "
                  f"{dict(zip(classes, counts))} — skipping.")
            return np.nan
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        fold_scores = []
        try:
            for train_idx, test_idx in cv.split(X_sub, y_N1_sub):
                X_train, X_test = X_sub[train_idx], X_sub[test_idx]
                y_train, y_test = y_N1_sub[train_idx], y_N1_sub[test_idx]
                sc = StandardScaler()
                X_train = sc.fit_transform(X_train)
                X_test  = sc.transform(X_test)
                clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
                clf.fit(X_train, y_train)
                fold_scores.append(balanced_accuracy_score(y_test, clf.predict(X_test)))
        except ValueError:
            pass
        return float(np.mean(fold_scores)) if fold_scores else np.nan

    hit_mask  = y_N == 0
    miss_mask = y_N == 1

    acc_direct_hit_n  = _run_direct(X_N[hit_mask],  y_N1[hit_mask],  'direct Hit-N')
    acc_direct_miss_n = _run_direct(X_N[miss_mask], y_N1[miss_mask], 'direct Miss-N')

    return acc_direct_hit_n, acc_direct_miss_n, n_units


# ── File-list helpers ─────────────────────────────────────────────────────────

def _parse_subject_dates(filtered_files, subjects_to_run, sessions_to_exclude):
    """Return sorted list of unique subject_date strings to process."""
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
    """Return (worst_day_sessions, late_day_sessions) as sets of subject_date strings."""
    early_df = threshold_df[threshold_df['Day'] == 1]
    early    = set('_'.join([subj, split('-*-', sid)[0]])
                   for subj, sid in zip(early_df['Subject'], early_df['Session']))

    late_df  = threshold_df[threshold_df['Day'].isin([10, 9, 8])]
    late     = set('_'.join([subj, split('-*-', sid)[0]])
                   for subj, sid in zip(late_df['Subject'], late_df['Session']))

    return early, late


# ── Output helpers ────────────────────────────────────────────────────────────

def _save_csv(results, settings):
    """Write one row per session summarizing all the N+1 decoding analyses to a CSV.

    Covers the standard decode, the sequential/direct-sequential variants,
    the threshold-transition test, decision-variance/correlation stats, and
    the behavioral-accuracy baseline, in one combined row per session — see
    the ``_decode_session_*`` functions above for how each field is computed.

    Args:
        results (dict): Maps subject_date -> per-session result dict with
            all the fields referenced below (accuracy/accuracy_seq/
            accuracy_seq_direct/acc_direct_threshold/threshold_pval/
            acc_direct_hit_n/acc_direct_miss_n/dec_var_*/dec_corr_*/
            dec_dprime/beh_acc_*/n_units/day_of_training/day_type).
        settings (dict): Pipeline settings; uses ``DECODER_N1_OUTPUT_FOLDER``,
            ``DECODER_N1_FILE_NAME_TAG``, ``DECODER_N1_METHOD``,
            ``DECODER_N1_EPOCH_START``/``DECODER_N1_EPOCH_END``.

    Returns:
        None. Writes ``LDAoutput_<file_name_tag>_<method>.csv`` under
        ``DECODER_N1_OUTPUT_FOLDER`` and prints its path.
    """
    output_folder = settings['DECODER_N1_OUTPUT_FOLDER']
    file_name_tag = settings['DECODER_N1_FILE_NAME_TAG']
    method        = settings['DECODER_N1_METHOD']
    epoch_start   = settings['DECODER_N1_EPOCH_START']
    epoch_end     = settings['DECODER_N1_EPOCH_END']

    csv_path = sep.join([output_folder, 'LDAoutput_' + file_name_tag + '_' + method + '.csv'])
    with open(csv_path, 'w', newline='', encoding='utf-8') as fh:
        writer = csv.writer(fh)
        writer.writerow(['Session', 'Method', 'Label_encoding',
                         'Epoch_start', 'Epoch_end',
                         'Accuracy', 'Accuracy_seq', 'Accuracy_seq_direct',
                         'Acc_direct_threshold', 'Threshold_pval',
                         'Acc_direct_hit_n', 'Acc_direct_miss_n',
                         'Dec_var_hit', 'Dec_var_miss', 'Dec_var_all', 'Dec_dprime',
                         'Dec_corr_all', 'Dec_corr_hit_n', 'Dec_corr_miss_n',
                         'Beh_acc_all', 'Beh_acc_hit_n', 'Beh_acc_miss_n',
                         'Unit_count', 'Day_of_training', 'Training_stage'])

        def _r(v):  return round(v * 100, 2) if not np.isnan(v) else np.nan
        def _rp(v): return round(v, 4)        if not np.isnan(v) else np.nan
        def _rv(v): return round(v, 6)        if not np.isnan(v) else np.nan

        for sd, r in results.items():
            writer.writerow([sd, method, 'actual',
                             epoch_start, epoch_end,
                             _r(r['accuracy']),
                             _r(r['accuracy_seq']),
                             _r(r['accuracy_seq_direct']),
                             _r(r['acc_direct_threshold']),
                             _rp(r['threshold_pval']),
                             _r(r['acc_direct_hit_n']),
                             _r(r['acc_direct_miss_n']),
                             _rv(r['dec_var_hit']),
                             _rv(r['dec_var_miss']),
                             _rv(r['dec_var_all']),
                             _rv(r['dec_dprime']),
                             _rv(r['dec_corr_all']),
                             _rv(r['dec_corr_hit_n']),
                             _rv(r['dec_corr_miss_n']),
                             _r(r['beh_acc_all']),
                             _r(r['beh_acc_hit_n']),
                             _r(r['beh_acc_miss_n']),
                             r['n_units'], r['day_of_training'], r['day_type']])
    print(f"Saved CSV → {csv_path}")


def _mean_se(values):
    """Compute the mean and standard error of the mean, ignoring NaNs.

    Args:
        values (Iterable[float]): Values to summarize.

    Returns:
        tuple[float, float]: ``(mean, sem)``, or ``(nan, nan)`` if no
        non-NaN values remain.
    """
    arr  = np.array([v for v in values if not np.isnan(v)], dtype=float)
    if len(arr) == 0:
        return np.nan, np.nan
    return np.nanmean(arr), np.nanstd(arr) / np.sqrt(len(arr))


def _bar_with_error(ax, positions, means, sems, colors, xlabels):
    """Draw a bar with an error bar for each (position, mean, sem) triple, skipping NaN means.

    Args:
        ax (matplotlib.axes.Axes): Axes to draw onto.
        positions (Iterable[float]): X positions for each bar.
        means (Iterable[float]): Bar heights.
        sems (Iterable[float]): Error-bar half-widths (standard errors).
        colors (Iterable): Bar face color per position.
        xlabels (Iterable[str]): Legend label per position.

    Returns:
        None.
    """
    for pos, mean, sem, color, label in zip(positions, means, sems, colors, xlabels):
        if np.isnan(mean):
            continue
        ax.bar(pos, mean, color=color, alpha=0.7, label=label, width=0.5)
        ax.errorbar(pos, mean, yerr=sem, fmt='none', color='black', capsize=4)


def _save_pdf(results, settings):
    """Plot bar charts (all/first/best day) for every N+1 decoding metric into one PDF.

    Mirrors ``_save_csv``'s scope: standard/sequential/direct-sequential
    accuracy, the threshold-transition test, decision-variance/correlation
    stats, and the behavioral baseline, each as its own page/panel.

    Args:
        results (dict): Maps subject_date -> per-session result dict, as in ``_save_csv``.
        settings (dict): Pipeline settings; uses ``DECODER_N1_OUTPUT_FOLDER`` and ``DECODER_N1_FILE_NAME_TAG``.

    Returns:
        None. Writes ``LDAoutput_<file_name_tag>_<method>.pdf`` under
        ``DECODER_N1_OUTPUT_FOLDER``.
    """
    def _subject_mean(results, day_type, key='accuracy', scale=100):
        """Average one metric within each subject, for one day_type, across sessions.

        Args:
            results (dict): Same per-session result dict as the enclosing function.
            day_type (str): Which day-type bucket to average within.
            key (str): Which field of each session's result dict to average.
            scale (float): Multiplier applied to each value before averaging (e.g. 100 for a percentage).

        Returns:
            list[float]: One mean value per subject that has non-NaN data for this day_type/key.
        """
        from collections import defaultdict
        by_subject = defaultdict(list)
        for sd, r in results.items():
            if r['day_type'] == day_type:
                subj = sd.split('_')[0]
                by_subject[subj].append(r[key] * scale)
        return [np.mean(v) for v in by_subject.values()]

    output_folder = settings['DECODER_N1_OUTPUT_FOLDER']
    file_name_tag = settings['DECODER_N1_FILE_NAME_TAG']
    method        = settings['DECODER_N1_METHOD']

    all_acc                  = [r['accuracy'] * 100                for r in results.values()]
    all_acc_seq              = [r['accuracy_seq'] * 100            for r in results.values() if not np.isnan(r['accuracy_seq'])]
    all_acc_seq_direct       = [r['accuracy_seq_direct'] * 100     for r in results.values() if not np.isnan(r['accuracy_seq_direct'])]
    all_acc_threshold        = [r['acc_direct_threshold'] * 100    for r in results.values() if not np.isnan(r['acc_direct_threshold'])]
    all_acc_hit_n            = [r['acc_direct_hit_n'] * 100        for r in results.values() if not np.isnan(r['acc_direct_hit_n'])]
    all_acc_miss_n           = [r['acc_direct_miss_n'] * 100       for r in results.values() if not np.isnan(r['acc_direct_miss_n'])]
    early_acc                = _subject_mean(results, 'early', 'accuracy')
    late_acc                 = _subject_mean(results, 'late',  'accuracy')
    early_acc_seq            = _subject_mean(results, 'early', 'accuracy_seq')
    late_acc_seq             = _subject_mean(results, 'late',  'accuracy_seq')
    early_acc_seq_direct     = _subject_mean(results, 'early', 'accuracy_seq_direct')
    late_acc_seq_direct      = _subject_mean(results, 'late',  'accuracy_seq_direct')
    early_acc_threshold      = _subject_mean(results, 'early', 'acc_direct_threshold')
    late_acc_threshold       = _subject_mean(results, 'late',  'acc_direct_threshold')
    early_acc_hit_n          = _subject_mean(results, 'early', 'acc_direct_hit_n')
    late_acc_hit_n           = _subject_mean(results, 'late',  'acc_direct_hit_n')
    early_acc_miss_n         = _subject_mean(results, 'early', 'acc_direct_miss_n')
    late_acc_miss_n          = _subject_mean(results, 'late',  'acc_direct_miss_n')

    pdf_path = sep.join([output_folder, 'LDAoutput_' + file_name_tag + '_' + method + '.pdf'])
    with PdfPages(pdf_path) as pdf:

        # Page 1: all sessions — atemporal vs two-stage sequential vs direct sequential
        fig, ax = plt.subplots()
        _bar_with_error(ax, [0, 1, 2],
                        [_mean_se(all_acc)[0], _mean_se(all_acc_seq)[0], _mean_se(all_acc_seq_direct)[0]],
                        [_mean_se(all_acc)[1], _mean_se(all_acc_seq)[1], _mean_se(all_acc_seq_direct)[1]],
                        ['#D4B483', '#7B9E87', '#A89BC2'],
                        ['Atemporal', 'Two-stage (N→N+1)', 'Direct (N→N+1)'])
        ax.axhline(50, color='black', linewidth=1, linestyle='--', label='Chance (50%)')
        ax.set_xticks([0, 1, 2]); ax.set_xticklabels(['Atemporal', 'Two-stage\n(N→N+1)', 'Direct\n(N→N+1)'])
        ax.set_ylabel('Decoding accuracy (%)'); ax.set_title('All training sessions')
        ax.legend(frameon=False)
        format_ax(ax); pdf.savefig(); plt.close()

        # Page 2: early vs late — atemporal
        fig, ax = plt.subplots()
        groups  = [early_acc, late_acc]
        colors  = ['#C1666B', '#4281A4']
        xlabels = ['Early', 'Late']
        _bar_with_error(ax, range(2),
                        [_mean_se(g)[0] for g in groups],
                        [_mean_se(g)[1] for g in groups],
                        colors, xlabels)
        ax.axhline(50, color='black', linewidth=1, linestyle='--', label='Chance (50%)')
        ax.set_xticks(range(2)); ax.set_xticklabels(xlabels)
        ax.set_ylabel('Decoding accuracy (%) — atemporal')
        ax.legend(handles=[patches.Patch(facecolor=c, alpha=0.7, label=l)
                            for c, l in [('#C1666B', 'Early'), ('#4281A4', 'Late')]],
                  frameon=False)
        format_ax(ax); pdf.savefig(); plt.close()

        # Page 3: early vs late — two-stage sequential
        fig, ax = plt.subplots()
        groups  = [early_acc_seq, late_acc_seq]
        _bar_with_error(ax, range(2),
                        [_mean_se(g)[0] for g in groups],
                        [_mean_se(g)[1] for g in groups],
                        colors, xlabels)
        ax.axhline(50, color='black', linewidth=1, linestyle='--', label='Chance (50%)')
        ax.set_xticks(range(2)); ax.set_xticklabels(xlabels)
        ax.set_ylabel('Decoding accuracy (%) — two-stage sequential (N→N+1)')
        ax.legend(handles=[patches.Patch(facecolor=c, alpha=0.7, label=l)
                            for c, l in [('#C1666B', 'Early'), ('#4281A4', 'Late')]],
                  frameon=False)
        format_ax(ax); pdf.savefig(); plt.close()

        # Page 4: early vs late — direct sequential
        fig, ax = plt.subplots()
        groups  = [early_acc_seq_direct, late_acc_seq_direct]
        _bar_with_error(ax, range(2),
                        [_mean_se(g)[0] for g in groups],
                        [_mean_se(g)[1] for g in groups],
                        colors, xlabels)
        ax.axhline(50, color='black', linewidth=1, linestyle='--', label='Chance (50%)')
        ax.set_xticks(range(2)); ax.set_xticklabels(xlabels)
        ax.set_ylabel('Decoding accuracy (%) — direct sequential (N→N+1)')
        ax.legend(handles=[patches.Patch(facecolor=c, alpha=0.7, label=l)
                            for c, l in [('#C1666B', 'Early'), ('#4281A4', 'Late')]],
                  frameon=False)
        format_ax(ax); pdf.savefig(); plt.close()

        # Page 5: threshold transition — early vs late + individual sessions coloured by p-value
        fig, ax = plt.subplots()
        stage_colors = ['#C1666B', '#4281A4']
        xlabels      = ['Early', 'Late']
        _bar_with_error(ax, range(2),
                        [_mean_se(early_acc_threshold)[0], _mean_se(late_acc_threshold)[0]],
                        [_mean_se(early_acc_threshold)[1], _mean_se(late_acc_threshold)[1]],
                        stage_colors, xlabels)
        # Overlay individual sessions, filled = p<0.05, open = n.s.
        stage_map = {'early': 0, 'late': 1}
        for sd, r in results.items():
            if np.isnan(r['acc_direct_threshold']):
                continue
            xpos  = stage_map.get(r['day_type'])
            if xpos is None:
                continue
            sig   = (not np.isnan(r['threshold_pval'])) and r['threshold_pval'] < 0.05
            color = stage_colors[xpos]
            ax.scatter(xpos, r['acc_direct_threshold'] * 100,
                       color=color if sig else 'none',
                       edgecolors=color, linewidths=1.2,
                       s=40, zorder=3, alpha=0.8)
        ax.axhline(50, color='black', linewidth=1, linestyle='--')
        ax.set_xticks(range(2)); ax.set_xticklabels(xlabels)
        ax.set_ylabel('Decoding accuracy (%)')
        ax.set_title('Threshold transition (ShockFlag 1→0)\nDoes trial N predict subthreshold detection?')
        # Legend: filled = p<0.05, open = n.s.
        ax.legend(handles=[
            patches.Patch(facecolor='#C1666B', alpha=0.7, label='Early'),
            patches.Patch(facecolor='#4281A4', alpha=0.7, label='Late'),
            plt.scatter([], [], color='grey', s=40, label='p < 0.05'),
            plt.scatter([], [], facecolors='none', edgecolors='grey', s=40, label='n.s.'),
        ], frameon=False, fontsize=7)
        format_ax(ax); pdf.savefig(); plt.close()

        # Page 6: Hit-N vs Miss-N direct decoder — all sessions
        fig, ax = plt.subplots()
        _bar_with_error(ax, [0, 1],
                        [_mean_se(all_acc_hit_n)[0], _mean_se(all_acc_miss_n)[0]],
                        [_mean_se(all_acc_hit_n)[1], _mean_se(all_acc_miss_n)[1]],
                        ['#5B8DB8', '#B85B5B'],
                        ['Hit-N', 'Miss-N'])
        ax.axhline(50, color='black', linewidth=1, linestyle='--', label='Chance (50%)')
        ax.set_xticks([0, 1]); ax.set_xticklabels(['Hit-N\n(direct)', 'Miss-N\n(direct)'])
        ax.set_ylabel('Decoding accuracy (%)'); ax.set_title('Direct N→N+1 by trial N type')
        ax.legend(frameon=False)
        format_ax(ax); pdf.savefig(); plt.close()

        # Page 6: Hit-N vs Miss-N by training stage
        fig, axes = plt.subplots(1, 2, sharey=True, figsize=(7, 4))
        for ax_sub, hit_groups, miss_groups, stage_label in [
            (axes[0], [early_acc_hit_n, early_acc_miss_n], None, 'Early'),
            (axes[1], [late_acc_hit_n,  late_acc_miss_n],  None, 'Late'),
        ]:
            hit_g, miss_g = (early_acc_hit_n, early_acc_miss_n) if stage_label == 'Early' \
                            else (late_acc_hit_n, late_acc_miss_n)
            _bar_with_error(ax_sub, [0, 1],
                            [_mean_se(hit_g)[0], _mean_se(miss_g)[0]],
                            [_mean_se(hit_g)[1], _mean_se(miss_g)[1]],
                            ['#5B8DB8', '#B85B5B'],
                            ['Hit-N', 'Miss-N'])
            ax_sub.axhline(50, color='black', linewidth=1, linestyle='--')
            ax_sub.set_xticks([0, 1]); ax_sub.set_xticklabels(['Hit-N', 'Miss-N'])
            ax_sub.set_title(stage_label)
            format_ax(ax_sub)
        axes[0].set_ylabel('Decoding accuracy (%) — direct (N→N+1)')
        plt.tight_layout(); pdf.savefig(); plt.close()

        # Page 7: above chance (50% subtracted) — all metrics
        fig, ax = plt.subplots(figsize=(10, 4))
        above_all              = [a - 50 for a in all_acc]
        above_all_seq          = [a - 50 for a in all_acc_seq]
        above_all_seq_direct   = [a - 50 for a in all_acc_seq_direct]
        above_all_threshold    = [a - 50 for a in all_acc_threshold]
        above_all_hit_n        = [a - 50 for a in all_acc_hit_n]
        above_all_miss_n       = [a - 50 for a in all_acc_miss_n]
        above_early            = [a - 50 for a in early_acc]
        above_late             = [a - 50 for a in late_acc]
        above_early_seq        = [a - 50 for a in early_acc_seq]
        above_late_seq         = [a - 50 for a in late_acc_seq]
        above_early_seq_direct = [a - 50 for a in early_acc_seq_direct]
        above_late_seq_direct  = [a - 50 for a in late_acc_seq_direct]
        above_early_threshold  = [a - 50 for a in early_acc_threshold]
        above_late_threshold   = [a - 50 for a in late_acc_threshold]
        above_early_hit_n      = [a - 50 for a in early_acc_hit_n]
        above_late_hit_n       = [a - 50 for a in late_acc_hit_n]
        above_early_miss_n     = [a - 50 for a in early_acc_miss_n]
        above_late_miss_n      = [a - 50 for a in late_acc_miss_n]
        groups = [above_all,    above_all_seq,    above_all_seq_direct, above_all_threshold, above_all_hit_n, above_all_miss_n,
                  above_early,  above_late,
                  above_early_seq,        above_late_seq,
                  above_early_seq_direct, above_late_seq_direct,
                  above_early_threshold,  above_late_threshold,
                  above_early_hit_n,      above_late_hit_n,
                  above_early_miss_n,     above_late_miss_n]
        bar_colors  = ['#D4B483', '#7B9E87', '#A89BC2', '#E8A838', '#5B8DB8', '#B85B5B',
                       '#C1666B', '#4281A4',
                       '#C1666B', '#4281A4',
                       '#C1666B', '#4281A4',
                       '#C1666B', '#4281A4',
                       '#C1666B', '#4281A4',
                       '#C1666B', '#4281A4']
        bar_xlabels = ['All\nAtemp.', 'All\n2-stg', 'All\nDirect', 'All\nThresh.', 'All\nHit-N', 'All\nMiss-N',
                       'Ear.\nAtemp.', 'Late\nAtemp.',
                       'Ear.\n2-stg',   'Late\n2-stg',
                       'Ear.\nDirect',  'Late\nDirect',
                       'Ear.\nThresh.', 'Late\nThresh.',
                       'Ear.\nHit-N',   'Late\nHit-N',
                       'Ear.\nMiss-N',  'Late\nMiss-N']
        _bar_with_error(ax, range(18),
                        [_mean_se(g)[0] for g in groups],
                        [_mean_se(g)[1] for g in groups],
                        bar_colors, bar_xlabels)
        ax.set_xticks(range(18)); ax.set_xticklabels(bar_xlabels, fontsize=6)
        ax.set_ylabel('Decoding accuracy (% above chance)')
        ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
        plt.tight_layout(); format_ax(ax); pdf.savefig(); plt.close()

        # Page 8: decision value variance + d' — early vs late
        early_var_hit  = _subject_mean(results, 'early', 'dec_var_hit',  scale=1)
        late_var_hit   = _subject_mean(results, 'late',  'dec_var_hit',  scale=1)
        early_var_miss = _subject_mean(results, 'early', 'dec_var_miss', scale=1)
        late_var_miss  = _subject_mean(results, 'late',  'dec_var_miss', scale=1)
        early_dprime   = _subject_mean(results, 'early', 'dec_dprime',   scale=1)
        late_dprime    = _subject_mean(results, 'late',  'dec_dprime',   scale=1)

        fig, axes = plt.subplots(1, 3, sharey=False, figsize=(10, 4))
        for ax_sub, e_vals, l_vals, ylabel, title in [
            (axes[0], early_var_hit,  late_var_hit,  'Decision value variance', 'Hit variance'),
            (axes[1], early_var_miss, late_var_miss, 'Decision value variance', 'Miss variance'),
            (axes[2], early_dprime,   late_dprime,   "d'",                      "d' (Hit vs Miss separation)"),
        ]:
            _bar_with_error(ax_sub, [0, 1],
                            [_mean_se(e_vals)[0], _mean_se(l_vals)[0]],
                            [_mean_se(e_vals)[1], _mean_se(l_vals)[1]],
                            ['#C1666B', '#4281A4'],
                            ['Early', 'Late'])
            ax_sub.set_xticks([0, 1]); ax_sub.set_xticklabels(['Early', 'Late'])
            ax_sub.set_title(title)
            ax_sub.set_ylabel(ylabel)
            format_ax(ax_sub)
        fig.suptitle('CV decision value statistics (atemporal LDA axis)', fontsize=9)
        plt.tight_layout(); pdf.savefig(); plt.close()

        # Page 9: decision value variance + d' vs day_of_training (scatter per session)
        fig, axes = plt.subplots(1, 3, sharey=False, figsize=(11, 4))
        for ax_sub, var_key, color, metric_label in [
            (axes[0], 'dec_var_hit',  '#5B8DB8', 'Hit variance'),
            (axes[1], 'dec_var_miss', '#B85B5B', 'Miss variance'),
            (axes[2], 'dec_dprime',   '#7B9E87', "d'"),
        ]:
            days, vals = [], []
            for sd, r in results.items():
                if r['day_of_training'] is not None and not np.isnan(r[var_key]):
                    days.append(r['day_of_training'])
                    vals.append(r[var_key])
            if days:
                ax_sub.scatter(days, vals, color=color, alpha=0.7, s=40)
                # Linear trend line
                z = np.polyfit(days, vals, 1)
                x_line = np.linspace(min(days), max(days), 100)
                ax_sub.plot(x_line, np.polyval(z, x_line),
                            color=color, linewidth=1.5, linestyle='--')
            ax_sub.set_xlabel('Day of training')
            ax_sub.set_ylabel(metric_label)
            ax_sub.set_title(metric_label)
            format_ax(ax_sub)
        fig.suptitle('Decision value statistics across training', fontsize=9)
        plt.tight_layout(); pdf.savefig(); plt.close()

        # Page 10: decision value → N+1 outcome correlation — early vs late
        early_corr_all    = _subject_mean(results, 'early', 'dec_corr_all',    scale=1)
        late_corr_all     = _subject_mean(results, 'late',  'dec_corr_all',    scale=1)
        early_corr_hit_n  = _subject_mean(results, 'early', 'dec_corr_hit_n',  scale=1)
        late_corr_hit_n   = _subject_mean(results, 'late',  'dec_corr_hit_n',  scale=1)
        early_corr_miss_n = _subject_mean(results, 'early', 'dec_corr_miss_n', scale=1)
        late_corr_miss_n  = _subject_mean(results, 'late',  'dec_corr_miss_n', scale=1)

        fig, axes = plt.subplots(1, 3, sharey=True, figsize=(10, 4))
        for ax_sub, e_vals, l_vals, title in [
            (axes[0], early_corr_all,    late_corr_all,    'All pairs'),
            (axes[1], early_corr_hit_n,  late_corr_hit_n,  'Hit-N pairs'),
            (axes[2], early_corr_miss_n, late_corr_miss_n, 'Miss-N pairs'),
        ]:
            _bar_with_error(ax_sub, [0, 1],
                            [_mean_se(e_vals)[0], _mean_se(l_vals)[0]],
                            [_mean_se(e_vals)[1], _mean_se(l_vals)[1]],
                            ['#C1666B', '#4281A4'],
                            ['Early', 'Late'])
            ax_sub.axhline(0, color='black', linewidth=0.8, linestyle='--')
            ax_sub.set_xticks([0, 1]); ax_sub.set_xticklabels(['Early', 'Late'])
            ax_sub.set_title(title)
            format_ax(ax_sub)
        axes[0].set_ylabel("Pearson r (trial-N decision value → trial-N+1 outcome)")
        fig.suptitle('Does trial-N neural state predict trial-N+1 outcome?\n'
                     'r ≠ 0 = structured variance; r = 0 = random noise', fontsize=9)
        plt.tight_layout(); pdf.savefig(); plt.close()

        # Page 11: correlation vs day_of_training (scatter per session)
        fig, axes = plt.subplots(1, 3, sharey=True, figsize=(11, 4))
        for ax_sub, corr_key, color, title in [
            (axes[0], 'dec_corr_all',    '#888888', 'All pairs'),
            (axes[1], 'dec_corr_hit_n',  '#5B8DB8', 'Hit-N pairs'),
            (axes[2], 'dec_corr_miss_n', '#B85B5B', 'Miss-N pairs'),
        ]:
            days, vals = [], []
            for sd, r in results.items():
                if r['day_of_training'] is not None and not np.isnan(r[corr_key]):
                    days.append(r['day_of_training'])
                    vals.append(r[corr_key])
            if days:
                ax_sub.scatter(days, vals, color=color, alpha=0.7, s=40)
                z = np.polyfit(days, vals, 1)
                x_line = np.linspace(min(days), max(days), 100)
                ax_sub.plot(x_line, np.polyval(z, x_line),
                            color=color, linewidth=1.5, linestyle='--')
            ax_sub.axhline(0, color='black', linewidth=0.8, linestyle='--')
            ax_sub.set_xlabel('Day of training')
            ax_sub.set_title(title)
            format_ax(ax_sub)
        axes[0].set_ylabel("Pearson r (trial-N decision value → trial-N+1 outcome)")
        fig.suptitle('Trial-N → trial-N+1 correlation across training', fontsize=9)
        plt.tight_layout(); pdf.savefig(); plt.close()

        # Page 12: behavioral baseline vs neural sequential — early vs late
        early_beh_all    = _subject_mean(results, 'early', 'beh_acc_all')
        late_beh_all     = _subject_mean(results, 'late',  'beh_acc_all')
        early_beh_hit_n  = _subject_mean(results, 'early', 'beh_acc_hit_n')
        late_beh_hit_n   = _subject_mean(results, 'late',  'beh_acc_hit_n')
        early_beh_miss_n = _subject_mean(results, 'early', 'beh_acc_miss_n')
        late_beh_miss_n  = _subject_mean(results, 'late',  'beh_acc_miss_n')

        fig, axes = plt.subplots(1, 3, sharey=False, figsize=(11, 4))
        for ax_sub, e_beh, l_beh, e_neural, l_neural, title in [
            (axes[0], early_beh_all,    late_beh_all,    early_acc_seq_direct, late_acc_seq_direct, 'All pairs'),
            (axes[1], early_beh_hit_n,  late_beh_hit_n,  early_acc_hit_n,      late_acc_hit_n,      'Hit-N pairs'),
            (axes[2], early_beh_miss_n, late_beh_miss_n, early_acc_miss_n,     late_acc_miss_n,     'Miss-N pairs'),
        ]:
            positions = [0, 0.6, 1.4, 2.0]
            vals      = [e_beh, l_beh, e_neural, l_neural]
            bar_cols  = ['#C1666B', '#4281A4', '#C1666B', '#4281A4']
            alphas    = [0.4, 0.4, 0.8, 0.8]   # lighter = behavioral, darker = neural
            for pos, val_list, col, alpha in zip(positions, vals, bar_cols, alphas):
                m, se = _mean_se(val_list)
                if np.isnan(m):
                    continue
                ax_sub.bar(pos, m, color=col, alpha=alpha, width=0.5)
                ax_sub.errorbar(pos, m, yerr=se, fmt='none', color='black', capsize=4)
            ax_sub.axhline(50, color='black', linewidth=0.8, linestyle='--')
            ax_sub.set_xticks(positions)
            ax_sub.set_xticklabels(['Early\nBeh.', 'Late\nBeh.',
                                    'Early\nNeural', 'Late\nNeural'], fontsize=7)
            ax_sub.set_title(title)
            ax_sub.set_ylabel('Balanced accuracy (%)')
            format_ax(ax_sub)
        fig.suptitle('Behavioral baseline (win-stay) vs neural sequential decoder\n'
                     'Light = behavior only; Dark = neural', fontsize=9)
        plt.tight_layout(); pdf.savefig(); plt.close()

    print(f"Saved PDF → {pdf_path}")


# ── Public entry point ────────────────────────────────────────────────────────

def run(filtered_files, SETTINGS_DICT):
    """
    Parameters
    ----------
    filtered_files : list of str
        JSON file paths (same list passed to zscore_timeSeries_fromJSON).
    SETTINGS_DICT  : dict
        Must contain the DECODER_N1_* keys documented at the top of this module.

    Returns
    -------
    results : dict keyed by subject_date:
        'accuracy'            : float (0–1)  atemporal decoder
        'accuracy_seq'        : float (0–1)  two-stage sequential N→N+1 decoder
        'accuracy_seq_direct' : float (0–1)  single-stage direct N→N+1 decoder
        'n_units'             : int
        'day_type'     : 'early' | 'late' | 'other'
    """
    settings = dict(SETTINGS_DICT)  # shallow copy — don't mutate caller's dict

    units_csv = settings.get('DECODER_N1_UNITS_CSV')
    settings['_UNITS_DF'] = pd.read_csv(units_csv) if units_csv else None

    threshold_df = pd.read_csv(settings['DECODER_N1_THRESHOLD_CSV'])
    early_day_sessions, late_day_sessions = _get_day_sessions(threshold_df)

    bin_cuts = np.arange(settings['DECODER_N1_START_TIME'],
                         settings['DECODER_N1_END_TIME'],
                         settings['DECODER_N1_BIN_SIZE'])

    # ── CSV mode: load pre-computed data file and derive sessions from it ──────
    feature_mode = settings.get('DECODER_N1_FEATURE_MODE', 'full_raster')
    if feature_mode == 'csv':
        csv_file = settings.get('DECODER_N1_CSV_FILE')
        if csv_file is None:
            raise ValueError(
                'DECODER_N1_FEATURE_MODE="csv" requires DECODER_N1_CSV_FILE to be set.')
        print(f"Loading CSV data from: {csv_file}")
        csv_df = pd.read_csv(csv_file)
        csv_df['_subject_date'] = csv_df['Unit'].apply(_unit_to_subject_date)
        settings['_CSV_DF'] = csv_df
        unique_sessions = _parse_subject_dates_from_csv(
            csv_df,
            settings.get('DECODER_N1_SUBJECTS'),
            settings.get('DECODER_N1_SESSIONS_TO_EXCLUDE'),
        )
    else:
        unique_sessions = _parse_subject_dates(
            filtered_files,
            settings.get('DECODER_N1_SUBJECTS'),
            settings.get('DECODER_N1_SESSIONS_TO_EXCLUDE'),
        )

    # Preprocess
    print(f"Preprocessing {len(unique_sessions)} sessions...")
    preprocessed = {}
    for sd in unique_sessions:
        spikes, outcomes, trial_ids, shock_flags = _preprocess_session(sd, filtered_files, bin_cuts, settings)
        if spikes is not None:
            preprocessed[sd] = {'spikes': spikes, 'outcomes': outcomes,
                                 'trial_ids': trial_ids, 'shock_flags': shock_flags}
    print(f"  → {len(preprocessed)} sessions retained")

    # ── Resolve (test, method) pairs ─────────────────────────────────────────
    tests_list     = settings.get('DECODER_N1_TESTS_TO_RUN',
                                  ['atemporal', 'two_stage', 'direct', 'threshold', 'by_ntype'])
    method_setting = settings.get('DECODER_N1_METHOD', 'lda')

    if isinstance(method_setting, str):
        test_method_pairs = [(t, method_setting) for t in tests_list]
    else:
        test_method_pairs = list(zip(tests_list, method_setting))

    # Group unique (test, method) pairs by method — one decode+save pass per method
    from collections import defaultdict
    method_to_tests = defaultdict(set)
    for test, meth in test_method_pairs:
        method_to_tests[meth].add(test)

    makedirs(settings['DECODER_N1_OUTPUT_FOLDER'], exist_ok=True)

    all_results = {}   # method → {subject_date → metrics}

    for meth, tests in method_to_tests.items():
        print(f"\nRunning classification [method={meth}, tests={', '.join(sorted(tests))}]...")

        # Inject method into a per-run settings copy so all decoders see it
        run_settings = dict(settings)
        run_settings['DECODER_N1_METHOD'] = meth

        results = {}
        for sd in sorted(preprocessed):
            sp = preprocessed[sd]
            n_units = sp['spikes'].shape[0]

            # ── atemporal ────────────────────────────────────────────────────
            if 'atemporal' in tests:
                acc, n_units = _decode_session(
                    sp['spikes'], sp['outcomes'], bin_cuts, run_settings)
            else:
                acc = np.nan

            # ── two-stage sequential ──────────────────────────────────────────
            if 'two_stage' in tests:
                acc_seq, _ = _decode_session_sequential(
                    sp['spikes'], sp['outcomes'], sp['trial_ids'], sp['shock_flags'],
                    bin_cuts, run_settings)
            else:
                acc_seq = np.nan

            # ── direct sequential ─────────────────────────────────────────────
            if 'direct' in tests:
                acc_seq_direct, _ = _decode_session_sequential_direct(
                    sp['spikes'], sp['outcomes'], sp['trial_ids'], sp['shock_flags'],
                    bin_cuts, run_settings)
            else:
                acc_seq_direct = np.nan

            # ── threshold transition + permutation test ───────────────────────
            if 'threshold' in tests:
                acc_direct_threshold, threshold_pval, _ = _decode_session_threshold_transition(
                    sp['spikes'], sp['outcomes'], sp['trial_ids'], sp['shock_flags'],
                    bin_cuts, run_settings)
            else:
                acc_direct_threshold, threshold_pval = np.nan, np.nan

            # ── by trial-N type ───────────────────────────────────────────────
            if 'by_ntype' in tests:
                acc_direct_hit_n, acc_direct_miss_n, _ = _decode_session_sequential_by_ntype(
                    sp['spikes'], sp['outcomes'], sp['trial_ids'], sp['shock_flags'],
                    bin_cuts, run_settings)
            else:
                acc_direct_hit_n, acc_direct_miss_n = np.nan, np.nan

            # ── decision value variance (stereotypy probe) ────────────────────
            if 'decision_variance' in tests:
                dec_var_hit, dec_var_miss, dec_var_all, dec_dprime, _ = _decode_session_decision_variance(
                    sp['spikes'], sp['outcomes'], bin_cuts, run_settings)
            else:
                dec_var_hit, dec_var_miss, dec_var_all, dec_dprime = np.nan, np.nan, np.nan, np.nan

            # ── behavioral baseline (win-stay, no neural data) ────────────────
            if 'behavioral_baseline' in tests:
                beh_acc_all, beh_acc_hit_n, beh_acc_miss_n = _decode_session_behavioral_baseline(
                    sp['outcomes'], sp['trial_ids'], sp['shock_flags'])
            else:
                beh_acc_all, beh_acc_hit_n, beh_acc_miss_n = np.nan, np.nan, np.nan

            # ── decision value → N+1 outcome correlation ──────────────────────
            if 'decision_correlation' in tests:
                dec_corr_all, dec_corr_hit_n, dec_corr_miss_n, _ = _decode_session_decision_correlation(
                    sp['spikes'], sp['outcomes'], sp['trial_ids'], sp['shock_flags'],
                    bin_cuts, run_settings)
            else:
                dec_corr_all, dec_corr_hit_n, dec_corr_miss_n = np.nan, np.nan, np.nan

            day_type = ('early' if sd in early_day_sessions else
                        'late'  if sd in late_day_sessions  else 'other')

            day_row = threshold_df[threshold_df.apply(
                lambda r: '_'.join([r['Subject'], split('-*-', r['Session'])[0]]) == sd, axis=1
            )]
            day_of_training = int(day_row['Day'].values[0]) if len(day_row) else None

            results[sd] = {
                'accuracy':              acc,
                'accuracy_seq':          acc_seq,
                'accuracy_seq_direct':   acc_seq_direct,
                'acc_direct_threshold':  acc_direct_threshold,
                'threshold_pval':        threshold_pval,
                'acc_direct_hit_n':      acc_direct_hit_n,
                'acc_direct_miss_n':     acc_direct_miss_n,
                'dec_var_hit':           dec_var_hit,
                'dec_var_miss':          dec_var_miss,
                'dec_var_all':           dec_var_all,
                'dec_dprime':            dec_dprime,
                'dec_corr_all':          dec_corr_all,
                'dec_corr_hit_n':        dec_corr_hit_n,
                'dec_corr_miss_n':       dec_corr_miss_n,
                'beh_acc_all':           beh_acc_all,
                'beh_acc_hit_n':         beh_acc_hit_n,
                'beh_acc_miss_n':        beh_acc_miss_n,
                'n_units':               n_units,
                'day_type':              day_type,
                'day_of_training':       day_of_training,
            }

            def _fmt(v):  return f'{v*100:.1f}%' if not np.isnan(v) else 'nan'
            def _fmtp(v): return f'p={v:.3f}'    if not np.isnan(v) else 'p=nan'
            print(f"  {sd}: acc={_fmt(acc)}  seq={_fmt(acc_seq)}  direct={_fmt(acc_seq_direct)}  "
                  f"threshold={_fmt(acc_direct_threshold)} {_fmtp(threshold_pval)}  "
                  f"dir_hit={_fmt(acc_direct_hit_n)}  dir_miss={_fmt(acc_direct_miss_n)}  "
                  f"units={n_units}  [{day_type}]")

        _save_csv(results, run_settings)
        _save_pdf(results, run_settings)
        all_results[meth] = results

    return all_results


# ── Standalone entry point ────────────────────────────────────────────────────

if __name__ == '__main__':
    INPUT_FOLDER = '.' + sep + sep.join(['Data', 'Output', 'JSON files'])
    all_json     = glob(INPUT_FOLDER + sep + '*json')
    run(all_json, _STANDALONE_SETTINGS)
