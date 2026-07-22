"""
lda_trial_outcome_decoding.py

Trial-outcome decoding from population spike rasters.
Designed to be called from an external pipeline after JSON preprocessing,
in the same style as zscore_timeSeries_fromJSON.

External pipeline usage
-----------------------
    import lda_trial_outcome_decoding

    lda_trial_outcome_decoding.run(filtered_files, SETTINGS_DICT)

All classification settings are read from SETTINGS_DICT (see SETTINGS keys below).

Standalone usage
----------------
    python lda_trial_outcome_decoding.py
"""

from os import sep, makedirs
import platform
import json
from glob import glob

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import patches
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import balanced_accuracy_score
from sklearn.multiclass import OneVsOneClassifier
from astropy.convolution import convolve_fft, Gaussian1DKernel
from sklearn.svm import LinearSVC, SVC
from matplotlib.backends.backend_pdf import PdfPages
from helpers.format_axes import format_ax
from scipy.stats import spearmanr, pearsonr, gaussian_kde

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
#   DECODER_N_METHOD               str   'lda' | 'lda_svc' | 'ovo' | 'svc'
#   DECODER_N_SESSION              str   'active' | 'pre' | 'post' | 'post1h'
#   DECODER_N_SPIKES_FIELDNAME     str   e.g. 'Response_spikes'
#   DECODER_N_SPIKE_TIME_FORMAT    bool  True if field contains spike times, False for pre-binned
#   DECODER_N_START_TIME           float signal window start (s)
#   DECODER_N_END_TIME             float signal window end   (s)
#   DECODER_N_BIN_SIZE             float bin width (s)
#   DECODER_N_PSTH_STEP_SIZE       float sliding-window step (s), usually == DECODER_N_BIN_SIZE
#   DECODER_N_EPOCH_START          float epoch start for decoding (s), <= DECODER_N_START_TIME
#   DECODER_N_EPOCH_END            float epoch end   for decoding (s), >= DECODER_N_END_TIME
#   DECODER_N_GAUSSIAN_SIGMA       float smoothing kernel SD (s); 0 = no smoothing
#   DECODER_N_SHOCK_ARTIFACT       list | None  [start, end] in seconds to blank, or None
#   DECODER_N_N_CV_SPLITS          int   number of stratified CV folds (default 5)
#   DECODER_N_RANDOM_STATE         int   RNG seed (default 0)
#   DECODER_N_SUBJECTS             list | None  subject IDs to include; None = all
#   DECODER_N_SESSIONS_TO_EXCLUDE  list | None  subject_date strings to drop; None = none
#   DECODER_N_UNITS_CSV            str | None   path to SU_list.csv; None = all units
#   DECODER_N_THRESHOLD_CSV        str          path to OFCPL_threshold_df.csv
#   DECODER_N_OUTPUT_FOLDER        str          folder for CSV + PDF output
#   DECODER_N_FILE_NAME_TAG        str          suffix for output file names
#   DECODER_N_DECODE_TARGET        str or list of str — one or more of:
#                                  'outcome'         (Hit vs Miss, default)
#                                  'amdepth'         (AM depth from all Hit+Miss trials)
#                                  'amdepth_hits'    (AM depth from Hit trials only)
#                                  'amdepth_misses'  (AM depth from Miss trials only)
#                                  If a list, all targets run on each session and results
#                                  appear as separate columns (CSV) / page groups (PDF).
#   DECODER_N_SHOCK_FLAG_FILTER    int | None | list — zipped with DECODER_N_DECODE_TARGET.
#                                  Scalar: same filter applied to every target.
#                                  List:   per-target filter, must match length of DECODER_N_DECODE_TARGET.
#                                  Values: 1 = ShockFlag==1 only (default)
#                                          0 = ShockFlag==0 only
#                                          None = all trials regardless of ShockFlag
#   DECODER_N_FEATURE_MODE         str   how to build the feature matrix for each trial:
#                                  'full_raster' (default) — flatten (units × epoch_bins)
#                                  'mean_rate'             — mean spike count per unit across epoch → (units,)
#                                  'isi'                   — KDE of log-ISI per unit evaluated at a
#                                                            fixed log-spaced grid → (units × n_grid,)
#                                                            requires DECODER_N_SPIKE_TIME_FORMAT=True
#   DECODER_N_ISI_GRID_MIN         float  shortest ISI included in KDE grid (s); default 0.001 (1 ms)
#   DECODER_N_ISI_GRID_MAX         float  longest  ISI included in KDE grid (s); default 2.0
#   DECODER_N_ISI_N_GRID           int    number of log-spaced grid evaluation points; default 20
#   DECODER_N_MIN_TRIALS_PER_CLASS int    drop any class with fewer than this many trials before CV;
#                                         0 = disabled (default). Useful for amdepth targets where
#                                         rare depths would otherwise force CV fold reduction.
#   DECODER_N_CSV_FILE             str | None  path to a pre-computed z-score CSV (e.g.
#                                              FR_timeSeries_data_zscore.csv).  Required when
#                                              DECODER_N_FEATURE_MODE='csv'; ignored otherwise.
#                                              Format: one row per unit×trial; columns include
#                                              Unit, Session, AnalysisID, Hit, Miss, FA,
#                                              ShockFlag, AMDepth, TrialID, and TP.1…TP.N.
#   DECODER_N_CSV_SAMPLING_RATE    float       sampling rate (Hz) of the TP columns.
#                                              TP.1 corresponds to DECODER_N_START_TIME;
#                                              TP.k is at START_TIME + (k-1)/SAMPLING_RATE.
#                                              Used to find the TP column for each bin in
#                                              bin_cuts: TP.{round((t-START_TIME)*RATE)+1}.
#   DECODER_N_CSV_ANALYSIS_ID      str | None  filter CSV rows to this AnalysisID value;
#                                              None = no filter (default: 'Response_timeSeries_zscore_globalBaseline').
#   DECODER_N_LOG1P_TRANSFORM      bool  apply np.log1p() to the feature matrix before classification;
#                                        default False.  Compresses right-skewed spike-count
#                                        distributions toward Gaussianity (helps LDA's covariance
#                                        assumption) and stabilises variance across firing-rate range.
#                                        Ignored for 'isi' and 'csv' feature modes (ISI features are
#                                        already log-scaled; CSV values are pre-z-scored).
#
# ─────────────────────────────────────────────────────────────────────────────


# ── Standalone defaults ───────────────────────────────────────────────────────

_STANDALONE_SETTINGS = {
    'DECODER_N_METHOD':              'lda',
    'DECODER_N_SESSION':             'active',
    'DECODER_N_SPIKES_FIELDNAME':    'Response_spikes',
    'DECODER_N_SPIKE_TIME_FORMAT':   True,
    'DECODER_N_START_TIME':          -2.0,
    'DECODER_N_END_TIME':             3.0,
    'DECODER_N_BIN_SIZE':             0.1,
    'DECODER_N_PSTH_STEP_SIZE':       0.1,
    'DECODER_N_EPOCH_START':         -2.0,
    'DECODER_N_EPOCH_END':            3.0,
    'DECODER_N_GAUSSIAN_SIGMA':       0,
    'DECODER_N_SHOCK_ARTIFACT':       None,
    'DECODER_N_N_CV_SPLITS':          5,
    'DECODER_N_RANDOM_STATE':         0,
    'DECODER_N_SUBJECTS':            ['SUBJ-ID-154', 'SUBJ-ID-389', 'SUBJ-ID-390',
                                'SUBJ-ID-1036', 'SUBJ-ID-1037', 'SUBJ-ID-1038'],
    'DECODER_N_SESSIONS_TO_EXCLUDE': None,
    'DECODER_N_UNITS_CSV':           '.' + sep + sep.join(['Data', 'Output', 'SU_list.csv']),
    'DECODER_N_THRESHOLD_CSV':       '.' + sep + sep.join(['Data', 'Output', 'OFCPL_threshold_df.csv']),
    'DECODER_N_OUTPUT_FOLDER':       '.' + sep + sep.join(['Data', 'Output']),
    'DECODER_N_FILE_NAME_TAG':       'TrialType_allTTs_lda_fullRaster',
    'DECODER_N_DECODE_TARGET':       'outcome',   # str or list: 'outcome'|'amdepth'|'amdepth_hits'|'amdepth_misses'
    'DECODER_N_SHOCK_FLAG_FILTER':   1,           # scalar or list zipped with DECODER_N_DECODE_TARGET: 1 | 0 | None
    'DECODER_N_FEATURE_MODE':        'full_raster',  # 'full_raster' | 'mean_rate' | 'isi' | 'csv'
    'DECODER_N_ISI_GRID_MIN':        0.001,           # seconds (1 ms)
    'DECODER_N_ISI_GRID_MAX':        2.0,             # seconds
    'DECODER_N_ISI_N_GRID':          20,              # log-spaced evaluation points
    'DECODER_N_MIN_TRIALS_PER_CLASS': 0,              # drop classes with fewer trials; 0 = disabled
    'DECODER_N_CSV_FILE':            None,            # path to z-score CSV; required for 'csv' mode
    'DECODER_N_CSV_SAMPLING_RATE':   100,             # Hz — TP column sampling rate
    'DECODER_N_CSV_ANALYSIS_ID':     'Response_timeSeries_zscore_globalBaseline',
    'DECODER_N_LOG1P_TRANSFORM':     False,           # apply log1p() to features; ignored for 'isi'/'csv'
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


# ── Confound analysis ─────────────────────────────────────────────────────────

def _compute_confound_stats(base_meta):
    """
    For each AM depth in base_meta, compute P(Miss | depth).

    Returns a dict with:
      depth_miss_corr   : Spearman r between AM depth and Miss fraction
                          (negative = harder/lower depths more likely Miss)
      depth_miss_corr_p : p-value of that correlation
      miss_frac_var     : variance of per-depth Miss fractions
                          (higher = stronger depth-outcome confound)
      miss_frac_by_depth: {depth: miss_fraction} for inspection
    """
    depths = np.sort(base_meta['AMdepth'].unique())
    miss_fracs = []
    for d in depths:
        mask = base_meta['AMdepth'] == d
        miss_fracs.append(float(base_meta.loc[mask, 'Miss'].mean()))

    miss_frac_by_depth = {float(d): f for d, f in zip(depths, miss_fracs)}
    miss_frac_var = float(np.var(miss_fracs, ddof=1)) if len(miss_fracs) > 1 else np.nan

    if len(depths) >= 3:
        r, p = spearmanr(depths, miss_fracs)
        depth_miss_corr   = float(r)
        depth_miss_corr_p = float(p)
    else:
        depth_miss_corr   = np.nan
        depth_miss_corr_p = np.nan

    return {
        'depth_miss_corr':    depth_miss_corr,
        'depth_miss_corr_p':  depth_miss_corr_p,
        'miss_frac_var':      miss_frac_var,
        'miss_frac_by_depth': miss_frac_by_depth,
    }


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

    Unit format: '{subject}_{date}_{cluster_info}'
    e.g. 'SUBJ-ID-1036_250830_concat_cluster200' → 'SUBJ-ID-1036_250830'

    Uses the same split logic as _parse_subject_dates so keys are identical.
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

    Each bin in bin_cuts maps to the single TP column whose time is nearest:
        tp_idx = round((bin_cuts[i] - START_TIME) * CSV_SAMPLING_RATE)
        column  = 'TP.{tp_idx + 1}'

    This produces a (n_units, n_trials, n_bins) array with the same shape as the
    spike raster from the JSON path, so all downstream decode functions work unchanged.
    Bins whose corresponding TP column is absent in the CSV receive NaN (filtered out
    by the existing valid-column check in _decode_session).

    Parameters
    ----------
    subject_date      : str   e.g. 'SUBJ-ID-1036_250830'
    bin_cuts          : np.ndarray  bin start times (seconds)
    csv_df            : pd.DataFrame  pre-loaded CSV with '_subject_date' column added
    shock_flag_filter : int | None
    settings          : dict
    prefix            : str   'DECODER_N' | 'DECODER_TS' | 'DECODER_N1'
    hit_only          : bool  keep Hit trials only
    miss_only         : bool  keep Miss trials only

    Returns
    -------
    session_arr : np.ndarray (n_units, n_trials, n_bins) or None
    base_meta   : pd.DataFrame or None
        Columns: Hit, Miss, AMdepth  (one row per trial, same order as session_arr axis 1)
    """
    start_time    = settings[f'{prefix}_START_TIME']
    sampling_rate = settings.get(f'{prefix}_CSV_SAMPLING_RATE', 100)
    # DECODER_N_CSV_ANALYSIS_ID takes precedence; fall back to SPIKES_FIELDNAME which
    # doubles as the AnalysisID when feature_mode='csv' (e.g.
    # 'Response_timeSeries_zscore_globalBaseline').
    analysis_id   = settings.get(f'{prefix}_CSV_ANALYSIS_ID')
    units_df      = settings.get('_UNITS_DF')

    # Map each bin_cut → TP column name
    tp_col_for_bin = [
        f'TP.{int(round((float(t) - start_time) * sampling_rate)) + 1}'
        for t in bin_cuts
    ]
    avail_tp = set(csv_df.columns)

    # Filter to this session
    sess_df = csv_df[csv_df['_subject_date'] == subject_date].copy()
    if analysis_id:
        sess_df = sess_df[sess_df['AnalysisID'] == analysis_id]
    if len(sess_df) == 0:
        return None, None

    # Trial mask: Hit/Miss only; FA rows excluded (the CSV may include them)
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

    # Unit filter (honours DECODER_*_UNITS_CSV)
    if units_df is not None:
        sess_df = sess_df[sess_df['Unit'].isin(units_df['Unit'].values)].copy()
    if len(sess_df) == 0:
        return None, None

    # Reference trial order: sort by TrialID from the first unit alphabetically
    units   = sorted(sess_df['Unit'].unique())
    ref_df  = (sess_df[sess_df['Unit'] == units[0]]
               .sort_values('TrialID')
               .reset_index(drop=True))
    trial_ids = ref_df['TrialID'].values

    # AMDepth column name may vary (CSV uses 'AMDepth'; JSON-derived uses 'AMdepth')
    amt_col  = 'AMDepth' if 'AMDepth' in ref_df.columns else 'AMdepth'
    base_meta = (ref_df[['Hit', 'Miss', amt_col]]
                 .rename(columns={amt_col: 'AMdepth'})
                 .reset_index(drop=True))

    # Build (n_units, n_trials, n_bins) array
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

def _preprocess_session(subject_date, filtered_files, bin_cuts, settings, target_filter_map):
    """
    Load and bin all units belonging to subject_date.

    target_filter_map : {target: shock_flag_filter}
        Targets that share the same shock_flag value share one binning pass.

    Returns
    -------
    dict : {target: (spikes np.ndarray (n_units, n_trials, n_bins),
                     labels np.ndarray (n_trials,))}
           Empty dict if no data loaded.
    """
    spikes_fieldname = settings['DECODER_N_SPIKES_FIELDNAME']
    session_type     = settings['DECODER_N_SESSION']
    spikeTime_format = settings['DECODER_N_SPIKE_TIME_FORMAT']
    BIN_SIZE         = settings['DECODER_N_BIN_SIZE']
    psth_step_size   = settings['DECODER_N_PSTH_STEP_SIZE']
    sigma            = settings['DECODER_N_GAUSSIAN_SIGMA']
    shock            = settings['DECODER_N_SHOCK_ARTIFACT']
    units_df         = settings.get('_UNITS_DF')
    feature_mode     = settings.get('DECODER_N_FEATURE_MODE', 'full_raster')
    isi_grid_min     = settings.get('DECODER_N_ISI_GRID_MIN', 0.001)
    isi_grid_max     = settings.get('DECODER_N_ISI_GRID_MAX', 2.0)
    isi_n_grid       = settings.get('DECODER_N_ISI_N_GRID',   20)
    log_grid         = (np.linspace(np.log(isi_grid_min), np.log(isi_grid_max), isi_n_grid)
                        if feature_mode == 'isi' else None)

    to_remove_mask = (
        ((bin_cuts >= shock[0]) & (bin_cuts < shock[1])) if shock is not None
        else np.zeros(len(bin_cuts), dtype=bool)
    )

    cur_jsons = [f for f in filtered_files if subject_date in f]

    # Group targets by shock_flag so each unique filter value bins trials only once
    from collections import defaultdict
    by_filter = defaultdict(list)
    for target, sf in target_filter_map.items():
        by_filter[sf].append(target)

    def _bin_base_pool(shock_flag_filter):
        """Bin all units for the given shock_flag filter.

        Returns (base_spikes, base_meta, pool_isi) or (None, None, None).
        pool_isi is (n_units, n_trials, n_grid) when feature_mode=='isi', else None.
        """
        unit_spikes      = []
        unit_spike_times = []   # list[units] of list[trials] of arrays; ISI mode only
        base_meta        = None

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

            base_mask = (
                (cur_data['Reminder'] == 0) &
                ((cur_data['Hit'] == 1) | (cur_data['Miss'] == 1))
            )
            if shock_flag_filter is not None:
                base_mask = base_mask & (cur_data['ShockFlag'] == shock_flag_filter)

            base_subset = cur_data[base_mask].reset_index(drop=True)
            if len(base_subset) == 0:
                continue

            cur_meta = base_subset[['Hit', 'Miss', 'AMdepth']].reset_index(drop=True)
            if base_meta is None:
                base_meta = cur_meta
            else:
                if not base_meta.equals(cur_meta):
                    print(f'Trial metadata mismatch across units in {subject_date}')
                    return None, None, None

            allTrial_spikes   = np.zeros((len(base_subset), len(bin_cuts)))
            trial_spike_times = []
            for row_idx, raw_spikes in enumerate(base_subset[spikes_fieldname].to_numpy()):
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

            unit_spikes.append(allTrial_spikes)
            if feature_mode == 'isi' and spikeTime_format:
                unit_spike_times.append(trial_spike_times)

        if not unit_spikes or base_meta is None:
            return None, None, None

        base_spikes = np.stack(unit_spikes, axis=0)  # (n_units, n_trials, n_bins)

        # Compute per-unit ISI KDE features over the decode epoch
        pool_isi = None
        if feature_mode == 'isi' and unit_spike_times:
            epoch_start = settings['DECODER_N_EPOCH_START']
            epoch_end   = settings['DECODER_N_EPOCH_END']
            pool_isi = np.stack(
                [_isi_kde_features(trial_times, epoch_start, epoch_end,
                                   log_grid, exclude_range=shock)
                 for trial_times in unit_spike_times],
                axis=0
            )  # (n_units, n_trials, n_grid)

        return base_spikes, base_meta, pool_isi

    def _labels_for_target(target, base_meta):
        hit_idx  = np.where(base_meta['Hit'].values  == 1)[0]
        miss_idx = np.where(base_meta['Miss'].values == 1)[0]
        all_idx  = np.arange(len(base_meta))

        if target == 'outcome':
            idx    = all_idx
            labels = np.zeros(len(idx))
            labels[base_meta['Miss'].values == 1] = 1
        elif target == 'amdepth':
            idx    = all_idx
            labels = base_meta['AMdepth'].values.astype(float)
        elif target == 'amdepth_hits':
            idx    = hit_idx
            labels = base_meta['AMdepth'].values[idx].astype(float)
        elif target == 'amdepth_misses':
            idx    = miss_idx
            labels = base_meta['AMdepth'].values[idx].astype(float)
        else:
            print(f'Unknown decode target "{target}". Skipping.')
            return None, None
        return idx, labels

    # ── CSV mode: load pre-computed z-score data instead of raw spike files ──────
    if feature_mode == 'csv':
        csv_df = settings.get('_CSV_DF')
        if csv_df is None:
            raise ValueError(
                'DECODER_N_FEATURE_MODE="csv" requires DECODER_N_CSV_FILE to be set '
                'in the settings dict.')
        result         = {}
        confound_by_sf = {}
        for shock_flag, group_targets in by_filter.items():
            session_arr, base_meta = _load_csv_session(
                subject_date, bin_cuts, csv_df, shock_flag, settings, 'DECODER_N')
            if session_arr is None:
                continue
            confound_by_sf[shock_flag] = _compute_confound_stats(base_meta)
            for target in group_targets:
                idx, labels = _labels_for_target(target, base_meta)
                if idx is None or len(idx) == 0:
                    continue
                # isi_slice is None — ISI features not applicable in CSV mode
                result[target] = (session_arr[:, idx, :], None, labels)
        result['_confound_by_sf'] = confound_by_sf
        return result

    # ── JSON / spike-file mode ────────────────────────────────────────────────
    result         = {}
    confound_by_sf = {}
    for shock_flag, group_targets in by_filter.items():
        base_spikes, base_meta, pool_isi = _bin_base_pool(shock_flag)
        if base_spikes is None:
            continue
        confound_by_sf[shock_flag] = _compute_confound_stats(base_meta)
        for target in group_targets:
            idx, labels = _labels_for_target(target, base_meta)
            if idx is None or len(idx) == 0:
                continue
            isi_slice = pool_isi[:, idx, :] if pool_isi is not None else None
            result[target] = (base_spikes[:, idx, :], isi_slice, labels)

    result['_confound_by_sf'] = confound_by_sf
    return result


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


def _decode_session(session_spikes, isi_features, outcomes, bin_cuts, settings):
    """
    Decode trial outcomes from the full population raster within the
    configured epoch window.

    Parameters
    ----------
    session_spikes : np.ndarray (n_units, n_trials, n_bins)
    isi_features   : np.ndarray (n_units, n_trials, n_grid) | None
        Pre-computed ISI KDE features; only used when DECODER_N_FEATURE_MODE=='isi'.

    Returns
    -------
    accuracy     : float  – mean balanced accuracy across CV folds (0–1)
    n_units      : int
    chance_level : float  – theoretical chance (1 / n_classes)
    """
    epoch_start  = settings['DECODER_N_EPOCH_START']
    epoch_end    = settings['DECODER_N_EPOCH_END']
    method       = settings['DECODER_N_METHOD']
    n_splits     = settings['DECODER_N_N_CV_SPLITS']
    random_state = settings['DECODER_N_RANDOM_STATE']
    feature_mode = settings.get('DECODER_N_FEATURE_MODE', 'full_raster')

    epoch_mask   = (bin_cuts >= epoch_start) & (bin_cuts < epoch_end)
    epoch_spikes = session_spikes[:, :, epoch_mask]           # (units, trials, epoch_bins)
    n_units, n_trials, _ = epoch_spikes.shape

    if feature_mode == 'mean_rate':
        X_data = epoch_spikes.mean(axis=2).T                  # (trials, units)
    elif feature_mode == 'isi':
        if isi_features is None:
            print('  ISI mode requires DECODER_N_SPIKE_TIME_FORMAT=True. Falling back to mean_rate.')
            X_data = epoch_spikes.mean(axis=2).T
        else:
            # isi_features: (units, trials, n_grid) → (trials, units*n_grid)
            X_data = isi_features.transpose(1, 0, 2).reshape(n_trials, -1)
    else:
        # 'full_raster' uses all epoch bins flattened; 'csv' does the same
        # (the CSV array is already aligned to bin_cuts, epoch_mask selects the epoch TPs)
        if feature_mode not in ('full_raster', 'csv'):
            print(f"  Unknown DECODER_N_FEATURE_MODE '{feature_mode}'. Using full_raster.")
        X_data = epoch_spikes.transpose(1, 0, 2).reshape(n_trials, -1)  # (trials, units*bins)

    # Optional log1p transform — compresses right-skewed spike counts toward Gaussianity.
    # Skipped for 'isi' (already log-scaled) and 'csv' (pre-z-scored).
    if settings.get('DECODER_N_LOG1P_TRANSFORM', False) and feature_mode not in ('isi', 'csv'):
        X_data = np.log1p(X_data)

    valid  = ~np.isnan(X_data).any(axis=0)
    X_data = X_data[:, valid]
    if X_data.shape[1] == 0:
        return np.nan, n_units, np.nan

    y_data = LabelEncoder().fit_transform(outcomes)

    classes, counts = np.unique(y_data, return_counts=True)
    print(f"  Class counts: { {int(c): int(n) for c, n in zip(classes, counts)} }")

    # Drop classes with too few trials
    min_trials = settings.get('DECODER_N_MIN_TRIALS_PER_CLASS', 0)
    if min_trials > 0:
        keep_classes = classes[counts >= min_trials]
        if len(keep_classes) < 2:
            print(f"  WARNING: fewer than 2 classes have ≥{min_trials} trials. Skipping session.")
            return np.nan, n_units, 1.0 / max(len(classes), 1)
        dropped = int((~np.isin(y_data, keep_classes)).sum())
        if dropped:
            print(f"  NOTE: dropping {dropped} trials from classes with <{min_trials} trials")
            keep_mask = np.isin(y_data, keep_classes)
            X_data    = X_data[keep_mask]
            y_data    = LabelEncoder().fit_transform(outcomes[keep_mask])
            classes, counts = np.unique(y_data, return_counts=True)

    n_classes    = len(classes)
    chance_level = 1.0 / n_classes
    min_count    = counts.min()
    if min_count < n_splits:
        print(f"  WARNING: min class count ({min_count}) < n_splits ({n_splits}). Skipping session.")
        return np.nan, n_units, chance_level

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
            if y_pred is not None:
                fold_scores.append(balanced_accuracy_score(y_test, y_pred))

    except ValueError:
        pass

    accuracy = float(np.mean(fold_scores)) if fold_scores else np.nan
    return accuracy, n_units, chance_level

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

def _save_csv(results, settings, targets, target_filter_map):
    """Write one row per session of decoding accuracy/chance/confound stats to a CSV.

    Args:
        results (dict): Maps subject_date -> per-session result dict (as
            produced by ``_decode_session``/``run``), with 'n_units',
            'day_of_training', 'day_type', 'targets' (per-target
            accuracy/chance), and 'confound_by_sf' (per-ShockFlag confound stats).
        settings (dict): Pipeline settings; uses ``DECODER_N_OUTPUT_FOLDER``,
            ``DECODER_N_FILE_NAME_TAG``, ``DECODER_N_METHOD``,
            ``DECODER_N_EPOCH_START``/``DECODER_N_EPOCH_END``.
        targets (Iterable[str]): Target names to include as columns (e.g. 'Hit_vs_Miss').
        target_filter_map (dict): Maps target name -> ShockFlag filter value
            used to select which confound stats to report for that target.

    Returns:
        None. Writes ``LDAoutput_<file_name_tag>.csv`` under
        ``DECODER_N_OUTPUT_FOLDER`` and prints its path.
    """
    output_folder = settings['DECODER_N_OUTPUT_FOLDER']
    file_name_tag = settings['DECODER_N_FILE_NAME_TAG']
    method        = settings['DECODER_N_METHOD']
    epoch_start   = settings['DECODER_N_EPOCH_START']
    epoch_end     = settings['DECODER_N_EPOCH_END']

    target_cols = []
    for t in targets:
        target_cols += [f'{t}_Accuracy', f'{t}_Chance', f'{t}_ShockFlag',
                        f'{t}_DepthMissCorr', f'{t}_DepthMissCorr_p', f'{t}_MissFracVar']

    csv_path = sep.join([output_folder, 'LDAoutput_' + file_name_tag + '.csv'])
    with open(csv_path, 'w', newline='', encoding='utf-8') as fh:
        writer = csv.writer(fh)
        writer.writerow(['Session', 'Method',
                         'Epoch_start', 'Epoch_end', 'Unit_count',
                         'Day_of_training', 'Training_stage'] + target_cols)
        for sd, r in results.items():
            row = [sd, method,
                   epoch_start, epoch_end,
                   r['n_units'], r['day_of_training'], r['day_type']]
            for t in targets:
                tr  = r['targets'].get(t, {})
                acc = tr.get('accuracy', np.nan)
                chc = tr.get('chance_level', np.nan)
                sf  = target_filter_map[t]
                cf  = r.get('confound_by_sf', {}).get(sf, {})
                dmc = cf.get('depth_miss_corr',   np.nan)
                dmp = cf.get('depth_miss_corr_p', np.nan)
                mfv = cf.get('miss_frac_var',      np.nan)
                def _fmt(v, digits=2):
                    return round(v, digits) if not np.isnan(v) else ''
                row += [
                    _fmt(acc * 100),
                    _fmt(chc * 100),
                    sf if sf is not None else 'all',
                    _fmt(dmc, 4),
                    _fmt(dmp, 4),
                    _fmt(mfv, 6),
                ]
            writer.writerow(row)
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


def _save_pdf(results, settings, targets, target_filter_map):
    """Plot per-target decoding accuracy (all/first/best day) as bar charts into a PDF.

    Args:
        results (dict): Maps subject_date -> per-session result dict, as in ``_save_csv``.
        settings (dict): Pipeline settings; uses ``DECODER_N_OUTPUT_FOLDER`` and ``DECODER_N_FILE_NAME_TAG``.
        targets (Iterable[str]): Target names, one figure/bar-group per target.
        target_filter_map (dict): Maps target name -> ShockFlag filter value (unused directly here, kept for a consistent call signature with ``_save_csv``).

    Returns:
        None. Writes ``LDAoutput_<file_name_tag>.pdf`` under ``DECODER_N_OUTPUT_FOLDER``.
    """
    def _subject_mean(results, day_type, target):
        """Average per-session accuracy (%) within each subject, for one day_type/target.

        Args:
            results (dict): Same per-session result dict as the enclosing function.
            day_type (str): Which day-type bucket to average within (e.g. 'first'/'best'/'all').
            target (str): Target name to read accuracy for.

        Returns:
            list[float]: One mean-accuracy value per subject that has data for this day_type/target.
        """
        from collections import defaultdict
        by_subject = defaultdict(list)
        for sd, r in results.items():
            if r['day_type'] == day_type:
                subj = sd.split('_')[0]
                acc  = r['targets'].get(target, {}).get('accuracy', np.nan)
                if not np.isnan(acc):
                    by_subject[subj].append(acc * 100)
        return [np.mean(v) for v in by_subject.values()]

    output_folder = settings['DECODER_N_OUTPUT_FOLDER']
    file_name_tag = settings['DECODER_N_FILE_NAME_TAG']

    def _chance_band(ax, x_lo, x_hi, mean, se, color='black', label=None):
        """Draw a shaded ±SE band with a centre line over [x_lo, x_hi]."""
        ax.fill_between([x_lo, x_hi], mean - se, mean + se,
                        color=color, alpha=0.15, linewidth=0)
        ax.plot([x_lo, x_hi], [mean, mean],
                color=color, linewidth=1, linestyle='--',
                label=label)

    pdf_path = sep.join([output_folder, 'LDAoutput_' + file_name_tag + '.pdf'])
    with PdfPages(pdf_path) as pdf:
        for target in targets:
            all_acc = [
                r['targets'].get(target, {}).get('accuracy', np.nan) * 100
                for r in results.values()
            ]
            all_acc = [a for a in all_acc if not np.isnan(a)]

            # Per-group chance distributions (per session, not averaged first)
            def _group_chance(day_type):
                return [
                    r['targets'].get(target, {}).get('chance_level', np.nan) * 100
                    for r in results.values()
                    if r['day_type'] == day_type and
                    not np.isnan(r['targets'].get(target, {}).get('chance_level', np.nan))
                ]

            all_chance   = [v for v in (r['targets'].get(target, {}).get('chance_level', np.nan) * 100
                                        for r in results.values()) if not np.isnan(v)]
            early_chance = _group_chance('early')
            late_chance  = _group_chance('late')

            all_c_mean,   all_c_se   = _mean_se(all_chance)
            early_c_mean, early_c_se = _mean_se(early_chance)
            late_c_mean,  late_c_se  = _mean_se(late_chance)

            early_acc = _subject_mean(results, 'early', target)
            late_acc  = _subject_mean(results, 'late',  target)

            # Page 1: all sessions — chance band spans full x range
            fig, ax = plt.subplots()
            _bar_with_error(ax, [0],
                            [_mean_se(all_acc)[0]],
                            [_mean_se(all_acc)[1]],
                            ['#D4B483'], ['Actual'])
            _chance_band(ax, -0.5, 0.5, all_c_mean, all_c_se,
                         label=f'Chance mean±SE ({all_c_mean:.0f}%)')
            ax.set_xticks([0]); ax.set_xticklabels(['Actual'])
            ax.set_ylabel('Decoding accuracy (%)')
            ax.set_title(f'All training sessions — {target}')
            ax.legend(frameon=False)
            format_ax(ax); pdf.savefig(); plt.close()

            # Page 2: early vs late — separate chance band per group, aligned to each bar
            fig, ax = plt.subplots()
            colors  = ['#C1666B', '#4281A4']
            xlabels = ['Early', 'Late']
            _bar_with_error(ax, range(2),
                            [_mean_se(g)[0] for g in [early_acc, late_acc]],
                            [_mean_se(g)[1] for g in [early_acc, late_acc]],
                            colors, xlabels)
            _chance_band(ax, -0.25, 0.25, early_c_mean, early_c_se,
                         color='#C1666B', label=f'Early chance ({early_c_mean:.0f}%)')
            _chance_band(ax, 0.75, 1.25, late_c_mean, late_c_se,
                         color='#4281A4', label=f'Late chance ({late_c_mean:.0f}%)')
            ax.set_xticks(range(2)); ax.set_xticklabels(xlabels)
            ax.set_ylabel('Decoding accuracy (%)')
            ax.set_title(target)
            ax.legend(handles=[
                patches.Patch(facecolor='#C1666B', alpha=0.7, label='Early'),
                patches.Patch(facecolor='#4281A4', alpha=0.7, label='Late'),
                patches.Patch(facecolor='#C1666B', alpha=0.15,
                              label=f'Early chance ({early_c_mean:.0f}%)'),
                patches.Patch(facecolor='#4281A4', alpha=0.15,
                              label=f'Late chance ({late_c_mean:.0f}%)'),
            ], frameon=False, fontsize=7)
            format_ax(ax); pdf.savefig(); plt.close()

            # Page 3: above chance — each group subtracts its own shuffled mean
            # (avoids bias when early/late sessions have different class imbalances)
            fig, ax = plt.subplots()
            above_all   = [a - all_c_mean   for a in all_acc]
            above_early = [a - early_c_mean for a in early_acc]
            above_late  = [a - late_c_mean  for a in late_acc]
            groups = [above_all, above_early, above_late]
            _bar_with_error(ax, range(3),
                            [_mean_se(g)[0] for g in groups],
                            [_mean_se(g)[1] for g in groups],
                            ['#D4B483', '#C1666B', '#4281A4'],
                            ['All', 'Early', 'Late'])
            ax.set_xticks(range(3)); ax.set_xticklabels(['All', 'Early', 'Late'])
            ax.set_ylabel('Decoding accuracy (% above shuffled chance)')
            ax.set_title(target)
            ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
            format_ax(ax); pdf.savefig(); plt.close()

            # Page 4 (amdepth targets only): accuracy vs. depth-outcome confound strength
            if 'amdepth' in target:
                sf = target_filter_map[target]
                stage_colors = {'early': '#C1666B', 'late': '#4281A4', 'other': '#888888'}

                conf_x, conf_y, conf_c = [], [], []
                for r in results.values():
                    acc = r['targets'].get(target, {}).get('accuracy', np.nan)
                    cf  = r.get('confound_by_sf', {}).get(sf, {})
                    mfv = cf.get('miss_frac_var', np.nan)
                    if not np.isnan(acc) and not np.isnan(mfv):
                        conf_x.append(mfv)
                        conf_y.append(acc * 100)
                        conf_c.append(stage_colors.get(r['day_type'], '#888888'))

                fig, ax = plt.subplots()
                ax.scatter(conf_x, conf_y, c=conf_c, alpha=0.7, s=40, zorder=3)

                stage_handles = [patches.Patch(facecolor=c, alpha=0.7, label=l)
                                 for c, l in [('#C1666B', 'Early'), ('#4281A4', 'Late')]]

                if len(conf_x) >= 3:
                    r_val, p_val = pearsonr(conf_x, conf_y)
                    z = np.polyfit(conf_x, conf_y, 1)
                    x_line = np.linspace(min(conf_x), max(conf_x), 100)
                    ax.plot(x_line, np.polyval(z, x_line),
                            color='black', linewidth=1.2, linestyle='--')
                    from matplotlib.lines import Line2D
                    stage_handles += [Line2D([0], [0], color='black', linewidth=1.2,
                                            linestyle='--',
                                            label=f'r = {r_val:.2f}, p = {p_val:.3f}')]

                ax.set_xlabel('Miss-fraction variance across AM depths\n(confound strength)')
                ax.set_ylabel('Decoding accuracy (%)')
                ax.set_title(f'{target} — accuracy vs. depth–outcome confound\n'
                             f'(r > 0 = decoder exploits Miss enrichment at low depths)')
                ax.axhline(all_c_mean, color='gray', linewidth=0.8, linestyle=':')
                ax.legend(handles=stage_handles, frameon=False, fontsize=7)
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
        Must contain the DECODER_N_* keys documented at the top of this module.

    Returns
    -------
    results : dict keyed by subject_date:
        'accuracy'    : float (0–1)
        'accuracy_sh' : float (0–1)
        'n_units'     : int
        'day_type'    : 'early' | 'late' | 'other'
    """
    settings = dict(SETTINGS_DICT)  # shallow copy — don't mutate caller's dict

    # Normalize DECODER_N_DECODE_TARGET and DECODER_N_SHOCK_FLAG_FILTER to aligned lists
    raw_targets = settings.get('DECODER_N_DECODE_TARGET', 'outcome')
    targets = [raw_targets] if isinstance(raw_targets, str) else list(raw_targets)

    raw_filters = settings.get('DECODER_N_SHOCK_FLAG_FILTER', 1)
    if isinstance(raw_filters, list):
        if len(raw_filters) != len(targets):
            raise ValueError(
                f'DECODER_N_SHOCK_FLAG_FILTER has {len(raw_filters)} entries but '
                f'DECODER_N_DECODE_TARGET has {len(targets)}. Lengths must match.'
            )
        filters = raw_filters
    else:
        filters = [raw_filters] * len(targets)

    target_filter_map = dict(zip(targets, filters))

    units_csv = settings.get('DECODER_N_UNITS_CSV')
    settings['_UNITS_DF'] = pd.read_csv(units_csv) if units_csv else None

    threshold_df = pd.read_csv(settings['DECODER_N_THRESHOLD_CSV'])
    early_day_sessions, late_day_sessions = _get_day_sessions(threshold_df)

    bin_cuts = np.arange(settings['DECODER_N_START_TIME'],
                         settings['DECODER_N_END_TIME'],
                         settings['DECODER_N_BIN_SIZE'])

    # ── CSV mode: load pre-computed data file and derive sessions from it ──────
    feature_mode = settings.get('DECODER_N_FEATURE_MODE', 'full_raster')
    if feature_mode == 'csv':
        csv_file = settings.get('DECODER_N_CSV_FILE')
        if csv_file is None:
            raise ValueError(
                'DECODER_N_FEATURE_MODE="csv" requires DECODER_N_CSV_FILE to be set.')
        print(f"Loading CSV data from: {csv_file}")
        csv_df = pd.read_csv(csv_file)
        # Precompute subject_date key once for fast per-session filtering
        csv_df['_subject_date'] = csv_df['Unit'].apply(_unit_to_subject_date)
        settings['_CSV_DF'] = csv_df
        unique_sessions = _parse_subject_dates_from_csv(
            csv_df,
            settings.get('DECODER_N_SUBJECTS'),
            settings.get('DECODER_N_SESSIONS_TO_EXCLUDE'),
        )
    else:
        unique_sessions = _parse_subject_dates(
            filtered_files,
            settings.get('DECODER_N_SUBJECTS'),
            settings.get('DECODER_N_SESSIONS_TO_EXCLUDE'),
        )

    # Preprocess — one JSON load + binning pass per unique ShockFlag value, split per target
    filter_summary = ', '.join(
        f'{t}→SF={sf if sf is not None else "all"}' for t, sf in target_filter_map.items()
    )
    print(f"Preprocessing {len(unique_sessions)} sessions ({filter_summary})...")
    preprocessed = {}
    for sd in unique_sessions:
        target_data = _preprocess_session(sd, filtered_files, bin_cuts, settings, target_filter_map)
        if target_data:
            preprocessed[sd] = target_data   # includes '_confound_by_sf' key
    print(f"  → {len(preprocessed)} sessions retained")

    # Decode
    print(f"Running classification for targets: {targets}")
    results = {}
    for sd in sorted(preprocessed):
        day_type = ('early' if sd in early_day_sessions else
                    'late'  if sd in late_day_sessions  else 'other')
        day_row = threshold_df[threshold_df.apply(
            lambda r: '_'.join([r['Subject'], split('-*-', r['Session'])[0]]) == sd, axis=1
        )]
        day_of_training = int(day_row['Day'].values[0]) if len(day_row) else None

        n_units        = None
        target_results = {}
        for target in targets:
            if target not in preprocessed[sd]:
                target_results[target] = {'accuracy': np.nan, 'chance_level': np.nan}
                continue
            spikes, isi_feats, outcomes = preprocessed[sd][target]
            acc, n_u, chance_level = _decode_session(spikes, isi_feats, outcomes, bin_cuts, settings)
            if n_units is None:
                n_units = n_u
            target_results[target] = {'accuracy': acc, 'chance_level': chance_level}
            print(f"  {sd} [{target}]: acc={acc*100:.1f}%  "
                  f"chance={chance_level*100:.0f}%  units={n_u}  [{day_type}]")

        confound_by_sf = preprocessed[sd].get('_confound_by_sf', {})

        results[sd] = {
            'n_units':         n_units,
            'day_type':        day_type,
            'day_of_training': day_of_training,
            'targets':         target_results,
            'confound_by_sf':  confound_by_sf,
        }

    makedirs(settings['DECODER_N_OUTPUT_FOLDER'], exist_ok=True)

    _save_csv(results, settings, targets, target_filter_map)
    _save_pdf(results, settings, targets, target_filter_map)
    return results


# ── Standalone entry point ────────────────────────────────────────────────────

if __name__ == '__main__':
    INPUT_FOLDER = '.' + sep + sep.join(['Data', 'Output', 'JSON files'])
    all_json     = glob(INPUT_FOLDER + sep + '*json')
    run(all_json, _STANDALONE_SETTINGS)