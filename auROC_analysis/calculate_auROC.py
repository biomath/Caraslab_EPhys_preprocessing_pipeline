"""auROC (area under the Receiver Operating Characteristic curve) calculations per unit/session.

Core primitive is ``_auROC_response_curve`` (Cohen et al., Nature 2012 method).
``run_calculate_auROC`` is the general-purpose entry point (parameterized by
``trial_or_response_aligned``/``trial_type``/``shock_flag``/``byAM_depth``/
``amdepth_subset``/etc.); ``run_auROC_extractions`` drives it with a list of
parameter dicts to compute several auROC variants in one call (see
``SETTINGS_DICT['AUROC_EXTRACTIONS_ACTIVE'/'_PASSIVE']`` in the notebook).
``run_auROC_pipeline``/``build_auROC_data_dict`` are the per-unit driver and
the pre/post/post1h/active data-loader consumed by
``auROC_heatmap_plotter``/``auROC_tslearnClustering_mp``.

``calculate_auROC_responseAligned_shock`` is kept as a standalone function
rather than folded into the parameterized switchboard: it re-derives its own
alignment point (the spout-offset event occurring during the shock window)
for Miss trials that have no formal response latency, which
``run_calculate_auROC``'s parameters can't express. Every other historical
``calculate_auROC_*`` variant (Hit/Miss/FA x trial/response-aligned x
shock-flag x AM-depth combinations) is fully reproducible as a
``run_calculate_auROC``/``run_auROC_extractions`` parameter spec and was
removed as redundant.
"""
from multiprocessing import Pool

import numpy as np
import pandas as pd
from copy import deepcopy
import copy

from helpers.get_JSON_data import _load_one_json
from helpers.write_json import write_json


def _auROC_response_curve(hist, edges, pre_stimulus_baseline_start, pre_stimulus_baseline_end, auroc_binsize=0.1):
    """
    Receiver Operating Characteristic curve (auROC) calculation
    From Cohen et al., Nature, 2012:
        a, Raster plot from 15 trials of 149 big-reward trials from a dopaminergic
        neuron. r1 and r2 correspond to two example 100-ms bins. b, Average firing rate of this neuron. c, Area
        under the receiver operating characteristic curve (auROC) for r1, in which the neuron increased its firing
        rate relative to baseline. We compared the histogram of spike counts during the baseline period (dashed
        line) to that during a given bin (solid line) by moving a criterion from zero to the maximum firing rate (in
        this example, 68 spikes/s). We then plotted the probability that the activity during r1 was greater than the
        criteria against the probability that the baseline activity was greater than the criteria. The area under this
        curve quantifies the degree of overlap between the two spike count distributions (i.e., the discriminability
        of the two).
    :param hist: numpy.ndarray
        A histogram resulting from numpy.hist
    :param edges: numpy.ndarray
        Histogram edges resulting from numpy.hist
    :param pre_stimulus_baseline_start:  number
        Start of period to calculate the baseline for the auROC in relation to trigger (negative means after); in seconds
    :param pre_stimulus_baseline_end: number
        End of period to calculate the baseline for the auROC in relation to trigger (negative means after); in seconds
    :param auroc_binsize: number; optional
        Bin size for auROC calculation; default is 0.1 s (Cohen et al., Nature, 2012)
    :return: auroc_curve: numpy.ndarray
        The auROC curve
    """

    # Grab baseline period histogram
    baseline_points_mask = (edges >= -pre_stimulus_baseline_start) & (edges < -pre_stimulus_baseline_end)
    baseline_hist = hist[baseline_points_mask[:-1]]

    max_criterion = np.max(hist) + 0.1  # Add a bit more to the max criterion so create a true-positive = 0

    # For every bin during response
    auroc_curve = np.array([])
    for start_bin in np.arange(edges[0], edges[-1], auroc_binsize):
        cur_points_mask = (edges >= start_bin) & (edges < start_bin + auroc_binsize)
        cur_hist_values = hist[cur_points_mask[:-1]]

        if max_criterion > 0:
            thresholds = np.linspace(0, max_criterion, int(max_criterion / 0.1), endpoint=True)
        else:
            thresholds = [0, 1]  # Fix for when there's zero spikes to still get auROC=0.5

        false_positive = []
        true_positive = []
        for t in thresholds:
            response_above_t = cur_hist_values >= t
            baseline_above_t = baseline_hist >= t

            false_positive.append(sum(baseline_above_t) / len(baseline_hist))
            true_positive.append(sum(response_above_t) / len(cur_hist_values))
        auroc_curve = np.append(auroc_curve, np.trapz(sorted(true_positive), sorted(false_positive)))
        # # For debugging
        # mpl.use('TkAgg')
        # plt.figure()
        # plt.plot(false_positive, true_positive)
        # plt.show()

    return auroc_curve


def run_calculate_auROC(cur_unitData: dict,
                        session_name: str,
                        trial_or_response_aligned: str,
                        pre_stimulus_baseline_start: float,
                        pre_stimulus_baseline_end: float,
                        pre_stimulus_raster: float,
                        post_stimulus_raster: float,
                        respLatency_filter: float=0,
                        shock_flag: str or int='All',
                        trial_type: str='GO',
                        byAM_depth: bool=False,
                        amdepth_subset: list or None=None,
                        psth_binsize: float=0.01,
                        auroc_binsize: float=0.1,
                        from_JSON: bool=False
                        ):
    """
    This function processes data for the area under Receiver Operating Characteristic curve (auROC) calculation

    From Cohen et al., Nature, 2012:
        a, Raster plot from 15 trials of 149 big-reward trials from a dopaminergic
        neuron. r1 and r2 correspond to two example 100-ms bins. b, Average firing rate of this neuron. c, Area
        under the receiver operating characteristic curve (auROC) for r1, in which the neuron increased its firing
        rate relative to baseline. We compared the histogram of spike counts during the baseline period (dashed
        line) to that during a given bin (solid line) by moving a criterion from zero to the maximum firing rate (in
        this example, 68 spikes/s). We then plotted the probability that the activity during r1 was greater than the
        criteria against the probability that the baseline activity was greater than the criteria. The area under this
        curve quantifies the degree of overlap between the two spike count distributions (i.e., the discriminability
        of the two).

    :param dict cur_unitData:
        A dictionary holding all relevant info about a unit's firing
    :param str session_name:
        The name of the session we're interested in calculating auROCs for
    :param str trial_or_response_aligned:
        'trialAligned' or 'responseAligned'
    :param number pre_stimulus_baseline_start:
        Start of period to calculate the baseline for the auROC in relation to trigger (negative means after); in seconds
    :param number pre_stimulus_baseline_end:
        End of period to calculate the baseline for the auROC in relation to trigger (negative means after); in seconds
    :param number pre_stimulus_raster:
        Start of PSTH in relation to trigger (negative means after); in seconds
    :param number post_stimulus_raster:
        End of PSTH in relation to trigger (negative means before, but not sure why you would use negative); in seconds
    :param str or int shock_flag:
        0, 1 or 'All' to get trials with shock flag off(0) or on(1)
    :param str trial_type:
        'Hit', 'Miss', 'FA' or 'GO'. Self-explanatory. 'GO' gets all go trials regarless of outcome
    :param bool byAM_depth:
        Separate output by AM depth
    :param number psth_binsize:
        Bin size for PSTH calculation; default is 0.01 s (Cohen et al., Nature, 2012)
    :param number auroc_binsize:
        Bin size for auROC calculation; default is 0.1 s (Cohen et al., Nature, 2012)
    :return: dict cur_unitData:
    """

    # For PSTH calculation
    bin_cuts = np.arange(-pre_stimulus_raster, post_stimulus_raster, psth_binsize)

    # Load data and calculate auROCs based on trial responses aligned to SpoutOff triggering a hit
    # Need to create a deep copy here or pandas will change original input (incredibly)
    key_filter = ['Trial_spikes', 'Response_spikes', 'Hit', 'Miss', 'FA', 'ShockFlag', 'AMdepth', 'RespLatency']
    if from_JSON:
        copy_relevant_unitData = {your_key: cur_unitData[your_key] for your_key in key_filter}
    else:
        copy_relevant_unitData = {your_key: cur_unitData["Session"][session_name][your_key] for your_key in key_filter}
    try:
        cur_df = pd.DataFrame.from_dict(copy_relevant_unitData)

        # Remove trials with response latencies lower than the filter
        cur_df = cur_df[cur_df['RespLatency'] >= respLatency_filter]

    except ValueError:
        print('Loading data for auROC calculation failed with ' + cur_unitData['Unit'] + ' ---- ' + session_name)
        return cur_unitData

    trial_type_switchboard = dict({
        'Hit': 0,
        'Miss': 0,
        'FA': 0,
        'GO': 0
    })

    assert trial_type in ('Hit', 'Miss', 'FA', 'GO'), 'Trial type must be Hit, Miss, FA or GO'

    if trial_type == 'Hit':
        trial_type_switchboard['Hit'] = 1
    elif trial_type == 'Miss':
        trial_type_switchboard['Miss'] = 1
    elif trial_type == 'FA':
        trial_type_switchboard['FA'] = 1
    else:  # assume 'GO'
        trial_type_switchboard['GO'] = 1

    assert trial_or_response_aligned in ('trialAligned', 'responseAligned'), \
        'trial_or_response_aligned must be trialAligned or responseAligned'
    if trial_or_response_aligned == 'trialAligned':
        spike_times_field = 'Trial_spikes'
    else:  # 'responseAligned'
        spike_times_field = 'Response_spikes'

    output_name = ''  # Example: responseAligned_Hit_ShockFlagAll_AMdepthAll
    if trial_or_response_aligned == 'trialAligned':
        output_name += 'TrialAligned_'
    else:
        output_name += 'ResponseAligned_'

    output_name += trial_type

    assert shock_flag in (0, 1, 'All'), 'ShockFlag must be 0, 1 or \'All\' '
    # if shock_flag == 'All':
    #     output_name += '_shockFlagAll_'
    if shock_flag == 1:
        output_name += '_shockFlagOn'
    elif shock_flag == 0:
        output_name += '_shockFlagOff'

    if byAM_depth:
        if amdepth_subset is None:
            amdepths = np.round(sorted(list(set(copy_relevant_unitData['AMdepth']))), 2)
        else:
            amdepths = amdepth_subset
    else:
        amdepths = ['Combined',]

    if respLatency_filter > 0:
        output_name += '_respLatencyFilter'

    output_name_suffix = ''
    for cur_amdepth in amdepths:
        # Grab spike times
        if byAM_depth:
            output_name_suffix = '_AMdepth' + str(cur_amdepth)
            if shock_flag == 'All':
                if trial_type_switchboard['GO'] == 0:
                    spike_times = cur_df[(cur_df['Hit'] == trial_type_switchboard['Hit']) &
                                         (cur_df['Miss'] == trial_type_switchboard['Miss']) &
                                         (cur_df['FA'] == trial_type_switchboard['FA']) &
                                         (cur_df['AMdepth'] == cur_amdepth)][spike_times_field]
                else:
                    spike_times = cur_df[((cur_df['Hit'] == 1) | (cur_df['Miss'] == 1)) &
                                         (cur_df['AMdepth'] == cur_amdepth)][spike_times_field]
            else:
                if trial_type_switchboard['GO'] == 0:
                    spike_times = cur_df[(cur_df['Hit'] == trial_type_switchboard['Hit']) &
                                         (cur_df['Miss'] == trial_type_switchboard['Miss']) &
                                         (cur_df['FA'] == trial_type_switchboard['FA']) &
                                         (cur_df['ShockFlag'] == shock_flag) &
                                         (cur_df['AMdepth'] == cur_amdepth)][spike_times_field]
                else:
                    spike_times = cur_df[((cur_df['Hit'] == 1) | (cur_df['Miss'] == 1)) &
                                         (cur_df['ShockFlag'] == shock_flag) &
                                         (cur_df['AMdepth'] == cur_amdepth)][spike_times_field]
        else:
            if amdepth_subset == None:
                if shock_flag == 'All':
                    if trial_type_switchboard['GO'] == 0:
                        spike_times = cur_df[(cur_df['Hit'] == trial_type_switchboard['Hit']) &
                                             (cur_df['Miss'] == trial_type_switchboard['Miss']) &
                                             (cur_df['FA'] == trial_type_switchboard['FA'])][spike_times_field]
                    else:
                        spike_times = cur_df[(cur_df['Hit'] == 1) | (cur_df['Miss'] == 1)][spike_times_field]
                else:
                    if trial_type_switchboard['GO'] == 0:
                        spike_times = cur_df[(cur_df['Hit'] == trial_type_switchboard['Hit']) &
                                             (cur_df['Miss'] == trial_type_switchboard['Miss']) &
                                             (cur_df['FA'] == trial_type_switchboard['FA']) &
                                             (cur_df['ShockFlag'] == shock_flag)][spike_times_field]
                    else:
                        spike_times = cur_df[((cur_df['Hit'] == 1) | (cur_df['Miss'] == 1)) &
                                             (cur_df['ShockFlag'] == shock_flag)][spike_times_field]

            else:
                output_name_suffix = '_middBs'
                if shock_flag == 'All':
                    if trial_type_switchboard['GO'] == 0:
                        spike_times = cur_df[(cur_df['Hit'] == trial_type_switchboard['Hit']) &
                                             (cur_df['Miss'] == trial_type_switchboard['Miss']) &
                                             (cur_df['FA'] == trial_type_switchboard['FA']) &
                                             (np.in1d(cur_df['AMdepth'].values, amdepth_subset))][spike_times_field]
                    else:
                        spike_times = cur_df[((cur_df['Hit'] == 1) | (cur_df['Miss'] == 1)) &
                                             (np.in1d(cur_df['AMdepth'].values, amdepth_subset))][spike_times_field]
                else:
                    if trial_type_switchboard['GO'] == 0:
                        spike_times = cur_df[(cur_df['Hit'] == trial_type_switchboard['Hit']) &
                                             (cur_df['Miss'] == trial_type_switchboard['Miss']) &
                                             (cur_df['FA'] == trial_type_switchboard['FA']) &
                                             (cur_df['ShockFlag'] == shock_flag) &
                                             (np.in1d(cur_df['AMdepth'].values, amdepth_subset))][spike_times_field]
                    else:
                        spike_times = cur_df[((cur_df['Hit'] == 1) | (cur_df['Miss'] == 1)) &
                                             (cur_df['ShockFlag'] == shock_flag) &
                                             (np.in1d(cur_df['AMdepth'].values, amdepth_subset))][spike_times_field]
        # If no trials, skip
        if len(spike_times) == 0:
            cur_unitData["Session"][session_name][output_name + output_name_suffix + '_psth'] = []
            cur_unitData["Session"][session_name][output_name + output_name_suffix + '_auroc'] = []
            continue

        # Flatten all trials into a 1D array
        zero_centered_spikes = np.concatenate(spike_times.values.ravel())

        # Generate a PSTH
        hist, edges = np.histogram(zero_centered_spikes, bins=bin_cuts)

        # Convert to Hz/trial
        hist = np.round((hist / len(spike_times.index)) / psth_binsize, 4)

        # Calculate auROC
        auroc_curve = _auROC_response_curve(hist, edges,
                                            pre_stimulus_baseline_start, pre_stimulus_baseline_end,
                                            auroc_binsize=auroc_binsize)
        if from_JSON:
            cur_unitData[output_name + output_name_suffix + '_psth'] = hist
            cur_unitData[output_name + output_name_suffix + '_auroc'] = auroc_curve
        else:
            cur_unitData["Session"][session_name][output_name + output_name_suffix + '_psth'] = hist
            cur_unitData["Session"][session_name][output_name + output_name_suffix + '_auroc'] = auroc_curve

    return cur_unitData


def run_auROC_extractions(cur_unitData, session_name, extraction_specs, common_kwargs=None):
    """Run several auROC extractions on one session in a single call.

    Thin orchestrator around ``run_calculate_auROC``: rather than writing a
    separate hardcoded function per trial-type/alignment/shock/AM-depth
    combination, describe each desired extraction as a dict of
    ``run_calculate_auROC`` keyword arguments and pass a list of them here.
    This is what drives ``SETTINGS_DICT['AUROC_EXTRACTIONS_ACTIVE'/'_PASSIVE']``
    in the notebook.

    ``run_calculate_auROC`` derives its output field name deterministically
    from its parameters (e.g. ``trial_type='Hit'``, ``trial_or_response_aligned=
    'responseAligned'``, ``shock_flag=1``, ``amdepth_subset=[...]`` ->
    ``'ResponseAligned_Hit_shockFlagOn_middBs_auroc'``) — see that function's
    docstring/body for the exact naming rules. So the field names referenced
    in ``AUROC_TRIALTYPES``/``TS_TRIALTYPES`` are produced automatically as
    long as ``extraction_specs`` includes a matching set of parameters; no
    separate name bookkeeping is needed here.

    Args:
        cur_unitData (dict): Unit data to update in place (passed through to
            ``run_calculate_auROC`` for every spec).
        session_name (str): Session key to compute auROCs for.
        extraction_specs (list[dict]): One dict per desired extraction, each
            containing the keyword arguments ``run_calculate_auROC`` accepts
            beyond ``cur_unitData``/``session_name`` (typically at least
            ``trial_or_response_aligned`` and ``trial_type``; optionally
            ``shock_flag``, ``byAM_depth``, ``amdepth_subset``,
            ``respLatency_filter``). Values here override ``common_kwargs``
            for that extraction.
        common_kwargs (dict, optional): Keyword arguments shared by every
            extraction (typically ``pre_stimulus_baseline_start/end``,
            ``pre_stimulus_raster``, ``post_stimulus_raster``,
            ``psth_binsize``, ``auroc_binsize``) so they don't need to be
            repeated in every spec.

    Returns:
        dict: ``cur_unitData``, updated with one set of ``<name>_psth``/
        ``<name>_auroc`` fields per extraction spec.
    """
    common_kwargs = common_kwargs or {}
    for spec in extraction_specs:
        cur_unitData = run_calculate_auROC(cur_unitData, session_name=session_name,
                                           **{**common_kwargs, **spec})
    return cur_unitData


def _resolve_amdepth_subset(cur_unitData, session_name, target_db_subset, tolerance_db=1.0):
    """Convert a nominal AM-depth subset (dB re:100%) into the actual linear AMdepth values present in one session.

    ``run_calculate_auROC``'s ``amdepth_subset`` matches trials by exact
    equality against the raw (linear) ``AMdepth`` column, but real stimulus
    calibration means the actual depth for a nominal target (e.g. -9 dB) may
    be stored as something like -9.1 dB rather than exactly -9.0. Resolving
    against real per-session depths (within a tolerance) avoids silently
    matching zero trials when nominal dB targets are used directly.

    Args:
        cur_unitData (dict): Unit data; only ``cur_unitData["Session"]
            [session_name]['AMdepth']`` is read.
        session_name (str): Session to read AM depths from.
        target_db_subset (list[float]): Nominal AM depths, dB re:100%
            (e.g. ``[-6, -9, -12]``).
        tolerance_db (float): Maximum allowed difference (dB) between a
            target and an actual depth for it to count as a match. AM depths
            in this task are spaced ~3 dB apart, so the default of 1 dB
            comfortably absorbs calibration drift without crossing into a
            neighboring nominal depth.

    Returns:
        list[float]: The actual (linear) AMdepth values present in this
        session closest to each target found within tolerance. Targets with
        no match within tolerance are skipped (a warning is printed).
    """
    actual_depths = np.array(cur_unitData['Session'][session_name]['AMdepth'], dtype=float)
    actual_depths = np.unique(actual_depths[actual_depths > 0])  # exclude 0/catch trials
    actual_db = 20 * np.log10(actual_depths)

    resolved = []
    for target in target_db_subset:
        diffs = np.abs(actual_db - target)
        idx = np.argmin(diffs)
        if diffs[idx] <= tolerance_db:
            resolved.append(float(actual_depths[idx]))
        else:
            print(f"No AM depth within {tolerance_db} dB of {target} dB found in session "
                  f"'{session_name}'; skipping that depth.")
    return resolved


def _is_active_session(session_name):
    """Classify a session name as 'active' (aversive/active behavior) vs passive (pre/post/post1h).

    Args:
        session_name (str): Session key, e.g. from ``cur_unitData["Session"].keys()``.

    Returns:
        bool: True if the session name contains 'active' or 'aversive' (case-insensitive).
    """
    s = session_name.lower()
    return 'active' in s or 'aversive' in s


def run_auROC_pipeline(input_list):
    """Compute all configured auROC extractions for every session of one unit, and persist to its JSON.

    For each session, dispatches to either
    ``SETTINGS_DICT['AUROC_EXTRACTIONS_ACTIVE']`` or
    ``SETTINGS_DICT['AUROC_EXTRACTIONS_PASSIVE']`` (via ``_is_active_session``)
    and runs those specs through ``run_auROC_extractions``, using shared
    baseline/raster/binsize settings for both. Results are written back onto
    the unit's JSON file so they're available to
    ``build_auROC_data_dict``/``auROC_heatmap_plotter``/``run_ts_clustering``
    without recomputation.

    Args:
        input_list (tuple): ``(file_name, SETTINGS_DICT)`` where ``file_name``
            is the unit JSON path and ``SETTINGS_DICT`` supplies
            ``AUROC_BASELINE_START``/``AUROC_BASELINE_END``,
            ``AUROC_PRE_STIMULUS_DURATION``/``AUROC_POST_STIMULUS_DURATION``,
            ``AUROC_PSTH_BINSIZE``, ``AUROC_BIN_SIZE``,
            ``AUROC_RESPLATENCY_FILTER``, ``AUROC_EXTRACTIONS_ACTIVE``, and
            ``AUROC_EXTRACTIONS_PASSIVE`` (each a list of
            ``run_calculate_auROC``-kwarg dicts, see ``run_auROC_extractions``),
            plus ``AUROC_MIDDB_SUBSET``/``AUROC_MIDDB_TOLERANCE_DB`` (default
            1.0 dB) used to resolve any spec with ``amdepth_subset ==
            'MIDDB_SUBSET'`` via ``_resolve_amdepth_subset``.

    Returns:
        None. Overwrites ``<unit_id>_unitData.json`` in place with the new
        auROC fields added under each session.
    """
    file_name, SETTINGS_DICT = input_list

    data_dict = _load_one_json(file_name)

    common_kwargs = dict(
        pre_stimulus_baseline_start=SETTINGS_DICT['AUROC_BASELINE_START'],
        pre_stimulus_baseline_end=SETTINGS_DICT['AUROC_BASELINE_END'],
        pre_stimulus_raster=SETTINGS_DICT['AUROC_PRE_STIMULUS_DURATION'],
        post_stimulus_raster=SETTINGS_DICT['AUROC_POST_STIMULUS_DURATION'],
        psth_binsize=SETTINGS_DICT['AUROC_PSTH_BINSIZE'],
        auroc_binsize=SETTINGS_DICT['AUROC_BIN_SIZE'],
        respLatency_filter=SETTINGS_DICT.get('AUROC_RESPLATENCY_FILTER', 0),
    )
    active_specs = SETTINGS_DICT['AUROC_EXTRACTIONS_ACTIVE']
    passive_specs = SETTINGS_DICT['AUROC_EXTRACTIONS_PASSIVE']
    middb_db_subset = SETTINGS_DICT.get('AUROC_MIDDB_SUBSET')
    middb_tolerance_db = SETTINGS_DICT.get('AUROC_MIDDB_TOLERANCE_DB', 1.0)

    for session_name in data_dict['Session'].keys():
        specs = active_specs if _is_active_session(session_name) else passive_specs

        # Specs that want the "middBs" subset mark it with the sentinel string
        # 'MIDDB_SUBSET' instead of an explicit amdepth_subset list, since the actual
        # per-session AM depths can drift slightly from the nominal dB targets
        # (calibration) and must be resolved fresh from this session's own data.
        resolved_specs = []
        for spec in specs:
            if spec.get('amdepth_subset') == 'MIDDB_SUBSET':
                spec = {**spec, 'amdepth_subset': _resolve_amdepth_subset(
                    data_dict, session_name, middb_db_subset, tolerance_db=middb_tolerance_db)}
            resolved_specs.append(spec)

        data_dict = run_auROC_extractions(data_dict, session_name, resolved_specs, common_kwargs=common_kwargs)

    write_json(data_dict, SETTINGS_DICT['OUTPUT_PATH'] + '/JSON files', data_dict['Unit'] + '_unitData.json')


def build_auROC_data_dict(filtered_files, session_type_resolver=_is_active_session):
    """Load every unit's JSON and bucket its sessions into pre/post/post1h/active.

    Produces the ``{unit_id: {'pre': {...fields...}, 'post': {...}, 'post1h':
    {...}, 'active': {...}}}`` structure expected by
    ``auROC_heatmap_plotter.run_auROC_heatmap_pipeline`` and
    ``auROC_tslearnClustering_mp.run_ts_clustering`` — neither of which loads
    JSON data itself.

    Args:
        filtered_files (Iterable[str]): Unit JSON paths to load (e.g. from
            ``get_JSON_data.get_JSON_data``), already populated with auROC
            fields via ``run_auROC_pipeline``.
        session_type_resolver (Callable[[str], bool], optional): Used to
            classify a session name as active; defaults to
            ``_is_active_session``. Passive sessions are further split into
            'pre'/'post1h'/'post' by substring match; anything matching none
            of these is skipped with a printed warning.

    Returns:
        dict: Maps unit ID -> {'pre'|'post'|'post1h'|'active': session data
        dict}. A unit/bucket is only present if a matching session was found
        for it.
    """
    data_dict = {}
    for file_name in filtered_files:
        unit_data = _load_one_json(file_name)
        if unit_data is None:
            continue
        buckets = {}
        for session_name, session_data in unit_data['Session'].items():
            if session_type_resolver(session_name):
                bucket = 'active'
            else:
                s = session_name.lower()
                if 'post1h' in s:
                    bucket = 'post1h'
                elif 'post' in s:
                    bucket = 'post'
                elif 'pre' in s:
                    bucket = 'pre'
                else:
                    print(f"Could not classify session '{session_name}' as pre/post/post1h/active for {unit_data['Unit']}; skipping.")
                    continue
            buckets[bucket] = session_data
        data_dict[unit_data['Unit']] = buckets
    return data_dict


def run_full_auROC_pipeline(filtered_files, SETTINGS_DICT):
    """Compute all configured auROC extractions for every unit in ``filtered_files``.

    This is the notebook-level driver: runs ``run_auROC_pipeline`` once per
    unit, serially or via a process pool per ``SETTINGS_DICT['MULTIPROCESS']``.

    Args:
        filtered_files (Iterable[str]): Unit JSON paths to process.
        SETTINGS_DICT (dict): Pipeline settings. Uses ``MULTIPROCESS`` and
            ``NUMBER_OF_CORES``; the rest are forwarded to ``run_auROC_pipeline``.

    Returns:
        None. Overwrites each unit's JSON in place with the new auROC fields
        (see ``run_auROC_pipeline``).
    """
    if not SETTINGS_DICT['MULTIPROCESS']:
        for fn in filtered_files:
            run_auROC_pipeline((fn, SETTINGS_DICT))
    else:
        input_list = [(fn, SETTINGS_DICT) for fn in filtered_files]
        with Pool(SETTINGS_DICT['NUMBER_OF_CORES']) as pool:
            for _ in pool.imap_unordered(run_auROC_pipeline, input_list, chunksize=1):
                pass


def calculate_auROC_responseAligned_shock(cur_unitData,
                                          session_name,
                                          pre_stimulus_baseline_start,
                                          pre_stimulus_baseline_end,
                                          pre_stimulus_raster,
                                          post_stimulus_raster,
                                          psth_binsize=0.01,
                                          auroc_binsize=0.1
                                          ):
    """
    This function processes data for the area under Receiver Operating Characteristic curve (auROC) calculation


    From Cohen et al., Nature, 2012:
        a, Raster plot from 15 trials of 149 big-reward trials from a dopaminergic
        neuron. r1 and r2 correspond to two example 100-ms bins. b, Average firing rate of this neuron. c, Area
        under the receiver operating characteristic curve (auROC) for r1, in which the neuron increased its firing
        rate relative to baseline. We compared the histogram of spike counts during the baseline period (dashed
        line) to that during a given bin (solid line) by moving a criterion from zero to the maximum firing rate (in
        this example, 68 spikes/s). We then plotted the probability that the activity during r1 was greater than the
        criteria against the probability that the baseline activity was greater than the criteria. The area under this
        curve quantifies the degree of overlap between the two spike count distributions (i.e., the discriminability
        of the two).

    :param cur_unitData: class UnitData
        An object holding all relevant info about a unit's firing
    :param session_name: string
        The name of the session we're interested in calculating auROCs for
    :param pre_stimulus_baseline_start: number
        Start of period to calculate the baseline for the auROC in relation to trigger (negative means after); in seconds
    :param pre_stimulus_baseline_end: number
        End of period to calculate the baseline for the auROC in relation to trigger (negative means after); in seconds
    :param pre_stimulus_raster: number
        Start of PSTH in relation to trigger (negative means after); in seconds
    :param post_stimulus_raster: number
        End of PSTH in relation to trigger (negative means before, but not sure why you would use negative); in seconds
    :param psth_binsize: number; optional
        Bin size for PSTH calculation; default is 0.01 s (Cohen et al., Nature, 2012)
    :param auroc_binsize: number; optional
        Bin size for auROC calculation; default is 0.1 s (Cohen et al., Nature, 2012)

    :return: cur_unitData: class UnitData
    """

    # For PSTH calculation
    bin_cuts = np.arange(-pre_stimulus_raster, post_stimulus_raster, psth_binsize)

    # Load data and calculate auROCs based on trial responses aligned to event
    # Need to create a deep copy here or pandas will change original input (incredibly)
    key_filter = ['Trial_spikes', 'Miss', 'ShockFlag', 'ResponseAligned_times_during_trial']
    copy_relevant_unitData = {your_key: cur_unitData["Session"][session_name][your_key] for your_key in key_filter}
    try:
        cur_df = pd.DataFrame.from_dict(copy_relevant_unitData)
    except ValueError:
        print('auROC spoutOffMisses failed with ' + cur_unitData['Unit'] + '----' + session_name)
        return cur_unitData

    # Grab spikes around Misses
    # Filter: Miss trials with ShockFlag == 1, re-aligned below to the spout-offset event
    # that occurs during the shock-artifact window (rather than to trial onset) -> 'ResponseAligned_misses_*'
    trial_spikes = cur_df[(cur_df['Miss'] == 1) & (cur_df['ShockFlag'] == 1)]['Trial_spikes']

    # If no misses, skip
    if len(trial_spikes) == 0:
        cur_unitData["Session"][session_name]['ResponseAligned_misses_psth'] = []
        cur_unitData["Session"][session_name]['ResponseAligned_misses_auroc'] = []
        return cur_unitData

    # Find spout offset immediately following the shock
    spoutOffset_all_triggers = cur_df[(cur_df['Miss'] == 1) & (cur_df['ShockFlag'] == 1)][
        'ResponseAligned_times_during_trial'].values
    spoutOffset_all_triggers = [np.array(t) for t in spoutOffset_all_triggers]

    # If no triggers around shock artifact (0.95 to 1.3, plus 0.5 s), skip
    try:
        if len(spoutOffset_all_triggers[0]) == 0:
            cur_unitData["Session"][session_name]['ResponseAligned_misses_psth'] = []
            cur_unitData["Session"][session_name]['ResponseAligned_misses_auroc'] = []
            return cur_unitData
    except TypeError:
        # Silently ignored (no message printed) if spoutOffset_all_triggers isn't indexable this way
        print()
    # try:
    #     spoutOffset_triggers = [cur_trial[(cur_trial > 0.95) & (cur_trial <= 1.8)][0] for cur_trial in spoutOffset_triggers]
    # except IndexError:
    #     print('Spout trigger during shock event was not found; possible timestamp alignment issue. Problem file: ' + session_name)
    #     cur_unitData["Session"][session_name]['ResponseAligned_misses_psth'] = []
    #     cur_unitData["Session"][session_name]['ResponseAligned_misses_auroc'] = []
    #     return cur_unitData
    #
    # # Zero-center spikes around the first spout offset event during shock period
    # zero_centered_spikes = deepcopy(np.array(trial_spikes.values))
    # for trial_idx, spoutOffset_trigger in enumerate(spoutOffset_triggers):
    #     # try:
    #     zero_centered_spikes[trial_idx] -= spoutOffset_trigger
    #     # except ValueError:
    #     #     print()
    spoutOffset_triggers = []
    for cur_trial in spoutOffset_all_triggers:
        spout_triggers = cur_trial[(cur_trial > 0.95) & (cur_trial < 1.8)]
        if len(spout_triggers) > 0:
            spoutOffset_triggers.append(spout_triggers[-1])
        else:
            spoutOffset_triggers.append(np.NaN)
    spoutOffset_triggers = np.array(spoutOffset_triggers)
    # spoutOffset_triggers = [cur_trial[(cur_trial > 0) & (cur_trial < 1)][-1] for cur_trial in spoutOffset_triggers]
    # except IndexError:
    #     print('Spout triggering FA not found; possible timestamp alignment issue. Problem file: ' + session_name)
    #
    #     cur_unitData["Session"][session_name]['ResponseAligned_FAs_psth'] = []
    #     cur_unitData["Session"][session_name]['ResponseAligned_FAs_auroc'] = []
    #     return cur_unitData

    # Remove NaNs from trials where spoutOffsets were not registered
    nan_trials = np.isnan(spoutOffset_triggers)
    print('Removed ' + str(np.sum(nan_trials)) + ' Miss trials without spout offset from: ' + session_name)
    trial_spikes = trial_spikes[~nan_trials]
    spoutOffset_triggers = spoutOffset_triggers[~nan_trials]
    # Zero-center spikes around those events
    zero_centered_spikes = deepcopy(np.array(trial_spikes.values))
    for trial_idx, spoutOffset_trigger in enumerate(spoutOffset_triggers):
        # try:
        zero_centered_spikes[trial_idx] -= spoutOffset_trigger
        # except ValueError:
        #     print()

    # Flatten all trials into a 1D array
    # zero_centered_spikes = np.concatenate(trial_spikes.values.ravel())
    zero_centered_spikes = np.concatenate(zero_centered_spikes.ravel())

    # Generate a PSTH
    hist, edges = np.histogram(zero_centered_spikes, bins=bin_cuts)

    # Convert to Hz/trial
    hist = np.round((hist / len(spoutOffset_triggers)) / psth_binsize, 4)

    # Calculate auROC
    auroc_curve = _auROC_response_curve(hist, edges,
                                        pre_stimulus_baseline_start, pre_stimulus_baseline_end,
                                        auroc_binsize=auroc_binsize)

    cur_unitData["Session"][session_name]['ResponseAligned_misses_psth'] = hist
    cur_unitData["Session"][session_name]['ResponseAligned_misses_auroc'] = auroc_curve

    return cur_unitData
