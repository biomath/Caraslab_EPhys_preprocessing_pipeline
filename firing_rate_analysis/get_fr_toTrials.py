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


def get_fr_toTrials(memory_path,
                    key_path_info,
                    unit_id,
                    output_path,
                    cur_unitData,
                    experiment_tag=None,
                    first_cell_flag=True,
                    breakpoint_offset=0,
                    nonAM_duration_for_fr: dict | int = 0.5,
                    trial_duration_for_fr: dict | int = 0.5,
                    trialOnset_duration_for_fr: dict | int = 0.4,
                    pre_stim_raster=2.,  # For timestamped spikeTimes
                    post_stim_raster=4.,  # For timestamped spikeTimes
                    aftertrial_FR_start: dict | int = 1.3,
                    # For calculating after-stimulus firing rate; useful for Misses
                    aftertrial_FR_end: dict | int = 2,
                    resptime_FR_start: dict | int = 0.3,  # For calculating after-response firing rate
                    resptime_FR_end: dict | int = 1.8,
                    beforeresp_FR_start: dict | int = 0.5,
                    # For calculating pre-response firing rate (will be converted to negative)
                    beforeresp_FR_end: dict | int = 0,
                    spoutreturn_FR_start: dict | int = 0,
                    # For calculating firing rate relative to the first spout return during the after-trial period
                    spoutreturn_FR_end: dict | int = 0.5
                    ):
    """Process spike data around AM-sound trials.

    For each Hit/Miss/FA trial (Reminders and CRs excluded — CRs are only
    used as baseline reference), computes firing rates in several windows
    (pre-trial baseline from the nearest valid preceding CR, trial period,
    trial onset, after-trial, around response time, and before response) and
    collects zero-centered spike rasters aligned to both trial onset and
    response time. Any of the window-size parameters may be passed either as
    a single number (used for all trial types) or as a dict keyed by trial
    type ('Hit'/'Miss'/'FA'/'CR'/'Passive') for per-type windows.

    Writes one row per trial per FR window to
    ``<experiment_tag>_AMsound_firing_rate.csv``, and stores the same data
    (plus per-trial spike rasters) onto ``cur_unitData`` for later
    timeseries/z-score analyses. Each output row/record also carries
    ``SpoutReturn_latency`` (lag from the outcome-triggering spout offset to the
    next spout onset), passed straight through from the trialInfo CSV when
    ``helpers.recalculate_ePsych_responseLatency`` has populated it and NaN
    otherwise (e.g. passive sessions, or sessions that pre-date that step).

    Args:
        memory_path (str): Path to the unit's spike-times file (whitespace-
            delimited timestamps, loaded with ``np.genfromtxt``).
        key_path_info (str): Path to this session's trialInfo CSV; also used
            (via its filename, sans '.csv') as the session key under
            ``cur_unitData["Session"]``. If 'passive' appears in this path,
            response-time-relative windows/baseline logic use the passive branch.
        unit_id (str): Unit identifier, written into the output CSV.
        output_path (str): Directory to write the firing-rate CSV into.
        cur_unitData (dict): Running per-unit data structure to update in place.
        experiment_tag (str, optional): Prefix for the output CSV name; if
            None, the CSV name is empty (effectively disabling the CSV write).
        first_cell_flag (bool): If True, (re)write the CSV with a header row;
            if False, append without a header (assumes the file already exists).
        breakpoint_offset (float): Seconds to add to each trial timestamp to
            align it with this unit's spike-time clock (for concatenated recordings).
        nonAM_duration_for_fr (dict or float): Window length (s), starting at
            the reference CR trial's onset, used for the pre-trial baseline FR.
        trial_duration_for_fr (dict or float): Window length (s) after trial
            onset used for the 'Trial' period FR.
        trialOnset_duration_for_fr (dict or float): Window length (s) after
            trial onset used for the 'TrialOnset' period FR (typically shorter
            than ``trial_duration_for_fr``).
        pre_stim_raster (float): Seconds of spikes to include before trial
            onset in the zero-centered raster.
        post_stim_raster (float): Seconds of spikes to include after trial
            onset in the zero-centered raster.
        aftertrial_FR_start (dict or float): Start offset (s, relative to
            trial onset) of the after-trial FR window; useful for Misses.
        aftertrial_FR_end (dict or float): End offset (s, relative to trial
            onset) of the after-trial FR window.
        resptime_FR_start (dict or float): Start offset (s, relative to
            response time) of the post-response FR window.
        resptime_FR_end (dict or float): End offset (s, relative to response
            time) of the post-response FR window.
        beforeresp_FR_start (dict or float): How far before response time (s)
            the pre-response FR window begins (subtracted from response time).
        beforeresp_FR_end (dict or float): How far before response time (s)
            the pre-response FR window ends (subtracted from response time).
        spoutreturn_FR_start (dict or float): Start offset (s, relative to the
            first spout onset that falls within the after-trial window) of the
            spout-return FR window.
        spoutreturn_FR_end (dict or float): End offset (s, relative to that
            first spout onset) of the spout-return FR window. If no spout onset
            occurs during the after-trial window, the spout-return FR is NaN.

    Returns:
        dict: ``cur_unitData``, updated with per-trial metadata, zero-centered
        trial/response spike rasters, and all FR arrays for this session.
    """
    def _spout_rate_in_window(event_times, win_start, win_end):
        """Count spout events of `event_times` falling in [win_start, win_end) and convert to Hz."""
        n_events = np.count_nonzero((event_times >= win_start) & (event_times < win_end))
        return n_events / (win_end - win_start)

    # Load key files
    info_key_times = read_csv(key_path_info)

    # Load spike times
    spike_times = np.genfromtxt(memory_path)

    # The default is 0, so this will only make a difference if set at function call
    breakpoint_offset_time = breakpoint_offset

    # Align trial timestamps to this unit's spike-time clock once, here, instead of
    # repeatedly adding breakpoint_offset_time at every point of use below. This must
    # happen before relevant_key_times is derived so both it and the previous_cr lookup
    # (which searches the full info_key_times, including CR rows) see the adjusted values.
    info_key_times['Trial_onset'] += breakpoint_offset_time
    info_key_times['Trial_offset'] += breakpoint_offset_time

    # Check for opto tags. Add dummy tags if non-existent
    if 'JitOnset' not in info_key_times.columns:
        info_key_times['JitOnset'] = 0
    if 'LED_TTL' not in info_key_times.columns:
        info_key_times['LED_TTL'] = 0

    # Grab all trials
    # These can be Hit, Miss or FA
    # CR trials are not included to save space, but you can remove this filter if you wish
    relevant_key_times = info_key_times[info_key_times['CR'] == 0].copy()

    # SpoutOffset_times/SpoutOnset_times only exist if recalculate_ePsych_responseLatency has been
    # run on this session's trialInfo.csv — and that function explicitly skips passive sessions —
    # so guard on column presence rather than assuming the columns are always there.
    has_spout_offset_col = 'SpoutOffset_times' in info_key_times.columns
    has_spout_onset_col = 'SpoutOnset_times' in info_key_times.columns

    # SpoutReturn_latency (lag from the outcome-triggering spout offset to the next spout onset) is
    # likewise only present once recalculate_ePsych_responseLatency has run, and never for passive
    # sessions. Ensure the column exists so the per-trial CSV rows and the cur_unitData copy below
    # always have something to reference; when it's missing every trial gets NaN.
    if 'SpoutReturn_latency' not in relevant_key_times.columns:
        relevant_key_times['SpoutReturn_latency'] = np.nan

    # Now grab spike times
    # Baseline will be the CR trial immediately preceding the current trial
    nonAM_FR_list = list()
    trial_FR_list = list()
    trialOnset_FR_list = list()
    aftertrial_FR_list = list()
    resptime_FR_list = list()
    beforeresp_FR_list = list()
    spoutreturn_FR_list = list()

    # Actual spike timestamps around trial and spout offset
    zerocentered_trial_spikes = list()
    zerocentered_response_spikes = list()

    # Per-trial spout offset/onset timestamps (zero-centered to trial onset, like the spike rasters)
    spout_offset_times_list = list()
    spout_onset_times_list = list()

    # Spout offset/onset event rates (Hz) within the same windows as the FR lists above
    spoutOffset_baseline_Hz_list = list()
    spoutOffset_trial_Hz_list = list()
    spoutOffset_trialOnset_Hz_list = list()
    spoutOffset_aftertrial_Hz_list = list()
    spoutOffset_resptime_Hz_list = list()
    spoutOffset_beforeresp_Hz_list = list()
    spoutOffset_spoutreturn_Hz_list = list()

    spoutOnset_baseline_Hz_list = list()
    spoutOnset_trial_Hz_list = list()
    spoutOnset_trialOnset_Hz_list = list()
    spoutOnset_aftertrial_Hz_list = list()
    spoutOnset_resptime_Hz_list = list()
    spoutOnset_beforeresp_Hz_list = list()
    spoutOnset_spoutreturn_Hz_list = list()

    for dummy_index, cur_trial in relevant_key_times.iterrows():
        # Get current trial's response time (i.e., latency)
        # Only relevant for active sessions
        # IMPORTANT NOTE: In miss trials, if latency value is during the AM period, this indicates that the animal
        #   attempted to withdraw from spout before the shock, but returned for some reason. Handle these trials as you wish
        #   For true undetected misses, respLatency (if it exists) will always be after the AM period during the shock period
        if 'passive' not in key_path_info.lower():
            # Get spike times around the current stimulus onset or response latency
            # For baseline, go to the previous trial that resulted in a correct rejection (NO-GO) which
            # may not be the immediately preceding trial
            # Also ensure the RespLatency is NaN, indicating that the animal was stable at the spout
            if cur_trial['Reminder'] == 1:
                # Reminder trials don't have a meaningful preceding CR to search for — just use
                # the baseline moment immediately before this trial's own onset instead.
                previous_cr_onset = cur_trial['Trial_onset']
                valid_baseline = True
            else:
                try:
                    previous_cr = info_key_times[(info_key_times['CR'] == 1) &
                                                 (info_key_times['TrialID'] < cur_trial['TrialID']) &
                                                 np.isnan(info_key_times['RespLatency'])].iloc[-1]
                    previous_cr_onset = previous_cr['Trial_onset']
                    valid_baseline = True
                except IndexError:  # In case there is no valid CR before current trial, make baseline firing = NaN
                    valid_baseline = False

            cur_resptime = cur_trial['RespLatency']  # Either NA or >0

            if cur_trial['Hit'] == 1:
                cur_trial_type = 'Hit'
            elif cur_trial['Miss'] == 1:
                cur_trial_type = 'Miss'
            elif cur_trial['FA'] == 1:
                cur_trial_type = 'FA'  # RespLatency is a different interpretation for FAs, since the start of the trial is arbitrary, but compute anyways
            else:
                cur_trial_type = 'CR'

            # Parse this trial's SpoutOffset_times/SpoutOnset_times (semicolon-joined strings written by
            # helpers.recalculate_ePsych_responseLatency) into arrays, and apply breakpoint_offset_time so
            # they land on the same clock as cur_trial_onset/spike_times. Guard on column presence since
            # recalculate_ePsych_responseLatency may not have been run on every session. An empty window
            # comes back from pandas as NaN (float), not an empty string, so that's checked explicitly too.
            if has_spout_offset_col:
                raw_spout_offsets = cur_trial['SpoutOffset_times']
                if isinstance(raw_spout_offsets, str) and raw_spout_offsets:
                    cur_spout_offsets = np.array([float(t) for t in raw_spout_offsets.split(';')]) + breakpoint_offset_time
                else:
                    cur_spout_offsets = np.array([])
            else:
                cur_spout_offsets = np.array([])

            if has_spout_onset_col:
                raw_spout_onsets = cur_trial['SpoutOnset_times']
                if isinstance(raw_spout_onsets, str) and raw_spout_onsets:
                    cur_spout_onsets = np.array([float(t) for t in raw_spout_onsets.split(';')]) + breakpoint_offset_time
                else:
                    cur_spout_onsets = np.array([])
            else:
                cur_spout_onsets = np.array([])

        else:
            # RespLatency is not considered for passive sessions when gathering the baseline trial
            try:
                previous_cr = info_key_times[(info_key_times['CR'] == 1) &
                                             (info_key_times['TrialID'] < cur_trial['TrialID'])].iloc[-1]
                previous_cr_onset = previous_cr['Trial_onset']
                valid_baseline = True
            except IndexError:  # In case there is no valid CR before current trial, make baseline firing = NaN
                valid_baseline = False
            cur_resptime = 0
            cur_trial_type = 'Passive'

            # Passive sessions never have spout-related data — recalculate_ePsych_responseLatency
            # skips them entirely, so SpoutOffset_times/SpoutOnset_times are never populated for these trials
            cur_spout_offsets = np.array([])
            cur_spout_onsets = np.array([])

        # Trial_onset/Trial_offset already include breakpoint_offset_time (applied once, up front)
        cur_trial_onset = cur_trial['Trial_onset']
        cur_trial_offset = cur_trial['Trial_offset']

        # Zero-center spout events to trial onset, matching the spike rasters
        spout_offset_times_list.append(cur_spout_offsets - cur_trial_onset)
        spout_onset_times_list.append(cur_spout_onsets - cur_trial_onset)

        # Use different onsets depending on trial type
        if type(trial_duration_for_fr) is dict:
            cur_trial_duration_for_fr_s = trial_duration_for_fr[cur_trial_type]
        else:
            cur_trial_duration_for_fr_s = trial_duration_for_fr

        if type(trialOnset_duration_for_fr) is dict:
            cur_trialOnset_duration_for_fr_s = trialOnset_duration_for_fr[cur_trial_type]
        else:
            cur_trialOnset_duration_for_fr_s = trialOnset_duration_for_fr

        if type(aftertrial_FR_start) is dict:
            cur_aftertrial_start = aftertrial_FR_start[cur_trial_type]
            cur_aftertrial_end = aftertrial_FR_end[cur_trial_type]
        else:
            cur_aftertrial_start = aftertrial_FR_start
            cur_aftertrial_end = aftertrial_FR_end

        if type(resptime_FR_start) is dict:
            cur_resptime_start = resptime_FR_start[cur_trial_type]
            cur_resptime_end = resptime_FR_end[cur_trial_type]
        else:
            cur_resptime_start = resptime_FR_start
            cur_resptime_end = resptime_FR_end

        if type(beforeresp_FR_start) is dict:
            cur_beforeresp_start = beforeresp_FR_start[cur_trial_type]
            cur_beforeresp_end = beforeresp_FR_end[cur_trial_type]
        else:
            cur_beforeresp_start = beforeresp_FR_start
            cur_beforeresp_end = beforeresp_FR_end

        if type(spoutreturn_FR_start) is dict:
            cur_spoutreturn_start = spoutreturn_FR_start[cur_trial_type]
            cur_spoutreturn_end = spoutreturn_FR_end[cur_trial_type]
        else:
            cur_spoutreturn_start = spoutreturn_FR_start
            cur_spoutreturn_end = spoutreturn_FR_end

        # SPIKE TIMES
        # Get spikes in the interval [trial_onset - pre_stim_raster; trial_onset + post_stim_raster]
        spikes_around_trial = spike_times[
            (spike_times >= (cur_trial_onset - pre_stim_raster)) &
            (spike_times < (cur_trial_onset + post_stim_raster))]

        # Zero center around trial onset
        zerocentered_trial_spikes.append(spikes_around_trial - cur_trial_onset)

        # Zero center around response time
        if np.isnan(cur_resptime):
            # No response latency for this trial: record zero response-aligned spikes
            # (an empty array, not a [None] sentinel, so np.round/downstream numeric
            # comparisons against this field don't break)
            zerocentered_response_spikes.append(np.array([]))
        else:
            zerocentered_response_spikes.append(
                spikes_around_trial - (cur_trial_onset + cur_resptime))

        if valid_baseline:
            nonAM_spikes = spike_times[
                (previous_cr_onset < spike_times) &
                (spike_times < previous_cr_onset + nonAM_duration_for_fr)]
        else:
            nonAM_spikes = None

        trial_spikes = spike_times[(spike_times >= cur_trial_onset) &
                                   (spike_times < (cur_trial_onset + cur_trial_duration_for_fr_s))]
        trialOnset_spikes = spike_times[(spike_times >= cur_trial_onset) &
                                        (spike_times < (cur_trial_onset + cur_trialOnset_duration_for_fr_s))]
        aftertrial_spikes = spike_times[
            (spike_times >= (cur_trial_onset + cur_aftertrial_start)) &
            (spike_times < (cur_trial_onset + cur_aftertrial_end))]

        resptime_spikes = spike_times[
            (spike_times >= (cur_trial_onset + cur_resptime + cur_resptime_start)) &
            (spike_times < (cur_trial_onset + cur_resptime + cur_resptime_end))]

        beforeresp_spikes = spike_times[
            (spike_times >= (cur_trial_onset + cur_resptime - cur_beforeresp_start)) &
            (spike_times < (cur_trial_onset + cur_resptime - cur_beforeresp_end))]

        # SpoutReturn window: anchored to the first spout onset (spout return) that falls within
        # this trial's after-trial window. If the animal never returns to the spout during that
        # window (or there is no spout data at all, e.g. passive sessions), the window is undefined
        # and every SpoutReturn quantity below is NaN.
        aftertrial_window_start = cur_trial_onset + cur_aftertrial_start
        aftertrial_window_end = cur_trial_onset + cur_aftertrial_end
        spout_returns_in_aftertrial = cur_spout_onsets[
            (cur_spout_onsets >= aftertrial_window_start) &
            (cur_spout_onsets < aftertrial_window_end)]
        if len(spout_returns_in_aftertrial) > 0:
            first_spout_return = spout_returns_in_aftertrial[0]
            spoutreturn_window_start = first_spout_return + cur_spoutreturn_start
            spoutreturn_window_end = first_spout_return + cur_spoutreturn_end
            spoutreturn_spikes = spike_times[
                (spike_times >= spoutreturn_window_start) &
                (spike_times < spoutreturn_window_end)]
        else:
            first_spout_return = np.nan
            spoutreturn_window_start = np.nan
            spoutreturn_window_end = np.nan
            spoutreturn_spikes = None

        # FR calculations
        if valid_baseline:
            cur_nonAM_FR = len(nonAM_spikes) / nonAM_duration_for_fr
        else:
            cur_nonAM_FR = np.nan
        cur_trial_FR = len(trial_spikes) / cur_trial_duration_for_fr_s
        cur_trialOnset_FR = len(trialOnset_spikes) / cur_trialOnset_duration_for_fr_s
        cur_aftertrial_fr = len(aftertrial_spikes) / (cur_aftertrial_end - cur_aftertrial_start)
        cur_resptime_fr = len(resptime_spikes) / (cur_resptime_end - cur_resptime_start)
        cur_beforeresp_fr = len(beforeresp_spikes) / (cur_beforeresp_start - cur_beforeresp_end)
        if spoutreturn_spikes is None:
            cur_spoutreturn_fr = np.nan
        else:
            cur_spoutreturn_fr = len(spoutreturn_spikes) / (cur_spoutreturn_end - cur_spoutreturn_start)

        nonAM_FR_list.append(cur_nonAM_FR)
        trial_FR_list.append(cur_trial_FR)
        trialOnset_FR_list.append(cur_trialOnset_FR)
        aftertrial_FR_list.append(cur_aftertrial_fr)
        resptime_FR_list.append(cur_resptime_fr)
        beforeresp_FR_list.append(cur_beforeresp_fr)
        spoutreturn_FR_list.append(cur_spoutreturn_fr)

        # SPOUT EVENT RATES (Hz) within the exact same windows as the FR calculations above
        if valid_baseline:
            cur_nonAM_spoutOffset_Hz = _spout_rate_in_window(
                cur_spout_offsets, previous_cr_onset, previous_cr_onset + nonAM_duration_for_fr)
            cur_nonAM_spoutOnset_Hz = _spout_rate_in_window(
                cur_spout_onsets, previous_cr_onset, previous_cr_onset + nonAM_duration_for_fr)
        else:
            cur_nonAM_spoutOffset_Hz = np.nan
            cur_nonAM_spoutOnset_Hz = np.nan

        cur_trial_spoutOffset_Hz = _spout_rate_in_window(
            cur_spout_offsets, cur_trial_onset, cur_trial_onset + cur_trial_duration_for_fr_s)
        cur_trial_spoutOnset_Hz = _spout_rate_in_window(
            cur_spout_onsets, cur_trial_onset, cur_trial_onset + cur_trial_duration_for_fr_s)

        cur_trialOnset_spoutOffset_Hz = _spout_rate_in_window(
            cur_spout_offsets, cur_trial_onset, cur_trial_onset + cur_trialOnset_duration_for_fr_s)
        cur_trialOnset_spoutOnset_Hz = _spout_rate_in_window(
            cur_spout_onsets, cur_trial_onset, cur_trial_onset + cur_trialOnset_duration_for_fr_s)

        cur_aftertrial_spoutOffset_Hz = _spout_rate_in_window(
            cur_spout_offsets, cur_trial_onset + cur_aftertrial_start, cur_trial_onset + cur_aftertrial_end)
        cur_aftertrial_spoutOnset_Hz = _spout_rate_in_window(
            cur_spout_onsets, cur_trial_onset + cur_aftertrial_start, cur_trial_onset + cur_aftertrial_end)

        cur_resptime_spoutOffset_Hz = _spout_rate_in_window(
            cur_spout_offsets, cur_trial_onset + cur_resptime + cur_resptime_start,
            cur_trial_onset + cur_resptime + cur_resptime_end)
        cur_resptime_spoutOnset_Hz = _spout_rate_in_window(
            cur_spout_onsets, cur_trial_onset + cur_resptime + cur_resptime_start,
            cur_trial_onset + cur_resptime + cur_resptime_end)

        cur_beforeresp_spoutOffset_Hz = _spout_rate_in_window(
            cur_spout_offsets, cur_trial_onset + cur_resptime - cur_beforeresp_start,
            cur_trial_onset + cur_resptime - cur_beforeresp_end)
        cur_beforeresp_spoutOnset_Hz = _spout_rate_in_window(
            cur_spout_onsets, cur_trial_onset + cur_resptime - cur_beforeresp_start,
            cur_trial_onset + cur_resptime - cur_beforeresp_end)

        if spoutreturn_spikes is None:
            cur_spoutreturn_spoutOffset_Hz = np.nan
            cur_spoutreturn_spoutOnset_Hz = np.nan
        else:
            cur_spoutreturn_spoutOffset_Hz = _spout_rate_in_window(
                cur_spout_offsets, spoutreturn_window_start, spoutreturn_window_end)
            cur_spoutreturn_spoutOnset_Hz = _spout_rate_in_window(
                cur_spout_onsets, spoutreturn_window_start, spoutreturn_window_end)

        spoutOffset_baseline_Hz_list.append(cur_nonAM_spoutOffset_Hz)
        spoutOffset_trial_Hz_list.append(cur_trial_spoutOffset_Hz)
        spoutOffset_trialOnset_Hz_list.append(cur_trialOnset_spoutOffset_Hz)
        spoutOffset_aftertrial_Hz_list.append(cur_aftertrial_spoutOffset_Hz)
        spoutOffset_resptime_Hz_list.append(cur_resptime_spoutOffset_Hz)
        spoutOffset_beforeresp_Hz_list.append(cur_beforeresp_spoutOffset_Hz)
        spoutOffset_spoutreturn_Hz_list.append(cur_spoutreturn_spoutOffset_Hz)

        spoutOnset_baseline_Hz_list.append(cur_nonAM_spoutOnset_Hz)
        spoutOnset_trial_Hz_list.append(cur_trial_spoutOnset_Hz)
        spoutOnset_trialOnset_Hz_list.append(cur_trialOnset_spoutOnset_Hz)
        spoutOnset_aftertrial_Hz_list.append(cur_aftertrial_spoutOnset_Hz)
        spoutOnset_resptime_Hz_list.append(cur_resptime_spoutOnset_Hz)
        spoutOnset_beforeresp_Hz_list.append(cur_beforeresp_spoutOnset_Hz)
        spoutOnset_spoutreturn_Hz_list.append(cur_spoutreturn_spoutOnset_Hz)

    # Cap floating point precision for optimized memory storage
    nonAM_FR_list = np.round(nonAM_FR_list, 4)
    trial_FR_list = np.round(trial_FR_list, 4)
    trialOnset_FR_list = np.round(trialOnset_FR_list, 4)
    aftertrial_FR_list = np.round(aftertrial_FR_list, 4)
    resptime_FR_list = np.round(resptime_FR_list, 4)
    beforeresp_FR_list = np.round(beforeresp_FR_list, 4)
    spoutreturn_FR_list = np.round(spoutreturn_FR_list, 4)
    zerocentered_trial_spikes = [np.round(_spikes, 4) for _spikes in zerocentered_trial_spikes]
    zerocentered_response_spikes = [np.round(_spikes, 4) for _spikes in zerocentered_response_spikes]
    spout_offset_times_list = [np.round(_spouts, 4) for _spouts in spout_offset_times_list]
    spout_onset_times_list = [np.round(_spouts, 4) for _spouts in spout_onset_times_list]

    spoutOffset_baseline_Hz_list = np.round(spoutOffset_baseline_Hz_list, 4)
    spoutOffset_trial_Hz_list = np.round(spoutOffset_trial_Hz_list, 4)
    spoutOffset_trialOnset_Hz_list = np.round(spoutOffset_trialOnset_Hz_list, 4)
    spoutOffset_aftertrial_Hz_list = np.round(spoutOffset_aftertrial_Hz_list, 4)
    spoutOffset_resptime_Hz_list = np.round(spoutOffset_resptime_Hz_list, 4)
    spoutOffset_beforeresp_Hz_list = np.round(spoutOffset_beforeresp_Hz_list, 4)
    spoutOffset_spoutreturn_Hz_list = np.round(spoutOffset_spoutreturn_Hz_list, 4)

    spoutOnset_baseline_Hz_list = np.round(spoutOnset_baseline_Hz_list, 4)
    spoutOnset_trial_Hz_list = np.round(spoutOnset_trial_Hz_list, 4)
    spoutOnset_trialOnset_Hz_list = np.round(spoutOnset_trialOnset_Hz_list, 4)
    spoutOnset_aftertrial_Hz_list = np.round(spoutOnset_aftertrial_Hz_list, 4)
    spoutOnset_resptime_Hz_list = np.round(spoutOnset_resptime_Hz_list, 4)
    spoutOnset_beforeresp_Hz_list = np.round(spoutOnset_beforeresp_Hz_list, 4)
    spoutOnset_spoutreturn_Hz_list = np.round(spoutOnset_spoutreturn_Hz_list, 4)

    relevant_key_times.loc[:, ['Trial_onset', 'Trial_offset', 'RespLatency', 'SpoutReturn_latency']] = (
        relevant_key_times.loc[:, ['Trial_onset', 'Trial_offset', 'RespLatency', 'SpoutReturn_latency']].round(4)
    )
    relevant_key_times.loc[:, 'AMdepth'] = relevant_key_times['AMdepth'].round(2)

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
                            ['JitOnset'] + ['LED_TTL'] + ['RespLatency'] + ['SpoutReturn_latency'] +
                            ['FR_Hz'] + ['SpoutOffset_Hz'] + ['SpoutOnset_Hz']
                            )
        for dummy_idx in range(0, len(nonAM_FR_list)):
            cur_row = relevant_key_times.iloc[dummy_idx, :]

            for (trial_period, FR_list, spoutOffset_Hz_list, spoutOnset_Hz_list) in zip(
                # Trial periods
                ('Baseline', 'Trial', 'TrialOnset',
                 'Aftertrial', 'RespTime', 'BeforeResp', 'SpoutReturn'),

                # FR lists
                (nonAM_FR_list, trial_FR_list, trialOnset_FR_list,
                 aftertrial_FR_list, resptime_FR_list, beforeresp_FR_list, spoutreturn_FR_list),

                # Spout offset Hz lists (same per-period windows as the FR lists above)
                (spoutOffset_baseline_Hz_list, spoutOffset_trial_Hz_list, spoutOffset_trialOnset_Hz_list,
                 spoutOffset_aftertrial_Hz_list, spoutOffset_resptime_Hz_list, spoutOffset_beforeresp_Hz_list,
                 spoutOffset_spoutreturn_Hz_list),

                # Spout onset Hz lists (same per-period windows as the FR lists above)
                (spoutOnset_baseline_Hz_list, spoutOnset_trial_Hz_list, spoutOnset_trialOnset_Hz_list,
                 spoutOnset_aftertrial_Hz_list, spoutOnset_resptime_Hz_list, spoutOnset_beforeresp_Hz_list,
                 spoutOnset_spoutreturn_Hz_list)
                ):
                try:
                    writer.writerow([unit_id] + [split(REGEX_SEP, key_path_info)[-1][:-4]] +
                                [cur_row['TrialID']] + [cur_row['AMdepth']] + [cur_row['Reminder']] +
                                [cur_row['ShockFlag']] +
                                [cur_row['Hit']] + [cur_row['Miss']] +
                                [cur_row['CR']] + [cur_row['FA']] +
                                [trial_period] + [cur_row['Trial_onset']] + [cur_row['Trial_offset']] +
                                [cur_row['JitOnset']] +
                                [cur_row['LED_TTL']] +
                                [cur_row['RespLatency']] +
                                [cur_row['SpoutReturn_latency']] +
                                [FR_list[dummy_idx]] +
                                [spoutOffset_Hz_list[dummy_idx]] +
                                [spoutOnset_Hz_list[dummy_idx]])
                except KeyError:
                    pass

    # Add all info to unitData
    trialInfo_filename = split(REGEX_SEP, key_path_info)[-1][:-4]
    for key_name in ('TrialID', 'Reminder', 'ShockFlag', 'JitOnset', 'LED_TTL', 'Hit', 'Miss', 'CR', 'FA',
                     'Trial_onset', 'Trial_offset', 'RespLatency', 'SpoutReturn_latency'):
        cur_unitData["Session"][trialInfo_filename][key_name] = relevant_key_times[key_name].values

    cur_unitData["Session"][trialInfo_filename]['AMdepth'] = relevant_key_times['AMdepth'].values
    cur_unitData["Session"][trialInfo_filename]['Trial_spikes'] = zerocentered_trial_spikes
    cur_unitData["Session"][trialInfo_filename]['Response_spikes'] = zerocentered_response_spikes
    cur_unitData["Session"][trialInfo_filename]['SpoutOffset_times'] = spout_offset_times_list
    cur_unitData["Session"][trialInfo_filename]['SpoutOnset_times'] = spout_onset_times_list
    cur_unitData["Session"][trialInfo_filename]['Baseline_FR'] = nonAM_FR_list
    cur_unitData["Session"][trialInfo_filename]['Trial_FR'] = trial_FR_list
    cur_unitData["Session"][trialInfo_filename]['TrialOnset_FR'] = trialOnset_FR_list
    cur_unitData["Session"][trialInfo_filename]['Aftertrial_FR'] = aftertrial_FR_list
    cur_unitData["Session"][trialInfo_filename]['ResponseTime_FR'] = resptime_FR_list
    cur_unitData["Session"][trialInfo_filename]['BeforeResponse_FR'] = beforeresp_FR_list
    cur_unitData["Session"][trialInfo_filename]['SpoutReturn_FR'] = spoutreturn_FR_list

    return cur_unitData