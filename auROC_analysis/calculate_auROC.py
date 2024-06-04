import numpy as np
import pandas as pd
from copy import deepcopy
import copy


def auROC_response_curve(hist, edges, pre_stimulus_baseline_start, pre_stimulus_baseline_end, auroc_binsize=0.1):
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


def calculate_auROC_spoutOffHit(cur_unitData,
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

    # Load data and calculate auROCs based on trial responses aligned to SpoutOff triggering a hit
    # Need to create a deep copy here or pandas will change original input (incredibly)
    #ShockFlag == 1 for direct comparison with Miss (shock)
    key_filter = ['Trial_spikes', 'Hit', 'SpoutOff_times_during_trial', 'ShockFlag']
    copy_relevant_unitData = {your_key: cur_unitData["Session"][session_name][your_key] for your_key in key_filter}
    try:
        cur_df = pd.DataFrame.from_dict(copy_relevant_unitData)
    except ValueError:
        print('auROC SpoutOffHits failed with ' + cur_unitData['Unit'] + '----' + session_name)
        return cur_unitData

    # Grab spikes around hits
    trial_spikes = cur_df[(cur_df['Hit'] == 1) & (cur_df['ShockFlag'] == 1)]['Trial_spikes']

    # If no hits, skip
    if len(trial_spikes) == 0:
        cur_unitData["Session"][session_name]['SpoutOff_hits_psth'] = []
        cur_unitData["Session"][session_name]['SpoutOff_hits_auroc'] = []
        return cur_unitData

    # Find spout offset that triggered Hit
    spoutOffset_all_triggers = cur_df[(cur_df['Hit'] == 1) & (cur_df['ShockFlag'] == 1)]['SpoutOff_times_during_trial'].values
    spoutOffset_all_triggers = [np.array(t) for t in spoutOffset_all_triggers]

    # If no triggers, skip
    # try:
    if len(spoutOffset_all_triggers[0]) == 0:
        cur_unitData["Session"][session_name]['SpoutOff_hits_psth'] = []
        cur_unitData["Session"][session_name]['SpoutOff_hits_auroc'] = []
        return cur_unitData
    # except TypeError:
    #     print()
    #
    # try:
    #     spoutOffset_triggers = [cur_trial[(cur_trial > 0) & (cur_trial < 1)][-1] for cur_trial in spoutOffset_triggers]
    # except IndexError:
    #     print('Spout triggering hit not found; possible timestamp alignment issue. Problem file: ' + session_name)
    #
    #     cur_unitData["Session"][session_name]['SpoutOff_hits_psth'] = []
    #     cur_unitData["Session"][session_name]['SpoutOff_hits_auroc'] = []
    #     return cur_unitData
    #
    # # Zero-center spikes around those events
    # zero_centered_spikes = deepcopy(np.array(trial_spikes.values))
    # for trial_idx, spoutOffset_trigger in enumerate(spoutOffset_triggers):
    #     # try:
    #     zero_centered_spikes[trial_idx] -= spoutOffset_trigger
    #     # except ValueError:
    #     #     print()
    spoutOffset_triggers = []
    for cur_trial in spoutOffset_all_triggers:
        spout_triggers = cur_trial[(cur_trial > 0) & (cur_trial < 1)]
        if len(spout_triggers) > 0:
            spoutOffset_triggers.append(spout_triggers[-1])
        else:
            spoutOffset_triggers.append(np.NaN)
    spoutOffset_triggers = np.array(spoutOffset_triggers)
    # spoutOffset_triggers = [cur_trial[(cur_trial > 0) & (cur_trial < 1)][-1] for cur_trial in spoutOffset_triggers]
    # except IndexError:
    #     print('Spout triggering FA not found; possible timestamp alignment issue. Problem file: ' + session_name)
    #
    #     cur_unitData["Session"][session_name]['SpoutOff_FAs_psth'] = []
    #     cur_unitData["Session"][session_name]['SpoutOff_FAs_auroc'] = []
    #     return cur_unitData

    # Remove NaNs from trials where spoutOffsets were not registered
    nan_trials = np.isnan(spoutOffset_triggers)
    print('Removed ' + str(np.sum(nan_trials)) + ' Hit trials without spout offset from: ' + session_name)
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
    auroc_curve = auROC_response_curve(hist, edges,
                                       pre_stimulus_baseline_start, pre_stimulus_baseline_end,
                                       auroc_binsize=auroc_binsize)

    cur_unitData["Session"][session_name]['SpoutOff_hits_psth'] = hist
    cur_unitData["Session"][session_name]['SpoutOff_hits_auroc'] = auroc_curve

    return cur_unitData


def calculate_auROC_hit(cur_unitData,
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
    For now, this function only calculates the auROC to a Hit trial;
        skip if no hits are found

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

    # Load data and calculate auROCs based on trial responses aligned to SpoutOff triggering a hit
    # Need to create a deep copy here or pandas will change original input (incredibly)
    key_filter = ['Trial_spikes', 'Hit', 'ShockFlag']
    copy_relevant_unitData = {your_key: cur_unitData["Session"][session_name][your_key] for your_key in key_filter}
    try:
        cur_df = pd.DataFrame.from_dict(copy_relevant_unitData)
    except ValueError:
        print('auROC Hit failed with ' + cur_unitData['Unit'] + '----' + session_name)
        return cur_unitData


    # Grab spikes around misses
    trial_spikes = cur_df[(cur_df['Hit'] == 1) & (cur_df['ShockFlag'] == 1)]['Trial_spikes']

    # If no Misses, skip
    if len(trial_spikes) == 0:
        cur_unitData["Session"][session_name]['Hit_psth'] = []
        cur_unitData["Session"][session_name]['Hit_auroc'] = []
        return cur_unitData

    # Flatten all trials into a 1D array
    zero_centered_spikes = np.concatenate(trial_spikes.values.ravel())

    # Generate a PSTH
    hist, edges = np.histogram(zero_centered_spikes, bins=bin_cuts)

    # Convert to Hz/trial
    hist = np.round((hist / len(trial_spikes.index)) / psth_binsize, 4)

    # Calculate auROC
    auroc_curve = auROC_response_curve(hist, edges,
                                       pre_stimulus_baseline_start, pre_stimulus_baseline_end,
                                       auroc_binsize=auroc_binsize)

    cur_unitData["Session"][session_name]['Hit_psth'] = hist
    cur_unitData["Session"][session_name]['Hit_auroc'] = auroc_curve

    return cur_unitData


def calculate_auROC_hitByShock(cur_unitData,
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
    key_filter = ['Trial_spikes', 'Hit', 'ShockFlag']
    copy_relevant_unitData = {your_key: cur_unitData["Session"][session_name][your_key] for your_key in key_filter}
    try:
        cur_df = pd.DataFrame.from_dict(copy_relevant_unitData)
    except ValueError:
        print('auROC missByShock failed with ' + cur_unitData['Unit'] + '----' + session_name)
        return cur_unitData


    # Grab spikes around misses
    shock_labels = ['Off', 'On']  # 0: Off, 1: On
    for shock_flag, shock_label in enumerate(shock_labels):
        trial_spikes = cur_df[(cur_df['Hit'] == 1) & (cur_df['ShockFlag'] == shock_flag)]['Trial_spikes']

        # The field that goes into the JSON file
        output_field = 'Hit_shock' + shock_labels[shock_flag]

        # If no trials, skip
        if len(trial_spikes) == 0:
            cur_unitData["Session"][session_name][output_field + '_psth'] = []
            cur_unitData["Session"][session_name][output_field + '_auroc'] = []
            continue

        # Flatten all trials into a 1D array
        zero_centered_spikes = np.concatenate(trial_spikes.values.ravel())

        # Generate a PSTH
        hist, edges = np.histogram(zero_centered_spikes, bins=bin_cuts)

        # Convert to Hz/trial
        hist = np.round((hist / len(trial_spikes.index)) / psth_binsize, 4)

        # Calculate auROC
        auroc_curve = auROC_response_curve(hist, edges,
                                           pre_stimulus_baseline_start, pre_stimulus_baseline_end,
                                           auroc_binsize=auroc_binsize)

        cur_unitData["Session"][session_name][output_field + '_psth'] = hist
        cur_unitData["Session"][session_name][output_field + '_auroc'] = auroc_curve

    return cur_unitData



def calculate_auROC_miss(cur_unitData,
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

    # Load data and calculate auROCs based on trial responses aligned to events
    # Need to create a deep copy here or pandas will change original input (incredibly)
    key_filter = ['Trial_spikes', 'Miss']
    copy_relevant_unitData = {your_key: cur_unitData["Session"][session_name][your_key] for your_key in key_filter}
    try:
        cur_df = pd.DataFrame.from_dict(copy_relevant_unitData)
    except ValueError:
        print('auROC missAllTrials failed with ' + cur_unitData['Unit'] + '----' + session_name)
        return cur_unitData


    # Grab spikes around trials
    trial_spikes = cur_df[cur_df['Miss'] == 1]['Trial_spikes']

    # If no Misses, skip
    if len(trial_spikes) == 0:
        cur_unitData["Session"][session_name]['Miss_psth'] = []
        cur_unitData["Session"][session_name]['Miss_auroc'] = []
        return cur_unitData

    # Flatten all trials into a 1D array
    zero_centered_spikes = np.concatenate(trial_spikes.values.ravel())

    # Generate a PSTH
    hist, edges = np.histogram(zero_centered_spikes, bins=bin_cuts)

    # Convert to Hz/trial
    hist = np.round((hist / len(trial_spikes.index)) / psth_binsize, 4)

    # Calculate auROC
    auroc_curve = auROC_response_curve(hist, edges,
                                       pre_stimulus_baseline_start, pre_stimulus_baseline_end,
                                       auroc_binsize=auroc_binsize)

    cur_unitData["Session"][session_name]['Miss_psth'] = hist
    cur_unitData["Session"][session_name]['Miss_auroc'] = auroc_curve

    return cur_unitData


def calculate_auROC_missByShock(cur_unitData,
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
    key_filter = ['Trial_spikes', 'Miss', 'ShockFlag']
    copy_relevant_unitData = {your_key: cur_unitData["Session"][session_name][your_key] for your_key in key_filter}
    try:
        cur_df = pd.DataFrame.from_dict(copy_relevant_unitData)
    except ValueError:
        print('auROC missByShock failed with ' + cur_unitData['Unit'] + '----' + session_name)
        return cur_unitData


    # Grab spikes around misses
    shock_labels = ['Off', 'On']  # 0: Off, 1: On
    for shock_flag, shock_label in enumerate(shock_labels):
        trial_spikes = cur_df[(cur_df['Miss'] == 1) & (cur_df['ShockFlag'] == shock_flag)]['Trial_spikes']

        # The field that goes into the JSON file
        output_field = 'Miss_shock' + shock_labels[shock_flag]

        # If no trials, skip
        if len(trial_spikes) == 0:
            cur_unitData["Session"][session_name][output_field + '_psth'] = []
            cur_unitData["Session"][session_name][output_field + '_auroc'] = []
            continue

        # Flatten all trials into a 1D array
        zero_centered_spikes = np.concatenate(trial_spikes.values.ravel())

        # Generate a PSTH
        hist, edges = np.histogram(zero_centered_spikes, bins=bin_cuts)

        # Convert to Hz/trial
        hist = np.round((hist / len(trial_spikes.index)) / psth_binsize, 4)

        # Calculate auROC
        auroc_curve = auROC_response_curve(hist, edges,
                                           pre_stimulus_baseline_start, pre_stimulus_baseline_end,
                                           auroc_binsize=auroc_binsize)

        cur_unitData["Session"][session_name][output_field + '_psth'] = hist
        cur_unitData["Session"][session_name][output_field + '_auroc'] = auroc_curve

    return cur_unitData


def calculate_auROC_spoutOffMiss(cur_unitData,
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
    key_filter = ['Trial_spikes', 'Miss', 'ShockFlag', 'SpoutOff_times_during_trial']
    copy_relevant_unitData = {your_key: cur_unitData["Session"][session_name][your_key] for your_key in key_filter}
    try:
        cur_df = pd.DataFrame.from_dict(copy_relevant_unitData)
    except ValueError:
        print('auROC spoutOffMisses failed with ' + cur_unitData['Unit'] + '----' + session_name)
        return cur_unitData

    # Grab spikes around Misses
    trial_spikes = cur_df[(cur_df['Miss'] == 1) & (cur_df['ShockFlag'] == 1)]['Trial_spikes']

    # If no misses, skip
    if len(trial_spikes) == 0:
        cur_unitData["Session"][session_name]['SpoutOff_misses_psth'] = []
        cur_unitData["Session"][session_name]['SpoutOff_misses_auroc'] = []
        return cur_unitData

    # Find spout offset immediately following the shock
    spoutOffset_all_triggers = cur_df[(cur_df['Miss'] == 1) & (cur_df['ShockFlag'] == 1)]['SpoutOff_times_during_trial'].values
    spoutOffset_all_triggers = [np.array(t) for t in spoutOffset_all_triggers]

    # If no triggers around shock artifact (0.95 to 1.3, plus 0.5 s), skip
    try:
        if len(spoutOffset_all_triggers[0]) == 0:
            cur_unitData["Session"][session_name]['SpoutOff_misses_psth'] = []
            cur_unitData["Session"][session_name]['SpoutOff_misses_auroc'] = []
            return cur_unitData
    except TypeError:
        print()
    # try:
    #     spoutOffset_triggers = [cur_trial[(cur_trial > 0.95) & (cur_trial <= 1.8)][0] for cur_trial in spoutOffset_triggers]
    # except IndexError:
    #     print('Spout trigger during shock event was not found; possible timestamp alignment issue. Problem file: ' + session_name)
    #     cur_unitData["Session"][session_name]['SpoutOff_misses_psth'] = []
    #     cur_unitData["Session"][session_name]['SpoutOff_misses_auroc'] = []
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
    #     cur_unitData["Session"][session_name]['SpoutOff_FAs_psth'] = []
    #     cur_unitData["Session"][session_name]['SpoutOff_FAs_auroc'] = []
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
    auroc_curve = auROC_response_curve(hist, edges,
                                       pre_stimulus_baseline_start, pre_stimulus_baseline_end,
                                       auroc_binsize=auroc_binsize)

    cur_unitData["Session"][session_name]['SpoutOff_misses_psth'] = hist
    cur_unitData["Session"][session_name]['SpoutOff_misses_auroc'] = auroc_curve

    return cur_unitData


def calculate_auROC_FA(cur_unitData,
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
    For now, this function only calculates the auROC to a False Alarm;
        skip if no hits are found

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

    # Load data and calculate auROCs based on trial responses aligned to SpoutOff triggering a hit
    # Need to create a deep copy here or pandas will change original input (incredibly)
    key_filter = ['Trial_spikes', 'FA']
    copy_relevant_unitData = {your_key: cur_unitData["Session"][session_name][your_key] for your_key in key_filter}
    try:
        cur_df = pd.DataFrame.from_dict(copy_relevant_unitData)
    except ValueError:
        print('auROC FA failed with ' + cur_unitData['Unit'] + '----' + session_name)
        return cur_unitData


    # Grab spikes around FAs
    trial_spikes = cur_df[cur_df['FA'] == 1]['Trial_spikes']

    # If no Misses, skip
    if len(trial_spikes) == 0:
        cur_unitData["Session"][session_name]['FA_psth'] = []
        cur_unitData["Session"][session_name]['FA_auroc'] = []
        return cur_unitData

    # Flatten all trials into a 1D array
    zero_centered_spikes = np.concatenate(trial_spikes.values.ravel())

    # Generate a PSTH
    hist, edges = np.histogram(zero_centered_spikes, bins=bin_cuts)

    # Convert to Hz/trial
    hist = np.round((hist / len(trial_spikes.index)) / psth_binsize, 4)

    # Calculate auROC
    auroc_curve = auROC_response_curve(hist, edges,
                                       pre_stimulus_baseline_start, pre_stimulus_baseline_end,
                                       auroc_binsize=auroc_binsize)

    cur_unitData["Session"][session_name]['FA_psth'] = hist
    cur_unitData["Session"][session_name]['FA_auroc'] = auroc_curve

    return cur_unitData


def calculate_auROC_spoutOffFA(cur_unitData,
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

    # Load data and calculate auROCs based on trial responses aligned to SpoutOff triggering a hit
    # Need to create a deep copy here or pandas will change original input (incredibly)
    key_filter = ['Trial_spikes', 'FA', 'SpoutOff_times_during_trial']
    copy_relevant_unitData = {your_key: cur_unitData["Session"][session_name][your_key] for your_key in key_filter}
    try:
        cur_df = pd.DataFrame.from_dict(copy_relevant_unitData)
    except ValueError:
        print('auROC SpoutOffHits failed with ' + cur_unitData['Unit'] + '----' + session_name)
        return cur_unitData

    # Grab spikes around FAs
    trial_spikes = cur_df[cur_df['FA'] == 1]['Trial_spikes']

    # If no FAs, skip
    if len(trial_spikes) == 0:
        cur_unitData["Session"][session_name]['SpoutOff_FAs_psth'] = []
        cur_unitData["Session"][session_name]['SpoutOff_FAs_auroc'] = []
        return cur_unitData

    # Find spout offset that triggered FA
    spoutOffset_all_triggers = cur_df[cur_df['FA'] == 1]['SpoutOff_times_during_trial'].values
    spoutOffset_all_triggers = [np.array(t) for t in spoutOffset_all_triggers]

    # If no triggers, skip
    # try:
    if len(spoutOffset_all_triggers[0]) == 0:
        cur_unitData["Session"][session_name]['SpoutOff_FAs_psth'] = []
        cur_unitData["Session"][session_name]['SpoutOff_FAs_auroc'] = []
        return cur_unitData
    # except TypeError:
    #     print()
    # try:
    spoutOffset_triggers = []
    for cur_trial in spoutOffset_all_triggers:
        spout_triggers = cur_trial[(cur_trial > 0) & (cur_trial < 1)]
        if len(spout_triggers) > 0:
            spoutOffset_triggers.append(spout_triggers[-1])
        else:
            spoutOffset_triggers.append(np.NaN)
    spoutOffset_triggers = np.array(spoutOffset_triggers)
    # spoutOffset_triggers = [cur_trial[(cur_trial > 0) & (cur_trial < 1)][-1] for cur_trial in spoutOffset_triggers]
    # except IndexError:
    #     print('Spout triggering FA not found; possible timestamp alignment issue. Problem file: ' + session_name)
    #
    #     cur_unitData["Session"][session_name]['SpoutOff_FAs_psth'] = []
    #     cur_unitData["Session"][session_name]['SpoutOff_FAs_auroc'] = []
    #     return cur_unitData

    # Remove NaNs from trials where spoutOffsets were not registered
    nan_trials = np.isnan(spoutOffset_triggers)
    print('Removed ' + str(np.sum(nan_trials)) + ' FA trials without spout offset from: ' + session_name)
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
    auroc_curve = auROC_response_curve(hist, edges,
                                       pre_stimulus_baseline_start, pre_stimulus_baseline_end,
                                       auroc_binsize=auroc_binsize)

    cur_unitData["Session"][session_name]['SpoutOff_FAs_psth'] = hist
    cur_unitData["Session"][session_name]['SpoutOff_FAs_auroc'] = auroc_curve

    return cur_unitData

def calculate_auROC_AMTrial(cur_unitData,
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
    For now, this function only calculates the auROC to AM trials;
        skip if none are found

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

    output_name = 'AMTrial'

    # For PSTH calculation
    bin_cuts = np.arange(-pre_stimulus_raster, post_stimulus_raster, psth_binsize)

    # Load data and calculate auROCs based on trial responses aligned to all AM trial
    # Need to create a deep copy here or pandas will change original input (incredibly)
    key_filter = ['Trial_spikes', 'Hit', 'Miss']
    copy_relevant_unitData = {x: cur_unitData["Session"][session_name][x] for x in key_filter}
    try:
        cur_df = pd.DataFrame.from_dict(copy_relevant_unitData)
    except ValueError:
        print('auROC AMTrial failed with ' + cur_unitData['Unit'] + '----' + session_name)

        # Not sure how this is possible but this one recording ended up with more trials than Trial_spikes entries
        # Eliminate the last to be able to run it. Try to identify the issue if this happens with more recordings
        copy_relevant_unitData['Trial_spikes'] = copy_relevant_unitData['Trial_spikes'][0:100]
        copy_relevant_unitData['Hit'] = copy_relevant_unitData['Hit'][0:100]
        copy_relevant_unitData['Miss'] = copy_relevant_unitData['Miss'][0:100]
        cur_df = pd.DataFrame.from_dict(copy_relevant_unitData)
        # return cur_unitData
    # Grab spikes around hits or misses
    trial_spikes = cur_df[(cur_df['Hit'] == 1) | cur_df['Miss'] == 1]['Trial_spikes']

    # If no spikes, skip
    if len(trial_spikes) == 0:
        cur_unitData["Session"][session_name][output_name + '_psth'] = []
        cur_unitData["Session"][session_name][output_name + '_auroc'] = []
        return cur_unitData

    # Flatten all trials into a 1D array
    zero_centered_spikes = np.concatenate(trial_spikes.values.ravel())

    # Generate a PSTH
    hist, edges = np.histogram(zero_centered_spikes, bins=bin_cuts)

    # Convert to Hz/trial
    hist = np.round((hist / len(trial_spikes.index)) / psth_binsize, 4)

    # Calculate auROC
    auroc_curve = auROC_response_curve(hist, edges,
                                       pre_stimulus_baseline_start, pre_stimulus_baseline_end,
                                       auroc_binsize=auroc_binsize)

    cur_unitData["Session"][session_name][output_name + '_psth'] = hist
    cur_unitData["Session"][session_name][output_name + '_auroc'] = auroc_curve

    return cur_unitData


def calculate_auROC_AMdepthHitVsMiss(cur_unitData,
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
    For now, this function only calculates the auROC to different AM depths;
        skip if none are found

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

    # Load data and calculate auROCs based on trial responses aligned to all AM trial
    # Need to create a deep copy here or pandas will change original input (incredibly)
    key_filter = ['Trial_spikes', 'Hit', 'Miss', 'AMdepth', 'ShockFlag']
    copy_relevant_unitData = {x: cur_unitData["Session"][session_name][x] for x in key_filter}
    try:
        cur_df = pd.DataFrame.from_dict(copy_relevant_unitData)
    except ValueError:
        print('auROC AMdepthHitVsMiss failed with ' + cur_unitData['Unit'] + '----' + session_name)
        return cur_unitData

    amdepths = np.round(sorted(list(set(copy_relevant_unitData['AMdepth']))), 2)
    shock_labels = ['Off', 'On', 'NA']  # 0: Off, 1: On
    for hit_or_miss in ('Hit', 'Miss'):
        for amdepth in amdepths:
            trials = cur_df[(np.round(cur_df['AMdepth'], 2) == amdepth) &
                                  (cur_df[hit_or_miss] == 1)]

            # Grab spikes around trials
            trial_spikes = trials['Trial_spikes']

            if len(trial_spikes) == 0:
                shockFlag = 2  # NA
            else:
                shockFlag = list(set(trials['ShockFlag']))[0]

            # The field that goes into the JSON file
            output_field = 'AMdepth_' + '_' + str(amdepth) + '_' + hit_or_miss + '_shock' + shock_labels[shockFlag]

            # If no spikes, skip
            if len(trial_spikes) == 0:
                cur_unitData["Session"][session_name][output_field + '_psth'] = []
                cur_unitData["Session"][session_name][output_field + '_auroc'] = []
                continue

            # Flatten all trials into a 1D array
            zero_centered_spikes = np.concatenate(trial_spikes.values.ravel())

            # Generate a PSTH
            hist, edges = np.histogram(zero_centered_spikes, bins=bin_cuts)

            # Convert to Hz/trial
            hist = np.round((hist / len(trial_spikes.index)) / psth_binsize, 4)

            # Calculate auROC
            auroc_curve = auROC_response_curve(hist, edges,
                                               pre_stimulus_baseline_start, pre_stimulus_baseline_end,
                                               auroc_binsize=auroc_binsize)

            cur_unitData["Session"][session_name][output_field + '_psth'] = hist
            cur_unitData["Session"][session_name][output_field + '_auroc'] = auroc_curve

    return cur_unitData


def calculate_auROC_AMdepthByAMdepth(cur_unitData,
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
    For now, this function only calculates the auROC to different AM depths;
        skip if none are found

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

    output_name = 'AMdepth'

    # For PSTH calculation
    bin_cuts = np.arange(-pre_stimulus_raster, post_stimulus_raster, psth_binsize)

    # Load data and calculate auROCs based on trial responses aligned to all AM trial
    # Need to create a deep copy here or pandas will change original input (incredibly)
    key_filter = ['Trial_spikes', 'Hit', 'Miss', 'AMdepth']
    copy_relevant_unitData = {x: cur_unitData["Session"][session_name][x] for x in key_filter}
    try:
        cur_df = pd.DataFrame.from_dict(copy_relevant_unitData)
    except ValueError:
        print('auROC AMdepthByAMdepth failed with ' + cur_unitData['Unit'] + '----' + session_name)
        return cur_unitData

    amdepths = np.round(sorted(list(set(copy_relevant_unitData['AMdepth']))), 2)

    for amdepth in amdepths:
        # Grab spikes around trials
        trial_spikes = cur_df[(np.round(cur_df['AMdepth'], 2) == amdepth)]['Trial_spikes']

        # If no spikes, skip
        if len(trial_spikes) == 0:
            cur_unitData["Session"][session_name][output_name + '_allTrials_' + str(amdepth) + '_psth'] = []
            cur_unitData["Session"][session_name][output_name + '_allTrials_' + str(amdepth) + '_auroc'] = []
            continue


        # Flatten all trials into a 1D array
        zero_centered_spikes = np.concatenate(trial_spikes.values.ravel())

        # Generate a PSTH
        hist, edges = np.histogram(zero_centered_spikes, bins=bin_cuts)

        # Convert to Hz/trial
        hist = np.round((hist / len(trial_spikes.index)) / psth_binsize, 4)

        # Calculate auROC
        auroc_curve = auROC_response_curve(hist, edges,
                                           pre_stimulus_baseline_start, pre_stimulus_baseline_end,
                                           auroc_binsize=auroc_binsize)

        cur_unitData["Session"][session_name][output_name + '_allTrials_' + str(amdepth) + '_psth'] = hist
        cur_unitData["Session"][session_name][output_name + '_allTrials_' + str(amdepth) + '_auroc'] = auroc_curve

    return cur_unitData


def calculate_auROC_allSpoutOnset(cur_unitData,
                            session_name,
                            pre_stimulus_baseline_start,
                            pre_stimulus_baseline_end,
                            pre_stimulus_raster,
                            post_stimulus_raster,
                            psth_binsize=0.01,
                            auroc_binsize=0.1
                            ):
    """

    """

    output_name = 'allSpoutOnset'

    # For PSTH calculation
    bin_cuts = np.arange(-pre_stimulus_raster, post_stimulus_raster, psth_binsize)

    # Load data and calculate auROCs based on trial responses aligned to events
    # Need to create a deep copy here or pandas will change original input (incredibly)
    key_filter = ['Onset_rasters']
    copy_relevant_unitData = {your_key: cur_unitData["Session"][session_name][your_key] for your_key in key_filter}
    try:
        cur_df = pd.DataFrame.from_dict(copy_relevant_unitData)
    except ValueError:
        print('auROC allSpoutOnset failed with ' + cur_unitData['Unit'] + '----' + session_name)
        return cur_unitData


    # Grab spikes around events
    trial_spikes = cur_df['Onset_rasters']

    # If no spikes, skip
    if len(trial_spikes) == 0:
        cur_unitData["Session"][session_name][output_name + '_psth'] = []
        cur_unitData["Session"][session_name][output_name + '_auroc'] = []
        return cur_unitData

    # Flatten all trials into a 1D array
    zero_centered_spikes = np.concatenate(trial_spikes.values.ravel())

    # Generate a PSTH
    hist, edges = np.histogram(zero_centered_spikes, bins=bin_cuts)

    # Convert to Hz/trial
    hist = np.round((hist / len(trial_spikes.index)) / psth_binsize, 4)

    # Calculate auROC
    auroc_curve = auROC_response_curve(hist, edges,
                                       pre_stimulus_baseline_start, pre_stimulus_baseline_end,
                                       auroc_binsize=auroc_binsize)

    cur_unitData["Session"][session_name][output_name + '_psth'] = hist
    cur_unitData["Session"][session_name][output_name + '_auroc'] = auroc_curve

    return cur_unitData


def calculate_auROC_allSpoutOffset(cur_unitData,
                                  session_name,
                                  pre_stimulus_baseline_start,
                                  pre_stimulus_baseline_end,
                                  pre_stimulus_raster,
                                  post_stimulus_raster,
                                  psth_binsize=0.01,
                                  auroc_binsize=0.1
                                  ):
    """

    """

    output_name = 'allSpoutOffset'

    # For PSTH calculation
    bin_cuts = np.arange(-pre_stimulus_raster, post_stimulus_raster, psth_binsize)

    # Load data and calculate auROCs based on trial responses aligned to all AM trial
    # Need to create a deep copy here or pandas will change original input (incredibly)
    key_filter = ['Offset_rasters']
    copy_relevant_unitData = {your_key: cur_unitData["Session"][session_name][your_key] for your_key in key_filter}
    try:
        cur_df = pd.DataFrame.from_dict(copy_relevant_unitData)
    except ValueError:
        print('auROC allSpoutOffset failed with ' + cur_unitData['Unit'] + '----' + session_name)
        return cur_unitData


    # Grab spikes around misses
    trial_spikes = cur_df['Offset_rasters']

    # If no spikes, skip
    if len(trial_spikes) == 0:
        cur_unitData["Session"][session_name][output_name + '_psth'] = []
        cur_unitData["Session"][session_name][output_name + '_auroc'] = []
        return cur_unitData

    # Flatten all trials into a 1D array
    zero_centered_spikes = np.concatenate(trial_spikes.values.ravel())

    # Generate a PSTH
    hist, edges = np.histogram(zero_centered_spikes, bins=bin_cuts)

    # Convert to Hz/trial
    hist = np.round((hist / len(trial_spikes.index)) / psth_binsize, 4)

    # Calculate auROC
    auroc_curve = auROC_response_curve(hist, edges,
                                       pre_stimulus_baseline_start, pre_stimulus_baseline_end,
                                       auroc_binsize=auroc_binsize)

    cur_unitData["Session"][session_name][output_name + '_psth'] = hist
    cur_unitData["Session"][session_name][output_name + '_auroc'] = auroc_curve

    return cur_unitData