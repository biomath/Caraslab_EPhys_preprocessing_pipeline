import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from matplotlib import patches
from glob import glob

from re import split
import csv
from pandas import read_csv
from os.path import sep
import platform
from helpers.format_axes import format_ax

# Tweak the regex file separator for cross-platform compatibility
if platform.system() == 'Windows':
    REGEX_SEP = sep * 2
else:
    REGEX_SEP = sep


def get_zscore_toTrials(memory_name,
                    key_path_info,
                    key_path_spout,
                    unit_name,
                    output_path,
                    cur_unitData,
                    csv_pre_name=None,
                    first_cell_flag=True,
                    breakpoint_offset=0,
                    bin_size_for_zscore=0.1,
                    baseline_window_for_zscore=(-2., -1.),
                    response_window_for_zscore=(-2., 4.)):  # Include baseline and afterTrial
    """
    Process spike data around trials
    Takes key files and writes two files:
    1. Firing rates within window (_AMsound_firing_rate.csv)
    2. Number of spout events during that window (_AMsound_firing_rate.csv)
    3. All timestamped spikes during that window for timeseries analyses (_AMsound_spikes.json)
    """
    # For zscore calculation
    bin_cuts = np.arange(response_window_for_zscore[0], response_window_for_zscore[1], bin_size_for_zscore)

    # Load key files
    info_key_times = read_csv(key_path_info)

    # Load spike times
    spike_times = np.genfromtxt(memory_name)

    # The default is 0, so this will only make a difference if set at function call
    # breakpoint_offset_time = breakpoint_offset / sampling_rate # OBSOLETE: offsets are now in seconds
    breakpoint_offset_time = breakpoint_offset

    # Grab GO trials and FA trials for stim response then walk back to get the immediately preceding CR trial
    # One complicating factor is spout onset/offset but I'll ignore this for now
    # Also ignore reminder trials
    relevant_key_times = info_key_times[
        ((info_key_times['TrialType'] == 0) | (info_key_times['FA'] == 1)) & (info_key_times['Reminder'] == 0)]

    zscore_list = list()
    trial_idx_list = list()
    for _, cur_trial in relevant_key_times.iterrows():
        trial_spikes = spike_times[
            (spike_times > cur_trial['Trial_onset'] + breakpoint_offset_time + response_window_for_zscore[0]) &
            (spike_times < (cur_trial[
                                'Trial_onset'] + breakpoint_offset_time + response_window_for_zscore[1]))]
        zero_centered_spikes = trial_spikes - (cur_trial['Trial_onset'] + breakpoint_offset_time)
        # bin trial_spikes
        hist, edges = np.histogram(zero_centered_spikes, bins=bin_cuts)

        baseline_points_mask = (edges >= baseline_window_for_zscore[0]) & (edges < baseline_window_for_zscore[1])
        baseline_hist = hist[baseline_points_mask[:-1]]

        baseline_mean = np.nanmean(baseline_hist)
        baseline_std = np.nanstd(baseline_hist, ddof=1)

        if baseline_std == 0:
            zscore_list.append(np.full(np.size(hist), np.nan))

        zscore_list.append(np.array((hist - baseline_mean) / baseline_std))
        trial_idx_list.append(relevant_key_times['TrialID'])

    hit_signals = np.array(zscore_list)[
        np.intersect1d(
            trial_idx_list,
            relevant_key_times[relevant_key_times['Hit'] == 1]['TrialID'].values, return_indices=True)[1]]
    missShock_signals = np.array(zscore_list)[
        np.intersect1d(
            trial_idx_list,
            relevant_key_times[(relevant_key_times['Miss'] == 1) & (relevant_key_times['ShockFlag'] == 1)]['TrialID'].values, return_indices=True)[1]]
    missNoShock_signals = np.array(zscore_list)[
        np.intersect1d(
            trial_idx_list,
            relevant_key_times[(relevant_key_times['Miss'] == 1) & (relevant_key_times['ShockFlag'] == 0)]['TrialID'].values, return_indices=True)[1]]
    fa_signals = np.array(zscore_list)[
        np.intersect1d(
            trial_idx_list,
            relevant_key_times[relevant_key_times['FA'] == 1]['TrialID'].values, return_indices=True)[1]]

    hit_color = '#60B2E5'
    missShock_color = '#C84630'
    missNoShock_color = '#F0A202'
    fa_color = '#CCCCCC'
    with PdfPages(sep.join([output_path, unit_name + '_trialZscore.pdf'])) as pdf:
        # fig, ax = plt.subplots(1, 1)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        # Trial shading
        ax.axvspan(0, 1, facecolor='black', alpha=0.1)
        legend_handles = list()
        for sigs, color, legend_label in zip([hit_signals, missNoShock_signals, missShock_signals, fa_signals],
                                                  [hit_color, missNoShock_color, missShock_color, fa_color],
                                                  ['Hit', 'Miss (no shock)', 'Miss (shock)', 'False alarm']):
            if np.size(sigs) == 0:
                continue

            # trial_type_list.append(legend_label)

            signals_mean = np.nanmean(sigs, axis=0)
            signals_std = np.nanstd(sigs, axis=0) / np.sqrt(np.count_nonzero(~np.isnan(sigs), axis=0))
            x_axis = np.linspace(response_window_for_zscore[0], response_window_for_zscore[1], len(signals_mean))
            ax.plot(x_axis, signals_mean, color=color)
            ax.fill_between(x_axis, signals_mean - signals_std, signals_mean + signals_std,
                            alpha=0.1, color=color, edgecolor='none')

            # trial_type_list.append(legend_label)

            legend_handles.append(patches.Patch(facecolor=color, edgecolor=None, alpha=0.5,
                                                label=legend_label))
            #
            # bounded_x_axis = x_axis[int((trapz_start + PRETRIAL_DURATION_FOR_SPIKETIMES) / BINSIZE):
            #                         int((trapz_end + PRETRIAL_DURATION_FOR_SPIKETIMES) / BINSIZE)]
            # bounded_signals = sigs[int((trapz_start + PRETRIAL_DURATION_FOR_SPIKETIMES) / BINSIZE):
            #                        int((trapz_end + PRETRIAL_DURATION_FOR_SPIKETIMES) / BINSIZE)]
            #
            # # auc = np.trapz(bounded_signals, bounded_x_axis, axis=1)     # idx=2
            # auc = simps(bounded_signals, bounded_x_axis)  # idx=2
            # peak = np.max(bounded_signals)  # idx=3
            #
            # output_dict.update({legend_label: (auc, peak)})

            # except ValueError:
            #     print()

        format_ax(ax)

        ax.set_xlabel("Time from trial onset (s)")
        ax.set_ylabel(r'Z-score')

        # Might want to make this a variable
        ax.set_ylim([-10, 10])

        labels = [h.get_label() for h in legend_handles]

        fig.legend(handles=legend_handles, labels=labels, frameon=False, numpoints=1)

        fig.tight_layout()

        # plt.show()
        pdf.savefig()
        plt.close()
