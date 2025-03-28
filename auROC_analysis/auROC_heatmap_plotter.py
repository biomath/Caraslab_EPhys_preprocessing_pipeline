from os import makedirs
from os.path import sep
import platform
import warnings

from re import split

import numpy as np
from matplotlib import pyplot as plt, rcParams,  patches
from matplotlib import colormaps

import pandas as pd

from matplotlib.backends.backend_pdf import PdfPages

# Tweak the regex file separator for cross-platform compatibility
if platform.system() == 'Windows':
    REGEX_SEP = sep * 2
else:
    REGEX_SEP = sep

def run_auROC_heatmap_pipeline(data_dict, SETTINGS_DICT):
    output_path = SETTINGS_DICT['OUTPUT_PATH'] + sep + 'auROC_heatmaps'
    makedirs(output_path, exist_ok=True)
    heatmap_binsize = SETTINGS_DICT['AUROC_BIN_SIZE']
    pretrial_duration = SETTINGS_DICT['AUROC_PRE_STIMULUS_DURATION']
    posttrial_duration = SETTINGS_DICT['AUROC_POST_STIMULUS_DURATION']

    trial_types = SETTINGS_DICT['AUROC_TRIALTYPES']
    sort_by_which_trial_type = SETTINGS_DICT['SORT_BY_WHICH_TRIALTYPE']

    if SETTINGS_DICT['AUROC_GROUPING_FILE'] is not None:
        auROC_grouping_file = pd.read_csv(SETTINGS_DICT['AUROC_GROUPING_FILE'])
        auROC_grouping_variable = SETTINGS_DICT['AUROC_GROUPING_VARIABLE']
        auROC_unique_groups = SETTINGS_DICT['AUROC_UNIQUE_GROUPS']
        if auROC_unique_groups is None:
            auROC_unique_groups = set(auROC_grouping_file[auROC_grouping_variable])

        auROC_group_colors = SETTINGS_DICT['AUROC_GROUP_COLORS']
        if auROC_group_colors is None:
            auROC_group_colors = colormaps['tab20'].colors
    else:
        auROC_grouping_file = None
        auROC_grouping_variable = None
        auROC_unique_groups = ['all',]
        auROC_group_colors = colormaps['tab20'].colors

    # Set plotting parameters
    label_font_size = 11
    tick_label_size = 7
    legend_font_size = 6
    line_thickness = 1

    rcParams['figure.dpi'] = 600
    rcParams['pdf.fonttype'] = 42
    rcParams['ps.fonttype'] = 42
    rcParams['font.family'] = 'Arial'
    rcParams['font.weight'] = 'regular'
    rcParams['axes.labelweight'] = 'regular'

    rcParams['font.size'] = label_font_size
    rcParams['axes.labelsize'] = label_font_size
    rcParams['axes.titlesize'] = label_font_size
    rcParams['axes.linewidth'] = line_thickness
    rcParams['legend.fontsize'] = legend_font_size
    rcParams['xtick.labelsize'] = tick_label_size
    rcParams['ytick.labelsize'] = tick_label_size
    rcParams['errorbar.capsize'] = label_font_size
    rcParams['lines.markersize'] = line_thickness
    rcParams['lines.linewidth'] = line_thickness
    rcParams['figure.figsize'] = (4, 5)

    auroc_list = []
    trialType_list = []
    unit_list = []
    unique_units = sorted(list(set((data_dict.keys()))))
    for unit in unique_units:
        # Grab  data from Pre, Task and Post.
        # Passive data are special cases because trial type is irrelevant; handle them first
        # Pre data
        try:
            cur_data = data_dict[unit]['pre']

            response_curve = np.array(cur_data['TrialAligned_GO_auroc'])  # This is the only relevant subfield

            # Fill no responses with NaNs
            if len(response_curve) == 0 or np.mean(response_curve) == 0:
                response_curve = np.nan

            auroc_list.append(response_curve)
            unit_list.append(unit)
            trialType_list.append('TrialAligned_PassivePre_auroc')
        except KeyError:
            warnings.warn('Could not find PassivePre data for {}.'.format(unit))
            response_curve = np.nan
            auroc_list.append(response_curve)
            unit_list.append(unit)
            trialType_list.append('TrialAligned_PassivePre_auroc')

        # Post data
        try:
            cur_data = data_dict[unit]['post']

            response_curve = np.array(cur_data['TrialAligned_GO_auroc'])

            # Fill no responses with NaNs
            if len(response_curve) == 0 or np.mean(response_curve) == 0:
                response_curve = np.nan

            auroc_list.append(response_curve)
            unit_list.append(unit)
            trialType_list.append('TrialAligned_PassivePost_auroc')
        except KeyError:
            warnings.warn('Could not find PassivePost data for {}.'.format(unit))
            response_curve = np.nan
            auroc_list.append(response_curve)
            unit_list.append(unit)
            trialType_list.append('TrialAligned_PassivePost_auroc')

        # Active data
        try:
            cur_data = data_dict[unit]['active']

            for trial_type in list(trial_types.keys()):
                if 'Passive' in trial_type:  # Just in case there was some mislabelling along the way

                    continue
                response_curve = np.array(cur_data[trial_type])

                # Fill no responses with NaNs
                if len(response_curve) == 0 or np.mean(response_curve) == 0:
                    response_curve = np.nan

                auroc_list.append(response_curve)
                unit_list.append(unit)
                trialType_list.append(trial_type)
        except KeyError:
            for trial_type in list(trial_types.keys()):
                if 'Passive' in trial_type:  # Just in case
                    warnings.warn('Active trial mislabeled as Passive for {}.'.format(unit))
                    continue
                else:
                    warnings.warn('Could not find Active data for {}.'.format(unit))
                    response_curve = np.nan
                    auroc_list.append(response_curve)
                    unit_list.append(unit)
                    trialType_list.append(trial_type)

    # Fill NaNs to match array dimensions
    auroc_len = np.round(np.mean([len(x) for x in auroc_list if not np.all(np.isnan(x))]), 0)
    plot_list = [np.full(int(auroc_len), np.nan) if np.all(np.isnan(x)) else x for x in auroc_list]
    plot_list = np.array(plot_list)

    # Plot
    # Reorder the data so they are grouped by unique_group and sorted by the desired subplot index
    overall_unit_order = list()
    colorbar_splits = [0,]
    for group_idx, cur_group in enumerate(auROC_unique_groups):
        # Get sorting indeces first
        trial_type = sort_by_which_trial_type

        if cur_group == 'all':
            cur_units = unique_units
        else:
            cur_units = auROC_grouping_file[auROC_grouping_file[auROC_grouping_variable] == cur_group]['Unit'].values

        # cur_units = UNITS_TO_RUN['Unit'].values
        cur_unit_filter = np.in1d(unit_list, cur_units)
        cur_trial_filter = np.array(trialType_list) == trial_type
        unit_trial_filter = cur_unit_filter * cur_trial_filter

        cur_units = np.array(unit_list)[unit_trial_filter]
        cur_resps = plot_list[unit_trial_filter]

        snippet_start = trial_types[trial_type][0]
        snippet_end = trial_types[trial_type][1]
        relevant_indices = np.arange(
            np.floor((snippet_start + pretrial_duration) / heatmap_binsize),
            np.floor((snippet_end + pretrial_duration) / heatmap_binsize) )

        relevant_snippet = np.array([cur_auroc[[int(idx) for idx in relevant_indices]]
                                     for cur_auroc in cur_resps])

        sorted_indices = np.argsort([np.mean(x) for x in relevant_snippet])[::-1]

        overall_unit_order.extend(cur_units[sorted_indices])
        colorbar_splits.append(len(overall_unit_order))


    colorbar_splits = [len(overall_unit_order) - x for x in colorbar_splits]

    f = plt.figure()
    plot_location_idx = 1
    with PdfPages(sep.join([output_path,  'auROC_heatmap.pdf'])) as pdf:
        for dummy_idx, trial_type in enumerate(trial_types.keys()):
            ax = f.add_subplot(1, len(trial_types.keys()), plot_location_idx)

            cur_trial_filter = np.array(trialType_list) == trial_type
            cur_resps = plot_list[cur_trial_filter]

            cur_units_unordered = np.array(unit_list)[cur_trial_filter]

            cur_units_ordered = []
            for cur_unit in overall_unit_order:
                cur_units_ordered.extend(np.where(cur_units_unordered == cur_unit)[0])

            sorted_plot_list = cur_resps[np.array(cur_units_ordered)]
            n_units = np.size(sorted_plot_list, axis=0)

            current_cmap = colormaps['plasma']
            current_cmap.set_bad('white')
            cax = ax.imshow(sorted_plot_list, vmin=0, vmax=1, interpolation='None', cmap=current_cmap,
                            extent=[0, np.size(sorted_plot_list, axis=1), n_units, 0],
                            aspect='auto')

            ax.axvline(x=pretrial_duration/heatmap_binsize, color='white', linestyle='--')

            # Add subject colors
            # These patches start at the bottom left corner so we need to use the reversed lists
            if len(auROC_unique_groups) > 1:
                for group_idx, cur_group in reversed(list(enumerate(auROC_unique_groups))):
                    y_start = (colorbar_splits[group_idx+1]) / n_units
                    y_height = (colorbar_splits[group_idx]) / n_units - y_start
                    rect = patches.Rectangle(
                        (-0.1, y_start), width=0.1, height=y_height, facecolor=auROC_group_colors[group_idx], transform=ax.transAxes,
                        clip_on=False, edgecolor='none'
                    )
                    ax.add_patch(rect)

            if plot_location_idx > 1:
                ax.yaxis.set_visible(False)

            ax.xaxis.set_ticks_position('bottom')

            ax.set_xticks(
                np.arange(0, (pretrial_duration + posttrial_duration + 1) / heatmap_binsize, 20))
            ax.set_xlim(
                [0, (pretrial_duration + posttrial_duration) / heatmap_binsize])

            new_ticks = np.round(ax.get_xticks() * heatmap_binsize - pretrial_duration, 1)
            ax.set_xticklabels(new_ticks.astype(int))
            ax.set_xlabel('Time (s)')

            ax.tick_params(axis='x', bottom='off')
            ax.tick_params(axis='y', left='off', right='off')

            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['bottom'].set_visible(False)

            plot_location_idx += 1

        plt.subplots_adjust(wspace=0.2)
        pdf.savefig()
        plt.close()