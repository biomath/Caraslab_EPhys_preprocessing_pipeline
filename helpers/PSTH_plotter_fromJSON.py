from os import remove, makedirs
from os.path import sep
from re import split
import platform
from time import time
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rcParams
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib
matplotlib.use('agg')  # Required to avoid known memory leak caused by matplotlib with Jupyter
from gc import collect

from helpers.format_axes import format_ax

# Tweak the regex file separator for cross-platform compatibility
if platform.system() == 'Windows':
    REGEX_SEP = sep * 2
else:
    REGEX_SEP = sep


def tic():
    return time()

def __common_psth_engine(spike_times,
                         pre_stimulus_raster, post_stimulus_raster,
                         key_times=None,
                         ax_raster=None, ax_psth=None, ax_gaussian=None,
                         breakpoint_offset=None,
                         hist_bin_size=0.01,
                         do_plot=True,
                         rasterize=True):
    bin_cuts = np.arange(-pre_stimulus_raster, post_stimulus_raster + hist_bin_size, hist_bin_size)

    # raster_trial_counter = 0
    if key_times is not None:
        number_of_stimulus_repetitions = len(key_times)
        # Loop through each stimulus presentation
        raster_trial_counter = number_of_stimulus_repetitions
        relative_times = list()
        for cur_stim_time in key_times:
            # offset_stim_time = cur_stim_time + breakpoint_offset / sampling_rate
            offset_stim_time = cur_stim_time + breakpoint_offset  # breakpoints are in seconds now

            # Get spike times around the current stimulus onset
            times_to_plot = spike_times[((offset_stim_time - pre_stimulus_raster) < spike_times) &
                                        (spike_times < (offset_stim_time + post_stimulus_raster))]

            # Zero-center spike times
            curr_relative_times = times_to_plot - offset_stim_time
            if do_plot:
                ax_raster.plot(curr_relative_times,
                               np.repeat(raster_trial_counter, len(curr_relative_times)),
                               'k|',
                               rasterized=rasterize)
                raster_trial_counter -= 1

            relative_times.append([x for x in curr_relative_times])

        relative_times = [item for sublist in relative_times for item in sublist]
    else:  # Assume zero-centered spikes already
        relative_times = [x[(x >= -pre_stimulus_raster) & (x < post_stimulus_raster)] for x in spike_times]
        number_of_stimulus_repetitions = len(relative_times)
        # Loop through each stimulus presentation
        raster_trial_counter = number_of_stimulus_repetitions
        for curr_relative_times in relative_times:
            if do_plot:
                ax_raster.plot(curr_relative_times,
                               np.repeat(raster_trial_counter, len(curr_relative_times)),
                               'k|',
                               rasterized=rasterize)
                raster_trial_counter -= 1
        else:
            pass
    flat_relative_times = list()
    for x in relative_times:
        flat_relative_times.extend(list(x))

    # Only to get y labels. Later, use this to plot as well
    hist, edges = np.histogram(flat_relative_times, bins=bin_cuts)

    # Change hist to spike rate before appending
    hist = np.round(hist / number_of_stimulus_repetitions / hist_bin_size, 2)

    if do_plot:
        ax_psth.bar(bin_cuts[:-1], hist, color='k', edgecolor='k', align='edge', width=hist_bin_size)

    return hist


def __plot_aligned_spikes(aligned_spikes, pre_stimulus_raster, post_stimulus_raster, psth_bin_size,
                          psth_fixed_ylim, raster_ylim, plot_suptitle, pdf_handle):

    aligned_spikes = [np.array(x) for x in aligned_spikes]
    # Plot
    plt.clf()
    f = plt.figure()
    ax_psth = f.add_subplot(212)
    ax_raster = f.add_subplot(211, sharex=ax_psth)

    __common_psth_engine(spike_times=aligned_spikes,
                         pre_stimulus_raster=pre_stimulus_raster,
                         post_stimulus_raster=post_stimulus_raster,
                         ax_psth=ax_psth, ax_raster=ax_raster,
                         hist_bin_size=psth_bin_size,
                         do_plot=True)

    # Format axs
    format_ax(ax_raster)
    format_ax(ax_psth)

    ax_raster.axis('off')

    ax_psth.set_ylim([0, psth_fixed_ylim])
    ax_raster.set_ylim([-0.5, raster_ylim])
    ax_psth.set_ylabel("Spike rate by trial (Hz)")
    ax_psth.set_xlabel("Time (s)")

    f.suptitle(plot_suptitle, fontsize='small')

    plt.tight_layout()

    pdf_handle.savefig()

    plt.clf()
    plt.close("all")


def __trialType_psth(cur_data, output_subfolder, unit_name, psth_bin_size, pre_stimulus_raster,
                     post_stimulus_raster, psth_fixed_ylim, raster_ylim, trial_types, align_to_response,
                     shock_artifact):
    with PdfPages(sep.join([output_subfolder, unit_name + '_PSTH_' + str(int(psth_bin_size*1000)) + 'ms.pdf'])) as pdf:
        for session in cur_data.keys():
            if session == 'active':
                for trial_type in trial_types:
                    if trial_type == 'Hit (shock)':
                        cur_trial_mask = np.all(
                            [np.array(cur_data[session]['Hit']) == 1,
                             np.array(cur_data[session]['ShockFlag']) == 1,
                             np.array(cur_data[session]['Reminder']) == 0],
                            axis=0)

                    elif trial_type == 'Hit (no shock)':
                        cur_trial_mask = np.all(
                            [np.array(cur_data[session]['Hit']) == 1,
                             np.array(cur_data[session]['ShockFlag']) == 0,
                             np.array(cur_data[session]['Reminder']) == 0],
                            axis=0)

                    elif trial_type == 'False alarm':
                        cur_trial_mask = np.all([np.array(cur_data[session]['FA']) == 1,
                                                 np.array(cur_data[session]['Reminder']) == 0], axis=0)

                    elif trial_type == 'Miss (shock)':
                        cur_trial_mask = np.all([np.array(cur_data[session]['Miss']) == 1,
                                                 np.array(cur_data[session]['ShockFlag']) == 1,
                                                 np.array(cur_data[session]['Reminder']) == 0], axis=0)

                    else:  # passive trials are handled in the scope above this loop
                        continue

                    if not align_to_response:
                        spike_times = [x for x_idx, x in enumerate(cur_data[session]['Trial_spikes']) if
                                       cur_trial_mask[x_idx]]
                    else:
                        spike_times = [x for x_idx, x in enumerate(cur_data[session]['Response_spikes']) if
                                       cur_trial_mask[x_idx]]

                    plot_suptitle = unit_name + "\n" + session + '\n' + trial_type
                    __plot_aligned_spikes(spike_times, pre_stimulus_raster, post_stimulus_raster, psth_bin_size,
                                          psth_fixed_ylim, raster_ylim, plot_suptitle, pdf)
                    collect()
            else:
                # Assume passive
                trial_type = 'passive'
                spike_times = cur_data[session]['Trial_spikes']
                plot_suptitle = unit_name + "\n" + session + '\n' + trial_type
                __plot_aligned_spikes(spike_times, pre_stimulus_raster, post_stimulus_raster, psth_bin_size,
                                      psth_fixed_ylim, raster_ylim, plot_suptitle, pdf)
                collect()


def __amDepth_psth(cur_data, output_subfolder, unit_name, psth_bin_size, pre_stimulus_raster,
                   post_stimulus_raster, psth_fixed_ylim, raster_ylim):
    with PdfPages(sep.join([output_subfolder, unit_name + '_PSTH_' + str(psth_bin_size) + 'ms.pdf'])) as pdf:
        for session in cur_data.keys():
            cur_amdepths = sorted(list(set(cur_data[session]['AMdepth'])))
            # 0 AMDepth is only for False alarms in the current version, non-AM psths can be extracted from before each AM trial
            cur_amdepths = [x for x in cur_amdepths if x > 0]
            for amdepth in cur_amdepths:
                cur_trial_mask = np.all(
                    [np.array(cur_data[session]['AMdepth']) == amdepth, np.array(cur_data[session]['Reminder']) == 0],
                    axis=0)
                spike_times = [np.array(x) for x_idx, x in enumerate(cur_data[session]['Trial_spikes']) if
                               cur_trial_mask[x_idx]]
                if amdepth > 0:
                    amdepth_log = np.round(20 * np.log10(amdepth), 1)
                else:
                    amdepth_log = 'NON-AM'

                plot_suptitle = unit_name + "\n" + session + '\n' + str(amdepth_log) + ' dB re:100%'
                __plot_aligned_spikes(spike_times, pre_stimulus_raster, post_stimulus_raster, psth_bin_size,
                                      psth_fixed_ylim, raster_ylim, plot_suptitle, pdf)
                collect()


def run_PSTH_pipeline(input_list):
    unit_name, data_dict, SETTINGS_DICT = input_list
    output_path = SETTINGS_DICT['OUTPUT_PATH'] + sep + 'PSTHs'
    psth_bin_size = SETTINGS_DICT['PSTH_BIN_SIZE']
    pre_stimulus_raster = SETTINGS_DICT['PSTH_PRE_STIMULUS_DURATION']
    post_stimulus_raster = SETTINGS_DICT['PSTH_POST_STIMULUS_DURATION']
    psth_fixed_ylim = SETTINGS_DICT['PSTH_FIXED_YLIM']
    raster_ylim = SETTINGS_DICT['PSTH_RASTER_YLIM']
    trial_types = SETTINGS_DICT['PSTH_TRIALTYPES']
    align_to_response = SETTINGS_DICT['PSTH_ALIGN_TO_RESPONSE']
    shock_artifact = SETTINGS_DICT['SHOCK_START_END']
    pipeline_switchboard = SETTINGS_DICT['PIPELINE_SWITCHBOARD']

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
    rcParams['figure.figsize'] = (2.5, 4)

    split_unit_name = split("_*_", unit_name)
    subject_id = split_unit_name[0]

    ''' PSTH separated by trial types '''
    if pipeline_switchboard['plot_trialType_PSTH']:
        output_subfolder = sep.join([output_path, subject_id, 'TrialType'])
        makedirs(output_subfolder, exist_ok=True)
        __trialType_psth(data_dict, output_subfolder, unit_name, psth_bin_size, pre_stimulus_raster,
                         post_stimulus_raster, psth_fixed_ylim, raster_ylim, trial_types, align_to_response,
                         shock_artifact)

    ''' PSTH separated by AM depth '''
    if pipeline_switchboard['plot_AMDepth_PSTH']:
        output_subfolder = sep.join([output_path, subject_id, 'AMDepth'])
        makedirs(output_subfolder, exist_ok=True)
        __amDepth_psth(data_dict, output_subfolder, unit_name, psth_bin_size, pre_stimulus_raster,
                       post_stimulus_raster, psth_fixed_ylim, raster_ylim)
