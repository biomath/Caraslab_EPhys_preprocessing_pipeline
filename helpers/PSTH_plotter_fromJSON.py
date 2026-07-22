from os import remove, makedirs
from os.path import sep
from re import split
import platform
from time import time
from multiprocessing import Pool
import numpy as np
import matplotlib
matplotlib.use('agg')  # Required to avoid known memory leak caused by matplotlib with Jupyter
from matplotlib import pyplot as plt
from matplotlib import rcParams
from matplotlib.backends.backend_pdf import PdfPages
from helpers import get_JSON_data

from gc import collect

from helpers.format_axes import format_ax

# Tweak the regex file separator for cross-platform compatibility
if platform.system() == 'Windows':
    REGEX_SEP = sep * 2
else:
    REGEX_SEP = sep


def tic():
    """Return the current time (seconds); pair with a second ``tic()`` call to time a block."""
    return time()

def __common_psth_engine(spike_times,
                         pre_stimulus_raster, post_stimulus_raster,
                         key_times=None,
                         ax_raster=None, ax_psth=None, ax_gaussian=None,
                         breakpoint_offset=None,
                         hist_bin_size=0.01,
                         do_plot=True,
                         rasterize=True):
    """Draw a raster plot and compute/optionally plot the corresponding PSTH histogram.

    Accepts spikes either as raw timestamps plus stimulus onset times to
    align to (``key_times`` given), or as already zero-centered per-trial
    spike arrays (``key_times=None``).

    Args:
        spike_times: If ``key_times`` is given, a flat array of raw spike
            timestamps to align per-trial. Otherwise, a list of per-trial
            arrays already centered on the event of interest.
        pre_stimulus_raster (float): Seconds before the event to include.
        post_stimulus_raster (float): Seconds after the event to include.
        key_times (Iterable[float], optional): Stimulus/event onset
            timestamps to align ``spike_times`` to; if None, ``spike_times``
            is assumed pre-aligned.
        ax_raster (matplotlib.axes.Axes, optional): Axes to draw the raster onto.
        ax_psth (matplotlib.axes.Axes, optional): Axes to draw the PSTH bar plot onto.
        ax_gaussian: Unused; kept for call-site compatibility.
        breakpoint_offset (float, optional): Seconds added to each entry in
            ``key_times`` to align with this unit's spike-time clock.
        hist_bin_size (float): PSTH histogram bin width (s).
        do_plot (bool): If False, only compute and return the histogram
            without drawing anything (``ax_raster``/``ax_psth`` unused).
        rasterize (bool): Whether to rasterize the raster line plot (for
            smaller vector output files with many trials).

    Returns:
        np.ndarray: PSTH firing rate (Hz) per bin, averaged across trials.
    """
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
    """Render one raster+PSTH figure page for a set of pre-aligned per-trial spike arrays.

    Args:
        aligned_spikes (Sequence): One zero-centered spike-time array per trial.
        pre_stimulus_raster (float): Seconds before the event shown.
        post_stimulus_raster (float): Seconds after the event shown.
        psth_bin_size (float): PSTH histogram bin width (s).
        psth_fixed_ylim (float): Upper y-limit for the PSTH rate axis.
        raster_ylim (float): Upper y-limit for the raster trial-count axis.
        plot_suptitle (str): Figure title (unit/session/condition label).
        pdf_handle (matplotlib.backends.backend_pdf.PdfPages): Open PDF handle to save into.

    Returns:
        None. Saves one page to ``pdf_handle`` and closes the figure.
    """
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
    """Generate one multi-page PDF of PSTHs, one page per session/trial-type combination.

    For active/aversive sessions, splits trials by outcome (per
    ``trial_types``: Hit, Hit (shock), Hit (no shock), False alarm, Miss
    (shock)), excluding reminder trials. Passive sessions are plotted as a
    single unsplit PSTH.

    Args:
        cur_data (dict): Unit JSON data (``cur_data['Session'][...]`` holds
            per-session trial outcome flags and spike arrays).
        output_subfolder (str): Directory to save the output PDF into.
        unit_name (str): Unit identifier, used in the output filename/titles.
        psth_bin_size (float): PSTH histogram bin width (s).
        pre_stimulus_raster (float): Seconds before the event shown.
        post_stimulus_raster (float): Seconds after the event shown.
        psth_fixed_ylim (float): Upper y-limit for the PSTH rate axis.
        raster_ylim (float): Upper y-limit for the raster trial-count axis.
        trial_types (Iterable[str]): Which trial-type categories to plot for
            active/aversive sessions.
        align_to_response (bool): If True, align spikes to response time
            ('Response_spikes') instead of trial onset ('Trial_spikes').
        shock_artifact: Unused here; kept for call-site compatibility.

    Returns:
        None. Writes ``<unit_name>_PSTH_<bin_ms>ms.pdf`` into ``output_subfolder``.
    """
    print('Plotting Trial type PSTH for ' + unit_name + '...')
    with PdfPages(sep.join([output_subfolder, unit_name + '_PSTH_' + str(int(psth_bin_size*1000)) + 'ms.pdf'])) as pdf:
        for session in cur_data['Session'].keys():
            if 'active' in session.lower() or 'aversive' in session.lower():
                for trial_type in trial_types:
                    if trial_type == 'Hit':
                        cur_trial_mask = np.all(
                            [np.array(cur_data['Session'][session]['Hit']) == 1,
                             np.array(cur_data['Session'][session]['Reminder']) == 0],
                            axis=0)
                    elif trial_type == 'Hit (shock)':
                        cur_trial_mask = np.all(
                            [np.array(cur_data['Session'][session]['Hit']) == 1,
                             np.array(cur_data['Session'][session]['ShockFlag']) == 1,
                             np.array(cur_data['Session'][session]['Reminder']) == 0],
                            axis=0)

                    elif trial_type == 'Hit (no shock)':
                        cur_trial_mask = np.all(
                            [np.array(cur_data['Session'][session]['Hit']) == 1,
                             np.array(cur_data['Session'][session]['ShockFlag']) == 0,
                             np.array(cur_data['Session'][session]['Reminder']) == 0],
                            axis=0)

                    elif trial_type == 'False alarm':
                        cur_trial_mask = np.all([np.array(cur_data['Session'][session]['FA']) == 1,
                                                 np.array(cur_data['Session'][session]['Reminder']) == 0], axis=0)

                    elif trial_type == 'Miss (shock)':
                        cur_trial_mask = np.all([np.array(cur_data['Session'][session]['Miss']) == 1,
                                                 np.array(cur_data['Session'][session]['ShockFlag']) == 1,
                                                 np.array(cur_data['Session'][session]['Reminder']) == 0], axis=0)

                    else:  # passive trials are handled in the scope above this loop
                        continue

                    if not align_to_response:
                        spike_times = [x for x_idx, x in enumerate(cur_data['Session'][session]['Trial_spikes']) if
                                       cur_trial_mask[x_idx]]
                    else:
                        spike_times = [x for x_idx, x in enumerate(cur_data['Session'][session]['Response_spikes']) if
                                       cur_trial_mask[x_idx]]

                    plot_suptitle = unit_name + "\n" + session + '\n' + trial_type
                    __plot_aligned_spikes(spike_times, pre_stimulus_raster, post_stimulus_raster, psth_bin_size,
                                          psth_fixed_ylim, raster_ylim, plot_suptitle, pdf)
                    collect()
            else:
                # Assume passive
                trial_type = 'passive'
                spike_times = cur_data['Session'][session]['Trial_spikes']
                plot_suptitle = unit_name + "\n" + session + '\n' + trial_type
                __plot_aligned_spikes(spike_times, pre_stimulus_raster, post_stimulus_raster, psth_bin_size,
                                      psth_fixed_ylim, raster_ylim, plot_suptitle, pdf)
                collect()


def __amDepth_psth(cur_data, output_subfolder, unit_name, psth_bin_size, pre_stimulus_raster,
                   post_stimulus_raster, psth_fixed_ylim, raster_ylim, responseLatency_filter):
    """Generate one multi-page PDF of PSTHs, one page per session/AM-depth combination.

    Splits trials by AM modulation depth (excluding reminder trials and
    trials below ``responseLatency_filter``), converting each depth to dB
    re: 100% for the plot title.

    Args:
        cur_data (dict): Unit JSON data (``cur_data['Session'][...]`` holds
            per-session AM depth, trial spikes, and response-latency arrays).
        output_subfolder (str): Directory to save the output PDF into.
        unit_name (str): Unit identifier, used in the output filename/titles.
        psth_bin_size (float): PSTH histogram bin width (s).
        pre_stimulus_raster (float): Seconds before the event shown.
        post_stimulus_raster (float): Seconds after the event shown.
        psth_fixed_ylim (float): Upper y-limit for the PSTH rate axis.
        raster_ylim (float): Upper y-limit for the raster trial-count axis.
        responseLatency_filter (float): Minimum response latency (s) for a
            trial to be included.

    Returns:
        None. Writes ``<unit_name>_PSTH_<bin_size>ms.pdf`` into ``output_subfolder``.
    """
    print('Plotting AM depth PSTH for ' + unit_name + '...')
    with PdfPages(sep.join([output_subfolder, unit_name + '_PSTH_' + str(psth_bin_size) + 'ms.pdf'])) as pdf:
        for session in cur_data['Session'].keys():
            cur_amdepths = sorted(list(set(cur_data['Session'][session]['AMdepth'])))
            # 0 AMDepth is only for False alarms in the current version, non-AM psths can be extracted from before each AM trial
            cur_amdepths = [x for x in cur_amdepths if x > 0]
            for amdepth in cur_amdepths:
                cur_trial_mask = np.all(
                    [np.array(cur_data['Session'][session]['AMdepth']) == amdepth, np.array(cur_data['Session'][session]['Reminder']) == 0,
                     np.array(cur_data['Session'][session]['RespLatency']) > responseLatency_filter],
                    axis=0)
                spike_times = [np.array(x) for x_idx, x in enumerate(cur_data['Session'][session]['Trial_spikes']) if
                               cur_trial_mask[x_idx]]
                if amdepth > 0:
                    amdepth_log = np.round(20 * np.log10(amdepth), 1)
                else:
                    amdepth_log = 'NON-AM'

                plot_suptitle = unit_name + "\n" + session + '\n' + str(amdepth_log) + ' dB re:100%'
                __plot_aligned_spikes(spike_times, pre_stimulus_raster, post_stimulus_raster, psth_bin_size,
                                      psth_fixed_ylim, raster_ylim, plot_suptitle, pdf)
                collect()


def __opto_psth(cur_data, output_subfolder, unit_name, psth_bin_size, pre_stimulus_raster,
                   post_stimulus_raster, psth_fixed_ylim, raster_ylim):
    """Generate one multi-page PDF of PSTHs aligned to opto LED onset and offset.

    Sessions without an 'LED_on_trialSpikes' key (no opto performed) are
    skipped with a printed message.

    Args:
        cur_data (dict): Unit JSON data (``cur_data['Session'][...]`` holds
            'LED_on_trialSpikes'/'LED_off_trialSpikes' arrays).
        output_subfolder (str): Directory to save the output PDF into.
        unit_name (str): Unit identifier, used in the output filename/titles.
        psth_bin_size (float): PSTH histogram bin width (s).
        pre_stimulus_raster (float): Seconds before the event shown.
        post_stimulus_raster (float): Seconds after the event shown.
        psth_fixed_ylim (float): Upper y-limit for the PSTH rate axis.
        raster_ylim (float): Upper y-limit for the raster trial-count axis.

    Returns:
        None. Writes ``<unit_name>_PSTH_<bin_size>ms.pdf`` into ``output_subfolder``.
    """
    print('Plotting Opto PSTH for ' + unit_name + '...')
    with PdfPages(sep.join([output_subfolder, unit_name + '_PSTH_' + str(psth_bin_size) + 'ms.pdf'])) as pdf:
        for session in cur_data['Session'].keys():
            # LED on
            try:
                spike_times = [np.array(x) for x in cur_data['Session'][session]['LED_on_trialSpikes']]
            except KeyError:
                print('LED_on_trialSpikes not found in session ' + session)
                continue

            plot_suptitle = unit_name + "\n" + session + '\n' + 'LED ON'
            __plot_aligned_spikes(spike_times, pre_stimulus_raster, post_stimulus_raster, psth_bin_size,
                                  psth_fixed_ylim, raster_ylim, plot_suptitle, pdf)
            collect()

            # LED off
            spike_times = [np.array(x) for x in cur_data['Session'][session]['LED_off_trialSpikes']]
            plot_suptitle = unit_name + "\n" + session + '\n' + 'LED OFF'
            __plot_aligned_spikes(spike_times, pre_stimulus_raster, post_stimulus_raster, psth_bin_size,
                                  psth_fixed_ylim, raster_ylim, plot_suptitle, pdf)
            collect()


def run_PSTH_pipeline(input_list):
    """Entry point: generate all PSTH PDFs (trial-type, AM-depth, opto) for one unit.

    Loads the unit's JSON data, applies uniform plotting style settings, and
    dispatches to ``__trialType_psth``, ``__amDepth_psth``, and
    ``__opto_psth`` in turn, each writing its own PDF under a subject-specific subfolder.

    Args:
        input_list (tuple): ``(file_name, SETTINGS_DICT)`` where ``file_name``
            is the unit JSON path and ``SETTINGS_DICT`` supplies
            ``OUTPUT_PATH`` and the various ``PSTH_*``/``RESPLATENCY_FILTER``/
            ``SHOCK_START_END`` settings.

    Returns:
        None. Writes PDFs under ``<OUTPUT_PATH>/PSTHs/<subject_id>/{TrialType,AMDepth,Opto}``.
    """
    file_name, SETTINGS_DICT = input_list

    data_dict = get_JSON_data._load_one_json(file_name)

    unit_name = data_dict['Unit']
    output_path = SETTINGS_DICT['OUTPUT_PATH'] + sep + 'PSTHs'
    psth_bin_size = SETTINGS_DICT['PSTH_BIN_SIZE']
    pre_stimulus_raster = SETTINGS_DICT['PSTH_PRE_STIMULUS_DURATION']
    post_stimulus_raster = SETTINGS_DICT['PSTH_POST_STIMULUS_DURATION']
    psth_fixed_ylim = SETTINGS_DICT['PSTH_FIXED_YLIM']
    raster_ylim = SETTINGS_DICT['PSTH_RASTER_YLIM']
    trial_types = SETTINGS_DICT['PSTH_TRIALTYPES']
    align_to_response = SETTINGS_DICT['PSTH_ALIGN_TO_RESPONSE']
    responseLatency_filter = SETTINGS_DICT['RESPLATENCY_FILTER']
    shock_artifact = SETTINGS_DICT['SHOCK_START_END']

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
    output_subfolder = sep.join([output_path, subject_id, 'TrialType'])
    makedirs(output_subfolder, exist_ok=True)
    __trialType_psth(data_dict, output_subfolder, unit_name, psth_bin_size, pre_stimulus_raster,
                     post_stimulus_raster, psth_fixed_ylim, raster_ylim, trial_types, align_to_response,
                     shock_artifact)

    ''' PSTH separated by AM depth '''
    output_subfolder = sep.join([output_path, subject_id, 'AMDepth'])
    makedirs(output_subfolder, exist_ok=True)
    __amDepth_psth(data_dict, output_subfolder, unit_name, psth_bin_size, pre_stimulus_raster,
                   post_stimulus_raster, psth_fixed_ylim, raster_ylim, responseLatency_filter)

    ''' PSTH aligned to opto LED onset and offset '''
    output_subfolder = sep.join([output_path, subject_id, 'Opto'])
    makedirs(output_subfolder, exist_ok=True)
    __opto_psth(data_dict, output_subfolder, unit_name, psth_bin_size, pre_stimulus_raster,
                   post_stimulus_raster, psth_fixed_ylim, raster_ylim)


def run_full_PSTH_pipeline(filtered_files, SETTINGS_DICT):
    """Generate PSTH PDFs for every unit in ``filtered_files``.

    This is the notebook-level driver: runs ``run_PSTH_pipeline`` once per
    unit, serially or via a process pool per ``SETTINGS_DICT['MULTIPROCESS']``.

    Args:
        filtered_files (Iterable[str]): Unit JSON paths to process.
        SETTINGS_DICT (dict): Pipeline settings. Uses ``MULTIPROCESS`` and
            ``NUMBER_OF_CORES``; the rest are forwarded to ``run_PSTH_pipeline``.

    Returns:
        None. Writes PDFs under ``<OUTPUT_PATH>/PSTHs/<subject_id>/{TrialType,AMDepth,Opto}``.
    """
    if not SETTINGS_DICT['MULTIPROCESS']:
        for fn in filtered_files:
            run_PSTH_pipeline((fn, SETTINGS_DICT))
    else:
        input_list = [(fn, SETTINGS_DICT) for fn in filtered_files]
        with Pool(SETTINGS_DICT['NUMBER_OF_CORES']) as pool:
            for _ in pool.imap_unordered(run_PSTH_pipeline, input_list, chunksize=1):
                pass