import numpy as np

from intrastim_correlation import *
import matplotlib as mpl
from pandas import read_csv, DataFrame
from os import makedirs
from os.path import sep
from re import split
from glob import glob
from datetime import datetime
from format_axes import *
from matplotlib.backends.backend_pdf import PdfPages
import platform
from scipy.interpolate import CubicSpline

import warnings

import json

# Tweak the regex file separator for cross-platform compatibility
if platform.system() == 'Windows':
    REGEX_SEP = sep*2
else:
    REGEX_SEP = sep


def plot_waveforms(waveform_file, wf_color='black', plot_err=True, use_std=True, f=None, ax=None, normalize=True):
    resampling_factor = 1000

    # Split path name to get subject, session and unit ID for prettier output
    split_json_path = split(REGEX_SEP, waveform_file)  # split path
    unit_id = split_json_path[-1][:-4]  # Example unit id: SUBJ-ID-26_SUBJ-ID-26_200711_concat_cluster41
    subj_id = split("_*_", unit_id)[0]

    if RECORDING_TYPE_DICT[subj_id] == 'synapse':
        sampling_rate = 24410.0625
    else:
        sampling_rate = 30000

    # file_name_pdf = OUTPUT_PATH + sep + unit_id + "_waveformSamples"

    # with PdfPages(file_name_pdf + '.pdf') as pdf:
    if f is None:
        f = plt.figure()
        if ax is None:
            ax = f.add_subplot(111)

    cax_list = list()
    x_time = None

    cur_wfs = read_csv(waveform_file, header=None, skiprows=1)
    if normalize:
        cur_wfs = cur_wfs.divide(np.nanmax(np.abs(cur_wfs), axis=1), axis='index')

    cur_mean = np.nanmean(cur_wfs, axis=0)

    if use_std:
        cur_error = np.nanstd(cur_wfs, axis=0)
    else:
        cur_error = np.nanstd(cur_wfs, axis=0) / np.sqrt(np.count_nonzero(~np.isnan(cur_wfs), axis=0))

    mean_f = CubicSpline(np.arange(0, len(cur_mean)), cur_mean)
    se_f = CubicSpline(np.arange(0, len(cur_error)), cur_error)

    x_time = np.linspace(0, len(cur_mean), resampling_factor)

    # for cur_raw_idx in cur_wfs.sample(500, replace=False):
    #     cur_raw = cur_wfs.iloc[cur_raw_idx, :]
    #     # oversampled_raw = resample(cur_raw, len(cur_raw)*resampling_factor)
    #     raw_f = CubicSpline(np.arange(0, len(cur_raw)), cur_raw)
    #     ax.errorbar(x_time, raw_f(x_time), color=color_waveforms, alpha = 0.1)

    cax_list.append(ax.errorbar(x_time, mean_f(x_time), color=wf_color))

    if plot_err:
        n_error = ax.fill_between(x_time,
                                  mean_f(x_time) - se_f(x_time),
                                  mean_f(x_time) + se_f(x_time), alpha=0.2,
                                  color=wf_color)

    # f.legend(cax_list, ['Cluster ' + str(x) for x in range(1, len(cax_list) + 1)], loc='upper right',
    #          bbox_transform=plt.gcf().transFigure)

    ax.set_xticklabels(np.round(ax.get_xticks() / sampling_rate * 1000, 1))
    ax.set_xlabel('Time (ms)')

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_yticks([])
    ax.xaxis.set_ticks_position('bottom')

        # pdf.savefig()
        # plt.close()
    return f, ax


def plot_overlaid_waveforms(waveform_files):
    color_mean = 'black'
    cbPalette = ( "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7", "#d55e00")
    f = plt.figure()
    ax = f.add_subplot(111)

    for dummy_idx, waveform_file in enumerate(waveform_files):
        if dummy_idx+1 < len(cbPalette):
            plot_waveforms(waveform_file, cbPalette[dummy_idx], normalize=True, use_std=True, f=f, ax=ax)
        else:
            break

    return f, ax
"""
Set global paths and variables
"""
warnings.filterwarnings("ignore")

OUTPUT_PATH = '.' + sep + sep.join(['Sample_data', 'Output', 'Waveform plots'])

ALL_WAVEFORMFILES = glob('.' + sep + sep.join(['Sample_data', 'Waveform samples', '*cluster*_waveforms.csv']))

# Load existing JSONs to add waveform info; will be empty if this is the first time running
ALL_JSON = glob('.' + sep + sep.join(['Sample_data', 'Output', 'JSON files', '*.json']))

# Only run these cells/su or None to run all
# CELLS_TO_RUN = ['SUBJ-ID-154_210504_concat_cluster5008', 'SUBJ-ID-154_210511_concat_cluster1976']

# tonic changes representatives
# CELLS_TO_RUN = ['SUBJ-ID-231_210705_concat_cluster1552', 'SUBJ-ID-390_220621_concat_cluster4318']

# Magazine figure representatives
CELLS_TO_RUN = ['SUBJ-ID-390_220621_concat_cluster4318', 'SUBJ-ID-390_220621_concat_cluster5764',
                'SUBJ-ID-390_220621_concat_cluster5747', 'SUBJ-ID-390_220621_concat_cluster6646',
                'SUBJ-ID-390_220621_concat_cluster6789']
# CELLS_TO_RUN = ['SUBJ-ID-390_220621']

OVERLAY = True

# CELLS_TO_RUN = None
SUBJECTS_TO_RUN = None

RECORDING_TYPE_DICT = {
    'SUBJ-ID-197': 'synapse',
    'SUBJ-ID-151': 'synapse',
    'SUBJ-ID-154': 'synapse',
    'SUBJ-ID-231': 'intan',
    'SUBJ-ID-232': 'intan',
    'SUBJ-ID-270': 'intan',
    'SUBJ-ID-389': 'intan',
    'SUBJ-ID-390': 'intan'
}

# Set plotting parameters
LABEL_FONT_SIZE = 15
TICK_LABEL_SIZE = 10
mpl.rcParams['figure.figsize'] = (12, 10)
mpl.rcParams['figure.dpi'] = 300
mpl.rcParams['font.size'] = LABEL_FONT_SIZE * 1.5
mpl.rcParams['axes.labelsize'] = LABEL_FONT_SIZE * 1.5
mpl.rcParams['axes.titlesize'] = LABEL_FONT_SIZE
mpl.rcParams['axes.linewidth'] = LABEL_FONT_SIZE / 12.
mpl.rcParams['legend.fontsize'] = LABEL_FONT_SIZE / 2.
mpl.rcParams['xtick.labelsize'] = TICK_LABEL_SIZE * 1.5
mpl.rcParams['ytick.labelsize'] = TICK_LABEL_SIZE * 1.5
mpl.rcParams['errorbar.capsize'] = LABEL_FONT_SIZE
mpl.rcParams['lines.markersize'] = LABEL_FONT_SIZE / 30.
mpl.rcParams['lines.markeredgewidth'] = LABEL_FONT_SIZE / 30.
mpl.rcParams['lines.linewidth'] = LABEL_FONT_SIZE / 8.

if __name__ == '__main__':
    # Generate a list of inputs to be passed to each worker
    input_lists = list()

    makedirs(OUTPUT_PATH, exist_ok=True)

    chosen_waveform_files = []
    for waveform_file in ALL_WAVEFORMFILES:
        if CELLS_TO_RUN is not None:
            if any([chosen for chosen in CELLS_TO_RUN if chosen in waveform_file]):
                pass
            else:
                continue

        if SUBJECTS_TO_RUN is not None:
            if any([chosen for chosen in SUBJECTS_TO_RUN if chosen in waveform_file]):
                pass
            else:
                continue
        chosen_waveform_files.append(waveform_file)

        # Plot each waveform in a separate file
        if not OVERLAY:
            # Split path name to get subject, session and unit ID for prettier output
            split_json_path = split(REGEX_SEP, waveform_file)  # split path
            unit_id = split_json_path[-1][:-4]  # Example unit id: SUBJ-ID-26_SUBJ-ID-26_200711_concat_cluster41
            file_name_pdf = OUTPUT_PATH + sep + unit_id + "_waveformSamples"

            with PdfPages(file_name_pdf + '.pdf') as pdf:
                f, ax = plot_waveforms(waveform_file)
                pdf.savefig()
                plt.close()

    # Overlay chosen waveforms in a single plot
    file_name_pdf = OUTPUT_PATH + sep + "OverlaidWfs_waveformSamples"
    with PdfPages(file_name_pdf + '.pdf') as pdf:
        f, ax = plot_overlaid_waveforms(chosen_waveform_files)
        pdf.savefig()
        plt.close()