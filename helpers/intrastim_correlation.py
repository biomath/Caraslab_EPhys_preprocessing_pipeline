from os import chdir, makedirs
from glob import glob
from time import time
from pattern_classifiers import *

import csv
import logging
from re import search, split
from os import listdir
from os.path import isdir


def run_correlation(memory_name,
                    key_name,
                    method,
                    sampling_rate,
                    pre_stimulus_time, post_stimulus_time,
                    pre_stimulus_raster, post_stimulus_raster,
                    brain_area,
                    write_master_sheet,
                    master_sheet_name,
                    unit_name,
                    make_baseline_stim=False,
                    count_bin_set=(2,),
                    master_sheet_columns=(),
                    stim_names=None,
                    sigma_set=(1, 2, 4, 8, 16, 32, 64, 128, 256),
                    save_plot=False,
                    save_rasters=False,
                    save_histograms=False,
                    spike_time_unit='s',
                    write_to_csv=False, file_name=None, pdf_handle=None):

    if method in ('rcorr', 'cross_corr'):
        peristim_dict = psd.load_rasters(memory_name, key_name=key_name, sampling_rate=sampling_rate,
                                         pre_stimulus_time=pre_stimulus_time,
                                         post_stimulus_time=post_stimulus_time,
                                         make_baseline_stim=make_baseline_stim, spike_time_unit=spike_time_unit)
        number_of_stimuli = len(peristim_dict.keys())

        for current_sigma in sigma_set:
            peristim_dict = psd.spike_timing_process_data(peristim_dict, sampling_rate=sampling_rate,
                                                          sigma=current_sigma)

            number_of_stimulus_repetitions = 1000  # Arbitrary large number
            for key in peristim_dict.keys():
                curr_stimulus_repetitions = len(peristim_dict[key])
                if curr_stimulus_repetitions < number_of_stimulus_repetitions:
                    number_of_stimulus_repetitions = curr_stimulus_repetitions

            current_corr_matrix_dict = dict([(key_train[0],
                                              np.zeros(
                                                  (number_of_stimulus_repetitions, number_of_stimulus_repetitions)))
                                             for key_train in peristim_dict.items()])
            current_summary_dict = dict([(key_train[0],
                                          list())
                                         for key_train in peristim_dict.items()])

            for i in np.arange(0, number_of_stimulus_repetitions):
                for j in np.arange(0, number_of_stimulus_repetitions):
                    for stim in peristim_dict.keys():
                        current_corr_matrix_dict[stim][i, j] = \
                            corr_function(peristim_dict[stim][i], peristim_dict[stim][j], method, normalize=True)

                        if i > j:
                            current_summary_dict[stim].append(current_corr_matrix_dict[stim][i, j])

            plot_stim_matrices(current_corr_matrix_dict, method, pdf_handle)
            plot_summary(current_summary_dict, method, pdf_handle)

            plot_rasters(memory_name, sampling_rate, pre_stimulus_raster, post_stimulus_raster, post_stimulus_time,
                         unit_name, method, sigma=current_sigma, make_baseline_stim=make_baseline_stim,
                         key_name=key_name, spike_time_unit=spike_time_unit,
                         stim_names=peristim_dict.keys(),
                         pdf_handle=pdf_handle)

            if write_master_sheet:
                with open(str(master_sheet_name + '_correlations.csv'), 'a', newline='') as file:
                    writer = csv.writer(file, delimiter=',')
                    for i, stimulus in enumerate(current_summary_dict.keys()):
                        writer.writerow(
                            [unit_name] + [method] +
                            [stimulus] + [np.nanmean(current_summary_dict[stimulus])] +
                            [np.nanstd(current_summary_dict[stimulus]) /
                             (np.sqrt(len(current_summary_dict[stimulus]) - np.sum(np.isnan(current_summary_dict[stimulus]))))])


def plot_stim_matrices(corr_matrix_dict, method, pdf_handle):
    import matplotlib.ticker as ticker
    f = plt.figure()
    f.set_size_inches((24, 13.3))
    ax_list = list()
    cax_list = list()
    for idx, stim in enumerate(corr_matrix_dict.keys()):
        ax = f.add_subplot(1, len(corr_matrix_dict.keys()), idx + 1)
        ax_list.append(ax)
        cax_list.append(ax.matshow(corr_matrix_dict[stim], interpolation='nearest', vmin=0, vmax=1, cmap='jet'))
        ax.set_xticklabels([''] + [str(x) for x in np.arange(1, len(corr_matrix_dict[stim]) + 1)])
        ax.set_yticklabels([''] + [str(x) for x in np.arange(1, len(corr_matrix_dict[stim]) + 1)])
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
        plt.title(stim)
        plt.xlabel('Trial number')
        plt.ylabel('Trial number')

    f.colorbar(cax_list[-1], fraction=0.046, pad=0.04)
    f.suptitle(method)

    pdf_handle.savefig()
    plt.close(f)


def plot_summary(summary_dict, method, pdf_handle):
    f = plt.figure()
    f.set_size_inches((24, 13.3))

    ind = np.arange(1, len(summary_dict.keys()) + 1)
    bar_width = 0.4

    means = list()
    stes = list()
    for stim in summary_dict.keys():
        means.append(np.nanmean(summary_dict[stim]))
        stes.append(np.nanstd(summary_dict[stim]) /
                    (np.sqrt(len(summary_dict[stim]) - np.sum(np.isnan(summary_dict[stim])))))

    ax = f.add_subplot(111)

    ax.bar(ind, means, bar_width,
           yerr=[np.zeros(len(ind)), stes],
           color='k', edgecolor='k')
    ax.set_ylim([0, 1])
    f.suptitle(method)

    pdf_handle.savefig()
    plt.close(f)


def plot_rasters(memory_name, sampling_rate, pre_stimulus_raster, post_stimulus_raster, post_stimulus_time, title,
                 method, make_baseline_stim, spike_time_unit='s',
                 downsample_q=None, key_name=None, sigma=None, stim_names=None, pdf_handle=None):
    """
    :param memory_name:
    :param sampling_rate:
    :param pre_stimulus_raster:
    :param post_stimulus_raster:
    :param post_stimulus_time:
    :param key_name:
    :param sigma:
    :param stim_names:
    :return:
    """

    rasters_dict = psd.load_rasters(memory_name, key_name=key_name, sampling_rate=sampling_rate,
                                     pre_stimulus_time=pre_stimulus_raster,
                                     post_stimulus_time=post_stimulus_raster,
                                     make_baseline_stim=make_baseline_stim, spike_time_unit=spike_time_unit)

    # Determining the smallest number of stimulus repetitions
    number_of_stimulus_repetitions = 1000  # Arbitrary large number
    for key in rasters_dict.keys():
        curr_stimulus_repetitions = len(rasters_dict[key])
        if curr_stimulus_repetitions < number_of_stimulus_repetitions:
            number_of_stimulus_repetitions = curr_stimulus_repetitions

    number_of_stimuli = len(rasters_dict.keys())

    if sigma is not None:
        rasters_dict = psd.spike_timing_process_data(
            rasters_dict, sampling_rate=sampling_rate, sigma=sigma)

    f, axarr = plt.subplots(number_of_stimulus_repetitions, number_of_stimuli, sharex='col', sharey='row')
    f.set_size_inches((24, 13.3))

    # Populate with data
    for plot_row_idx in np.arange(0, number_of_stimulus_repetitions):
        for plot_col_idx, stim_name in enumerate(rasters_dict.keys()):
            if method is not 'count':
                train_to_plot = normalize_vector(rasters_dict[stim_name][plot_row_idx])

            if method is not 'count':
                axarr[plot_row_idx, plot_col_idx].plot(
                    np.linspace(-pre_stimulus_raster, post_stimulus_raster, train_to_plot.size),
                    train_to_plot,
                    color='black')
            else:
                train_to_plot[train_to_plot == 0.0] = np.nan
                axarr[plot_row_idx, plot_col_idx].plot(
                    np.linspace(-pre_stimulus_raster, post_stimulus_raster, train_to_plot.size),
                    train_to_plot, "|",
                    markersize=mpl.rcParams['font.size'],  # Change from default on purpose. 1.5 is the scale
                    markeredgewidth=mpl.rcParams['font.size'] / 100,
                    color='black', alpha=0.8)
                # axarr[plot_row_idx, plot_col_idx].set_ylim([0.9, 1.1])

            axarr[plot_row_idx, plot_col_idx].axvspan(xmin=0, xmax=post_stimulus_time, color='black', alpha=0.2)

            axarr[plot_row_idx, plot_col_idx].axis('off')

    for idx, stim_name in enumerate(stim_names):
        axarr[0, idx].set_title(stim_name)
    plt.setp([a.get_yticklabels() for a in axarr[:, 0]], visible=False)
    plt.setp([a.get_xticklabels() for a in axarr[number_of_stimulus_repetitions - 1, :]])
    f.suptitle(title)

    plt.subplots_adjust(hspace=0)

    pdf_handle.savefig()
    plt.close(f)

    return f
