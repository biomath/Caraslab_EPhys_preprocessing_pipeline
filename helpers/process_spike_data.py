__author__ = 'Matheus'
import numpy as np
from astropy.convolution import convolve_fft
from astropy.convolution import Gaussian1DKernel
from pandas import read_csv, DataFrame, concat
from decimal import Decimal


def time_to_binary(spike_times, sampling_rate, max_time=None):
    if max_time is None:
        max_time = max(spike_times)

    # Rounding spike times according to desired sampling rate
    sampling_decimal_accuracy = abs(Decimal(str(sampling_rate)).as_tuple().exponent)
    spike_times = np.round(spike_times, sampling_decimal_accuracy)

    time = np.arange(0, max_time, sampling_rate)
    spikes = np.zeros(len(time))

    indeces = np.in1d(time, spike_times)

    for i in np.where(indeces):
        spikes[i] = 1

    return spikes


def binary_to_time(raster, sampling_rate):
    spike_indices, = np.where(raster == 1)
    spike_indices = spike_indices.astype(float)

    return spike_indices * sampling_rate


def gaussian_moving_average_fft(spikes, sampling_rate, sigma, dtype='times'):
    if dtype is 'times':
        binary_spikes = time_to_binary(spike_times=spikes, sampling_rate=sampling_rate)
        return convolve_fft(binary_spikes, Gaussian1DKernel(stddev=sigma))
    elif dtype is 'binary':
        return convolve_fft(spikes, Gaussian1DKernel(stddev=sigma))


def spike_timing_process_data(rasters_dict, sampling_rate, sigma):
    gaussians_dict = dict((key, []) for key in rasters_dict.keys())
    for key in rasters_dict.keys():
        for sweep in rasters_dict[key]:
            gaussians_dict[key].append(gaussian_moving_average_fft(sweep, sampling_rate, sigma, dtype='binary'))

    return gaussians_dict


def bin_train(train, sampling_rate, bin_ms):
    binned_train = list()

    step_size = int(bin_ms * sampling_rate * 1000)

    start_offset = 0
    while start_offset < len(train):
        if len(train) - start_offset > step_size:
            binned_train.append(np.sum(train[start_offset:(start_offset + step_size)]))
        else:
            binned_train.append(np.sum(train[start_offset:]))

        start_offset += step_size

    return np.asarray(binned_train)


def load_rasters(memory_name, sampling_rate, pre_stimulus_time, post_stimulus_time,
                          key_name=None, make_baseline_stim=False, spike_time_unit='s'):
    """
    Function takes the recording name and the desired gaussian width (sigma) and returns
    a dictionary (pre, ne, post) containing a subdictionary (stim49, stim50, stim51, stim52),
    which contains a list of peristimulus spike rasters smoothed by a gaussian filter
    :param recording: recording name.
    Key file names have to have the format (e.g.) _PRE_KEY.csv
    Memory file names have to have the format (e.g.) _PRE_MEMORY.txt
    :param sigma: gaussian width
    :return:  (pre, ne, post) containing a subdictionary (stim49, stim50, stim51, stim52),
    which contains a list of peristimulus spike rasters smoothed by a gaussian filter
    """
    if key_name is None:
        key_times = read_csv(memory_name)

        spike_times = np.genfromtxt(memory_name)

    else:
        # Check for header
        key_times = read_csv(key_name, header=None)
        if isinstance(key_times.iloc[0, 0], str):
            key_times = read_csv(key_name, header=0)

        spike_times = np.genfromtxt(memory_name)

    if spike_time_unit == 'ms':
        spike_times = spike_times / 1000
    else:
        pass

    if make_baseline_stim:  # grab one baseline per stimulus (2s before;
        # half from beginning, half from end presentations)
        # Count how many baseline rasters we need
        number_of_stimulus_repetitions = 1000  # Arbitrary large number
        for key in set(key_times.iloc[:, 1]):  # set of all stimuli
            curr_stimulus_repetitions = len(key_times[key_times.iloc[:, 1] == key])
            if curr_stimulus_repetitions < number_of_stimulus_repetitions:
                number_of_stimulus_repetitions = curr_stimulus_repetitions

        # Grab half of the timestamps from the first presentations and half from the last presentations
        baseline_timestamps = list()
        dummy_idx = 0
        for stim_idx in np.arange(0, len(key_times.iloc[:, 1])):
            if dummy_idx < number_of_stimulus_repetitions//2:
                baseline_timestamps.append(key_times.iloc[stim_idx, 0] - 2)
                dummy_idx += 1
            else:
                break
        for stim_idx in np.flip(np.arange(0, len(key_times.iloc[:, 1])), 0):  # loop backwards
            if dummy_idx != number_of_stimulus_repetitions:
                baseline_timestamps.append(key_times.iloc[stim_idx, 0] - 2)
                dummy_idx += 1
            else:
                break
        to_append = dict(zip(baseline_timestamps, np.repeat('Baseline', number_of_stimulus_repetitions)))
        df = DataFrame.from_dict(to_append, orient='index').reset_index()
        df.columns = [0, 1]

        key_times = concat([key_times, df], axis=0, ignore_index=True)
        key_times = key_times.sort_values(0)

    rasters_dict = generate_rasters_from_csv(spike_times, key_times, sampling_rate,
                                                       pre_stimulus_time=pre_stimulus_time,
                                                       post_stimulus_time=post_stimulus_time)
    return rasters_dict


def generate_rasters_from_csv(spike_times, markers_df, sampling_rate, pre_stimulus_time,
                                                 post_stimulus_time):
    binary_spikes = time_to_binary(spike_times, sampling_rate)

    # number_of_stimuli = len(set(markers_df.iloc[:, 1]))
    # sort key list by value, just in case. Because later on in the classification, the order will be important
    key_list = sorted(list(set(markers_df.iloc[:, 1])))
    rasters_dict = dict((key, []) for key in key_list)

    for i in np.arange(0, len(markers_df)):
        min_bound = int(np.round((markers_df.iloc[i, 0] - pre_stimulus_time) / sampling_rate))
        max_bound = int(np.round((markers_df.iloc[i, 0] + post_stimulus_time) / sampling_rate))

        # Fix for some rasters having imprecisions in the order of sampling_rate
        # If more than max_bound-min_bound delete last data point (1ms usually)
        # If less, add 0 to the end
        if min_bound < 0:  # in case the recording starts later than the first min bound
            min_bound = 0
        spikes_to_append = binary_spikes[min_bound:max_bound]
        if len(spikes_to_append) > ((post_stimulus_time + pre_stimulus_time) / sampling_rate):
            while len(spikes_to_append) > ((post_stimulus_time + pre_stimulus_time) / sampling_rate):
                spikes_to_append = np.delete(spikes_to_append, len(spikes_to_append) - 1)
        elif len(spikes_to_append) < ((post_stimulus_time + pre_stimulus_time) / sampling_rate):
            while len(spikes_to_append) < ((post_stimulus_time + pre_stimulus_time) / sampling_rate):
                spikes_to_append = np.append(spikes_to_append, 0)

        rasters_dict[markers_df.iloc[i, 1]].append(spikes_to_append)

    return rasters_dict


# OBSOLETE CODE BELOW
# # Four stimuli; one condition
# def four_stim_load_rasters(memory_name, sampling_rate, pre_stimulus_time, post_stimulus_time,
#                            key_name=None):
#     """
#     Function takes the recording name and the desired gaussian width (sigma) and returns
#     a dictionary (pre, ne, post) containing a subdictionary (stim49, stim50, stim51, stim52),
#     which contains a list of peristimulus spike rasters smoothed by a gaussian filter
#     :param recording: recording name.
#     Key file names have to have the format (e.g.) _PRE_KEY.csv
#     Memory file names have to have the format (e.g.) _PRE_MEMORY.txt
#     :param sigma: gaussian width
#     :return:  (pre, ne, post) containing a subdictionary (stim49, stim50, stim51, stim52),
#     which contains a list of peristimulus spike rasters smoothed by a gaussian filter
#     """
#     if key_name is None:
#         key_times = read_csv(memory_name)
#
#         spike_times = np.genfromtxt(memory_name)
#     else:
#         key_times = read_csv(key_name)
#
#         spike_times = np.genfromtxt(memory_name)
#
#     pre_raster_stim49, pre_raster_stim50, pre_raster_stim51, pre_raster_stim52 \
#         = four_stim_generate_rasters_from_csv(spike_times, key_times, sampling_rate=sampling_rate,
#                                               pre_stimulus_time=pre_stimulus_time,
#                                               post_stimulus_time=post_stimulus_time)
#
#     rasters_dict = {'stim49': pre_raster_stim49,
#                     'stim50': pre_raster_stim50,
#                     'stim51': pre_raster_stim51,
#                     'stim52': pre_raster_stim52}
#     return rasters_dict
#
#
# def multi_load_rasters(recording, sampling_rate, pre_stimulus_time, post_stimulus_time):
#     """
#     Function takes the recording name and the desired gaussian width (sigma) and returns
#     a dictionary (pre, ne, post) containing a subdictionary (stim49, stim50, stim51, stim52),
#     which contains a list of peristimulus spike rasters smoothed by a gaussian filter
#     :param recording: recording name.
#     Key file names have to have the format (e.g.) _PRE_KEY.csv
#     Memory file names have to have the format (e.g.) _PRE_MEMORY.txt
#     :param sigma: gaussian width
#     :return:  (pre, ne, post) containing a subdictionary (stim49, stim50, stim51, stim52),
#     which contains a list of peristimulus spike rasters smoothed by a gaussian filter
#     """
#     key_times = read_csv(str(recording + '_key.csv'))
#
#     spike_times = np.genfromtxt(str(recording + '.txt'))
#
#     pre_raster_stim49, pre_raster_stim50, pre_raster_stim51, pre_raster_stim52 \
#         = four_stim_generate_rasters_from_csv(spike_times, key_times, sampling_rate=sampling_rate,
#                                               pre_stimulus_time=pre_stimulus_time,
#                                               post_stimulus_time=post_stimulus_time)
#
#     rasters_dict = {'stim49': pre_raster_stim49,
#                     'stim50': pre_raster_stim50,
#                     'stim51': pre_raster_stim51,
#                     'stim52': pre_raster_stim52}
#     return rasters_dict
#
#
# def four_stim_spike_timing_process_data(rasters_dict, sampling_rate, sigma=None):
#     """
#     Function takes the recording name and the desired gaussian width (sigma) and returns
#     a dictionary (pre, ne, post) containing a subdictionary (stim49, stim50, stim51, stim52),
#     which contains a list of peristimulus spike rasters smoothed by a gaussian filter
#     :param recording: recording name.
#     Key file names have to have the format (e.g.) _PRE_KEY.csv
#     Memory file names have to have the format (e.g.) _PRE_MEMORY.txt
#     :param sigma: gaussian width
#     :return:  (pre, ne, post) containing a subdictionary (stim49, stim50, stim51, stim52),
#     which contains a list of peristimulus spike rasters smoothed by a gaussian filter
#     """
#
#     pre_gaussian_stim49 = list()
#     for sweep in rasters_dict['stim49']:
#         pre_gaussian_stim49.append(gaussian_moving_average_fft(sweep, sampling_rate, sigma, dtype='binary'))
#
#     pre_gaussian_stim50 = list()
#     for sweep in rasters_dict['stim50']:
#         pre_gaussian_stim50.append(gaussian_moving_average_fft(sweep, sampling_rate, sigma, dtype='binary'))
#
#     pre_gaussian_stim51 = list()
#     for sweep in rasters_dict['stim51']:
#         pre_gaussian_stim51.append(gaussian_moving_average_fft(sweep, sampling_rate, sigma, dtype='binary'))
#
#     pre_gaussian_stim52 = list()
#     for sweep in rasters_dict['stim52']:
#         pre_gaussian_stim52.append(gaussian_moving_average_fft(sweep, sampling_rate, sigma, dtype='binary'))
#
#     gaussians_dict = {'stim49': pre_gaussian_stim49,
#                       'stim50': pre_gaussian_stim50,
#                       'stim51': pre_gaussian_stim51,
#                       'stim52': pre_gaussian_stim52}
#
#     return gaussians_dict
#
#
# def four_stim_generate_rasters_from_csv(spike_times, key_times, pre_stimulus_time,
#                                         post_stimulus_time, sampling_rate):
#     binary_spikes = time_to_binary(spike_times, sampling_rate)
#
#     spikes_stim49 = list()
#     spikes_stim50 = list()
#     spikes_stim51 = list()
#     spikes_stim52 = list()
#
#     for i in np.arange(0, len(key_times)):
#         min_bound = (key_times.iloc[i, 0] - pre_stimulus_time) / sampling_rate
#         max_bound = (key_times.iloc[i, 0] + post_stimulus_time) / sampling_rate
#
#         # Fix for some rasters having imprecisions in the order of sampling_rate
#         # If more than max_bound-min_bound delete last data point (1ms usually)
#         # If less, add 0 to the end
#         spikes_to_append = binary_spikes[min_bound:max_bound]
#         if len(spikes_to_append) - round(max_bound - min_bound, 0) >= 1:
#             while len(spikes_to_append) - round(max_bound - min_bound, 0) >= 1:
#                 spikes_to_append = np.delete(spikes_to_append, len(spikes_to_append) - 1)
#         elif len(spikes_to_append) - round(max_bound - min_bound, 0) <= -1:
#             while len(spikes_to_append) - round(max_bound - min_bound, 0) <= -1:
#                 spikes_to_append = np.append(spikes_to_append, 0)
#
#         if key_times.iloc[i, 1] == 49:
#             spikes_stim49.append(spikes_to_append)
#         elif key_times.iloc[i, 1] == 50:
#             spikes_stim50.append(spikes_to_append)
#         elif key_times.iloc[i, 1] == 51:
#             spikes_stim51.append(spikes_to_append)
#         elif key_times.iloc[i, 1] == 52:
#             spikes_stim52.append(spikes_to_append)
#
#     return (spikes_stim49,
#             spikes_stim50,
#             spikes_stim51,
#             spikes_stim52)
#
#
# # Four stimuli; three conditions: PRE, NE, POST
# def ne_spike_timing_process_data(rasters_dict, sampling_rate,
#                                  sigma=None, pre_sigma=None, ne_sigma=None, post_sigma=None):
#     """
#     Function takes the recording name and the desired gaussian width (sigma) and returns
#     a dictionary (pre, ne, post) containing a subdictionary (stim49, stim50, stim51, stim52),
#     which contains a list of peristimulus spike rasters smoothed by a gaussian filter
#     :param recording: recording name.
#     Key file names have to have the format (e.g.) _PRE_KEY.csv
#     Memory file names have to have the format (e.g.) _PRE_MEMORY.txt
#     :param sigma: gaussian width
#     :return:  (pre, ne, post) containing a subdictionary (stim49, stim50, stim51, stim52),
#     which contains a list of peristimulus spike rasters smoothed by a gaussian filter
#     """
#     if sigma is not None:
#         pre_sigma = ne_sigma = post_sigma = sigma
#
#     pre_gaussian_stim49 = list()
#     for sweep in rasters_dict['pre']['stim49']:
#         pre_gaussian_stim49.append(gaussian_moving_average_fft(sweep, sampling_rate, pre_sigma, dtype='binary'))
#
#     pre_gaussian_stim50 = list()
#     for sweep in rasters_dict['pre']['stim50']:
#         pre_gaussian_stim50.append(gaussian_moving_average_fft(sweep, sampling_rate, pre_sigma, dtype='binary'))
#
#     pre_gaussian_stim51 = list()
#     for sweep in rasters_dict['pre']['stim51']:
#         pre_gaussian_stim51.append(gaussian_moving_average_fft(sweep, sampling_rate, pre_sigma, dtype='binary'))
#
#     pre_gaussian_stim52 = list()
#     for sweep in rasters_dict['pre']['stim52']:
#         pre_gaussian_stim52.append(gaussian_moving_average_fft(sweep, sampling_rate, pre_sigma, dtype='binary'))
#
#     ne_gaussian_stim49 = list()
#     for sweep in rasters_dict['ne']['stim49']:
#         ne_gaussian_stim49.append(gaussian_moving_average_fft(sweep, sampling_rate, ne_sigma, dtype='binary'))
#
#     ne_gaussian_stim50 = list()
#     for sweep in rasters_dict['ne']['stim50']:
#         ne_gaussian_stim50.append(gaussian_moving_average_fft(sweep, sampling_rate, ne_sigma, dtype='binary'))
#
#     ne_gaussian_stim51 = list()
#     for sweep in rasters_dict['ne']['stim51']:
#         ne_gaussian_stim51.append(gaussian_moving_average_fft(sweep, sampling_rate, ne_sigma, dtype='binary'))
#
#     ne_gaussian_stim52 = list()
#     for sweep in rasters_dict['ne']['stim52']:
#         ne_gaussian_stim52.append(gaussian_moving_average_fft(sweep, sampling_rate, ne_sigma, dtype='binary'))
#
#     post_gaussian_stim49 = list()
#     for sweep in rasters_dict['post']['stim49']:
#         post_gaussian_stim49.append(gaussian_moving_average_fft(sweep, sampling_rate, post_sigma, dtype='binary'))
#
#     post_gaussian_stim50 = list()
#     for sweep in rasters_dict['post']['stim50']:
#         post_gaussian_stim50.append(gaussian_moving_average_fft(sweep, sampling_rate, post_sigma, dtype='binary'))
#
#     post_gaussian_stim51 = list()
#     for sweep in rasters_dict['post']['stim51']:
#         post_gaussian_stim51.append(gaussian_moving_average_fft(sweep, sampling_rate, post_sigma, dtype='binary'))
#
#     post_gaussian_stim52 = list()
#     for sweep in rasters_dict['post']['stim52']:
#         post_gaussian_stim52.append(gaussian_moving_average_fft(sweep, sampling_rate, post_sigma, dtype='binary'))
#
#     gaussians_dict = {'pre': {'stim49': pre_gaussian_stim49,
#                               'stim50': pre_gaussian_stim50,
#                               'stim51': pre_gaussian_stim51,
#                               'stim52': pre_gaussian_stim52},
#                       'ne': {'stim49': ne_gaussian_stim49,
#                              'stim50': ne_gaussian_stim50,
#                              'stim51': ne_gaussian_stim51,
#                              'stim52': ne_gaussian_stim52},
#                       'post': {'stim49': post_gaussian_stim49,
#                                'stim50': post_gaussian_stim50,
#                                'stim51': post_gaussian_stim51,
#                                'stim52': post_gaussian_stim52}}
#
#     return gaussians_dict
#
#
# def ne_load_rasters(recording, sampling_rate, pre_stimulus_time, post_stimulus_time):
#     """
#     Function takes the recording name and the desired gaussian width (sigma) and returns
#     a dictionary (pre, ne, post) containing a subdictionary (stim49, stim50, stim51, stim52),
#     which contains a list of peristimulus spike rasters smoothed by a gaussian filter
#     :param recording: recording name.
#     Key file names have to have the format (e.g.) _PRE_KEY.csv
#     Memory file names have to have the format (e.g.) _PRE_MEMORY.txt
#     :return:  (pre, ne, post) containing a subdictionary (stim49, stim50, stim51, stim52),
#     which contains a list of peristimulus spike rasters smoothed by a gaussian filter
#     """
#     pre_key_times = read_csv(str(recording + '_PRE_KEY.csv'))
#     ne_key_times = read_csv(str(recording + '_NE_KEY.csv'))
#     post_key_times = read_csv(str(recording + '_POST_KEY.csv'))
#
#     pre_spike_times = np.genfromtxt(str(recording + '_PRE_MEMORY.txt'))
#     ne_spike_times = np.genfromtxt(str(recording + '_NE_MEMORY.txt'))
#     post_spike_times = np.genfromtxt(str(recording + '_POST_MEMORY.txt'))
#
#     pre_raster_stim49, pre_raster_stim50, pre_raster_stim51, pre_raster_stim52 \
#         = four_stim_generate_rasters_from_csv(pre_spike_times, pre_key_times, sampling_rate=sampling_rate,
#                                               pre_stimulus_time=pre_stimulus_time,
#                                               post_stimulus_time=post_stimulus_time)
#
#     ne_raster_stim49, ne_raster_stim50, ne_raster_stim51, ne_raster_stim52 \
#         = four_stim_generate_rasters_from_csv(ne_spike_times, ne_key_times, sampling_rate=sampling_rate,
#                                               pre_stimulus_time=pre_stimulus_time,
#                                               post_stimulus_time=post_stimulus_time)
#
#     post_raster_stim49, post_raster_stim50, post_raster_stim51, post_raster_stim52 \
#         = four_stim_generate_rasters_from_csv(post_spike_times, post_key_times, sampling_rate=sampling_rate,
#                                               pre_stimulus_time=pre_stimulus_time,
#                                               post_stimulus_time=post_stimulus_time)
#
#     rasters_dict = {'pre': {'stim49': pre_raster_stim49,
#                             'stim50': pre_raster_stim50,
#                             'stim51': pre_raster_stim51,
#                             'stim52': pre_raster_stim52},
#                     'ne': {'stim49': ne_raster_stim49,
#                            'stim50': ne_raster_stim50,
#                            'stim51': ne_raster_stim51,
#                            'stim52': ne_raster_stim52},
#                     'post': {'stim49': post_raster_stim49,
#                              'stim50': post_raster_stim50,
#                              'stim51': post_raster_stim51,
#                              'stim52': post_raster_stim52}}
#
#     return rasters_dict
#
#
# # Four stimuli; four conditions: PRE, LOW, HIGH, POST
# def ne_gonist_load_rasters(unit_name, sampling_rate, pre_stimulus_time, post_stimulus_time, key_name=None):
#     if key_name is None:
#         key_name = unit_name
#
#     pre_key_times = read_csv(str(key_name + '_PRE_KEY.csv'))
#     low_key_times = read_csv(str(key_name + '_LOW_KEY.csv'))
#     high_key_times = read_csv(str(key_name + '_HIGH_KEY.csv'))
#     post_key_times = read_csv(str(key_name + '_POST_KEY.csv'))
#
#     pre_spike_times = np.genfromtxt(str(unit_name + '_PRE_MEMORY.txt'))
#     low_spike_times = np.genfromtxt(str(unit_name + '_LOW_MEMORY.txt'))
#     high_spike_times = np.genfromtxt(str(unit_name + '_HIGH_MEMORY.txt'))
#     post_spike_times = np.genfromtxt(str(unit_name + '_POST_MEMORY.txt'))
#
#     pre_raster_stim49, pre_raster_stim50, pre_raster_stim51, pre_raster_stim52 \
#         = four_stim_generate_rasters_from_csv(pre_spike_times, pre_key_times,
#                                               pre_stimulus_time=pre_stimulus_time,
#                                               post_stimulus_time=post_stimulus_time,
#                                               sampling_rate=sampling_rate)
#
#     low_raster_stim49, low_raster_stim50, low_raster_stim51, low_raster_stim52 \
#         = four_stim_generate_rasters_from_csv(low_spike_times, low_key_times,
#                                               pre_stimulus_time=pre_stimulus_time,
#                                               post_stimulus_time=post_stimulus_time,
#                                               sampling_rate=sampling_rate)
#
#     high_raster_stim49, high_raster_stim50, high_raster_stim51, high_raster_stim52 \
#         = four_stim_generate_rasters_from_csv(high_spike_times, high_key_times,
#                                               pre_stimulus_time=pre_stimulus_time,
#                                               post_stimulus_time=post_stimulus_time,
#                                               sampling_rate=sampling_rate)
#
#     post_raster_stim49, post_raster_stim50, post_raster_stim51, post_raster_stim52 \
#         = four_stim_generate_rasters_from_csv(post_spike_times, post_key_times,
#                                               pre_stimulus_time=pre_stimulus_time,
#                                               post_stimulus_time=post_stimulus_time,
#                                               sampling_rate=sampling_rate)
#
#     rasters_dict = {'pre': {'stim49': pre_raster_stim49,
#                             'stim50': pre_raster_stim50,
#                             'stim51': pre_raster_stim51,
#                             'stim52': pre_raster_stim52},
#                     'low': {'stim49': low_raster_stim49,
#                             'stim50': low_raster_stim50,
#                             'stim51': low_raster_stim51,
#                             'stim52': low_raster_stim52},
#                     'high': {'stim49': high_raster_stim49,
#                              'stim50': high_raster_stim50,
#                              'stim51': high_raster_stim51,
#                              'stim52': high_raster_stim52},
#                     'post': {'stim49': post_raster_stim49,
#                              'stim50': post_raster_stim50,
#                              'stim51': post_raster_stim51,
#                              'stim52': post_raster_stim52}}
#
#     return rasters_dict
#
#
# def ne_gonist_spike_timing_process_data(rasters_dict, sampling_rate, sigmas):
#     """
#     Function takes the recording name and the desired gaussian width (sigma) and returns
#     a dictionary (pre, ne, post) containing a subdictionary (stim49, stim50, stim51, stim52),
#     which contains a list of peristimulus spike rasters smoothed by a gaussian filter
#     :param recording: recording name.
#     Key file names have to have the format (e.g.) _PRE_KEY.csv
#     Memory file names have to have the format (e.g.) _PRE_MEMORY.txt
#     :param sigma: gaussian width
#     :return:  (pre, ne, post) containing a subdictionary (stim49, stim50, stim51, stim52),
#     which contains a list of peristimulus spike rasters smoothed by a gaussian filter
#     """
#     if not isinstance(sigmas, dict):
#         sigmas = {'pre': sigmas, 'low': sigmas, 'high': sigmas, 'post': sigmas}
#     gaussians_dict = {}
#     for condition in ['pre', 'low', 'high', 'post']:
#         gaussians_dict[condition] = {}
#         for stim in ['stim49', 'stim50', 'stim51', 'stim52']:
#             temp_gaussian_list = list()
#             for train in rasters_dict[condition][stim]:
#
#                 temp_gaussian_list.append(gaussian_moving_average_fft(train, sampling_rate,
#                                                                       sigmas[condition], dtype='binary'))
#             gaussians_dict[condition].update({stim: temp_gaussian_list})
#
#     return gaussians_dict
#
#
# # Four stimuli; five conditions: PRE, PREFAD, NEFAD, POSTFAD, POST
# def fad_ne_load_rasters(recording, sampling_rate, pre_stimulus_time, post_stimulus_time):
#     """
#     Function takes the recording name and the desired gaussian width (sigma) and returns
#     a dictionary (pre, ne, post) containing a subdictionary (stim49, stim50, stim51, stim52),
#     which contains a list of peristimulus spike rasters smoothed by a gaussian filter
#     :param recording: recording name.
#     Key file names have to have the format (e.g.) _PRE_KEY.csv
#     Memory file names have to have the format (e.g.) _PRE_MEMORY.txt
#     :param sigma: gaussian width
#     :return:  (pre, ne, post) containing a subdictionary (stim49, stim50, stim51, stim52),
#     which contains a list of peristimulus spike rasters smoothed by a gaussian filter
#     """
#     pre_key_times = read_csv(str(recording + '_PRE_KEY.csv'))
#     prefad_key_times = read_csv(str(recording + '_PREFAD_KEY.csv'))
#     fadne_key_times = read_csv(str(recording + '_FADNE_KEY.csv'))
#     postfad_key_times = read_csv(str(recording + '_POSTFAD_KEY.csv'))
#     post_key_times = read_csv(str(recording + '_POST_KEY.csv'))
#
#     pre_spike_times = np.genfromtxt(str(recording + '_PRE_MEMORY.txt'))
#     prefad_spike_times = np.genfromtxt(str(recording + '_PREFAD_MEMORY.txt'))
#     fadne_spike_times = np.genfromtxt(str(recording + '_FADNE_MEMORY.txt'))
#     postfad_spike_times = np.genfromtxt(str(recording + '_POSTFAD_MEMORY.txt'))
#     post_spike_times = np.genfromtxt(str(recording + '_POST_MEMORY.txt'))
#
#     pre_raster_stim49, pre_raster_stim50, pre_raster_stim51, pre_raster_stim52 \
#         = four_stim_generate_rasters_from_csv(pre_spike_times, pre_key_times, sampling_rate=sampling_rate,
#                                               pre_stimulus_time=pre_stimulus_time,
#                                               post_stimulus_time=post_stimulus_time)
#
#     prefad_raster_stim49, prefad_raster_stim50, prefad_raster_stim51, prefad_raster_stim52 \
#         = four_stim_generate_rasters_from_csv(prefad_spike_times, prefad_key_times, sampling_rate=sampling_rate,
#                                               pre_stimulus_time=pre_stimulus_time,
#                                               post_stimulus_time=post_stimulus_time)
#
#     fadne_raster_stim49, fadne_raster_stim50, fadne_raster_stim51, fadne_raster_stim52 \
#         = four_stim_generate_rasters_from_csv(fadne_spike_times, fadne_key_times, sampling_rate=sampling_rate,
#                                               pre_stimulus_time=pre_stimulus_time,
#                                               post_stimulus_time=post_stimulus_time)
#
#     postfad_raster_stim49, postfad_raster_stim50, postfad_raster_stim51, postfad_raster_stim52 \
#         = four_stim_generate_rasters_from_csv(postfad_spike_times, postfad_key_times, sampling_rate=sampling_rate,
#                                               pre_stimulus_time=pre_stimulus_time,
#                                               post_stimulus_time=post_stimulus_time)
#
#     post_raster_stim49, post_raster_stim50, post_raster_stim51, post_raster_stim52 \
#         = four_stim_generate_rasters_from_csv(post_spike_times, post_key_times, sampling_rate=sampling_rate,
#                                               pre_stimulus_time=pre_stimulus_time,
#                                               post_stimulus_time=post_stimulus_time)
#
#     rasters_dict = {'pre': {'stim49': pre_raster_stim49,
#                             'stim50': pre_raster_stim50,
#                             'stim51': pre_raster_stim51,
#                             'stim52': pre_raster_stim52},
#                     'prefad': {'stim49': prefad_raster_stim49,
#                                'stim50': prefad_raster_stim50,
#                                'stim51': prefad_raster_stim51,
#                                'stim52': prefad_raster_stim52},
#                     'fadne': {'stim49': fadne_raster_stim49,
#                               'stim50': fadne_raster_stim50,
#                               'stim51': fadne_raster_stim51,
#                               'stim52': fadne_raster_stim52},
#                     'postfad': {'stim49': postfad_raster_stim49,
#                                 'stim50': postfad_raster_stim50,
#                                 'stim51': postfad_raster_stim51,
#                                 'stim52': postfad_raster_stim52},
#                     'post': {'stim49': post_raster_stim49,
#                              'stim50': post_raster_stim50,
#                              'stim51': post_raster_stim51,
#                              'stim52': post_raster_stim52}}
#
#     return rasters_dict
#
#
# def fad_ne_spike_timing_process_data(rasters_dict, sampling_rate, sigma=None, pre_sigma=None, prefad_sigma=None,
#                                      fadne_sigma=None,
#                                      postfad_sigma=None, post_sigma=None):
#     """
#     Function takes the recording name and the desired gaussian width (sigma) and returns
#     a dictionary (pre, ne, post) containing a subdictionary (stim49, stim50, stim51, stim52),
#     which contains a list of peristimulus spike rasters smoothed by a gaussian filter
#     :param recording: recording name.
#     Key file names have to have the format (e.g.) _PRE_KEY.csv
#     Memory file names have to have the format (e.g.) _PRE_MEMORY.txt
#     :param sigma: gaussian width
#     :return:  (pre, ne, post) containing a subdictionary (stim49, stim50, stim51, stim52),
#     which contains a list of peristimulus spike rasters smoothed by a gaussian filter
#     """
#     if sigma is not None:
#         pre_sigma = prefad_sigma = fadne_sigma = postfad_sigma = post_sigma = sigma
#
#     pre_gaussian_stim49 = list()
#     for sweep in rasters_dict['pre']['stim49']:
#         pre_gaussian_stim49.append(gaussian_moving_average_fft(sweep, sampling_rate, pre_sigma, dtype='binary'))
#
#     pre_gaussian_stim50 = list()
#     for sweep in rasters_dict['pre']['stim50']:
#         pre_gaussian_stim50.append(gaussian_moving_average_fft(sweep, sampling_rate, pre_sigma, dtype='binary'))
#
#     pre_gaussian_stim51 = list()
#     for sweep in rasters_dict['pre']['stim51']:
#         pre_gaussian_stim51.append(gaussian_moving_average_fft(sweep, sampling_rate, pre_sigma, dtype='binary'))
#
#     pre_gaussian_stim52 = list()
#     for sweep in rasters_dict['pre']['stim52']:
#         pre_gaussian_stim52.append(gaussian_moving_average_fft(sweep, sampling_rate, pre_sigma, dtype='binary'))
#
#     prefad_gaussian_stim49 = list()
#     for sweep in rasters_dict['prefad']['stim49']:
#         prefad_gaussian_stim49.append(gaussian_moving_average_fft(sweep, sampling_rate, prefad_sigma, dtype='binary'))
#
#     prefad_gaussian_stim50 = list()
#     for sweep in rasters_dict['prefad']['stim50']:
#         prefad_gaussian_stim50.append(gaussian_moving_average_fft(sweep, sampling_rate, prefad_sigma, dtype='binary'))
#
#     prefad_gaussian_stim51 = list()
#     for sweep in rasters_dict['prefad']['stim51']:
#         prefad_gaussian_stim51.append(gaussian_moving_average_fft(sweep, sampling_rate, prefad_sigma, dtype='binary'))
#
#     prefad_gaussian_stim52 = list()
#     for sweep in rasters_dict['prefad']['stim52']:
#         prefad_gaussian_stim52.append(gaussian_moving_average_fft(sweep, sampling_rate, prefad_sigma, dtype='binary'))
#
#     fadne_gaussian_stim49 = list()
#     for sweep in rasters_dict['fadne']['stim49']:
#         fadne_gaussian_stim49.append(gaussian_moving_average_fft(sweep, sampling_rate, fadne_sigma, dtype='binary'))
#
#     fadne_gaussian_stim50 = list()
#     for sweep in rasters_dict['fadne']['stim50']:
#         fadne_gaussian_stim50.append(gaussian_moving_average_fft(sweep, sampling_rate, fadne_sigma, dtype='binary'))
#
#     fadne_gaussian_stim51 = list()
#     for sweep in rasters_dict['fadne']['stim51']:
#         fadne_gaussian_stim51.append(gaussian_moving_average_fft(sweep, sampling_rate, fadne_sigma, dtype='binary'))
#
#     fadne_gaussian_stim52 = list()
#     for sweep in rasters_dict['fadne']['stim52']:
#         fadne_gaussian_stim52.append(gaussian_moving_average_fft(sweep, sampling_rate, fadne_sigma, dtype='binary'))
#
#     postfad_gaussian_stim49 = list()
#     for sweep in rasters_dict['postfad']['stim49']:
#         postfad_gaussian_stim49.append(gaussian_moving_average_fft(sweep, sampling_rate, postfad_sigma, dtype='binary'))
#
#     postfad_gaussian_stim50 = list()
#     for sweep in rasters_dict['postfad']['stim50']:
#         postfad_gaussian_stim50.append(gaussian_moving_average_fft(sweep, sampling_rate, postfad_sigma, dtype='binary'))
#
#     postfad_gaussian_stim51 = list()
#     for sweep in rasters_dict['postfad']['stim51']:
#         postfad_gaussian_stim51.append(gaussian_moving_average_fft(sweep, sampling_rate, postfad_sigma, dtype='binary'))
#
#     postfad_gaussian_stim52 = list()
#     for sweep in rasters_dict['postfad']['stim52']:
#         postfad_gaussian_stim52.append(gaussian_moving_average_fft(sweep, sampling_rate, postfad_sigma, dtype='binary'))
#
#     post_gaussian_stim49 = list()
#     for sweep in rasters_dict['post']['stim49']:
#         post_gaussian_stim49.append(gaussian_moving_average_fft(sweep, sampling_rate, post_sigma, dtype='binary'))
#
#     post_gaussian_stim50 = list()
#     for sweep in rasters_dict['post']['stim50']:
#         post_gaussian_stim50.append(gaussian_moving_average_fft(sweep, sampling_rate, post_sigma, dtype='binary'))
#
#     post_gaussian_stim51 = list()
#     for sweep in rasters_dict['post']['stim51']:
#         post_gaussian_stim51.append(gaussian_moving_average_fft(sweep, sampling_rate, post_sigma, dtype='binary'))
#
#     post_gaussian_stim52 = list()
#     for sweep in rasters_dict['post']['stim52']:
#         post_gaussian_stim52.append(gaussian_moving_average_fft(sweep, sampling_rate, post_sigma, dtype='binary'))
#
#     gaussians_dict = {'pre': {'stim49': pre_gaussian_stim49,
#                               'stim50': pre_gaussian_stim50,
#                               'stim51': pre_gaussian_stim51,
#                               'stim52': pre_gaussian_stim52},
#                       'prefad': {'stim49': prefad_gaussian_stim49,
#                                  'stim50': prefad_gaussian_stim50,
#                                  'stim51': prefad_gaussian_stim51,
#                                  'stim52': prefad_gaussian_stim52},
#                       'fadne': {'stim49': fadne_gaussian_stim49,
#                                 'stim50': fadne_gaussian_stim50,
#                                 'stim51': fadne_gaussian_stim51,
#                                 'stim52': fadne_gaussian_stim52},
#                       'postfad': {'stim49': postfad_gaussian_stim49,
#                                   'stim50': postfad_gaussian_stim50,
#                                   'stim51': postfad_gaussian_stim51,
#                                   'stim52': postfad_gaussian_stim52},
#                       'post': {'stim49': post_gaussian_stim49,
#                                'stim50': post_gaussian_stim50,
#                                'stim51': post_gaussian_stim51,
#                                'stim52': post_gaussian_stim52}}
#
#     return gaussians_dict
#
#
# def six_stim_four_stim_generate_rasters_from_csv(spike_times, markers_df, sampling_rate, pre_stimulus_time,
#                                                  post_stimulus_time):
#     binary_spikes = time_to_binary(spike_times, sampling_rate)
#
#     spikes_stim49 = list()
#     spikes_stim50 = list()
#     spikes_stim51 = list()
#     spikes_stim52 = list()
#     spikes_stim53 = list()
#     spikes_stim54 = list()
#
#     for i in np.arange(0, len(markers_df)):
#         min_bound = (markers_df.iloc[i, 0] - pre_stimulus_time) / sampling_rate
#         max_bound = (markers_df.iloc[i, 0] + post_stimulus_time) / sampling_rate
#
#         # Fix for some rasters having imprecisions in the order of sampling_rate
#         # If more than max_bound-min_bound delete last data point (1ms usually)
#         # If less, add 0 to the end
#         spikes_to_append = binary_spikes[min_bound:max_bound]
#         if len(spikes_to_append) - round(max_bound - min_bound, 0) >= 1:
#             while len(spikes_to_append) - round(max_bound - min_bound, 0) >= 1:
#                 spikes_to_append = np.delete(spikes_to_append, len(spikes_to_append) - 1)
#         elif len(spikes_to_append) - round(max_bound - min_bound, 0) <= -1:
#             while len(spikes_to_append) - round(max_bound - min_bound, 0) <= -1:
#                 spikes_to_append = np.append(spikes_to_append, 0)
#
#         if markers_df.iloc[i, 1] == 49:
#             spikes_stim49.append(spikes_to_append)
#         elif markers_df.iloc[i, 1] == 50:
#             spikes_stim50.append(spikes_to_append)
#         elif markers_df.iloc[i, 1] == 51:
#             spikes_stim51.append(spikes_to_append)
#         elif markers_df.iloc[i, 1] == 52:
#             spikes_stim52.append(spikes_to_append)
#         elif markers_df.iloc[i, 1] == 53:
#             spikes_stim53.append(spikes_to_append)
#         elif markers_df.iloc[i, 1] == 54:
#             spikes_stim54.append(spikes_to_append)
#
#     return (spikes_stim49,
#             spikes_stim50,
#             spikes_stim51,
#             spikes_stim52,
#             spikes_stim53,
#             spikes_stim54)
#
#
# # Six stimuli; one condition
# def six_stim_spike_timing_process_data(rasters_dict, sampling_rate, sigma):
#     """
#     Function takes the recording name and the desired gaussian width (sigma) and returns
#     a dictionary (pre, ne, post) containing a subdictionary (stim49, stim50, stim51, stim52),
#     which contains a list of peristimulus spike rasters smoothed by a gaussian filter
#     :param recording: recording name.
#     Key file names have to have the format (e.g.) _PRE_KEY.csv
#     Memory file names have to have the format (e.g.) _PRE_MEMORY.txt
#     :param sigma: gaussian width
#     :return:  (pre, ne, post) containing a subdictionary (stim49, stim50, stim51, stim52),
#     which contains a list of peristimulus spike rasters smoothed by a gaussian filter
#     """
#     gaussian_stim49 = list()
#     for sweep in rasters_dict['stim49']:
#         gaussian_stim49.append(gaussian_moving_average_fft(sweep, sampling_rate, sigma, dtype='binary'))
#
#     gaussian_stim50 = list()
#     for sweep in rasters_dict['stim50']:
#         gaussian_stim50.append(gaussian_moving_average_fft(sweep, sampling_rate, sigma, dtype='binary'))
#
#     gaussian_stim51 = list()
#     for sweep in rasters_dict['stim51']:
#         gaussian_stim51.append(gaussian_moving_average_fft(sweep, sampling_rate, sigma, dtype='binary'))
#
#     gaussian_stim52 = list()
#     for sweep in rasters_dict['stim52']:
#         gaussian_stim52.append(gaussian_moving_average_fft(sweep, sampling_rate, sigma, dtype='binary'))
#
#     gaussian_stim53 = list()
#     for sweep in rasters_dict['stim53']:
#         gaussian_stim53.append(gaussian_moving_average_fft(sweep, sampling_rate, sigma, dtype='binary'))
#
#     gaussian_stim54 = list()
#     for sweep in rasters_dict['stim54']:
#         gaussian_stim54.append(gaussian_moving_average_fft(sweep, sampling_rate, sigma, dtype='binary'))
#
#     gaussians_dict = {'stim49': gaussian_stim49,
#                       'stim50': gaussian_stim50,
#                       'stim51': gaussian_stim51,
#                       'stim52': gaussian_stim52,
#                       'stim53': gaussian_stim53,
#                       'stim54': gaussian_stim54}
#
#     return gaussians_dict
#
#
# def six_stim_load_rasters(memory_name, sampling_rate, pre_stimulus_time, post_stimulus_time,
#                           number_of_stimulus_repetitions=20,
#                           key_name=None, key_file_has_header=0):
#     """
#     Function takes the recording name and the desired gaussian width (sigma) and returns
#     a dictionary (pre, ne, post) containing a subdictionary (stim49, stim50, stim51, stim52),
#     which contains a list of peristimulus spike rasters smoothed by a gaussian filter
#     :param recording: recording name.
#     Key file names have to have the format (e.g.) _PRE_KEY.csv
#     Memory file names have to have the format (e.g.) _PRE_MEMORY.txt
#     :param sigma: gaussian width
#     :return:  (pre, ne, post) containing a subdictionary (stim49, stim50, stim51, stim52),
#     which contains a list of peristimulus spike rasters smoothed by a gaussian filter
#     """
#     if key_name is None:
#         key_times = read_csv(memory_name)
#
#         spike_times = np.genfromtxt(memory_name)
#     else:
#         key_times = read_csv(key_name, header=key_file_has_header)
#
#         spike_times = np.genfromtxt(memory_name)
#
#     assert key_times.shape[0] == 6 * number_of_stimulus_repetitions
#
#     raster_stim49, raster_stim50, raster_stim51, raster_stim52, raster_stim53, raster_stim54 \
#         = six_stim_four_stim_generate_rasters_from_csv(spike_times, key_times, sampling_rate,
#                                                        pre_stimulus_time=pre_stimulus_time,
#                                                        post_stimulus_time=post_stimulus_time)
#
#     rasters_dict = {'stim49': raster_stim49,
#                     'stim50': raster_stim50,
#                     'stim51': raster_stim51,
#                     'stim52': raster_stim52,
#                     'stim53': raster_stim53,
#                     'stim54': raster_stim54}
#
#     return rasters_dict
#
