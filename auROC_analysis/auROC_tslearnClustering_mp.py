from os import remove, makedirs
from os.path import sep
import platform
from time import time
from glob import glob
from multiprocessing import Pool, current_process
from numba import cuda

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rcParams
from tslearn.clustering import TimeSeriesKMeans
import pandas as pd
import seaborn as sns
from auROC_analysis.calculate_auROC import *


custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(style="ticks", rc=custom_params)
from matplotlib.backends.backend_pdf import PdfPages

# Tweak the regex file separator for cross-platform compatibility
if platform.system() == 'Windows':
    REGEX_SEP = sep * 2
else:
    REGEX_SEP = sep


def tic():
    return time()


def toc(t0, pre_message='Processing time:'):
    tkend = time() - t0
    print(pre_message + ' %d min, %.3f sec' % (tkend // 60, np.round(tkend % 60, 3)))


def load_timeSeriesKMeans(n_clusters):
    # Helper function to standardize all runs
    km = TimeSeriesKMeans(n_clusters=n_clusters, tol=1e-6, max_iter_barycenter=500, metric='euclidean', init='k-means++')
    return km


def mp_TSKMeans(mp_input_list, output_folder):
    # Output inertia from each run
    # Currently unused function
    k, data = mp_input_list
    tmp_file_name = output_folder + sep + current_process().name + "_dataTSK_tsClustering.npy"

    # Cluster data set to get inertia (aka MSE(within)/MSE(between) clusters)
    km = load_timeSeriesKMeans(n_clusters=k)
    km.fit(data)

    with open(tmp_file_name, mode='a') as file:
        np.save(file, (k, km.inertia_))


def mp_randomTSKMeans(mp_input_list):

    # Create new random reference set and cluster to get inertia
    boot_k_combination, data_shape, data_inertia, output_folder = mp_input_list

    tmp_file_name = output_folder + sep + current_process().name + "_randomTSK_tsClustering.txt"
    # boot_k_gap_array = np.zeros(np.shape(boot_k_combination) + np.array([0, 1]))

    randomReference = np.random.random_sample(size=data_shape)

    km = load_timeSeriesKMeans(n_clusters=boot_k_combination[1])
    km.fit(randomReference)

    reference_inertia = km.inertia_

    # Calculate gap statistic
    gap = np.log(reference_inertia) - np.log(data_inertia)

    boot_k_gap_array = np.array((boot_k_combination[0], boot_k_combination[1], gap))

    with open(tmp_file_name, "ab") as f:
        np.savetxt(f, boot_k_gap_array.reshape(1, boot_k_gap_array.shape[0]), fmt='%d,%d,%.5f', newline='\n')



def gpu_randomTSKMeans(gpu_input_list, output_folder):
        @cuda.jit(nopython=False)
        def compute_randomTSKMeans(boot_k_combination_list, data_shape_list, data_inertia_list, gpu_output_list):
            cuda_idx = cuda.grid(1)
            if cuda_idx < len(gpu_output_list):
                boot_k_combination = boot_k_combination_list[cuda_idx]
                data_shape = data_shape_list[cuda_idx]
                data_inertia = data_inertia_list[cuda_idx]
                # Create new random reference set and cluster to get inertia
                randomReference = np.random.random_sample(size=data_shape)

                km = load_timeSeriesKMeans(n_clusters=boot_k_combination[1])
                km.fit(randomReference)

                reference_inertia = km.inertia_

                # Calculate gap statistic
                gap = np.log(reference_inertia) - np.log(data_inertia)

                gpu_output_list[cuda_idx] = np.array((boot_k_combination[0], boot_k_combination[1], gap))

        threadsperblock  = 32
        blockspergrid = (len(gpu_input_list[0]) + (threadsperblock - 1)) // threadsperblock
        config = [blockspergrid, threadsperblock]
        gpu_output_list = np.empty(len(gpu_input_list[0]))
        boot_k_gap_array = compute_randomTSKMeans[config](gpu_input_list[0], gpu_input_list[1], gpu_input_list[2], gpu_output_list)

        tmp_file_name = output_folder + sep + current_process().name + "_randomTSK_tsClustering.txt"

        # boot_k_gap_array = np.zeros(np.shape(boot_k_combination) + np.array([0, 1]))

        for r_output in boot_k_gap_array:
            with open(tmp_file_name, "ab") as f:
                np.savetxt(f, r_output.reshape(1, r_output.shape[0]), fmt='%d,%d,%.5f', newline='\n')

def mp_optimalK(data, output_folder, number_of_cores=1, maxClusters=10, boot_n=10, sk_factor=1,
                multiProcess=False, use_gpu=True, use_tibshirani_criterion=True):
    """
    This function calculates the optimal number of clusters from time series
    It is optimal for time series because it uses dynamic time warping (DTW) as a distance metric
    The optimal number of clusters is derived from an implementation of the gap-statistic by Tibshirani et al., 2001

    - **parameters**, **types**, **return** and **return types**::
        :param data: time-series data
        :param maxClusters: maximum number of clusters to be evaluated
        :param boot_n: number of iterations (maximum of 50 recommended due to computing time for DTW)
        :param sk_factor: error multiplication factor for choosing best number of clusters
        :param multiProcess: use multiprocessing to speed up computation
        :param use_gpu: use gpu to speed up computation; not implemented yet
        :param use_tibshirani_criterion: If true, use f(k) ≥ f(k+1) - s{k+1}; if false, use the stricter f(k) + s{k} ≥ f(k+1) - s{k+1}
        :return:
            seMax_cluster: optimal number of clusters
            (mean_per_cluster, sks_per_cluster): decision parameters for plotting the gap-statistic graph

        :type data: 2D-numpy.array or list of lists containing numbers
        :type maxClusters: int
        :type boot_n: int
        :type sk_factor: int
        :rtype:
            seMax_cluster: int
            (mean_per_cluster, sks_per_cluster): tuple of numpy.arrays
    """
    print('OptimalK start...')

    data_shape = data.shape
    k_range = range(1, maxClusters + 1)
    data_inertia_list = np.zeros(len(k_range))

    for dummy_idx, k in enumerate(k_range):
        print('Clustering data using %d cluster(s)...' % k)
        tk = tic()
        # Cluster data set to get inertia (aka MSE(within)/MSE(between) clusters)
        km = load_timeSeriesKMeans(n_clusters=k)
        km.fit(data)
        data_inertia_list[dummy_idx] = km.inertia_
        toc(tk)


    boot_k_combinations = np.array(np.meshgrid(range(0, boot_n), k_range)).T.reshape(-1, 2)

    # Remove files leftover from previous runs
    for file in glob(output_folder + sep + '*_randomTSK_tsClustering.txt'):
        remove(file)

    if not multiProcess:
        number_of_cores = 1

    print('Initializing %d process(es) for clustering random distributions in %d simulations x %d clusters = %d iterations' %
          (number_of_cores, boot_n, maxClusters, boot_n * maxClusters))

    tk = tic()
    if multiProcess:
        # Prepare data for multiprocessing
        mp_input_list = \
            list(
                zip(
                    boot_k_combinations,
                    np.tile(data_shape, [len(boot_k_combinations), 1]),
                    np.tile(data_inertia_list, len(range(0, boot_n))),
                    np.repeat(output_folder, len(boot_k_combinations))
                )
            )

        pool = Pool(number_of_cores)
        # Feed each worker with all memory paths from one unit
        pool_map_result = pool.map(mp_randomTSKMeans, mp_input_list)
        pool.close()
        pool.join()
    elif use_gpu:
        # Prepare data for CUDA
        gpu_input_list = [
                boot_k_combinations,
                np.tile(data_shape, [len(boot_k_combinations), 1]),
                np.tile(data_inertia_list, len(range(0, boot_n)))
            ]
        gpu_randomTSKMeans(gpu_input_list, output_folder)
    else:
        input_list = \
            list(
                zip(
                    boot_k_combinations,
                    np.tile(data_shape, [len(boot_k_combinations), 1]),
                    np.tile(data_inertia_list, len(range(0, boot_n))),
                    np.repeat(output_folder, len(boot_k_combinations))
                )
            )

        for mp_input in input_list:
            mp_randomTSKMeans(mp_input)

    toc(tk)

    print('Compiling data and computing gap-statistic...')
    tk = tic()
    # Open all processes temp files and append them
    tmp_file_names = glob(output_folder + sep + '*_randomTSK_tsClustering.txt')
    boot_k_gap_array = np.empty([0, 3], float)
    for tmp_file_name in tmp_file_names:
        cur_bkg_array = np.loadtxt(tmp_file_name, delimiter=',', dtype=float, ndmin=2)
        boot_k_gap_array = np.append(boot_k_gap_array, cur_bkg_array, axis=0)

    # Remove temp files
    for file in glob(output_folder + sep + '*_randomTSK_tsClustering.txt'):
        remove(file)

    mean_per_cluster = np.zeros(len(k_range))
    std_per_cluster = np.zeros(len(k_range))
    for dummy_idx, k in enumerate(k_range):
        cur_k_gaps = [bkg[2] for bkg in boot_k_gap_array if bkg[1] == k]
        mean_per_cluster[dummy_idx] = np.mean(cur_k_gaps)
        std_per_cluster[dummy_idx] = np.std(cur_k_gaps, ddof=0)

    # Tibshirani et al., 2001's error: s(k)_factor * s(k)
    sks_per_cluster = sk_factor * (np.sqrt(1 + 1 / boot_n) * std_per_cluster)
    tibs_score_lower_bound = np.array([cur_mean - cur_sk for cur_mean, cur_sk in zip(mean_per_cluster, sks_per_cluster)])
    tibs_score_upper_bound = np.array([cur_mean + cur_sk for cur_mean, cur_sk in zip(mean_per_cluster, sks_per_cluster)])

    first_seMax = -1000  # arbitrarily large
    seMax_found = False
    # Loop through clusters in order until criterion is met
    # “Tibshirani et al (2001) proposed: the smallest k such that f(k) ≥ f(k+1) - s{k+1}”
    # For a stricter criterion without increasing sk_factor you can use the upper bound of the scores,
    #   i.e., f(k) + s{k} ≥ f(k+1) - s{k+1}
    for cluster_idx in np.arange(0, maxClusters - 1):
        if seMax_found:
            break
        if use_tibshirani_criterion:
            if mean_per_cluster[cluster_idx] < tibs_score_lower_bound[cluster_idx + 1]:
                first_seMax = mean_per_cluster[cluster_idx]
            else:
                first_seMax = mean_per_cluster[cluster_idx]
                seMax_found = True
        else:
            if tibs_score_upper_bound[cluster_idx] < tibs_score_lower_bound[cluster_idx + 1]:
                first_seMax = mean_per_cluster[cluster_idx]
            else:
                first_seMax = mean_per_cluster[cluster_idx]
                seMax_found = True

    if not seMax_found:
        seMax_cluster = maxClusters
    else:
        seMax_cluster = np.where(mean_per_cluster == first_seMax)[0][0] + 1

    toc(tk)

    return seMax_cluster, (mean_per_cluster, sks_per_cluster)


def run_ts_clustering(data_dict, SETTINGS_DICT):
    output_path = SETTINGS_DICT['OUTPUT_PATH'] + sep + 'TS_clustering_SU'
    makedirs(output_path, exist_ok=True)

    # Retrieve parameters
    # These were parameters used in constructing the PSTHs or auROC curves; should not change if kept default
    pretrial_duration_for_spiketimes = SETTINGS_DICT['AUROC_PRE_STIMULUS_DURATION']
    posttrial_duration_for_spiketimes = SETTINGS_DICT['AUROC_POST_STIMULUS_DURATION']
    binsize = SETTINGS_DICT['AUROC_BIN_SIZE']

    number_of_cores = SETTINGS_DICT['NUMBER_OF_CORES']
    do_multiprocess = SETTINGS_DICT['MULTIPROCESS']
    use_gpu = SETTINGS_DICT['USE_GPU']

    # Turn off CPU multiprocessing if GPU is enabled
    if use_gpu:
        do_multiprocess = False

    # optimalK clustering and gap-stat parameters
    maxclusters = SETTINGS_DICT['MAXCLUSTERS']
    boot_n = SETTINGS_DICT['BOOT_N']
    sk_factor = SETTINGS_DICT['SK_FACTOR']
    use_tibshirani_criterion = SETTINGS_DICT['USE_TIBSHIRANI_CRITERION']

    # Events you want to cluster and the timespan to use for the clustering
    run_dict = SETTINGS_DICT['TS_TRIALTYPES']

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
    rcParams["legend.loc"] = 'upper right'

    rcParams['figure.figsize'] = (5, 5)

    for cur_col in run_dict.keys():
        print("Running gap-stat clustering for metric %s with %d maxclusters and %d simulations" %
              (cur_col, maxclusters, boot_n))
        t0 = tic()
        unit_list = []
        auroc_list = []
        print("Loading data in JSONs...")
        unique_units = sorted(list(set((data_dict.keys()))))
        for unit in unique_units:

            # Passive data are special cases because trial type is irrelevant; handle them first
            try:
                if 'Passive' in cur_col:
                    if 'Pre' in cur_col:
                        cur_data = data_dict[unit]['pre']
                    elif 'Post1h' in cur_col:
                        cur_data = data_dict[unit]['post1h']
                    elif 'Post' in cur_col:
                        cur_data = data_dict[unit]['post']
                    else:
                        continue
                    response_curve = np.array(cur_data['TrialAligned_GO_auroc'])
                else:
                    # Active data, thus trial type is relevant
                    cur_data = data_dict[unit]['active']
                    response_curve = np.array(cur_data[cur_col])

                # Exclude units without responses (no FA trials for example)
                if len(response_curve) == 0 or np.mean(response_curve) == 0:
                    continue

            except KeyError:
                psth_binsize = SETTINGS_DICT['PSTH_BIN_SIZE']
                auroc_binsize = SETTINGS_DICT['AUROC_BIN_SIZE']
                # cur_col absent from JSON file, try to generate a new field if it's implemented
                if cur_col == 'TrialAligned_Hit_middBs_auroc':
                    cur_data = data_dict[unit]['active']
                    # Include only AM depths in the middle of the range (-12:-6 dB) which are the ones to be learned
                    amdepth_subset = [0.18, 0.25, 0.35, 0.5]
                    cur_data = run_calculate_auROC(cur_data,
                                                       session_name=cur_data['Session'],
                                                       trial_or_response_aligned='trialAligned',
                                                       pre_stimulus_baseline_start=pretrial_duration_for_spiketimes,
                                                       pre_stimulus_baseline_end=pretrial_duration_for_spiketimes - 1,
                                                       pre_stimulus_raster=pretrial_duration_for_spiketimes,
                                                       post_stimulus_raster=posttrial_duration_for_spiketimes,
                                                       shock_flag='All',
                                                       trial_type='Hit',
                                                       byAM_depth=False,
                                                       amdepth_subset=amdepth_subset,
                                                       psth_binsize=psth_binsize,
                                                       auroc_binsize=auroc_binsize,
                                                       from_JSON=True
                                                       )
                elif cur_col == 'ResponseAligned_Hit_middBs_auroc':
                    cur_data = data_dict[unit]['active']
                    # Include only AM depths in the middle of the range (-12:-6 dB) which are the ones to be learned
                    amdepth_subset = [0.18, 0.25, 0.35, 0.5]
                    cur_data = run_calculate_auROC(cur_data,
                                                   session_name=cur_data['Session'],
                                                   trial_or_response_aligned='responseAligned',
                                                   pre_stimulus_baseline_start=pretrial_duration_for_spiketimes,
                                                   pre_stimulus_baseline_end=pretrial_duration_for_spiketimes - 1,
                                                   pre_stimulus_raster=pretrial_duration_for_spiketimes,
                                                   post_stimulus_raster=posttrial_duration_for_spiketimes,
                                                   shock_flag='All',
                                                   trial_type='Hit',
                                                   byAM_depth=False,
                                                   amdepth_subset=amdepth_subset,
                                                   psth_binsize=psth_binsize,
                                                   auroc_binsize=auroc_binsize,
                                                   from_JSON=True
                                                   )
                elif cur_col == 'TrialAligned_Miss_middBs_auroc':
                    cur_data = data_dict[unit]['active']
                    # Include only AM depths in the middle of the range (-12:-6 dB) which are the ones to be learned
                    amdepth_subset = [0.18, 0.25, 0.35, 0.5]
                    cur_data = run_calculate_auROC(cur_data,
                                                       session_name=cur_data['Session'],
                                                       trial_or_response_aligned='trialAligned',
                                                       pre_stimulus_baseline_start=pretrial_duration_for_spiketimes,
                                                       pre_stimulus_baseline_end=pretrial_duration_for_spiketimes - 1,
                                                       pre_stimulus_raster=pretrial_duration_for_spiketimes,
                                                       post_stimulus_raster=posttrial_duration_for_spiketimes,
                                                       shock_flag='All',
                                                       trial_type='Miss',
                                                       byAM_depth=False,
                                                       amdepth_subset=amdepth_subset,
                                                       psth_binsize=psth_binsize,
                                                       auroc_binsize=auroc_binsize,
                                                       from_JSON=True
                                                       )
                elif cur_col == 'ResponseAligned_Miss_shockFlagOn_byAMdepth_auroc':
                    cur_data = data_dict[unit]['active']
                    # Include only AM depths in the middle of the range (-12:-6 dB) which are the ones to be learned
                    amdepth_subset = [0.18, 0.25, 0.35, 0.5]
                    cur_data = run_calculate_auROC(cur_data,
                                                       session_name=cur_data['Session'],
                                                       trial_or_response_aligned='responseAligned',
                                                       pre_stimulus_baseline_start=pretrial_duration_for_spiketimes,
                                                       pre_stimulus_baseline_end=pretrial_duration_for_spiketimes - 1,
                                                       pre_stimulus_raster=pretrial_duration_for_spiketimes,
                                                       post_stimulus_raster=posttrial_duration_for_spiketimes,
                                                       shock_flag='On',
                                                       trial_type='Miss',
                                                       byAM_depth=False,
                                                       amdepth_subset=amdepth_subset,
                                                       psth_binsize=psth_binsize,
                                                       auroc_binsize=auroc_binsize,
                                                       from_JSON=True
                                                       )
                elif cur_col == 'ResponseAligned_Hit_respLatencyFilter_middBs_auroc':
                    cur_data = data_dict[unit]['active']
                    respLatency_filter = SETTINGS_DICT['AUROC_RESPLATENCY_FILTER']
                    # Include only AM depths in the middle of the range (-12:-6 dB) which are the ones to be learned
                    amdepth_subset = [0.18, 0.25, 0.35, 0.5]
                    cur_data = run_calculate_auROC(cur_data,
                                                   session_name=cur_data['Session'],
                                                   trial_or_response_aligned='responseAligned',
                                                   pre_stimulus_baseline_start=pretrial_duration_for_spiketimes,
                                                   pre_stimulus_baseline_end=pretrial_duration_for_spiketimes - 1,
                                                   pre_stimulus_raster=pretrial_duration_for_spiketimes,
                                                   post_stimulus_raster=posttrial_duration_for_spiketimes,
                                                   respLatency_filter=respLatency_filter,
                                                   shock_flag='All',
                                                   trial_type='Hit',
                                                   byAM_depth=False,
                                                   amdepth_subset=amdepth_subset,
                                                   psth_binsize=psth_binsize,
                                                   auroc_binsize=auroc_binsize,
                                                   from_JSON=True
                                                   )
                else:
                    continue

                response_curve = np.array(cur_data[cur_col])

            auroc_list.append(response_curve)
            unit_list.append(unit)
            # Option to smooth and Z-score firing rate if using non-normalized data
            # boxcar_points = 10
            # # response_curve = convolve_fft(response_curve, Box1DKernel(boxcar_points))
            # response_curve = resample(response_curve, len(response_curve) // boxcar_points)
            #
            # relevant_indices = np.arange((-PRETRIAL_DURATION_FOR_SPIKETIMES) /
            #                              BINSIZE, (-PRETRIAL_DURATION_FOR_SPIKETIMES + BASELINE_DURATION_FOR_FR) / BINSIZE,dtype=np.int32)
            # response_curve_baseline_mean = np.mean(response_curve[relevant_indices])
            # response_curve_baseline_sd = np.std(response_curve[relevant_indices], ddof=1)
            # if response_curve_baseline_sd == 0:
            #     continue
            # response_curve = (response_curve - response_curve_baseline_mean)/response_curve_baseline_sd

        auroc_list = np.array(auroc_list)

        # Rescale and remove nan if zscore
        # plot_list = [ (x - np.min(x)) / (np.max(x) - np.min(x)) for x in zscore_miss_list if ~np.isnan(np.sum(x))]
        # plot_list = [ x for x in zscore_miss_list if ~np.isnan(np.sum(x))]

        clustering_time_start = run_dict[cur_col][0]
        clustering_time_end = run_dict[cur_col][1]
        file_pre_name = cur_col + '_' + str(clustering_time_start) + 's-' + str(clustering_time_end) + 's'

        relevant_indices = np.arange((clustering_time_start + pretrial_duration_for_spiketimes) /
                                     binsize, (clustering_time_end + pretrial_duration_for_spiketimes) / binsize)

        relevant_snippet = np.array([cur_auroc[[int(idx) for idx in relevant_indices]] for cur_auroc in auroc_list])

        k, (gapStat_means, gapStat_sks) = mp_optimalK(relevant_snippet, output_folder=output_path,
                                                      number_of_cores=number_of_cores,
                                                      maxClusters=maxclusters, boot_n=boot_n,
                                                      sk_factor=sk_factor,
                                                      multiProcess=do_multiprocess, use_gpu=use_gpu,
                                                      use_tibshirani_criterion=use_tibshirani_criterion)
        gapstat_df = pd.DataFrame(
            {"Cluster_n": np.arange(1, maxclusters + 1), "Gap_mean": gapStat_means, "Gap_sks": gapStat_sks})
        # Save gap-stat data
        gapstat_df.to_csv(sep.join([output_path, file_pre_name + '_gapStat.csv']), index=False)

        print("Gap-statistic found %d clusters in data" % k)

        with PdfPages(sep.join([output_path, file_pre_name + '_gapStat.pdf'])) as pdf:
            fig, ax = plt.subplots()
            ax.errorbar(np.arange(1, maxclusters + 1), gapStat_means, gapStat_sks)
            ax.axvline(x=k, color='black', linestyle='--')
            ax.set_xlabel('Clusters')
            ax.set_ylabel('Gap-statistic')
            pdf.savefig()
            plt.close()

        # Final clustering
        row_link = load_timeSeriesKMeans(n_clusters=k)
        row_link.fit(relevant_snippet)

        clusters = row_link.labels_ + 1

        # Color cluster separation
        unique_clusters = np.unique(clusters.flatten())
        palette = sns.husl_palette(len(unique_clusters))

        color_dict = dict()
        for dummy_idx, unique_cluster in enumerate(unique_clusters):
            color_dict.update({unique_cluster: palette[dummy_idx]})

        cluster_df = pd.DataFrame({"Cluster_id": clusters})

        row_colors = cluster_df.Cluster_id.map(color_dict)
        cluster_df['Cluster_color'] = [row_color for row_color in row_colors.values]
        cluster_df = cluster_df.sort_values(by='Cluster_id')

        with PdfPages(sep.join([output_path, file_pre_name + '_HClustering.pdf'])) as pdf:
            plt.figure()

            # Reorder input using index of max(abs(auROC))
            sorted_plot_list = list()
            sorted_indices = list()
            for cluster_id in sorted(list(set(clusters))):
                cur_indices = cluster_df[cluster_df['Cluster_id'] == cluster_id].index.tolist()
                sorted_indices.extend(cur_indices)
                cur_resps = auroc_list[cur_indices]
                cur_abs = np.abs(relevant_snippet[cur_indices])
                idx_sort = np.argsort([np.argmax(x) for x in cur_abs])
                sorted_plot_list.extend(cur_resps[idx_sort])

            # abs_values = np.abs(relevant_snippet)
            # idx_max = [np.argmax(x) for x in abs_values]
            # idx_sort = np.lexsort((np.argsort(idx_max), cluster_df.index.tolist(),))
            # plot_list = plot_list[idx_sort]

            # Plot with seaborn
            g = sns.clustermap(sorted_plot_list, row_cluster=False, col_cluster=False,
                               row_colors=row_colors[sorted_indices].to_numpy())
            g.ax_heatmap.set_xticklabels([np.round(float(a.get_text()) * binsize - pretrial_duration_for_spiketimes, 1)
                                          for a in g.ax_heatmap.get_xticklabels()], size='xx-small')
            g.ax_heatmap.set_xlabel('Time relative to event (s)')
            g.ax_heatmap.set_ylabel('Unit # / clustering')
            g.ax_heatmap.set_title(cur_col)
            g.ax_heatmap.axvline(x=(clustering_time_start + pretrial_duration_for_spiketimes) / binsize,
                                 color='lightcyan', linestyle='--')
            g.ax_heatmap.axvline(x=(clustering_time_end + pretrial_duration_for_spiketimes) / binsize,
                                 color='lightcyan', linestyle='--')

            # Trim a bit of the ResponseAligned data because the offset to get the spout off event sometimes goes beyond
            # what I used to calculate the auROC so we end up with blank spaces at the end
            if 'ResponseAligned' in cur_col:
                g.ax_heatmap.set_xlim(
                    [0, (posttrial_duration_for_spiketimes + pretrial_duration_for_spiketimes - 1) / binsize])

            pdf.savefig()
            plt.close()

        # Plot average response by cluster group
        # Transform responses and cluster id into a dataframe first
        x_axis = np.arange(0, np.size(auroc_list, 1)) * binsize - pretrial_duration_for_spiketimes
        df = pd.DataFrame({'Unit': np.repeat(unit_list, np.size(auroc_list, 1)),
                           'Time_s': np.tile(x_axis, np.size(auroc_list, 0)),
                           'auROC': auroc_list.flatten(),
                           'Cluster': np.repeat(clusters, np.size(auroc_list, 1))})
        df = df.convert_dtypes()
        df = df.astype({'Cluster': 'category'})

        # Save data
        df.to_csv(sep.join([output_path, file_pre_name + '_HClustering.csv']), index=False)

        # Plot average+-95%CI responses per cluster
        with PdfPages(sep.join([output_path, file_pre_name + '_meanResponse.pdf'])) as pdf:
            # fig, ax = plt.subplots()
            g = sns.relplot(data=df, x="Time_s", y="auROC", hue="Cluster", kind='line', ax=ax,
                            palette=sns.husl_palette(len(unique_clusters)))  # 95% CI is the default
            g.ax.axvline(x=0, color='black',
                         linestyle='--')
            g.ax.fill_betweenx(y=[0, 1], x1=clustering_time_start, x2=clustering_time_end, facecolor='black', alpha=0.1)
            g.ax.set_xlabel('Time relative to event (s)')

            if 'auROC' in cur_col:
                g.ax.set_ylabel('auROC')
                g.ax.set_ylim([0, 1])
            elif 'psth' in cur_col:
                g.ax.set_ylabel('Spikes/s')

            # Trim a bit of the ResponseAligned data heatmap because the offset to get the spout off event sometimes goes beyond
            # what I used to calculate the auROC so we end up with blank spaces at the end
            if 'ResponseAligned' in cur_col:
                g.ax.set_xlim([-pretrial_duration_for_spiketimes, posttrial_duration_for_spiketimes - 1])

            sns.despine()
            pdf.savefig()
            plt.close()
        toc(t0, "Total runtime:")
        print("---------------------------------------------------------------------------------------------------")
