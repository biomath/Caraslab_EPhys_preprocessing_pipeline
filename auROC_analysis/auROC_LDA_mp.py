from os import remove
from os.path import sep
import platform
import json
from time import time
from glob import glob
from multiprocessing import Pool, cpu_count, current_process
from pathlib import Path
from re import split
from copy import deepcopy

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import patches
import scipy.stats as st
from tslearn.clustering import TimeSeriesKMeans
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.utils import shuffle
from sklearn import svm
import pandas as pd
from format_axes import format_ax
# import seaborn as sns

custom_params = {"axes.spines.right": False, "axes.spines.top": False}
# sns.set_theme(style="ticks", rc=custom_params)
from matplotlib.backends.backend_pdf import PdfPages

import warnings

warnings.filterwarnings("ignore")

# Tweak the regex file separator for cross-platform compatibility
if platform.system() == 'Windows':
    REGEX_SEP = sep * 2
else:
    REGEX_SEP = sep


class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def tic():
    return time()


def toc(t0, pre_message='Processing time:'):
    tkend = time() - t0
    print(pre_message + ' %d min, %.3f sec' % (tkend // 60, np.round(tkend % 60, 3)))


'''
Define globals
'''
'''
Pseudocode

for each subject
    for each day_of_training
        for each JSON file
            gather spikes for Reminder == 0 & TrialType == 0
            bin in 100 ms bins between 0 and 1 s
        for each bin
            for each trial
                LDA using all cells to predict Hit
'''

# IO locations
OUTPUT_FOLDER = '.' + sep + sep.join(['Data', 'Output'])
INPUT_FOLDER = '.' + sep + sep.join(['Data', 'Output', 'JSON files'])
# Load data and calculate auROCs based on trial responses aligned to SpoutOff triggering a hit
all_json = glob(INPUT_FOLDER + sep + '*json')

threshold_df = pd.read_csv(OUTPUT_FOLDER + sep + 'OFCPL_threshold_df.csv')

SUBJECTS_TO_RUN = ['SUBJ-ID-151', 'SUBJ-ID-154', 'SUBJ-ID-231', 'SUBJ-ID-389', 'SUBJ-ID-390']
# SUBJECTS_TO_RUN = None

SESSIONS_TO_EXCLUDE = ['SUBJ-ID-390_220627', 'SUBJ-ID-154_210520']

# UNITS_TO_RUN = np.loadtxt('.' + sep + sep.join([OUTPUT_FOLDER, 'SU_list_230201.csv']), delimiter=',', dtype=str)
UNITS_TO_RUN = pd.read_csv('.' + sep + sep.join([OUTPUT_FOLDER, 'allUnits_list_240219.csv']), delimiter=',', dtype=str)

# UNITS_TO_RUN = UNITS_TO_RUN[UNITS_TO_RUN['Cluster_quality'] == 'good']
UNITS_TO_RUN = UNITS_TO_RUN[UNITS_TO_RUN['ActiveTrial_modulation_direction'] == 'increase']

file_name_tag = 'hitVsMiss_tonicIncrease'

# Sessions you want to cluster; keep in mind passive sessions do not contain true trial outcomes or spout events and
# should only be clustered using all AMTrials or by AMdepth
which_session = ['active']  # pre, active, post, post1h


def run_FR_LDA(data_dict, SETTINGS_DICT):
    # This is for the non-overlapping histogram stuff. The overlapping histogram uses BINSIZE for window duration but
    # a fixed 0.0
    # BINSIZE = 0.08
    PRE_DURATION = 1
    POST_DURATION = 1
    # bin_cuts = np.arange(-PRE_DURATION, POST_DURATION, BINSIZE)

    WINDOW_SIZE = 0.1
    step_size = 0.02
    bin_cuts = np.arange(-PRE_DURATION, POST_DURATION, step_size)

    LDA_outcome1 = SETTINGS_DICT['LDA_OUTCOME1']
    LDA_outcome2 = SETTINGS_DICT['LDA_OUTCOME2']

    cur_data = pd.DataFrame.from_dict({key: data_dict[key] for key in ['Reminder', 'Hit',
                                                                      'Miss', 'FA', 'ShockFlag',
                                                                      'AMdepth', 'Trial_spikes',
                                                                      'SpoutOff_times_during_trial']})
    # subset relevant trials from Trial_spikes
    subset_data = cur_data[(cur_data['Reminder'] == 0) & ((cur_data[LDA_outcome1] == 1) | (cur_data[LDA_outcome2] == 1))]

            # subset_data = cur_data[(cur_data['Reminder'] == 0) &
            #                        (((cur_data['Hit'] == 1) & (cur_data['ShockFlag'] == 1)) | (cur_data['FA'] == 1))]

            # Find spout offset that triggered Hits/FAs and end of shock artifact (~1.25 s)
            # Zero-center spikes around those events
            # Also skip neurons that don't fire for more than 50% of the trials
            invalid_trial_counter = 0

            shock_artifact_start_end = np.float64([0.95, 1.25])
            # shock_artifact_start_end = np.float64([-10, -10])  #  outside range (do not remove anything)

            for row_idx, df_row in subset_data.iterrows():
                subset_data.at[row_idx, 'Trial_spikes'] = np.array(subset_data.at[row_idx, 'Trial_spikes'])
                if len(df_row['Trial_spikes']) == 0:
                    invalid_trial_counter += 1
                    continue
                # Eliminate shock artifact period (even from hit trials for fair comparison)
                # to_remove_mask = (subset_data.at[row_idx, 'Trial_spikes'] >= shock_artifact_start_end[0]) & \
                #                  (subset_data.at[row_idx, 'Trial_spikes'] <= shock_artifact_start_end[1])
                # subset_data.at[row_idx, 'Trial_spikes'] = subset_data.at[row_idx, 'Trial_spikes'][
                #     np.where(to_remove_mask == False)[0]]
                #
                if (df_row['Hit'] == 1) | (df_row['FA'] == 1):
                    spoutOffset_trigger = np.array(df_row['SpoutOff_times_during_trial'])
                    try:
                        spoutOffset_trigger = \
                            spoutOffset_trigger[(spoutOffset_trigger > 0) & (spoutOffset_trigger < 1)][-1]
                    except IndexError:
                        continue

                    # Zero-center
                    subset_data.at[row_idx, 'Trial_spikes'] -= spoutOffset_trigger
                #
                # elif df_row['Miss'] == 1:
                #     # Zero-center using the end of shock artifact if Miss
                #     shock_trigger = shock_artifact_start_end[1]  # pandas doesn't like regular floats for this...
                #     subset_data.at[row_idx, 'Trial_spikes'] -= shock_trigger

            # if invalid_trial_counter / len(subset_data) > 0.5:
            #     print('Unit: %s was discarded due to 0 firing in %d%% of trials' %
            #           (cur_dict['Unit'], np.round(100*invalid_trial_counter / len(subset_data))))
            #     continue

            # Non-overlapping bins
            # subset_data['FR_binned'] = [np.histogram(trial_spikes, bins=bin_cuts)[0] for
            #                             trial_spikes in subset_data['Trial_spikes']]

            # Overlapping (sliding) bins
            def sliding_hist(trial_spikes, bin_cuts, window_size):
                ret_hist = np.zeros(len(bin_cuts))
                for step_counter, step_start in enumerate(bin_cuts):
                    ret_hist[step_counter] += np.sum((trial_spikes >= step_start) &
                                                     (trial_spikes < (step_start + window_size)))
                return ret_hist

            subset_data['FR_binned'] = [sliding_hist(trial_spikes, bin_cuts=bin_cuts, window_size=WINDOW_SIZE) for
                                        trial_spikes in subset_data['Trial_spikes']]

            all_trial_spikes.append(subset_data['FR_binned'].values)

            outcomes.append(subset_data['Hit'].values)

        raw_input_dict.update({subject_date: all_trial_spikes})

        # try:
        class_dict.update({subject_date: outcomes[0]})
        # except IndexError:
        #     print()

        session_accuracy = np.zeros(len(bin_cuts[:-1]))
        session_accuracy_shuf = np.zeros(len(bin_cuts[:-1]))
        for bin_idx, bin_value in enumerate(bin_cuts[:-1]):
            cur_bin_spikes = np.empty(np.shape(np.transpose(all_trial_spikes)))
            for unit_idx, unit_spikes in enumerate(all_trial_spikes):
                for trial_idx, trial_spikes in enumerate(unit_spikes):
                    cur_bin_spikes[trial_idx, unit_idx] = trial_spikes[bin_idx]

            '''
            From Basu et al., 2021
            We applied a decoder based on LDA that assigns individual class prob- abilities by 
            setting class boundaries between multivariate Gaussian distributions fitted to data. 
            In brief, a dataset from each recording ses- sion was divided into a training dataset 
            and a test dataset, and a decoder was constructed from the training dataset by employing 
            multiclass one-versus-one LDA using the ‘fitcecoc’ function of MATLAB with a regu- larization 
            factor of 0.5 to reduce overfitting. We used uniform priors for all decoders. Next, we used the 
            ‘predict’ function of MATLAB to obtain decoding probabilities of individual wells from the test 
            dataset. This function uses an algorithm described by Hastie and Tibshirani to compute 
            posterior probabilities from the pairwise conditional probabilities obtained using multiclass 
            one-versus-one decoders.
            '''
            model_y = outcomes[0]
            model_y_shuf = shuffle(model_y, random_state=0)
            try:
                X_train, X_test, y_train, y_test = train_test_split(cur_bin_spikes, model_y,
                                                                    test_size=0.2,
                                                                    random_state=0, shuffle=True)

                X_train_shuf, X_test_shuf, y_train_shuf, y_test_shuf = train_test_split(cur_bin_spikes, model_y_shuf,
                                                                                        test_size=0.2,
                                                                                        random_state=0, shuffle=True)

            except ValueError:
                continue

            # sc = StandardScaler()
            # sc = MinMaxScaler()
            sc = RobustScaler()
            X_train = sc.fit_transform(X_train)  # sc object learns mean and var from training data
            X_test = sc.transform(X_test)  # sc object applies mean and var from training to test data

            # sc_shuf = StandardScaler()
            # sc_shuf = MinMaxScaler()
            sc_shuf = RobustScaler()
            X_train_shuf = sc_shuf.fit_transform(X_train_shuf)  # sc object learns mean and var from training data
            X_test_shuf = sc_shuf.transform(X_test_shuf)  # sc object applies mean and var from training to test data

            lda = LinearDiscriminantAnalysis()
            lda_shuf = LinearDiscriminantAnalysis()
            try:
                X_train = lda.fit_transform(X_train, y_train)
                X_train_shuf = lda_shuf.fit_transform(X_train_shuf, y_train_shuf)
            except ValueError:
                continue

            X_test = lda.transform(X_test)
            X_test_shuf = lda_shuf.transform(X_test_shuf)

            classifier = svm.SVC(kernel='linear', C=.5)
            classifier_shuf = svm.SVC(kernel='linear', C=.5)
            # classifier = svm.LinearSVC(C=.5, max_iter=10000)
            try:
                classifier.fit(X_train, y_train)
                classifier_shuf.fit(X_train_shuf, y_train_shuf)
            except ValueError:
                continue
            y_pred = classifier.predict(X_test)
            y_pred_shuf = classifier_shuf.predict(X_test_shuf)

            session_accuracy[bin_idx] = accuracy_score(y_test, y_pred)
            session_accuracy_shuf[bin_idx] = accuracy_score(y_test_shuf, y_pred_shuf)

        # chance_value = (sum(outcomes[0] == 1) / len(outcomes[0])) ** 2 + (sum(outcomes[0] == 0) / len(outcomes[0])) ** 2
        # chance_value = sum(outcomes[0] == 1) / len(outcomes[0])  # Chance value is the proportion of outcomes == 1

        # all_chance_values.append(chance_value*100)

        if subject_date in worst_day_sessions:
            first_days_accuracies.append(session_accuracy*100)
            first_days_chance_values.append(session_accuracy_shuf*100)

        elif subject_date in best_days_sessions:
            best_days_accuracies.append(session_accuracy*100)
            best_days_chance_values.append(session_accuracy_shuf*100)

        all_sessions_accuracies.append(session_accuracy*100)
        all_chance_values.append(session_accuracy_shuf * 100)

    plot_ci = False
    with PdfPages(sep.join([OUTPUT_FOLDER, 'LDAoutput_' + file_name_tag + '.pdf'])) as pdf:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        legend_handles = list()

        # Plot all data
        # Plot chance line
        temp_color = '#D4B483'
        temp_sessions_mean = np.nanmean(all_chance_values, axis=0)
        # temp_sessions_se = np.nanstd(all_chance_values, axis=0) / np.sqrt(np.shape(all_chance_values)[0])
        temp_sessions_se = np.nanstd(all_chance_values, axis=0) / np.sqrt(np.count_nonzero(~np.isnan(all_chance_values), axis=0))

        if not plot_ci:
            # Plot standard errors
            ax.fill_between(bin_cuts[:-1], temp_sessions_mean - temp_sessions_se, temp_sessions_mean + temp_sessions_se,
                             alpha=0.5,
                             color=temp_color)
        else:
            # Or plot 95% CI
            temp_sessions_se = st.t.interval(confidence=0.95, df=np.shape(first_days_accuracies)[0] - 1,
                                             loc=temp_sessions_mean, scale=temp_sessions_se)
            ax.fill_between(bin_cuts[:-1], temp_sessions_se[0], temp_sessions_se[1], alpha=0.5,
                             color=temp_color)

        # ax.plot(bin_cuts[:-1], np.repeat(temp_sessions_mean, len(bin_cuts[:-1])), color=temp_color, linestyle='--')
        ax.plot(bin_cuts[:-1], temp_sessions_mean, color=temp_color, linestyle='--')

        # Then data
        temp_color = '#D4B483'
        temp_label = 'All training'
        temp_sessions_mean = np.nanmean(all_sessions_accuracies, axis=0)
        temp_sessions_se = np.nanstd(all_sessions_accuracies, axis=0) / np.sqrt(np.count_nonzero(~np.isnan(all_sessions_accuracies), axis=0))
        if not plot_ci:
            # Plot standard errors
            ax.fill_between(bin_cuts[:-1], temp_sessions_mean - temp_sessions_se, temp_sessions_mean + temp_sessions_se,
                             alpha=0.5,
                             color=temp_color)
        else:
            # Or plot 95% CI
            temp_sessions_se = st.t.interval(confidence=0.95, df=np.shape(first_days_accuracies)[0] - 1,
                                             loc=temp_sessions_mean, scale=temp_sessions_se)
            plt.fill_between(bin_cuts[:-1], temp_sessions_se[0], temp_sessions_se[1], alpha=0.5,
                             color=temp_color)

        ax.plot(bin_cuts[:-1], temp_sessions_mean, color=temp_color)

        ax.set_ylabel('Decoding accuracy for Hits')
        ax.set_xlabel('Time (s)')

        format_ax(ax)
        ax.set_ylim([40, 80])
        ax.set_xlim([-PRE_DURATION, POST_DURATION])

        pdf.savefig()

        ##### EARLY VS LATE TRAINING #####
        # Plot early training data
        fig = plt.figure()
        ax = fig.add_subplot(111)
        legend_handles = list()

        temp_color = '#C1666B'
        temp_label = 'Early training'
        # Chance line
        temp_sessions_mean = np.nanmean(first_days_chance_values, axis=0)
        # temp_sessions_se = np.nanstd(all_chance_values, axis=0) / np.sqrt(np.shape(all_chance_values)[0])
        temp_sessions_se = np.nanstd(first_days_chance_values, axis=0) / np.sqrt(np.count_nonzero(~np.isnan(first_days_chance_values), axis=0))

        if not plot_ci:
            # Plot standard errors
            ax.fill_between(bin_cuts[:-1], temp_sessions_mean - temp_sessions_se, temp_sessions_mean + temp_sessions_se,
                             alpha=0.5,
                             color=temp_color)
        else:
            # Or plot 95% CI
            temp_sessions_se = st.t.interval(confidence=0.95, df=np.shape(first_days_accuracies)[0] - 1,
                                             loc=temp_sessions_mean, scale=temp_sessions_se)
            plt.fill_between(bin_cuts[:-1], temp_sessions_se[0], temp_sessions_se[1], alpha=0.5,
                             color=temp_color)

        # try:
        # ax.plot(bin_cuts[:-1], np.repeat(temp_sessions_mean, len(bin_cuts[:-1])), color=temp_color, linestyle='--')
        ax.plot(bin_cuts[:-1], temp_sessions_mean, color=temp_color, linestyle='--')

        # except ValueError:
        #     print()
        # Then data
        temp_sessions_mean = np.nanmean(first_days_accuracies, axis=0)
        temp_sessions_se = np.nanstd(first_days_accuracies, axis=0) / np.sqrt(np.count_nonzero(~np.isnan(first_days_accuracies), axis=0))

        if not plot_ci:
            # Plot standard errors
            ax.fill_between(bin_cuts[:-1], temp_sessions_mean - temp_sessions_se, temp_sessions_mean + temp_sessions_se,
                             alpha=0.5,
                             color=temp_color)
        else:
            # Or plot 95% CI
            temp_sessions_se = st.t.interval(confidence=0.95, df=np.shape(first_days_accuracies)[0] - 1,
                                             loc=temp_sessions_mean, scale=temp_sessions_se)
            plt.fill_between(bin_cuts[:-1], temp_sessions_se[0], temp_sessions_se[1], alpha=0.5,
                             color=temp_color)

        ax.plot(bin_cuts[:-1], temp_sessions_mean, color=temp_color)
        legend_handles.append(patches.Patch(facecolor=temp_color, edgecolor=None, alpha=0.5,
                                            label=temp_label))

        # Plot late training data
        temp_color = '#4281A4'
        temp_label = 'Late training'

        # Chance line
        temp_sessions_mean = np.nanmean(best_days_chance_values, axis=0)
        temp_sessions_se = np.nanstd(best_days_chance_values, axis=0) / np.sqrt(
            np.count_nonzero(~np.isnan(best_days_chance_values), axis=0))

        if not plot_ci:
            # Plot standard errors
            ax.fill_between(bin_cuts[:-1], temp_sessions_mean - temp_sessions_se, temp_sessions_mean + temp_sessions_se,
                             alpha=0.5,
                             color=temp_color)
        else:
            # Or plot 95% CI
            temp_sessions_se = st.t.interval(confidence=0.95, df=np.shape(first_days_accuracies)[0] - 1,
                                             loc=temp_sessions_mean, scale=temp_sessions_se)
            plt.fill_between(bin_cuts[:-1], temp_sessions_se[0], temp_sessions_se[1], alpha=0.5,
                             color=temp_color)

        # ax.plot(bin_cuts[:-1], np.repeat(temp_sessions_mean, len(bin_cuts[:-1])), color=temp_color, linestyle='--')
        ax.plot(bin_cuts[:-1], temp_sessions_mean, color=temp_color, linestyle='--')

        # Then data
        temp_sessions_mean = np.nanmean(best_days_accuracies, axis=0)
        temp_sessions_se = np.nanstd(best_days_accuracies, axis=0) / np.sqrt(np.count_nonzero(~np.isnan(best_days_accuracies), axis=0))

        if not plot_ci:
            # Plot standard errors
            ax.fill_between(bin_cuts[:-1], temp_sessions_mean - temp_sessions_se, temp_sessions_mean + temp_sessions_se,
                             alpha=0.5,
                             color=temp_color)
        else:
            # Or plot 95% CI
            temp_sessions_se = st.t.interval(alpha=0.95, df=np.shape(first_days_accuracies)[0] - 1,
                                             loc=temp_sessions_mean, scale=temp_sessions_se)
            plt.fill_between(bin_cuts[:-1], temp_sessions_se[0], temp_sessions_se[1], alpha=0.5,
                             color=temp_color)

        ax.plot(bin_cuts[:-1], temp_sessions_mean, color=temp_color)
        legend_handles.append(patches.Patch(facecolor=temp_color, edgecolor=None, alpha=0.5,
                                            label=temp_label))

        labels = [h.get_label() for h in legend_handles]

        fig.legend(handles=legend_handles, labels=labels, frameon=False, numpoints=1)

        ax.set_ylabel('Decoding accuracy for Hits')
        ax.set_xlabel('Time (s)')

        format_ax(ax)
        ax.set_ylim([40, 100])
        ax.set_xlim([-PRE_DURATION, POST_DURATION])

        pdf.savefig()


        ##### DECODING ACCURACY ABOVE CHANCE #####
        fig = plt.figure()
        ax = fig.add_subplot(111)
        legend_handles = list()

        temp_color = '#D4B483'

        temp_values = [(x - chance) / chance * 100 for x, chance in
                       zip(all_sessions_accuracies, all_chance_values)]

        temp_sessions_mean = np.nanmean(temp_values, axis=0)
        temp_sessions_se = np.nanstd(temp_values, axis=0) / np.sqrt(np.count_nonzero(~np.isnan(temp_values), axis=0))
        if not plot_ci:
            # Plot standard errors
            ax.fill_between(bin_cuts[:-1], temp_sessions_mean - temp_sessions_se, temp_sessions_mean + temp_sessions_se,
                             alpha=0.5,
                             color=temp_color)
        else:
            # Or plot 95% CI
            temp_sessions_se = st.t.interval(confidence=0.95, df=np.shape(all_sessions_accuracies)[0] - 1,
                                             loc=temp_sessions_mean, scale=temp_sessions_se)
            plt.fill_between(bin_cuts[:-1], temp_sessions_se[0], temp_sessions_se[1], alpha=0.5,
                             color=temp_color)

        ax.plot(bin_cuts[:-1], temp_sessions_mean, color=temp_color)

        ax.set_ylabel('Decoding accuracy for Hits (% above chance)')
        ax.set_xlabel('Time (s)')

        format_ax(ax)
        ax.set_ylim([0, 40])
        ax.set_xlim([-PRE_DURATION, POST_DURATION])

        pdf.savefig()
        plt.close()

        ##### early vs late #####
        fig = plt.figure()
        ax = fig.add_subplot(111)
        legend_handles = list()

        # Plot early training data
        temp_color = '#C1666B'
        temp_label = 'Early training'
        temp_values = [(x - chance) / chance * 100 for x, chance in
                       zip(first_days_accuracies, first_days_chance_values)]
        temp_sessions_mean = np.nanmean(temp_values, axis=0)
        temp_sessions_se = np.nanstd(temp_values, axis=0) / np.sqrt(np.count_nonzero(~np.isnan(temp_values), axis=0))
        if not plot_ci:
            # Plot standard errors
            ax.fill_between(bin_cuts[:-1], temp_sessions_mean - temp_sessions_se, temp_sessions_mean + temp_sessions_se,
                             alpha=0.5,
                             color=temp_color)
        else:
            # Or plot 95% CI
            temp_sessions_se = st.t.interval(confidence=0.95, df=np.shape(first_days_accuracies)[0] - 1,
                                             loc=temp_sessions_mean, scale=temp_sessions_se)
            plt.fill_between(bin_cuts[:-1], temp_sessions_se[0], temp_sessions_se[1], alpha=0.5,
                             color=temp_color)

        ax.plot(bin_cuts[:-1], temp_sessions_mean, color=temp_color)
        legend_handles.append(patches.Patch(facecolor=temp_color, edgecolor=None, alpha=0.5,
                                            label=temp_label))

        # Plot late training data
        temp_color = '#4281A4'
        temp_label = 'Late training'
        temp_values = [(x - chance) / chance * 100 for x, chance in zip(best_days_accuracies, best_days_chance_values)]
        temp_sessions_mean = np.nanmean(temp_values, axis=0)
        temp_sessions_se = np.nanstd(temp_values, axis=0) / np.sqrt(np.count_nonzero(~np.isnan(temp_values), axis=0))

        if not plot_ci:
            # Plot standard errors
            ax.fill_between(bin_cuts[:-1], temp_sessions_mean - temp_sessions_se, temp_sessions_mean + temp_sessions_se,
                             alpha=0.5,
                             color=temp_color)
        else:
            # Or plot 95% CI
            temp_sessions_se = st.t.interval(confidence=0.95, df=np.shape(first_days_accuracies)[0] - 1,
                                             loc=temp_sessions_mean, scale=temp_sessions_se)
            plt.fill_between(bin_cuts[:-1], temp_sessions_se[0], temp_sessions_se[1], alpha=0.5,
                             color=temp_color)

        ax.plot(bin_cuts[:-1], temp_sessions_mean, color=temp_color)
        legend_handles.append(patches.Patch(facecolor=temp_color, edgecolor=None, alpha=0.5,
                                            label=temp_label))

        labels = [h.get_label() for h in legend_handles]

        fig.legend(handles=legend_handles, labels=labels, frameon=False, numpoints=1)

        ax.set_ylabel('Decoding accuracy for Hits (% above chance)')
        ax.set_xlabel('Time from trigger (spout off or shock)')

        format_ax(ax)
        ax.set_ylim([0, 60])
        ax.set_xlim([-PRE_DURATION, POST_DURATION])

        pdf.savefig()
        plt.close()

    # # # Save raw input and classes as json files
    # json_filename = 'Macae_input_data.json'
    # with open(sep.join([OUTPUT_FOLDER, json_filename]), 'w') as cur_json:
    #     cur_json.write(json.dumps(raw_input_dict, cls=NumpyEncoder, indent=4))
    #
    # json_filename = 'Macae_input_class.json'
    # with open(sep.join([OUTPUT_FOLDER, json_filename]), 'w') as cur_json:
    #     cur_json.write(json.dumps(class_dict, cls=NumpyEncoder, indent=4))
