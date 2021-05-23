import pandas as pd
from sklearn.utils import shuffle
import numpy as np
from dir_path import benchmark_dir
# from testing_parameters import pulsar_seed, non_pulsar_seed

# benchmark data set
PALFA_pulsars_120 = benchmark_dir + "/PALFA120Pulsars_20181020.txt"
PALFA_non_pulsars_1200 = benchmark_dir + "/PALFA1200NonPulsars.txt"
PALFA_benchmark_data_full = benchmark_dir + "/PALFA_all_pulsars_non_pulsars.txt"


def get_pulsar_beams(survey='GBT'):
    if survey == 'GBT':
        pulsar_beams = GBT_pulsars_120
    elif survey == 'PALFA':
        pulsar_beams = PALFA_pulsars_120
    else:
        exit("Survey name not valid!")

    # pulsars in benchmark data set
    pulsars = pd.read_csv(pulsar_beams, header=None, names=['beam'])

    pulsars_list = pulsars['beam'].tolist()
    # print(pulsars_list)
    n_pulsar_total = len(pulsars_list)
    print("Survey ", survey, ", total pulsars in benchmark: ", n_pulsar_total)
    return pulsars_list


def get_survey_beams(survey='GBT', random_seed=None):
    if survey == 'GBT':
        pulsar_beams = GBT_pulsars_120
        non_pulsar_beams = GBT_non_pulsars_1200
    elif survey == 'PALFA':
        pulsar_beams = PALFA_pulsars_120
        non_pulsar_beams = PALFA_non_pulsars_1200
    else:
        exit("Survey name not valid!")

    # pulsars in GBT benchmark data set
    pulsars = pd.read_csv(pulsar_beams, header=None, names=['beam'])
    pulsars_shuffled = shuffle(pulsars, random_state=random_seed)

    # non-pulsars in GBT benchmark data set
    non_pulsars = pd.read_csv(non_pulsar_beams, header=None, names=['beam'])
    non_pulsars_shuffled = shuffle(non_pulsars, random_state=random_seed + 123)

    n_pulsar_total = pulsars.shape[0]
    print("Survey ", survey, ", total pulsars in benchmark: ", n_pulsar_total)
    n_non_pulsar_total = non_pulsars.shape[0]
    print("Survey ", survey, "total non-pulsars in benchmark: ", n_non_pulsar_total)

    return pulsars_shuffled, non_pulsars_shuffled


def get_folds(survey='GBT', n_folds=6, rdn_seed=None):
    #  get beams
    pulsars_shuffled, non_pulsars_shuffled = get_survey_beams(survey=survey, random_seed=rdn_seed)
    all_folds = []
    # number of pulsars in each fold
    each_fold_pulsars = 120 // n_folds
    each_fold_non_pulsars = 1200 // n_folds

    for i in range(n_folds):
        pulsar_start_idx = each_fold_pulsars * i
        pulsar_stop_idx = each_fold_pulsars * i + each_fold_pulsars
        fold_pulsars = pulsars_shuffled[pulsar_start_idx:pulsar_stop_idx]

        non_pulsar_start_idx = each_fold_non_pulsars * i
        non_pulsar_stop_idx = each_fold_non_pulsars * i + each_fold_non_pulsars
        fold_non_pulsars = non_pulsars_shuffled[non_pulsar_start_idx:non_pulsar_stop_idx]

        fold = pd.concat([fold_pulsars, fold_non_pulsars])
        all_folds.append(fold)
    return all_folds


def get_spegs(survey='GBT', speg_type='all'):
    if survey == 'GBT':
        benchmark_data_full = GBT_benchmark_data_full
    elif survey == 'PALFA':
        benchmark_data_full = PALFA_benchmark_data_full
    else:
        exit("Survey name not valid!")

    speg_df = pd.read_csv(benchmark_data_full, sep=",", skipinitialspace=True)
    # only keep clean labeled data
    speg_df_clean = speg_df.loc[(speg_df['label'] == "YES") | (speg_df['label'] == "NO")]

    print("Survey ", survey, "total labeled SPEGs: ")
    print(speg_df_clean.shape)

    # MAKE A COPY
    df = speg_df_clean.copy()
    df['class_label'] = df['label'] == "YES"
    # print(df_GBT.shape)
    df = df * 1

    # calculate features
    # size ratio
    df['size_ratio'] = df['size'] * 1.0 / df['sizeU']
    # cluster_density
    df['cluster_density'] = df['cluster_number'] * 1.0 / (df['obs_length'] * df['DM_channel_number'])
    # DM_range
    df['DM_range'] = df['max_DM'] - df['min_DM']
    # time_range
    df['time_range'] = df['max_time'] - df['min_time']
    # pulse_width
    df['pulse_width'] = df['peak_time'] / df['peak_sampling'] * df['peak_downfact']
    # time_ratio
    df['time_ratio'] = df['time_range'] / df['pulse_width']

    df_learning = df[['filename', 'SPEG_rank', 'group_rank', 'group_max_SNR', 'group_median_SNR',
                      'peak_SNR', 'centered_DM', 'clipped_SPEG', 'SNR_sym_index', 'DM_sym_index',
                      'peak_score', 'bright_recur_times', 'recur_times', 'size_ratio', 'cluster_density',
                      'DM_range', 'time_range', 'pulse_width', 'time_ratio', 'n_SPEGs_zero_DM',
                      'n_brighter_SPEGs_zero_DM', 'SPEGs_zero_DM_min_DM', 'brighter_SPEGs_zero_DM_peak_DM',
                      'brighter_SPEGs_zero_DM_peak_SNR', 'class_label']]

    df_learning_with_central_peak = df[['filename', 'SPEG_rank', 'group_rank', 'group_max_SNR', 'group_median_SNR',
                                        'peak_SNR', 'centered_DM', 'clipped_SPEG', 'SNR_sym_index', 'DM_sym_index',
                                        'peak_score', 'bright_recur_times', 'recur_times', 'size_ratio', 'cluster_density',
                                        'DM_range', 'time_range', 'pulse_width', 'time_ratio', 'n_SPEGs_zero_DM',
                                        'n_brighter_SPEGs_zero_DM', 'SPEGs_zero_DM_min_DM', 'brighter_SPEGs_zero_DM_peak_DM',
                                        'brighter_SPEGs_zero_DM_peak_SNR', 'class_label',
                                        'center_startDM', 'center_stopDM', 'min_DM', 'max_DM',
                                        'peak_time', 'min_time', 'max_time']]

    df_learning_no_grouping = df[['filename', 'SPEG_rank', 'peak_SNR', 'centered_DM', 'clipped_SPEG',
                                  'SNR_sym_index', 'DM_sym_index', 'peak_score', 'size_ratio', 'cluster_density',
                                  'DM_range', 'time_range', 'pulse_width', 'time_ratio', 'n_SPEGs_zero_DM',
                                  'n_brighter_SPEGs_zero_DM', 'SPEGs_zero_DM_min_DM', 'brighter_SPEGs_zero_DM_peak_DM',
                                  'brighter_SPEGs_zero_DM_peak_SNR', 'class_label']]

    df_learning_dim = df_learning[df_learning['peak_SNR'] < 8]

    df_learning_no_grouping_bright = df_learning_no_grouping[df_learning_no_grouping['peak_SNR'] > 7.99]

    if speg_type == 'all':
        return df_learning
    elif speg_type == 'all_with_central_peak':
        return df_learning_with_central_peak
    elif speg_type == 'bright':
        return df_learning_no_grouping_bright
    elif speg_type == 'dim':
        return df_learning_dim
    else:
        exit("speg_type invalid!")


# helper function to evaluate the performance by SPEG
def get_speg_metrics(y_test, y_prediction):
    y_test = y_test.tolist()
    test_result = {'label': y_test, 'prediction': y_prediction}
    df = pd.DataFrame(test_result)

    # true positive by SPEG
    tp_SPEG = df.loc[(df['prediction'] == 1) & (df['label'] == 1)]
    fn_SPEG = df.loc[(df['prediction'] == 0) & (df['label'] == 1)]
    fp_SPEG = df.loc[(df['prediction'] == 1) & (df['label'] == 0)]
    tn_SPEG = df.loc[(df['prediction'] == 0) & (df['label'] == 0)]

    n_tp_SPEG = tp_SPEG.shape[0]
    n_fn_SPEG = fn_SPEG.shape[0]
    n_fp_SPEG = fp_SPEG.shape[0]
    n_tn_SPEG = tn_SPEG.shape[0]


    return n_tp_SPEG, n_fn_SPEG, n_fp_SPEG, n_tn_SPEG


# helper function to evaluate the combined performance by beam
def get_beam_combined_metrics(df_test_bright, y_test_bright, y_bright_prediction,
                              df_test_dim, y_test_dim, y_dim_prediction):

    pulsar_id_test_bright = df_test_bright['filename'].tolist()
    y_test_bright = y_test_bright.tolist()

    test_bright_result = {'pulsar': pulsar_id_test_bright, 'label': y_test_bright, 'prediction': y_bright_prediction}
    df_bright = pd.DataFrame(test_bright_result)

    pulsar_id_test_dim = df_test_dim['filename'].tolist()
    y_test_dim = y_test_dim.tolist()

    test_dim_result = {'pulsar': pulsar_id_test_dim, 'label': y_test_dim, 'prediction': y_dim_prediction}
    df_dim = pd.DataFrame(test_dim_result)
    # print(df_valid)

    df = pd.concat([df_bright, df_dim])
    # combine data frames

    # brighttrue positive by SPEG
    tp_SPEG = df.loc[(df['prediction'] == 1) & (df['label'] == 1)]
    fn_SPEG = df.loc[(df['prediction'] == 0) & (df['label'] == 1)]
    fp_SPEG = df.loc[(df['prediction'] == 1) & (df['label'] == 0)]
    # tn_SPEG = df.loc[(df['prediction'] == 0) & (df['label'] == 0)]

    # true positive by pulsar
    tp_pulsar = set(tp_SPEG['pulsar'].unique())
    n_tp_pulsar = len(tp_pulsar)
    print("TP pulsars: ", n_tp_pulsar)

    # false negative by pulsar
    fn_pulsar = set(fn_SPEG['pulsar'].unique())
    real_fn_pulsar = fn_pulsar.difference(tp_pulsar)
    missed_pulsars = '#'.join(real_fn_pulsar)
    n_real_fn_pulsar = len(real_fn_pulsar)
    print("FN pulsars: ", n_real_fn_pulsar)

    # false positive by pulsar
    fp_pulsar = set(fp_SPEG['pulsar'].unique())
    real_fp_pulsar = fp_pulsar.difference(tp_pulsar)
    n_real_fp_pulsar = len(real_fp_pulsar)
    print("FP pulars: ", n_real_fp_pulsar)

    n_tn_pulsar = len(set(df['pulsar'].unique())) - n_tp_pulsar - \
                         n_real_fn_pulsar - n_real_fp_pulsar
    print("TN pulsars: ", n_tn_pulsar)

    return n_tp_pulsar, n_real_fn_pulsar, n_real_fp_pulsar, n_tn_pulsar, missed_pulsars


# helper function to evaluate the performance by beam
def get_beam_metrics(df_test, y_test, y_prediction):
    pulsar_id_test = df_test['filename'].tolist()
    y_test = y_test.tolist()

    test_result = {'pulsar': pulsar_id_test, 'label': y_test, 'prediction': y_prediction}
    df = pd.DataFrame(test_result)
    # print(df_valid)

    # true positive by SPEG
    tp_SPEG = df.loc[(df['prediction'] == 1) & (df['label'] == 1)]
    fn_SPEG = df.loc[(df['prediction'] == 0) & (df['label'] == 1)]
    fp_SPEG = df.loc[(df['prediction'] == 1) & (df['label'] == 0)]
    # tn_SPEG = df.loc[(df['prediction'] == 0) & (df['label'] == 0)]

    # true positive by pulsar
    tp_pulsar = set(tp_SPEG['pulsar'].unique())
    n_tp_pulsar = len(tp_pulsar)
    print("TP pulsars: ", n_tp_pulsar)

    # false negative by pulsar
    fn_pulsar = set(fn_SPEG['pulsar'].unique())
    real_fn_pulsar = fn_pulsar.difference(tp_pulsar)
    missed_pulsars = '#'.join(real_fn_pulsar)
    n_real_fn_pulsar = len(real_fn_pulsar)
    print("FN pulsars: ", n_real_fn_pulsar)

    # false positive by pulsar
    fp_pulsar = set(fp_SPEG['pulsar'].unique())
    real_fp_pulsar = fp_pulsar.difference(tp_pulsar)
    n_real_fp_pulsar = len(real_fp_pulsar)
    print("FP pulars: ", n_real_fp_pulsar)

    n_tn_pulsar = len(set(df['pulsar'].unique())) - n_tp_pulsar - n_real_fn_pulsar - n_real_fp_pulsar
    print("TN pulsars: ", n_tn_pulsar)

    return n_tp_pulsar, n_real_fn_pulsar, n_real_fp_pulsar, n_tn_pulsar, missed_pulsars


def shuffle_batch(X, y, batch_size):
    rnd_idx = np.random.permutation(len(X))
    n_batches = len(X) // batch_size
    for batch_idx in np.array_split(rnd_idx, n_batches):
        X_batch, y_batch = X[batch_idx], y[batch_idx]
        yield X_batch, y_batch


def get_unlabeld_spegs(survey, SPEG_file):
    speg_df = pd.read_csv(SPEG_file, sep=",", skipinitialspace=True)

    print("Survey ", survey, "total SPEGs to be classified: ")
    print(speg_df.shape)

    # MAKE A COPY
    df = speg_df.copy()
    # df['class_label'] = 0
    # print(df_GBT.shape)
    df = df * 1

    # calculate features
    # size ratio
    df['size_ratio'] = df['size'] * 1.0 / df['sizeU']
    # cluster_density
    df['cluster_density'] = df['cluster_number'] * 1.0 / (df['obs_length'] * df['DM_channel_number'])
    # DM_range
    df['DM_range'] = df['max_DM'] - df['min_DM']
    # time_range
    df['time_range'] = df['max_time'] - df['min_time']
    # pulse_width
    df['pulse_width'] = df['peak_time'] / df['peak_sampling'] * df['peak_downfact']
    # time_ratio
    df['time_ratio'] = df['time_range'] / df['pulse_width']

    df_to_be_learned = df[['filename', 'SPEG_rank', 'group_rank', 'group_max_SNR', 'group_median_SNR',
                           'peak_SNR', 'centered_DM', 'clipped_SPEG', 'SNR_sym_index', 'DM_sym_index',
                           'peak_score', 'bright_recur_times', 'recur_times', 'size_ratio', 'cluster_density',
                           'DM_range', 'time_range', 'pulse_width', 'time_ratio', 'n_SPEGs_zero_DM',
                           'n_brighter_SPEGs_zero_DM', 'SPEGs_zero_DM_min_DM', 'brighter_SPEGs_zero_DM_peak_DM',
                           'brighter_SPEGs_zero_DM_peak_SNR']]

    # s_no_grouping = df[['filename', 'SPEG_rank', 'peak_SNR', 'centered_DM', 'clipped_SPEG',
    #                               'SNR_sym_index', 'DM_sym_index', 'peak_score', 'size_ratio', 'cluster_density',
    #                               'DM_range', 'time_range', 'pulse_width', 'time_ratio', 'n_SPEGs_zero_DM',
    #                               'n_brighter_SPEGs_zero_DM', 'SPEGs_zero_DM_min_DM', 'brighter_SPEGs_zero_DM_peak_DM',
    #                               'brighter_SPEGs_zero_DM_peak_SNR', 'class_label']]

    # df_learning_dim = df_learning[df_learning['peak_SNR'] < 8]

    # df_learning_no_grouping_bright = df_learning_no_grouping[df_learning_no_grouping['peak_SNR'] > 7.99]

    # if speg_type == 'all':
    #     return df_learning
    # elif speg_type == 'bright':
    #     return df_learning_no_grouping_bright
    # elif speg_type == 'dim':
    #     return df_learning_dim
    # else:
    #     exit("speg_type invalid!")
    return df_to_be_learned, df


# def recall(y_true, y_pred):
#     """Recall metric.
#
#     Only computes a batch-wise average of recall.
#
#     Computes the recall, a metric for multi-label classification of
#     how many relevant items are selected.
#     """
#     true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
#     possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
#     recall = true_positives / (possible_positives + K.epsilon())
#     return recall


# def f1(y_true, y_pred):
#     def recall(y_true, y_pred):
#         """Recall metric.
#
#         Only computes a batch-wise average of recall.
#
#         Computes the recall, a metric for multi-label classification of
#         how many relevant items are selected.
#         """
#         true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
#         possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
#         recall = true_positives / (possible_positives + K.epsilon())
#         return recall
#
#     def precision(y_true, y_pred):
#         """Precision metric.
#
#         Only computes a batch-wise average of precision.
#
#         Computes the precision, a metric for multi-label classification of
#         how many selected items are relevant.
#         """
#         true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
#         predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
#         precision = true_positives / (predicted_positives + K.epsilon())
#         return precision
#     precision = precision(y_true, y_pred)
#     recall = recall(y_true, y_pred)
#     return 2*((precision*recall)/(precision+recall+K.epsilon()))

def get_benchmark_pulsars(survey='GBT'):
    if survey == 'GBT':
        benchmark_data_full = GBT_benchmark_data_full
    elif survey == 'PALFA':
        benchmark_data_full = PALFA_benchmark_data_full
    else:
        exit("Survey name not valid!")

    speg_df = pd.read_csv(benchmark_data_full, sep=",", skipinitialspace=True)

    # only keep labeled astro SPEGs
    speg_df_astro = speg_df.loc[speg_df['label'] == "YES"]

    print("Survey ", survey, "total labeled SPEGs: ")
    print(speg_df_astro.shape)

    # MAKE A COPY
    df = speg_df_astro.copy()
    pulsars = df['filename'].unique()

    pulsar_list = []


    for pulsar in pulsars:
        # print(pulsar)
        df_pulsar = df.loc[df['filename'] == pulsar]
        # print(df_pulsar)
        n_pulses = df_pulsar.shape[0]
        brightest_SNR = df_pulsar['peak_SNR'].max()
        cur_pulsar = benchmarkPulsar(pulsar, n_pulses, brightest_SNR)
        pulsar_list.append(cur_pulsar)

    return pulsar_list



class benchmarkPulsar(object):
    """
    This is the class for benchmark pulsars with the main following attributes:
    beam: the name of the beam in which the pulsar is detected
    n_pulses: the number of astrophysical pulses
    brightest_SNR: the SNR of the detected brightest astrophysical pulse
    """

    __slots__ = ["beam", "n_pulses", "brightest_SNR"]

    def __init__(self, beam, n_pulses, brightest_SNR):
        """
        benchmarkPulsar constructor.
        :param current_list: a list of attributes of the brightest trial single pulse event within the cluster
        """

        self.beam = beam
        self.n_pulses = n_pulses
        self.brightest_SNR = brightest_SNR

    def __str__(self):
        # print cur_cluster
        s = ["\tpulsar beam: \t\t%s" % self.beam,
             "\tnumber of pulses: %4d" % self.n_pulses,
             "\tbrightest SNR: \t\t%3.2f" % self.brightest_SNR,
             "--------------------------------"
             ]
        return '\n'.join(s)
