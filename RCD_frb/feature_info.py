from scipy.stats import median_absolute_deviation
import numpy as np
import pandas as pd
import statistics

feature_list = ['SPEG_rank', 'group_rank', 'group_max_SNR', 'group_median_SNR', 'peak_SNR',
                'centered_DM', 'clipped_SPEG', 'SNR_sym_index', 'DM_sym_index', 'peak_score',
                'bright_recur_times', 'recur_times', 'size_ratio', 'cluster_density', 'DM_range',
                'time_range', 'pulse_width', 'time_ratio', 'n_SPEGs_zero_DM', 'n_brighter_SPEGs_zero_DM'
                ]

data_types = {'SPEG_rank': 'interval',
              'group_rank': 'interval',
              'group_max_SNR': 'interval',
              'group_median_SNR': 'interval',
              'peak_SNR': 'interval',
              'centered_DM': 'interval',
              'clipped_SPEG': 'categorical',
              'SNR_sym_index': 'interval',
              'DM_sym_index': 'interval',
              'peak_score': 'ordinal',
              'bright_recur_times': 'interval',
              'recur_times': 'interval',
              'size_ratio': 'interval',
              'cluster_density': 'interval',
              'DM_range': 'interval',
              'time_range': 'interval',
              'pulse_width': 'interval',
              'time_ratio': 'interval',
              'n_SPEGs_zero_DM': 'interval',
              'n_brighter_SPEGs_zero_DM': 'interval'
              }

# bins based on quantile=true
bins = {'SPEG_rank': [0, 10, 20, 30, 40, 50, 60, 80, 100, 120, 140, 160, 180, 200, 225, 250, 275, 300, 350, 400,
                      500, 750, 1000, 1500, 2200],
        'group_rank': [0, 1, 2, 3, 4, 5, 6, 7, 9, 11, 13, 17, 21, 25, 30, 50, 345],
        'group_max_SNR': 20,
        'group_median_SNR':20,
        'peak_SNR':20,
        'centered_DM':25,
        'SNR_sym_index': 20,
        'DM_sym_index': 25,
        'bright_recur_times': [0, 1, 2, 3, 4, 5, 10, 20, 30, 50, 70, 90, 110, 140, 170, 200, 240, 280, 320, 370,
                               420, 500, 800, 1200, 1613],
        'recur_times': [0, 1, 2, 3, 5, 7, 10, 20, 40, 60, 100, 140, 180, 220, 260, 300, 350, 400, 450, 500, 550,
                        600, 700, 800, 1500, 2553],
        'size_ratio': 20,
        'cluster_density': 20,
        'DM_range': 20,
        'time_range': 20,
        'pulse_width': 20,
        'time_ratio': 20,
        'n_SPEGs_zero_DM': 10,
        'n_brighter_SPEGs_zero_DM': 10
        }
# bins = {'mileage':5, 'driver_age':[18,25,35,45,55,65,125]}


def get_sim_value_SPEGs(feature = None, df=None, target_SPEG=None, candidate_SPEG=None):
    # print("cur feature: ", feature)
    target_value = getattr(target_SPEG, feature)
    candidate_value = getattr(candidate_SPEG, feature)
    cur_values = df[feature]
    if feature in ['SPEG_rank', 'group_max_SNR', 'group_median_SNR', 'peak_SNR', 'centered_DM',
                   'SNR_sym_index', 'DM_sym_index', 'cluster_density', 'DM_range',
                   'time_range', 'pulse_width', 'time_ratio']:
        # log transform
        cur_values = np.log(cur_values)
        cur_MAD = median_absolute_deviation(cur_values)
        cur_MAD = max(cur_MAD, 0.000001)
        cur_sim_value = max(0, 1 - abs(np.log(target_value) - np.log(candidate_value)) / (3 * 1.483 * cur_MAD))

        # print("target_value:", target_value)
        # print("candidate_value:", candidate_value)
        # print("cur_MAD:", cur_MAD)

    elif feature in ['group_rank', 'size_ratio']:
        # log transform
        cur_values = np.log(cur_values)
        # min_max
        cur_sim_value = max(0, 1 - abs(np.log(target_value) - np.log(candidate_value)) / (max(cur_values) - min(cur_values)))
        # print("target_value:", target_value)
        # print("candidate_value:", candidate_value)

    elif feature in ['bright_recur_times', 'recur_times']:
        # log transform
        cur_values = np.log(cur_values)
        cur_MAD = median_absolute_deviation(cur_values)
        cur_MAD = max(cur_MAD, 0.000001)
        # print("cur_MAD:", cur_MAD)
        # do not multiply by 3
        # milder penalty for over shooting
        delta = 1.0 * (candidate_value > target_value)
        alpha = 0.5
        cur_sim_value = max(0, 1 - abs(np.log(target_value) - np.log(candidate_value)) / (3 * 1.483 * cur_MAD) +
                            alpha * delta * abs(np.log(target_value) - np.log(candidate_value)) / (3 * 1.483 * cur_MAD))

    elif feature == 'clipped_SPEG':
        cur_sim_value = 1 - abs(target_value - candidate_value)

    elif feature == 'peak_score':
        cur_sim_value = 1 - abs(target_value - candidate_value) / (6 - 2)

    elif feature in ['n_SPEGs_zero_DM', 'n_brighter_SPEGs_zero_DM']:
        max_proxy = max(1, np.quantile(cur_values, 0.95))
        # print("max_proxy: ", max_proxy)
        cur_sim_value = max(0, 1 - abs(target_value - candidate_value) / max_proxy)

    return cur_sim_value


def get_outlyingness(feature=None, df=None, target_value=None):
    target_value_count = 0
    # cur_values = df[feature]
    if feature in ['SPEG_rank', 'group_rank']:
        # find the one theat is closest to current value
        df_sort = df.iloc[(df[feature] - target_value).abs().argsort()[:1]]
        # print(df[feature])
        cur_proxy_value = df_sort[feature].tolist()
        print("cur_proxy_value: ", cur_proxy_value)

        cur_values_count = df[feature].value_counts()
        cur_mode_count = cur_values_count.max()
        print("cur_mode: ", cur_mode_count)
        target_value_count = float(cur_values_count[cur_proxy_value])
        print("target_value_count: ", target_value_count)

        cur_outlyingness = np.log(cur_mode_count / target_value_count)

    elif feature in ['group_max_SNR', 'group_median_SNR', 'peak_SNR', 'centered_DM', 'SNR_sym_index', 'DM_sym_index',
                     'bright_recur_times', 'recur_times', 'DM_range', 'time_range', 'pulse_width', 'time_ratio']:
        cur_values = df[feature]
        # log transform
        cur_values = np.log(cur_values)
        cur_median = statistics.median(cur_values)
        cur_MAD = median_absolute_deviation(cur_values)
        cur_z_score = abs(np.log(target_value) - cur_median) / (1.483 * cur_MAD)
        cur_outlyingness = cur_z_score

    elif feature in ['peak_score']:
        df_tmp = df.copy()
        bin_results = pd.cut(df_tmp[feature], bins=[1, 3, 4, 5, 6]).value_counts()
        print("df_cut----------")
        print(bin_results)

        cur_mode_count = bin_results.max()
        print("cur_mode: ", cur_mode_count)
        print("target_value: ", target_value)
        for cur_bin in bin_results.index:
            # print(each_bin)
            if target_value in cur_bin:
                target_value_count = bin_results.at[cur_bin]
                print("found: ", target_value_count)
                break
        cur_outlyingness = (cur_mode_count / target_value_count)

    elif feature in ['size_ratio']:
        df_tmp = df.copy()
        bin_results = pd.cut(np.log(df_tmp[feature]), bins=[-0.1, 0.1, 0.2, 0.3, 2]).value_counts()
        print("df_cut----------")
        print(bin_results)

        cur_mode_count = bin_results.max()
        print("target_value: ", target_value)
        for cur_bin in bin_results.index:
            if np.log(target_value) in cur_bin:
                target_value_count = bin_results.at[cur_bin]
                print("found: ", target_value_count)
                break

        # TODO: log or not
        if target_value_count < 1:
            target_value_count = 1
        cur_outlyingness = (cur_mode_count / target_value_count)

    elif feature in ['cluster_density']:
        df_tmp = df.copy()
        bin_results = pd.cut(df_tmp[feature], bins=np.linspace(0, 0.035, 8)).value_counts()
        print("df_cut----------")
        print(bin_results)

        cur_mode_count = bin_results.max()
        # print("cur_mode: ", cur_mode_count)
        for cur_bin in bin_results.index:
            # print(each_bin)
            if target_value in cur_bin:
                target_value_count = bin_results.at[cur_bin]
                print("found: ", target_value_count)
                break
        print("searching: ", target_value)
        # TODO: log or not
        if target_value_count < 1:
            target_value_count = 1
        cur_outlyingness = np.sqrt(cur_mode_count / target_value_count)

    elif feature in ['clipped_SPEG']:
        # find the one theat is closest to current value
        df_sort = df.iloc[(df[feature] - target_value).abs().argsort()[:1]]
        # print(df[feature])
        cur_proxy_value = df_sort[feature].tolist()
        print("cur_proxy_value: ", cur_proxy_value)

        cur_values_count = df[feature].value_counts()
        cur_mode_count = cur_values_count.max()
        print("cur_mode: ", cur_mode_count)
        target_value_count = float(cur_values_count[cur_proxy_value])
        print("target_value_count: ", target_value_count)

        cur_outlyingness = np.sqrt(cur_mode_count / target_value_count)

    elif feature in ['n_SPEGs_zero_DM']:
        df_tmp = df.copy()
        bin_results = pd.cut(df_tmp[feature], bins=[-1, 1, 2, 8]).value_counts()
        print("df_cut----------")
        print(bin_results)

        cur_mode_count = bin_results.max()
        # print("cur_mode: ", cur_mode_count)
        for cur_bin in bin_results.index:
            # print(each_bin)
            if target_value in cur_bin:
                target_value_count = bin_results.at[cur_bin]
                print("found: ", target_value_count)
                break
        # TODO: log or not
        if target_value_count < 1:
            target_value_count = 1
        cur_outlyingness = np.sqrt((cur_mode_count / target_value_count))

    elif feature in ['n_brighter_SPEGs_zero_DM']:
        cur_outlyingness = 1

    print("cur_outlyingness: ", cur_outlyingness)
    # set minimumn values 1
    cur_outlyingness = max(1, cur_outlyingness)
    return cur_outlyingness


# def get_outlyingness2(feature=None, df=None, target_value=None):
#     target_value_count = 0
#     # cur_values = df[feature]
#     if feature in ['SPEG_rank', 'group_rank']:
#         # find the one theat is closest to current value
#         df_sort = df.iloc[(df[feature] - target_value).abs().argsort()[:1]]
#         # print(df[feature])
#         cur_proxy_value = df_sort[feature].tolist()
#         print("cur_proxy_value: ", cur_proxy_value)
#
#         cur_values_count = df[feature].value_counts()
#         cur_mode_count = cur_values_count.max()
#         print("cur_mode: ", cur_mode_count)
#         target_value_count = float(cur_values_count[cur_proxy_value])
#         print("target_value_count: ", target_value_count)
#
#         cur_outlyingness = np.log(cur_mode_count / target_value_count)
#
#     # different bins
#     elif feature in ['group_max_SNR', 'peak_SNR']:
#         df_tmp = df.copy()
#         bin_results = pd.cut(df_tmp[feature], bins=np.linspace(0, 80, 9)).value_counts()
#         print("df_cut----------")
#         print(bin_results)
#
#         cur_mode_count = bin_results.max()
#         # print("cur_mode: ", cur_mode_count)
#         for cur_bin in bin_results.index:
#             # print(each_bin)
#             if target_value in cur_bin:
#                 target_value_count = bin_results.at[cur_bin]
#                 # print("found: ", target_value_count)
#         cur_outlyingness = (cur_mode_count / target_value_count)
#         # print("cur_outlyingness: ", cur_outlyingness)
#
#     elif feature in ['bright_recur_times', 'recur_times']:
#         df_tmp = df.copy()
#         bin_results = pd.cut(np.log(df_tmp[feature]), bins=np.linspace(0, 8, 9)).value_counts()
#         print("df_cut----------")
#         print(bin_results)
#
#         cur_mode_count = bin_results.max()
#         # print("cur_mode: ", cur_mode_count)
#         for cur_bin in bin_results.index:
#             # print(each_bin)
#             if np.log(target_value) in cur_bin:
#                 target_value_count = bin_results.at[cur_bin]
#                 print("found: ", target_value_count)
#                 break
#         # TODO: np.sqrt or not
#         if target_value_count < 1:
#             target_value_count = 1
#         cur_outlyingness = np.sqrt(cur_mode_count / target_value_count)
#
#     # elif feature in ['recur_times']:
#     #     df_tmp = df.copy()
#     #     bin_results = pd.cut(np.log(df_tmp[feature]), bins=np.linspace(1, 8, 8)).value_counts()
#     #     print("df_cut----------")
#     #     print(bin_results)
#     #
#     #     cur_mode_count = bin_results.max()
#     #     # print("cur_mode: ", cur_mode_count)
#     #     for cur_bin in bin_results.index:
#     #         # print(each_bin)
#     #         if np.log(target_value) in cur_bin:
#     #             target_value_count = bin_results.at[cur_bin]
#     #             print("found: ", target_value_count)
#     #             break
#     #     # TODO: log or not
#     #     if target_value_count < 1:
#     #         target_value_count = 1
#     #     cur_outlyingness = np.sqrt(cur_mode_count / target_value_count)
#
#     elif feature in ['DM_range']:
#         df_tmp = df.copy()
#         bin_results = pd.cut(np.log(df_tmp[feature]), bins=np.linspace(0, 7, 8)).value_counts()
#         print("df_cut----------")
#         print(bin_results)
#
#         cur_mode_count = bin_results.max()
#         # print("cur_mode: ", cur_mode_count)
#         for cur_bin in bin_results.index:
#             # print(each_bin)
#             if np.log(target_value) in cur_bin:
#                 target_value_count = bin_results.at[cur_bin]
#                 print("found: ", target_value_count)
#                 break
#         # TODO: log or not
#         if target_value_count < 1:
#             target_value_count = 1
#         cur_outlyingness = cur_mode_count / target_value_count
#
#     elif feature in ['time_range']:
#         df_tmp = df.copy()
#         bin_results = pd.cut(np.log(df_tmp[feature]), bins=np.linspace(-8, -1, 8)).value_counts()
#         print("df_cut----------")
#         print(bin_results)
#
#         cur_mode_count = bin_results.max()
#         # print("cur_mode: ", cur_mode_count)
#         for cur_bin in bin_results.index:
#             # print(each_bin)
#             if np.log(target_value) in cur_bin:
#                 target_value_count = bin_results.at[cur_bin]
#                 print("found: ", target_value_count)
#                 break
#         # TODO: log or not
#         if target_value_count < 1:
#             target_value_count = 1
#         cur_outlyingness = np.log(cur_mode_count / target_value_count)
#
#     elif feature in ['pulse_width']:
#         df_tmp = df.copy()
#         bin_results = pd.cut(np.log(df_tmp[feature]), bins=10).value_counts()
#         print("df_cut----------")
#         print(bin_results)
#
#         cur_mode_count = bin_results.max()
#         # print("cur_mode: ", cur_mode_count)
#         for cur_bin in bin_results.index:
#             # print(each_bin)
#             if np.log(target_value) in cur_bin:
#                 target_value_count = bin_results.at[cur_bin]
#                 print("found: ", target_value_count)
#                 break
#         # TODO: log or not
#         if target_value_count < 1:
#             target_value_count = 1
#         cur_outlyingness = np.log(cur_mode_count / target_value_count)
#
#     elif feature in ['peak_score']:
#         df_tmp = df.copy()
#         bin_results = pd.cut(df_tmp[feature], bins=[1, 3, 4, 5, 6]).value_counts()
#         print("df_cut----------")
#         print(bin_results)
#
#         cur_mode_count = bin_results.max()
#         print("cur_mode: ", cur_mode_count)
#         print("target_value: ", target_value)
#         for cur_bin in bin_results.index:
#             # print(each_bin)
#             if target_value in cur_bin:
#                 target_value_count = bin_results.at[cur_bin]
#                 print("found: ", target_value_count)
#                 break
#         cur_outlyingness = (cur_mode_count / target_value_count)
#
#     elif feature in ['size_ratio']:
#         df_tmp = df.copy()
#         bin_results = pd.cut(np.log(df_tmp[feature]), bins=[-0.1, 0.1, 0.2, 0.3, 2]).value_counts()
#         print("df_cut----------")
#         print(bin_results)
#
#         cur_mode_count = bin_results.max()
#         # print("cur_mode: ", cur_mode_count)
#         for cur_bin in bin_results.index:
#             # print(each_bin)
#             if np.log(target_value) in cur_bin:
#                 target_value_count = bin_results.at[cur_bin]
#                 print("found: ", target_value_count)
#                 break
#         # TODO: log or not
#         if target_value_count < 1:
#             target_value_count = 1
#         cur_outlyingness = (cur_mode_count / target_value_count)
#
#     elif feature in ['cluster_density']:
#         df_tmp = df.copy()
#         bin_results = pd.cut(df_tmp[feature], bins=np.linspace(0, 0.035, 8)).value_counts()
#         print("df_cut----------")
#         print(bin_results)
#
#         cur_mode_count = bin_results.max()
#         # print("cur_mode: ", cur_mode_count)
#         for cur_bin in bin_results.index:
#             # print(each_bin)
#             if target_value in cur_bin:
#                 target_value_count = bin_results.at[cur_bin]
#                 print("found: ", target_value_count)
#                 break
#         # TODO: log or not
#         if target_value_count < 1:
#             target_value_count = 1
#         cur_outlyingness = np.sqrt(cur_mode_count / target_value_count)
#
#     elif feature in ['group_median_SNR']:
#         df_tmp = df.copy()
#         bin_results = pd.cut(df_tmp[feature], bins=np.linspace(5, 25, 11)).value_counts()
#         # print("df_cut----------")
#         print(bin_results)
#
#         cur_mode_count = bin_results.max()
#         print("cur_mode: ", cur_mode_count)
#         for cur_bin in bin_results.index:
#             # print(each_bin)
#             if target_value in cur_bin:
#                 target_value_count = bin_results.at[cur_bin]
#                 print("found: ", target_value_count)
#                 break
#         # TODO: log or not
#         cur_outlyingness = abs((cur_mode_count / target_value_count))
#
#     # do not bin every feature
#     elif feature in ['centered_DM', 'SNR_sym_index', 'DM_sym_index', 'time_ratio']:
#         cur_values = df[feature]
#         # log transform
#         cur_values = np.log(cur_values)
#         cur_median = statistics.median(cur_values)
#         cur_MAD = median_absolute_deviation(cur_values)
#         cur_z_score = abs(np.log(target_value) - cur_median) / (1.483 * cur_MAD)
#         cur_outlyingness = cur_z_score
#
#     elif feature in ['clipped_SPEG']:
#         # find the one theat is closest to current value
#         df_sort = df.iloc[(df[feature] - target_value).abs().argsort()[:1]]
#         # print(df[feature])
#         cur_proxy_value = df_sort[feature].tolist()
#         print("cur_proxy_value: ", cur_proxy_value)
#
#         cur_values_count = df[feature].value_counts()
#         cur_mode_count = cur_values_count.max()
#         print("cur_mode: ", cur_mode_count)
#         target_value_count = float(cur_values_count[cur_proxy_value])
#         print("target_value_count: ", target_value_count)
#
#         cur_outlyingness = np.sqrt(cur_mode_count / target_value_count)
#
#     elif feature in ['n_SPEGs_zero_DM']:
#         df_tmp = df.copy()
#         bin_results = pd.cut(df_tmp[feature], bins=[-1, 1, 2, 8]).value_counts()
#         print("df_cut----------")
#         print(bin_results)
#
#         cur_mode_count = bin_results.max()
#         # print("cur_mode: ", cur_mode_count)
#         for cur_bin in bin_results.index:
#             # print(each_bin)
#             if target_value in cur_bin:
#                 target_value_count = bin_results.at[cur_bin]
#                 print("found: ", target_value_count)
#                 break
#         # TODO: log or not
#         if target_value_count < 1:
#             target_value_count = 1
#         cur_outlyingness = np.sqrt(cur_mode_count / target_value_count)
#
#     elif feature in ['n_brighter_SPEGs_zero_DM']:
#         cur_outlyingness = 1
#
#     print("cur_outlyingness: ", cur_outlyingness)
#     # set minimumn values 1
#     cur_outlyingness = max(1, cur_outlyingness)
#     return cur_outlyingness

