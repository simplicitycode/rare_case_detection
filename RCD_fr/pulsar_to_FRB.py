from SPEG_Candidate_FRB import SPEG_Candidate
import numpy as np
import pandas as pd
import random


def pulsar_to_FRB(df_spegs=None, cur_pulsar=None, speg_rank=None, double_DM=False,
                  benchmark_path=None, save_dir=None):
    """
    this is the newest version of pulsar_to_frb, keep all dim SPEGs
    :param df_spegs: from which beam spegs are selected
    :param cur_pulsar: the pulsar to be converted
    :param speg_rank: the SPEG rank of selected pulse
    :param double_DM: whether the peak DM of the simulated FRB is doubled
    :return:
    """
    ####################################################################################################333333##########
    df_beam = df_spegs.loc[df_spegs['filename'] == cur_pulsar].copy()
    print("df_beam:", df_beam.shape[0])
    dim_SNR_threshold = df_beam.iloc[-1, :].peak_SNR
    print("dim_SNR_threshold: ", dim_SNR_threshold)

    #######
    df_astro_tmp = df_beam.loc[df_beam['SPEG_rank'] == speg_rank]

    # print("df astro tmp shape", df_astro_tmp.shape)
    if df_astro_tmp.shape[0] < 1:
        return None
    else:
        print("df_beam_astro: ", df_astro_tmp.shape[0])

    non_astro_SPEGs = []

    for index, row in df_beam.iterrows():
        cur_row = row.tolist()
        cur_cand = SPEG_Candidate(cur_row)
        if cur_cand.class_label == 0:
            non_astro_SPEGs.append(cur_cand)
        # only one positive pulse
        elif cur_cand.SPEG_rank == speg_rank:
            # this is the FRB pulse
            cur_FRB_pulse = cur_cand

    # class label of the FRB
    cur_FRB_pulse.class_label = 2
    # TODO: double DM, startDM, stopDM
    if double_DM:
        centered_DM_orginal = cur_FRB_pulse.centered_DM
        center_startDM_original = cur_FRB_pulse.center_startDM
        center_stopDM_original = cur_FRB_pulse.center_stopDM

        cur_FRB_pulse.centered_DM = 2 * centered_DM_orginal
        # update
        cur_FRB_pulse.center_startDM = cur_FRB_pulse.centered_DM - (centered_DM_orginal -  center_startDM_original)
        cur_FRB_pulse.center_stopDM = cur_FRB_pulse.centered_DM + (center_stopDM_original - centered_DM_orginal)

        min_DM_original = cur_FRB_pulse.min_DM
        max_DM_original = cur_FRB_pulse.max_DM

        cur_FRB_pulse.min_DM = cur_FRB_pulse.centered_DM - (centered_DM_orginal -  min_DM_original)
        cur_FRB_pulse.max_DM = cur_FRB_pulse.centered_DM + (max_DM_original - centered_DM_orginal)

    selected_SPEGs = non_astro_SPEGs
    selected_SPEGs.append(cur_FRB_pulse)

    #################@@@@@@@@1
    # # must not be dense plot
    # if dim_SNR_threshold > 6.01:
    #     return None
    # astro
    df_beam_astro = df_beam.loc[df_beam['class_label'] == 1].copy()

    # # astro_moderate_pulses sampled
    # df_beam_astro_moderate = df_beam_astro.loc[df_beam_astro['peak_SNR'] <= snr_max]
    # # must have at least three pulses with peak_SNR great than 6, less than 7
    # if df_beam_astro_moderate.shape[0] < 3:
    #     return None
    # else:
    #     print("df_beam_astro_moderate: ", df_beam_astro_moderate.shape[0])
    # # only choose 3 to 6 astro pulses
    # astro_moderate_ranks = df_beam_astro_moderate['SPEG_rank'].tolist()
    # print("astro_moderate_ranks: ", astro_moderate_ranks)
    # # random selection
    # # random.seed(1)
    # n_astro_pulses = random.randint(3, min(6, len(astro_moderate_ranks)))
    # print("n_dim_pulses: ", n_astro_pulses)
    #
    # selected_astro_moderate_ranks = random.sample(astro_moderate_ranks, n_astro_pulses)
    # # selected SPEGs
    # print("selected astro_moderate_ranks: ", selected_astro_moderate_ranks)
    #
    # # selected df
    # df_beam_astro_selected = df_beam_astro_moderate.loc[
    #     df_beam_astro_moderate['SPEG_rank'].isin(selected_astro_moderate_ranks)]

    # non-astro
    df_beam_non_astro = df_beam.loc[df_beam['class_label'] == 0].copy()

    # orignal SPEGs
    slash_index = cur_pulsar.find('/')
    cur_beam_name = cur_pulsar[slash_index + 1:]

    SPEG_all_file = benchmark_path + '/' + cur_pulsar + '/' + cur_beam_name + '_SPEG_all.csv'

    df_beam_all_SPEGs = pd.read_csv(SPEG_all_file)
    print("df_beam_all_SPEGs: ", df_beam_all_SPEGs.shape[0])

    # dim SPEGs, not labeled
    df_beam_dim_SPEGs = df_beam_all_SPEGs.loc[df_beam_all_SPEGs['peak_SNR'] < dim_SNR_threshold].copy()

    # add extra features
    df_beam_dim_SPEGs['class_label'] = 0

    df_beam_dim_SPEGs = df_beam_dim_SPEGs * 1

    # calculate features
    # size ratio
    df_beam_dim_SPEGs['size_ratio'] = df_beam_dim_SPEGs['size'] * 1.0 / df_beam_dim_SPEGs['sizeU']
    # cluster_density
    df_beam_dim_SPEGs['cluster_density'] = df_beam_dim_SPEGs['cluster_number'] * 1.0 / \
                                           (df_beam_dim_SPEGs['obs_length'] * df_beam_dim_SPEGs['DM_channel_number'])
    # DM_range
    df_beam_dim_SPEGs['DM_range'] = df_beam_dim_SPEGs['max_DM'] - df_beam_dim_SPEGs['min_DM']
    # time_range
    df_beam_dim_SPEGs['time_range'] = df_beam_dim_SPEGs['max_time'] - df_beam_dim_SPEGs['min_time']
    # pulse_width
    df_beam_dim_SPEGs['pulse_width'] = df_beam_dim_SPEGs['peak_time'] / df_beam_dim_SPEGs['peak_sampling'] \
                                       * df_beam_dim_SPEGs['peak_downfact']
    # time_ratio
    df_beam_dim_SPEGs['time_ratio'] = df_beam_dim_SPEGs['time_range'] / df_beam_dim_SPEGs['pulse_width']

    # with center_startDM and center_stopDM
    df_beam_dim_learning_w_central_peak = df_beam_dim_SPEGs[['filename', 'SPEG_rank', 'group_rank', 'group_max_SNR',
                                                             'group_median_SNR', 'peak_SNR', 'centered_DM',
                                                             'clipped_SPEG', 'SNR_sym_index', 'DM_sym_index',
                                                             'peak_score', 'bright_recur_times', 'recur_times',
                                                             'size_ratio', 'cluster_density', 'DM_range', 'time_range',
                                                             'pulse_width', 'time_ratio', 'n_SPEGs_zero_DM',
                                                             'n_brighter_SPEGs_zero_DM', 'SPEGs_zero_DM_min_DM',
                                                             'brighter_SPEGs_zero_DM_peak_DM',
                                                             'brighter_SPEGs_zero_DM_peak_SNR',
                                                             'class_label', 'center_startDM', 'center_stopDM',
                                                             'min_DM', 'max_DM', 'peak_time', 'min_time', 'max_time']]

    # append dim pulses
    for index, row in df_beam_dim_learning_w_central_peak.iterrows():
        cur_SPEG = SPEG_Candidate(row)
        selected_SPEGs.append(cur_SPEG)

    # sort SPEGs by peak_SNR in descending order
    selected_SPEGs.sort(key=lambda x: x.peak_SNR, reverse=True)
    # # dim_SNR_threshold = round(selected_SPEGs[-1].peak_SNR)
    # selected_group_ranks = [x.group_rank for x in selected_SPEGs]
    # selected_group_ranks = list(set(selected_group_ranks))
    # # sort group rank in ascending order
    # selected_group_ranks.sort(reverse=False)
    #
    for cur_SPEG in selected_SPEGs:
        cur_SPEG.grouped = False

    # get the grouping and ranking features, bright SPEGs only
    # define median function
    def median(lst):
        return np.median(np.array(lst))

    # TODO: remove really dim SPEGs
    bright_SPEGs = [cur_SPEG for cur_SPEG in selected_SPEGs if cur_SPEG.peak_SNR >= dim_SNR_threshold]
    print(len(bright_SPEGs))
    n_bright_SPEGs = len(bright_SPEGs)
    cur_group_rank = 1

    dim_SPEGs = [cur_SPEG for cur_SPEG in selected_SPEGs if cur_SPEG.peak_SNR < dim_SNR_threshold]
    print(len(dim_SPEGs))
    # exit()
    # grouping, must have at least one bright pulse
    for i in range(n_bright_SPEGs):
        cur_group = []
        cur_group_SNRs = []
        cur_SPEG = bright_SPEGs[i]

        cur_SPEG.SPEG_rank = i + 1
        if not cur_SPEG.grouped:
            # the first element
            cur_group.append(cur_SPEG)
            cur_group_SNRs.append(cur_SPEG.peak_SNR)
            cur_max_SNR = cur_SPEG.peak_SNR
            cur_group_peak_DM = cur_SPEG.centered_DM
            # check other clusters
            for j in range(i + 1, n_bright_SPEGs):
                ano_SPEG = bright_SPEGs[j]
                # not grouped yet
                if not ano_SPEG.grouped:
                    # # contains the peak DM of the group
                    # if ano_SPEG.min_DM <= cur_SPEG.centered_DM <= ano_SPEG.max_DM:
                    # central part overlap
                    if (cur_SPEG.center_stopDM >= ano_SPEG.center_startDM and
                            cur_SPEG.center_startDM <= ano_SPEG.center_stopDM):
                        # and ano_SPEG.min_DM <= cur_group_peak_DM <= ano_SPEG.max_DM):
                        # include into current group
                        cur_group.append(ano_SPEG)
                        cur_group_SNRs.append(ano_SPEG.peak_SNR)
                        ano_SPEG.grouped = True
            # before including dim SPEG
            n_bright_recur_times = len(cur_group)

            cur_group_dim = []
            # check dim SPEG
            for each_dim in dim_SPEGs:
                # not grouped yet
                if not each_dim.grouped:
                    # central part overlap
                    if (cur_SPEG.center_stopDM >= each_dim.center_startDM and
                            cur_SPEG.center_startDM <= each_dim.center_stopDM and
                            each_dim.min_DM <= cur_group_peak_DM <= each_dim.max_DM):
                        cur_group_dim.append(each_dim)
                        # cur_group_SNRs.append(each_dim.peak_SNR)
                        # cur_group.append(each_dim)
                        # cur_group_SNRs.append(each_dim.peak_SNR)
                        each_dim.grouped = True

            # non-astrophysical pulses
            if cur_SPEG.class_label == 0:
                cur_dim_SNRs = [cur_dim.peak_SNR for cur_dim in cur_group_dim]
                cur_group_SNRs.extend(cur_dim_SNRs)
                cur_group.extend(cur_group_dim)
            # only class_label:  0 or 2
            # elif cur_SPEG.class_label == 1:
            #     print("cur_astr0_SPEG:", cur_SPEG.SPEG_rank)
            #     # select 0 to 2 from dim pulses
            #     # random.seed(1)
            #     n_dim_astro = random.randint(0, 2)
            #     print("dim astro SPEGs: ", n_dim_astro)
            #     dim_astro_SEPGs = random.sample(cur_group_dim, min(n_dim_astro, len(cur_group_dim)))
            #     cur_dim_SNRs = [cur_dim_astro.peak_SNR for cur_dim_astro in dim_astro_SEPGs]
            #     print("cur_dim_SNRs: ", cur_dim_SNRs)
            #
            #     cur_group_SNRs.extend(cur_dim_SNRs)
            #     cur_group.extend(dim_astro_SEPGs)

            # after including dim SPEG
            n_recur_times = len(cur_group)
            cur_median_SNR = median(cur_group_SNRs)

            # assign the group rank and times of recurrences
            for each_SPEG in cur_group:
                each_SPEG.bright_recur_times = n_bright_recur_times
                each_SPEG.recur_times = n_recur_times
                each_SPEG.group_rank = cur_group_rank
                each_SPEG.group_median_SNR = cur_median_SNR
                each_SPEG.group_max_SNR = cur_max_SNR
                each_SPEG.group_peak_DM = cur_group_peak_DM
            # print "cur_group_rank", cur_group_rank
            cur_group_rank += 1

    #####################################################################################################################

    # # sort SPEGs by peak_SNR in descending order
    # selected_SPEGs.sort(key=lambda x: x.peak_SNR, reverse=True)
    # # dim_SNR_threshold = round(selected_SPEGs[-1].peak_SNR)
    # selected_group_ranks = [x.group_rank for x in selected_SPEGs]
    # selected_group_ranks = list(set(selected_group_ranks))
    # # sort group rank in ascending order
    # selected_group_ranks.sort(reverse=False)
    #
    # # get the grouping and ranking features, bright SPEGs only
    # # define median function
    # def median(lst):
    #     return np.median(np.array(lst))
    #
    # ranks_affected = []
    # ranks_not_affected = []
    # groups_not_affected = []
    # # change this
    #
    # for i, cur_group_rank in enumerate(selected_group_ranks):
    #     cur_SPEG_group = [x for x in selected_SPEGs if x.group_rank == cur_group_rank]
    #     # update group_rank, group_max_SNR
    #     SPEG_first = cur_SPEG_group[0]
    #
    #     # check which groups have been affected
    #     bright_SPEG_record = SPEG_first.bright_recur_times
    #     bright_SPEG_count = len(cur_SPEG_group)
    #     if bright_SPEG_count != bright_SPEG_record:
    #         # groups affected, may needs new group_rank, new grouping
    #         ranks_affected.append(cur_group_rank)
    #     else:
    #         # unaffected groups
    #         ranks_not_affected.append(cur_group_rank)
    #         cur_group_max_SNR = SPEG_first.peak_SNR
    #         # used to re-assign group rank
    #         groups_not_affected.append([cur_group_max_SNR, cur_group_rank, cur_SPEG_group])
    #
    # SPEGs_ungrouped = []
    # # these are SPEGs needed to be regrouped
    # for cur_group_rank in ranks_affected:
    #     # iterate through all affected groups
    #     for cur_SPEG in selected_SPEGs:
    #         if cur_SPEG.group_rank == cur_group_rank:
    #             # -2 means not grouped
    #             cur_SPEG.group_rank = -2
    #             SPEGs_ungrouped.append(cur_SPEG)
    #
    # # group the ungrouped SPEGs
    # new_sub_groups = []
    # groups_rearranged = []
    # # n_sub_groups = 0
    # for i in range(len(SPEGs_ungrouped)):
    #     cur_SPEG = SPEGs_ungrouped[i]
    #     if cur_SPEG.group_rank == -2:
    #         # grouped flag: -1
    #         cur_SPEG.group_rank = -1
    #         # start a new group
    #         cur_group = [cur_SPEG]
    #         for j in range(i + 1, len(SPEGs_ungrouped)):
    #             ano_SPEG = SPEGs_ungrouped[j]
    #             # not grouped either
    #             if ano_SPEG.group_rank == -2:
    #                 if (cur_SPEG.center_stopDM >= ano_SPEG.center_startDM and
    #                         cur_SPEG.center_startDM <= ano_SPEG.center_stopDM):
    #                     ano_SPEG.group_rank = -1
    #                     # include it into current group
    #                     cur_group.append(ano_SPEG)
    #
    #         # n_sub_groups += 1
    #         new_sub_groups.append(cur_group)
    #
    # # extract features from new groups
    # for cur_sub_group in new_sub_groups:
    #     head_SPEG = cur_sub_group[0]
    #
    #     bright_SNRs = [x.peak_SNR for x in cur_sub_group]
    #     cur_bright_rec_times = len(bright_SNRs)
    #
    #     # number of dim pulses in the orginal group
    #     dim_rec_times_record = head_SPEG.recur_times - head_SPEG.bright_recur_times
    #     n_dim_new = round(dim_rec_times_record // (head_SPEG.bright_recur_times / cur_bright_rec_times))
    #     # print(dim_rec_times_record, head_SPEG.bright_recur_times,cur_bright_rec_times, n_dim_new)
    #     dim_SNRs = []
    #     for k in range(n_dim_new):
    #         cur_dim_SNR = random.uniform(dim_SNR_threshold - 1, dim_SNR_threshold)
    #         dim_SNRs.append(cur_dim_SNR)
    #
    #     bright_SNRs.extend(dim_SNRs)
    #     # combined SNRs
    #     combined_SNRs = bright_SNRs
    #     cur_rec_times = len(combined_SNRs)
    #
    #     cur_median_SNR = median(combined_SNRs)
    #     cur_max_SNR = combined_SNRs[0]
    #
    #     # update new features
    #     for each_SPEG in cur_sub_group:
    #         each_SPEG.group_max_SNR = cur_max_SNR
    #         each_SPEG.group_median_SNR = cur_median_SNR
    #         each_SPEG.bright_recur_times = cur_bright_rec_times
    #         each_SPEG.recur_times = cur_rec_times
    #         # if each_SPEG.class_label == 2:
    #         #     print(each_SPEG)
    #
    #     # -1 placeholder for group rank
    #     groups_rearranged.append([cur_max_SNR, -1, cur_sub_group])
    #
    # # update the group rank
    #
    # # all SPEG groups
    # SPEG_groups = groups_not_affected
    # SPEG_groups.extend(groups_rearranged)
    #
    # # sort in descending order by cur_max_SNR in descending order
    # SPEG_groups.sort(key=lambda x: x[0], reverse=True)
    #
    # SPEGs_group_rank_updated = []
    # for index, cur_SPEG_group in enumerate(SPEG_groups):
    #     # group of SPEGs has rank 2, update SPEG rank
    #     for cur_SPEG in cur_SPEG_group[2]:
    #         # group rank starts from rank 1
    #         cur_SPEG.group_rank = index + 1
    #         SPEGs_group_rank_updated.append(cur_SPEG)
    #
    # # sort by peak_SNR in descending order
    # SPEGs_group_rank_updated.sort(key=lambda x: x.peak_SNR, reverse=True)
    #
    # # TODO: (remove) only keep bright SPEGs
    # # SPEGs_group_rank_updated_bright = [x for x in SPEGs_group_rank_updated if x.peak_SNR >= snr_min]
    # # # update the SPEG_rank
    # # SPEGs_ranks_updated = list()
    # # for index, cur_SPEG in enumerate(SPEGs_group_rank_updated_bright):
    # #     new_SPEG_rank = index + 1
    # #     cur_SPEG.SPEG_rank = new_SPEG_rank
    # #     SPEGs_ranks_updated.append(cur_SPEG)
    #
    # # update the SPEG_rank
    # SPEGs_ranks_updated = list()
    # for index, cur_SPEG in enumerate(SPEGs_group_rank_updated):
    #     new_SPEG_rank = index + 1
    #     cur_SPEG.SPEG_rank = new_SPEG_rank
    #     SPEGs_ranks_updated.append(cur_SPEG)
    #
    # # return a list of updated SPEGs
    # # TODO: only double in the end
    # if double_DM:
    #     SPEGs_ranks_updated2 = list()
    #     for cur_SPEG in SPEGs_ranks_updated:
    #         if cur_SPEG.class_label == 2:
    #             cur_SPEG.centered_DM = 2 * cur_SPEG.centered_DM
    #         SPEGs_ranks_updated2.append(cur_SPEG)
    #
    # # test output
    # cur_beam_info = cur_pulsar.split('/')
    # cur_beam = cur_beam_info[1]
    if save_dir:
        output_file = save_dir + '/' + cur_beam_name + '_SPEG_simulated_FRB.csv'
        output_fp = open(output_file, 'w')
        header = "filename,group_rank,group_peak_DM,group_max_SNR,group_median_SNR,SPEG_rank,centered_DM," \
                 "center_startDM,center_stopDM,peak_time," \
                 "peak_SNR,min_time,max_time,min_DM,max_DM,bright_recur_times,recur_times,DM_sym_index,peak_score,class_label"

        output_fp.write(header + '\n')

        for cur_SPEG in bright_SPEGs:
            # save bright and dim SPEGs
            cur_line = cur_SPEG.filename + ',' + str(cur_SPEG.group_rank) + ',' + str(cur_SPEG.group_peak_DM) + ',' \
                       + str(cur_SPEG.group_max_SNR) + ',' \
                       + str(cur_SPEG.group_median_SNR) + ',' + str(cur_SPEG.SPEG_rank) + ',' \
                       + str(cur_SPEG.centered_DM) + ',' + str(cur_SPEG.center_startDM) + ',' + str(cur_SPEG.center_stopDM) + ',' \
                       + str(cur_SPEG.peak_time) + ',' + str(cur_SPEG.peak_SNR) + ',' + str(cur_SPEG.min_time) + ',' \
                       + str(cur_SPEG.max_time) + ',' + str(cur_SPEG.min_DM) + ',' + str(cur_SPEG.max_DM) + ',' \
                       + str(cur_SPEG.bright_recur_times) + ',' + str(cur_SPEG.recur_times) + ',' \
                       + str(cur_SPEG.DM_sym_index) + ',' + str(cur_SPEG.peak_score) + ',' + str(cur_SPEG.class_label)

            output_fp.write(cur_line + '\n')

        for cur_SPEG in dim_SPEGs:
            # save bright and dim SPEGs
            cur_line = cur_SPEG.filename + ',' + str(cur_SPEG.group_rank) + ',' + str(cur_SPEG.group_peak_DM) + ',' \
                       + str(cur_SPEG.group_max_SNR) + ',' \
                       + str(cur_SPEG.group_median_SNR) + ',' + str(cur_SPEG.SPEG_rank) + ',' \
                       + str(cur_SPEG.centered_DM) + ',' + str(cur_SPEG.center_startDM) + ',' + str(cur_SPEG.center_stopDM) + ',' \
                       + str(cur_SPEG.peak_time) + ',' + str(cur_SPEG.peak_SNR) + ',' + str(cur_SPEG.min_time) + ',' \
                       + str(cur_SPEG.max_time) + ',' + str(cur_SPEG.min_DM) + ',' + str(cur_SPEG.max_DM) + ',' \
                       + str(cur_SPEG.bright_recur_times) + ',' + str(cur_SPEG.recur_times) + ',' \
                       + str(cur_SPEG.DM_sym_index) + ',' + str(cur_SPEG.peak_score) + ',' + str(cur_SPEG.class_label)

            output_fp.write(cur_line + '\n')

        output_fp.close()

        # # remove some dim pulses
        # bright_SPEGs_updated = []
        # for cur_SPEG in bright_SPEGs:
        #     if cur_SPEG.class_label == 1:
        #         cur_SPEG.recur_times = min(round(cur_SPEG.bright_recur_times * 1.5), cur_SPEG.recur_times)
        #     bright_SPEGs_updated.append(cur_SPEG)

        return bright_SPEGs



