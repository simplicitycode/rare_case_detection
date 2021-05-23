from sklearn.feature_selection import mutual_info_classif
import numpy as np
import phik
from feature_info import data_types, bins, get_outlyingness, feature_list
import multiprocessing
from functools import partial
from weighted_similarity import weighted_similarity
import time
import random

from sklearn import preprocessing

from SPEGID_utils import get_folds, get_spegs, get_benchmark_pulsars
from pulsar_to_FRB import pulsar_to_FRB
from SPEG_Candidate_FRB import SPEG_Candidate
from dir_path import simulated_frb_dir, weights_dir, results_dir, benchmark_path

import argparse



def main():
    if X_source == 'PALFA':
        all_folds = get_folds(survey='PALFA', n_folds=number_of_folds, rdn_seed=run_id)
        df_learning = get_spegs(survey='PALFA', speg_type=speg_type)
    else:
        exit("Invalid source")

    print(X_source, "all spegs: ", df_learning.shape, "\n")

    pulsar_beams = get_benchmark_pulsars(survey=X_source)
    benchmark_pulsars = [x.beam for x in pulsar_beams]

    # only pulsars with less than 30 pulses
    # TODO: change this n_pulses
    cand_benchmark_pulsars = [x.beam for x in pulsar_beams if (x.dimmest_SNR <= snr_max and
                                                               x.brightest_SNR >= snr_min and
                                                               x.n_pulses <= max_astro_pulses and
                                                               x.peak_DM >= DM_min)]

    print(cand_benchmark_pulsars)
    print(len(cand_benchmark_pulsars))

    # 1320 beams
    all_beams = []
    for each_fold in all_folds:
        cur_beams = each_fold['beam'].tolist()
        # print(type(cur_beams))
        # print(cur_beams)
        all_beams.extend(cur_beams)

    j = 0
    # no validation fold
    test_fold = all_folds[j]
    test_fold = test_fold['beam'].tolist()
    train_folds = [x for x in all_beams if (x not in test_fold)]

    # pulsars
    pulsars_train = [x for x in train_folds if x in benchmark_pulsars]
    pulsars_test = [x for x in test_fold if x in benchmark_pulsars]

    df_train_full = df_learning.loc[df_learning['filename'].isin(train_folds)].copy()
    df_test_full = df_learning.loc[df_learning['filename'].isin(test_fold)].copy()

    cand_train_pulsars = [x for x in pulsars_train if x in cand_benchmark_pulsars]
    cand_test_pulsars = [x for x in pulsars_test if x in cand_benchmark_pulsars]

    # generate 3 FRBs in the training data, and 3 in the test data
    reference_frbs = []

    # a list of 3 lists of SPEGs
    reference_frb_SPEGs = []
    while len(reference_frbs) < 3:
        # this is a list
        cur_selected_pulsar = random.sample(cand_train_pulsars, 1)
        cur_selected_pulsar = cur_selected_pulsar[0]

        df_beam = df_train_full.loc[df_train_full['filename'] == cur_selected_pulsar].copy()
        df_beam_astro_bright = df_beam.loc[(df_beam['class_label'] == 1)
                                           & (df_beam['peak_SNR'] >= snr_min)
                                           & (df_beam['peak_SNR'] <= snr_max) &
                                           (df_beam['centered_DM'] >= DM_min)]

        print("training_pulsar: ", cur_selected_pulsar)
        astro_ranks = df_beam_astro_bright['SPEG_rank'].tolist()
        print("astro_ranks: ", astro_ranks)
        cur_speg_rank = random.sample(astro_ranks, 1)[0]
        print("training SPEG rank: ", cur_speg_rank)
        if cur_selected_pulsar not in reference_frbs:

            # pulsar beam to synthetic frb beam
            cur_frb_SPEGs = pulsar_to_FRB(df_spegs=df_train_full, cur_pulsar=cur_selected_pulsar,
                                          speg_rank=cur_speg_rank, double_DM=double_DM,
                                          benchmark_path=benchmark_path, save_dir=simulated_frb_dir)

            # not None
            if cur_frb_SPEGs is not None:
                reference_frbs.append(cur_selected_pulsar)
                # keep the first SPEG
                for cur_SEPG in cur_frb_SPEGs:
                    # print(cur_SEPG.class_label)
                    if cur_SEPG.class_label > 1:
                        reference_frb_SPEGs.append(cur_SEPG)
                        # only the brightest pulsar SPEG
                        break


    # add test fold
    print("***************** creating testing data *********************")

    test_frbs = []
    test_SPEGs_from_frb_beam = []

    while len(test_frbs) < 3:
        cur_test_pulsar = random.sample(cand_test_pulsars, 1)
        cur_test_pulsar = cur_test_pulsar[0]

        df_beam_test = df_test_full.loc[df_test_full['filename'] == cur_test_pulsar].copy()
        df_beam_test_astro_bright = df_beam_test.loc[(df_beam_test['class_label'] == 1)
                                                     & (df_beam_test['peak_SNR'] >= snr_min)
                                                     & (df_beam_test['peak_SNR'] <= snr_max)
                                                     & (df_beam_test['centered_DM'] >= DM_min)]

        print("test_pulsar: ", cur_test_pulsar)
        astro_ranks_test = df_beam_test_astro_bright['SPEG_rank'].tolist()
        print("astro_ranks_test: ", astro_ranks_test)
        test_speg_rank = random.sample(astro_ranks_test, 1)[0]
        print("test SPEG rank: ", test_speg_rank)

        if cur_test_pulsar not in test_frbs:
            # pulsar beam to synthetic frb beam
            frb_SPEGs = pulsar_to_FRB(df_spegs=df_test_full, cur_pulsar=cur_test_pulsar,
                                      speg_rank=test_speg_rank, double_DM=double_DM,
                                      benchmark_path=benchmark_path, save_dir=simulated_frb_dir)
            # not None
            if frb_SPEGs is not None:
                test_frbs.append(cur_test_pulsar)
                # recored SPEGs
                test_SPEGs_from_frb_beam.extend(frb_SPEGs)

    # non-pulsar SPEGs, exclude all pulsar beams
    test_non_pulsar_SPEGs = []
    for index, row in df_test_full.iterrows():
        if row['filename'] not in pulsars_test:
            cur_SPEG = SPEG_Candidate(row)
            test_non_pulsar_SPEGs.append(cur_SPEG)
            # print("test_non_pulsar_SPEGs: ", len(test_non_pulsar_SPEGs))

    # extend returns None
    test_SPEGs_from_frb_beam.extend(test_non_pulsar_SPEGs)

    test_SPEGs_all = test_SPEGs_from_frb_beam
    print("test SPEGs count: ", len(test_SPEGs_all))

    # test_SPEGs_all = [each_SPEG for each_SPEG in test_SPEGs_all if each_SPEG.peak_SNR < 8]
    # print("test SPEGs count: ", len(test_SPEGs_all))

    print("ref: ", reference_frbs)
    print("test: ", test_frbs)
    ################ uniform
    # dependence_list = ['uniform', 'MI', 'psik']
    # for dependence in dependence_list:
    #     if dependence == 'uniform':
    #         outlyingness_list = [0]
    #     else:
    #         outlyingness_list = [0, 1, 2]
    #######################
    # dependence = 'uniform'
    # outlyingness_flag = 0

    # for outlyingness_flag in outlyingness_list:
    # exclude reference pulsars
    df_train_full_final = df_train_full.loc[~df_train_full['filename'].isin(reference_frbs)]
    df_train_full_final_astro = df_train_full_final.loc[df_train_full_final['class_label'] == 1]

    # the first astrophysical SPEG
    df_train_final_astro_head = df_train_full_final_astro.groupby('filename').first()

    print(df_train_full.shape)
    print(df_train_full_final.shape)
    print(df_train_full_final_astro.shape)
    print(df_train_final_astro_head.head())
    # exit()

    # exclude some columns
    df_train = df_train_full_final[df_train_full.columns.intersection(selected_columns)]

    X_train = df_train.loc[:, ~df_train.columns.isin(['class_label'])]
    y_train = df_train['class_label']


    ''' 
    repeat for every dependence, MI, psik, uniform 
    'MI'
    '''
    dependence = 'MI'
    outlyingness_list = [0, 1]

    # print(X_train.columns)
    if dependence == 'MI':
        mi = mutual_info_classif(X_train, y_train)
        print(mi)
        # weights (dc: dependence coefficients)
        dc_list = mi
    elif dependence == 'psik':
        # coefficient psik
        interval_cols = [col for col, v in data_types.items() if v == 'interval' and col in df_train.columns]
        print(interval_cols)
        phik_overview = df_train.phik_matrix(interval_cols=interval_cols, bins=bins)
        print(phik_overview)
        print(phik_overview['class_label'])
        dc_list = phik_overview['class_label'].tolist()[:-1]
    elif dependence == 'uniform':
        dc_list = np.ones(X_train.shape[1])

    for outlyingness_flag in outlyingness_list:
        for reference_SPEG in reference_frb_SPEGs:
            print("reference")
            print(reference_SPEG)

            outlyingness_scores = []
            for cur_feature in X_train.columns:
                # cur_feature = 'n_SPEGs_zero_DM'
                print("cur_feature: ", cur_feature)
                print("target_value: ", getattr(reference_SPEG, cur_feature))

                if outlyingness_flag == 1:
                    cur_outlyingness = get_outlyingness(feature=cur_feature, df=df_train_final_astro_head,
                                                        target_value=getattr(reference_SPEG, cur_feature))
                # elif outlyingness_flag == 2:
                #     cur_outlyingness = get_outlyingness2(feature=cur_feature, df=df_train_final_astro_head,
                #                                          target_value=getattr(reference_SPEG, cur_feature))
                elif outlyingness_flag == 0:
                    cur_outlyingness = 1

                outlyingness_scores.append(cur_outlyingness)

            cur_weights = np.multiply(outlyingness_scores, dc_list)
            # normalize
            # cur_weights = cur_weights / sum(cur_weights)
            print("\n cur outlyingness_scores:")
            print(outlyingness_scores)

            print("\n cur dc_list:")
            print(dc_list)

            print("\n cur weights:")
            print(cur_weights)

            # save weights
            weights_header = X_train.columns.tolist()
            weights_output_file = weights_dir + '/' + dependence + '_' + str(outlyingness_flag) + \
                                  '_fold_' + str(j) + '_weights.txt'

            with open(weights_output_file, 'a') as weights_fp:
                weights_fp.write(str(weights_header) + '\n')
                weights_fp.write(str(cur_weights) + '\n')

            output_test_SPEG_file = results_dir + '/' + str(run_id) + '_' + dependence + '_' + \
                                    str(outlyingness_flag) + '_fold_' + str(j) + '_results.txt'

            output_test_SPEG_fp = open(output_test_SPEG_file, 'a')
            print(output_test_SPEG_file)

            print("****** calculating similarity testing fold *******")
            pool = multiprocessing.Pool(processes=6)
            calc_similarity = partial(weighted_similarity, target_pulse=reference_SPEG, df=X_train, weights=cur_weights)

            test_SPEGs_ordered = pool.map(calc_similarity, test_SPEGs_all)
            # # sort test_SPEGs_sorted in descending order of similarity
            test_SPEGs_ordered.sort(key=lambda x: x.similarity, reverse=True)

            n_test_SPEGs = len(test_SPEGs_ordered)
            for k in range(n_test_SPEGs):
                cur_SPEG = test_SPEGs_ordered[k]
                cur_SPEG_line = str(k) + ',' + cur_SPEG.filename + ',' + str(cur_SPEG.SPEG_rank) + ',' \
                                + str(cur_SPEG.class_label) + ',' + str(cur_SPEG.peak_SNR) + ',' \
                                + str(cur_SPEG.centered_DM) + ',' + str(cur_SPEG.similarity) + ',' \
                                + str(reference_SPEG.filename) + ',' + str(reference_SPEG.SPEG_rank) + ',' \
                                + str(reference_SPEG.centered_DM) + ',' + str(reference_SPEG.peak_SNR)

                output_test_SPEG_fp.write(cur_SPEG_line + '\n')
            output_test_SPEG_fp.close()

            time.sleep(30)

    ''' 
    repeat for every dependence, MI, psik, uniform 
    'psik'
    '''
    dependence = 'psik'
    outlyingness_list = [0, 1]

    # print(X_train.columns)
    if dependence == 'MI':
        mi = mutual_info_classif(X_train, y_train)
        print(mi)
        # weights (dc: dependence coefficients)
        dc_list = mi
    elif dependence == 'psik':
        # coefficient psik
        interval_cols = [col for col, v in data_types.items() if v == 'interval' and col in df_train.columns]
        print(interval_cols)
        phik_overview = df_train.phik_matrix(interval_cols=interval_cols, bins=bins)
        print(phik_overview)
        print(phik_overview['class_label'])
        dc_list = phik_overview['class_label'].tolist()[:-1]
    elif dependence == 'uniform':
        dc_list = np.ones(X_train.shape[1])

    for outlyingness_flag in outlyingness_list:
        for reference_SPEG in reference_frb_SPEGs:
            print("reference")
            print(reference_SPEG)

            outlyingness_scores = []
            for cur_feature in X_train.columns:
                # cur_feature = 'n_SPEGs_zero_DM'
                print("cur_feature: ", cur_feature)
                print("target_value: ", getattr(reference_SPEG, cur_feature))

                if outlyingness_flag == 1:
                    cur_outlyingness = get_outlyingness(feature=cur_feature, df=df_train_final_astro_head,
                                                        target_value=getattr(reference_SPEG, cur_feature))
                # elif outlyingness_flag == 2:
                #     cur_outlyingness = get_outlyingness2(feature=cur_feature, df=df_train_final_astro_head,
                #                                          target_value=getattr(reference_SPEG, cur_feature))
                elif outlyingness_flag == 0:
                    cur_outlyingness = 1

                outlyingness_scores.append(cur_outlyingness)

            cur_weights = np.multiply(outlyingness_scores, dc_list)
            # normalize
            # cur_weights = cur_weights / sum(cur_weights)
            print("\n cur outlyingness_scores:")
            print(outlyingness_scores)

            print("\n cur dc_list:")
            print(dc_list)

            print("\n cur weights:")
            print(cur_weights)

            # save weights
            weights_header = X_train.columns.tolist()
            weights_output_file = weights_dir + '/' + dependence + '_' + str(outlyingness_flag) + \
                                  '_fold_' + str(j) + '_weights.txt'

            with open(weights_output_file, 'a') as weights_fp:
                weights_fp.write(str(weights_header) + '\n')
                weights_fp.write(str(cur_weights) + '\n')

            output_test_SPEG_file = results_dir + '/' + str(run_id) + '_' + dependence + '_' + \
                                    str(outlyingness_flag) + '_fold_' + str(j) + '_results.txt'

            output_test_SPEG_fp = open(output_test_SPEG_file, 'a')
            print(output_test_SPEG_file)

            print("****** calculating similarity testing fold *******")
            pool = multiprocessing.Pool(processes=6)
            calc_similarity = partial(weighted_similarity, target_pulse=reference_SPEG, df=X_train, weights=cur_weights)

            test_SPEGs_ordered = pool.map(calc_similarity, test_SPEGs_all)
            # # sort test_SPEGs_sorted in descending order of similarity
            test_SPEGs_ordered.sort(key=lambda x: x.similarity, reverse=True)

            n_test_SPEGs = len(test_SPEGs_ordered)
            for k in range(n_test_SPEGs):
                cur_SPEG = test_SPEGs_ordered[k]
                cur_SPEG_line = str(k) + ',' + cur_SPEG.filename + ',' + str(cur_SPEG.SPEG_rank) + ',' \
                                + str(cur_SPEG.class_label) + ',' + str(cur_SPEG.peak_SNR) + ',' \
                                + str(cur_SPEG.centered_DM) + ',' + str(cur_SPEG.similarity) + ',' \
                                + str(reference_SPEG.filename) + ',' + str(reference_SPEG.SPEG_rank) + ',' \
                                + str(reference_SPEG.centered_DM) + ',' + str(reference_SPEG.peak_SNR)

                output_test_SPEG_fp.write(cur_SPEG_line + '\n')
            output_test_SPEG_fp.close()

            time.sleep(30)

    ''' 
    repeat for every dependence, uniform, MI, psik
    'uniform'
    '''
    dependence = 'uniform'
    outlyingness_flag = 0

    # print(X_train.columns)
    if dependence == 'MI':
        mi = mutual_info_classif(X_train, y_train)
        print(mi)
        # weights (dc: dependence coefficients)
        dc_list = mi
    elif dependence == 'psik':
        # coefficient psik
        interval_cols = [col for col, v in data_types.items() if v == 'interval' and col in df_train.columns]
        print(interval_cols)
        phik_overview = df_train.phik_matrix(interval_cols=interval_cols, bins=bins)
        print(phik_overview)
        print(phik_overview['class_label'])
        dc_list = phik_overview['class_label'].tolist()[:-1]
    elif dependence == 'uniform':
        dc_list = np.ones(X_train.shape[1])

    for reference_SPEG in reference_frb_SPEGs:
        print("reference")
        print(reference_SPEG)

        outlyingness_scores = []
        for cur_feature in X_train.columns:
            # cur_feature = 'n_SPEGs_zero_DM'
            print("cur_feature: ", cur_feature)
            print("target_value: ", getattr(reference_SPEG, cur_feature))

            if outlyingness_flag == 1:
                cur_outlyingness = get_outlyingness(feature=cur_feature, df=df_train_final_astro_head,
                                                    target_value=getattr(reference_SPEG, cur_feature))
            # elif outlyingness_flag == 2:
            #     cur_outlyingness = get_outlyingness2(feature=cur_feature, df=df_train_final_astro_head,
            #                                          target_value=getattr(reference_SPEG, cur_feature))
            elif outlyingness_flag == 0:
                cur_outlyingness = 1

            outlyingness_scores.append(cur_outlyingness)

        cur_weights = np.multiply(outlyingness_scores, dc_list)
        # normalize
        # cur_weights = cur_weights / sum(cur_weights)
        print("\n cur outlyingness_scores:")
        print(outlyingness_scores)

        print("\n cur dc_list:")
        print(dc_list)

        print("\n cur weights:")
        print(cur_weights)

        # save weights
        weights_header = X_train.columns.tolist()
        weights_output_file = weights_dir + '/' + dependence + '_' + str(outlyingness_flag) + \
                              '_fold_' + str(j) + '_weights.txt'

        with open(weights_output_file, 'a') as weights_fp:
            weights_fp.write(str(weights_header) + '\n')
            weights_fp.write(str(cur_weights) + '\n')

        output_test_SPEG_file = results_dir + '/' + str(run_id) + '_' + dependence + '_' + \
                                str(outlyingness_flag) + '_fold_' + str(j) + '_results.txt'

        output_test_SPEG_fp = open(output_test_SPEG_file, 'a')
        print(output_test_SPEG_file)

        print("****** calculating similarity testing fold *******")
        pool = multiprocessing.Pool(processes=6)
        calc_similarity = partial(weighted_similarity, target_pulse=reference_SPEG, df=X_train, weights=cur_weights)

        test_SPEGs_ordered = pool.map(calc_similarity, test_SPEGs_all)
        # # sort test_SPEGs_sorted in descending order of similarity
        test_SPEGs_ordered.sort(key=lambda x: x.similarity, reverse=True)

        n_test_SPEGs = len(test_SPEGs_ordered)
        for k in range(n_test_SPEGs):
            cur_SPEG = test_SPEGs_ordered[k]
            cur_SPEG_line = str(k) + ',' + cur_SPEG.filename + ',' + str(cur_SPEG.SPEG_rank) + ',' \
                            + str(cur_SPEG.class_label) + ',' + str(cur_SPEG.peak_SNR) + ',' \
                            + str(cur_SPEG.centered_DM) + ',' + str(cur_SPEG.similarity) + ',' \
                            + str(reference_SPEG.filename) + ',' + str(reference_SPEG.SPEG_rank) + ',' \
                            + str(reference_SPEG.centered_DM) + ',' + str(reference_SPEG.peak_SNR)

            output_test_SPEG_fp.write(cur_SPEG_line + '\n')
        output_test_SPEG_fp.close()

        time.sleep(30)


if __name__ == "__main__":
    X_source = 'PALFA'
    speg_type = "all_with_central_peak"
    number_of_folds = 3

    double_DM = True
    snr_min = 8
    snr_max = 60
    snr_step = 4
    max_astro_pulses = 400
    DM_min = 50.0

    run_id = 11

    # benchmark_path = "/home/user/Documents/AstroData/Benchmark/PALFA_Pulsars"

    # simulated_frb_dir = simulated_frb_dir + '/' + str(run_id)

    # TODO: save weights
    # weights_dir = "/home/beaver/PycharmProjects/RS_FRB_Dim2/calculated_weights/" + str(run_id)
    #
    # results_dir = "/home/beaver/PycharmProjects/RS_FRB_Dim2/results/final"

    # parser = argparse.ArgumentParser()
    #
    # parser.add_argument("run_id")
    # parser.add_argument("valid_fold", type=int, choices=[0, 1, 2, 3])
    # parser.add_argument("dependence", choices=['MI', 'psik', 'uniform'])
    # parser.add_argument("outlyingness", type=int, choices=[1, 2, 0])
    #
    # args = parser.parse_args()
    # print("run_id: ", args.run_id)
    # run_id = args.run_id
    #
    # print("validation fold: ", args.valid_fold)
    # i = args.valid_fold
    #
    # j = (i + 1) % num_of_folds
    #
    # print("dependence measure: ", args.dependence)
    # dependence = args.dependence
    #
    # print("outlyingness flag: ", args.outlyingness)
    # outlyingness_flag = args.outlyingness

    selected_columns = feature_list.copy()
    selected_columns.append('class_label')
    start = time.perf_counter()

    # output_valid_SPEG_file = results_dir + '/' + run_id + '_' + dependence + '_' + str(outlyingness_flag) + "_valid.txt"
    # output_valid_SPEG_fp = open(output_valid_SPEG_file, 'a')
    #
    # output_test_SPEG_file = results_dir + '/' + run_id + '_' + dependence + '_' + str(outlyingness_flag) + "_test.txt"
    # output_test_SPEG_fp = open(output_test_SPEG_file, 'a')

    main()
    # output_valid_SPEG_fp.close()
    # output_test_SPEG_fp.close()

    finish = time.perf_counter()
    print(f'Finished in {round(finish - start, 2)} seconds(s)')