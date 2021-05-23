import numpy as np
from feature_info import get_sim_value_SPEGs, feature_list


def weighted_similarity(candidate_SPEG=None, target_pulse=None, df=None, weights=None):
    # calculate here
    # pulse_features = target_pulse.get_all_attr()
    # candidate_features = candidate_SPEG.get_all_attr()

    sim_values = []
    for cur_feature in feature_list:
        cur_sim_value = get_sim_value_SPEGs(feature = cur_feature, df=df, target_SPEG=target_pulse,
                                            candidate_SPEG=candidate_SPEG)
        sim_values.append(cur_sim_value)

    # divide by sum of the weights
    cur_similarity = sum(np.multiply(weights, sim_values)) / sum(weights)
    # print("cur_similarity: ", cur_similarity)

    candidate_SPEG.similarity = cur_similarity
    return candidate_SPEG

