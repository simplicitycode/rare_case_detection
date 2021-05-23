from feature_info import feature_list

class SPEG_Candidate(object):
    """
    This is the class of single-pulse events group (SPEG) candidate with a list of features, compared with SPEG,
    SPEG_Candidate_FRB also includes self.center_startDM, self.center_stopDM which are used to re-group the SPEGs


    filename: in which the SEPG is identified
    SPEG_rank: rank of the SPEG by the maximum SNR in decreasing order
    group_rank: rank of the SPEG group by group_SNR_max in decreasing order
    group_max_SNR: maximum SNR of the brightest SPEG within the SPEG group
    group_median_SNR: median of the maximum SNRs of SPEGs within the SPEG group

    peak_SNR: SNR (Signal-to-Noise Ratio) of the brightest trial single-pulse event within SPEG
    centered_DM: for a regular SPEG, it's peak_DM; for a clipped SPEG, the fitted central DM (refer to the paper)
    clipped_SPEG: boolean value representing whether the cluster (or SPEG) is clipped
    SNR_sym_index: numerical value measuring the symmetry of the SPEG by SNR
    DM_sym_index: numerical value measuring the symmetry of the SPEG by DM
    peak_score: peak score of the SNR vs. DM curve the SPEG

    bright_recur_times: number of bright SPEGs in the group
    recur_times: total number of SPEGs in the group
    size_ratio: the ratio between number of spes and DM channels within SPEG

    cluster_density: density of spe cluster of the beam
    DM_range: DM range of the SPEG
    time_range: time DM range of the SPEG
    pulse_width: pulse width from boxcar match filtering
    time_ratio: ratio of timewidth (SPEG’s width in time (maximum time of the SPEG minus minimum time)) over pulse_width

    # n_SPEGs_zero_DM: number of SPEGs appearing at close time and lower DM.
    # n_brighter_SPEGs_zero_DM: number of brighter SPEGs appearing at close time and lower DM.
    # SPEGs_zero_DM_min_DM: minimum DM of SPEGs_zero_DM.
    # brighter_SPEGs_zero_DM_peak_DM: peakDM of the brightest SPEGs_BrRFI .
    # brighter_SPEGs_zero_DM_peak_SNR: Peak S/N of the brightest SPEGs_BrRFI.

    n_SPEGs_zero_DM: number of SPEGs appearing at close time and lower DM
    n_brighter_SPEGs_zero_DM: number of brighter SPEGs appearing at close time and lower DM
    SPEGs_zero_DM_min_DM: minimum DM of SPEGs appearing at close time and lower DM
    brighter_SPEGs_zero_DM_peak_DM: peak DM of the brightest SPEGs appearing at close time and lower DM
    brighter_SPEGs_zero_DM_peak_SNR: maximum SNR of the brightest SPEGs appearing at close time and lower DM
    """
    __slots__ = ['filename', 'SPEG_rank', 'group_rank', 'group_max_SNR',
                 'group_median_SNR', 'peak_SNR', 'centered_DM', 'clipped_SPEG',
                 'SNR_sym_index', 'DM_sym_index', 'peak_score', 'bright_recur_times',
                 'recur_times', 'size_ratio', 'cluster_density', 'DM_range',
                 'time_range', 'pulse_width', 'time_ratio', 'n_SPEGs_zero_DM',
                 'n_brighter_SPEGs_zero_DM', 'SPEGs_zero_DM_min_DM',
                 'brighter_SPEGs_zero_DM_peak_DM', 'brighter_SPEGs_zero_DM_peak_SNR',
                 'class_label', 'center_startDM','center_stopDM', 'min_DM', 'max_DM',
                 'peak_time', 'min_time', 'max_time',
                 'group_peak_DM', 'similarity', 'grouped']

    def __init__(self, current_list):
        """
        SinglePulseEventGroup constructor.
        :param current_list: a list of attributes of the brightest trial single pulse event within the cluster
        """
        self.filename = current_list[0]
        self.SPEG_rank = current_list[1]
        self.group_rank = current_list[2]
        self.group_max_SNR = current_list[3]
        self.group_median_SNR = current_list[4]

        self.peak_SNR = current_list[5]
        self.centered_DM = current_list[6]
        self.clipped_SPEG = current_list[7]
        self.SNR_sym_index = current_list[8]
        self.DM_sym_index = current_list[9]
        self.peak_score = current_list[10]

        self.bright_recur_times = current_list[11]
        self.recur_times = current_list[12]
        self.size_ratio = current_list[13]
        self.cluster_density = current_list[14]

        self.DM_range = current_list[15]
        self.time_range = current_list[16]
        self.pulse_width = current_list[17]
        self.time_ratio = current_list[18]

        self.n_SPEGs_zero_DM = current_list[19]
        self.n_brighter_SPEGs_zero_DM = current_list[20]
        self.SPEGs_zero_DM_min_DM = current_list[21]
        self.brighter_SPEGs_zero_DM_peak_DM = current_list[22]
        self.brighter_SPEGs_zero_DM_peak_SNR = current_list[23]

        self.class_label =current_list[24]

        self.center_startDM = current_list[25]
        self.center_stopDM = current_list[26]

        self.min_DM = current_list[27]
        self.max_DM = current_list[28]
        self.peak_time = current_list[29]

        self.min_time = current_list[30]
        self.max_time = current_list[31]

        self.group_peak_DM = -1
        self.similarity = -1
        self.grouped = False

    def get_all_attr(self):
        feature_values = []
        for each_feature in feature_list:
            each_value = getattr(self, each_feature)
            feature_values.append(each_value)
        return feature_values

        # return[self.SPEG_rank, self.group_rank, self.group_max_SNR, self.group_median_SNR,
        #        self.peak_SNR, self.centered_DM, self.clipped_SPEG, self.SNR_sym_index,
        #        self.DM_sym_index, self.peak_score, self.bright_recur_times, self.recur_times,
        #        self.size_ratio, self.cluster_density, self.DM_range, self.time_range,
        #        self.pulse_width, self.time_ratio, self.n_SPEGs_zero_DM,
        #        self.n_brighter_SPEGs_zero_DM]

    def __str__(self):
        # print cur_cluster
        s = ["\tfilename %s " % self.filename,
             "\tSPEG_rank %5d " % self.SPEG_rank,
             "\tgroup_rank %5d " % self.group_rank,
             "\tgroup_max_SNR %5.2f " % self.group_max_SNR,
             "\tgroup_median_SNR %5.2f " % self.group_median_SNR,
             "\tpeak_SNR %5.2f " % self.peak_SNR,
             "\tcentered_DM %5.2f " % self.centered_DM,
             "\tclipped_SPEG %5d " % self.clipped_SPEG,
             "\tSNR_sym_index %5.4f " % self.SNR_sym_index,
             "\tDM_sym_index %5.4f " % self.DM_sym_index,
             "\tpeak_score %5d " % self.peak_score,
             "\tbright_recur_times %5d " % self.bright_recur_times,
             "\trecur_times %5d " % self.recur_times,
             "\tsize_ratio %5.2f " % self.size_ratio,
             "\tcluster_density %5.4f " % self.cluster_density,
             "\tDM_range %5.2f " % self.DM_range,
             "\ttime_range %5.6f " % self.time_range,
             "\tpulse_width %5.6f " % self.pulse_width,
             "\ttime_ratio %5.3f " % self.time_ratio,
             "\tn_SPEGs_zero_DM %5d " % self.n_SPEGs_zero_DM,
             "\tn_brighter_SPEGs_zero_DM %5d " % self.n_brighter_SPEGs_zero_DM,
             "\tSPEGs_zero_DM_min_DM %5.2f " % self.SPEGs_zero_DM_min_DM,
             "\tbrighter_SPEGs_zero_DM_peak_DM %5.2f " % self.brighter_SPEGs_zero_DM_peak_DM,
             "\tbrighter_SPEGs_zero_DM_peak_SNR %5.2f " % self.brighter_SPEGs_zero_DM_peak_SNR,
             "\tclass_label %5d " % self.class_label,
             "\tsimilarity %5.8f " % self.similarity,
             "--------------------------------"
        ]
        return '\n'.join(s)