import pandas as pd
import numpy as np
import mislabeling_tests
import CLOUD
import datetime
from joblib import Parallel, delayed
from multiprocessing import cpu_count

all_multi_taxa = [1.5, 3, 5, 10, 20, 100]
freq_of_taxa_to_change = 0.01
def contamination_test(normal_df, anomaly_df, normals_subsample_size, anomalies_subsample_size,
                     weights_to_test, replacement_to_test, subsample_size_to_test, pc_method_to_test, repeats = 100,
                     number_of_trees = 100, max_depth = 100, min_samples_to_split = 2, splitting_method = "pcoa",
                     paral = True, cpu = None, verbose = True):
    all_multi_anomalies_list = []
    if cpu == None:
        cpu = cpu_count()
    for cur_multi_taxa in all_multi_taxa:
        anomalies_list = []
        for replacement in replacement_to_test:
            for weights in weights_to_test:
                for subsample_size in subsample_size_to_test:
                    for pc_method in pc_method_to_test:
                        if verbose:
                            now = datetime.datetime.now()
                            print("starting:", cur_multi_taxa, replacement, weights, subsample_size, pc_method, "Current time: ", now.time())
                        if paral:
                            tmp_anomaly_list = Parallel(n_jobs=cpu)(
                                delayed(contamination_single_test_single_taxon)(normal_df = normal_df, anomaly_df = anomaly_df,
                                                        normals_subsample_size = normals_subsample_size, anomalies_subsample_size = anomalies_subsample_size,
                                                        weights = weights, replacement = replacement, subsample_size = subsample_size, pc_method = pc_method,
                                                        number_of_trees = number_of_trees, max_depth = max_depth, min_samples_to_split = min_samples_to_split, splitting_method = splitting_method,
                                                        paral=paral, cpu=cpu, seed = i, multi_taxa = cur_multi_taxa) for i in range(repeats))
                            for i in range(repeats):
                                tmp_anomaly_list[i]["iteration"] = i
                                anomalies_list.append(tmp_anomaly_list[i])
                        else:
                            for i in range(repeats):
                                anomalies_list.append(contamination_single_test_single_taxon(normal_df = normal_df, anomaly_df = anomaly_df,
                                                    normals_subsample_size = normals_subsample_size, anomalies_subsample_size = anomalies_subsample_size,
                                                    weights = weights, replacement = replacement, subsample_size = subsample_size, pc_method = pc_method,
                                                    number_of_trees = number_of_trees, max_depth = max_depth, min_samples_to_split = min_samples_to_split, splitting_method = splitting_method,
                                                    paral=paral, cpu=cpu, seed = i, multi_taxa = cur_multi_taxa))
                                anomalies_list[-1]["iteration"] = i
                        if verbose:
                            now = datetime.datetime.now()
                            print("finished:", replacement, weights, subsample_size, pc_method, "Current time: ", now.time())

        anomaly_summary = pd.concat(anomalies_list, axis  = 0)
        anomaly_summary.reset_index(drop = True, inplace = True)
        anomaly_summary["multi_taxa"] = cur_multi_taxa
        all_multi_anomalies_list.append(anomaly_summary)

    anomaly_summary_all = pd.concat(all_multi_anomalies_list, axis  = 0)

    return anomaly_summary_all


def contamination_single_test_single_taxon(normal_df, anomaly_df, normals_subsample_size, anomalies_subsample_size,
                            weights, replacement, subsample_size, pc_method,
                            number_of_trees, max_depth, min_samples_to_split, splitting_method, multi_taxa = 5,
                            paral = True, cpu = None, seed = None):
    np.random.seed(seed)

    normal_df = normal_df + 0.5

    samples_to_use = normal_df.sample(normals_subsample_size + anomalies_subsample_size)
    normal_samples = samples_to_use.iloc[:normals_subsample_size]
    anomaly_samples = samples_to_use.iloc[normals_subsample_size:]

    col_list = list(anomaly_samples.columns)
    np.random.shuffle(col_list)
    random_taxa =  col_list[:np.max([int(anomaly_samples.shape[1] * freq_of_taxa_to_change), 5])]
    anomaly_samples.loc[:, random_taxa] = anomaly_samples.loc[:, random_taxa] * multi_taxa


    normal_samples = normal_samples.div(normal_samples.sum(axis=1), axis=0)
    anomaly_samples = anomaly_samples.div(anomaly_samples.sum(axis=1), axis=0)
    cur_comt_df = mislabeling_tests.mislabeling_single_test(normal_samples, anomaly_samples,
                                              normals_subsample_size, anomalies_subsample_size,
                                              weights, replacement, subsample_size, pc_method,
                                              number_of_trees, max_depth, min_samples_to_split, splitting_method,
                                              paral, cpu, seed)
    return cur_comt_df