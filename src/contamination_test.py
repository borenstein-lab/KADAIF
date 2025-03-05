import pandas as pd
import numpy as np
import mislabeling_tests
import CLOUD
import datetime
from joblib import Parallel, delayed
from multiprocessing import cpu_count

def contamination_test(normal_df, anomaly_df, normals_subsample_size, anomalies_subsample_size,
                     precentages_to_test,
                     weights_to_test, replacement_to_test, subsample_size_to_test, pc_method_to_test, repeats = 100,
                     number_of_trees = 100, max_depth = 100, min_samples_to_split = 2, splitting_method = "pcoa",
                     paral = True, cpu = None, verbose = True):
    anomalies_list = []
    if cpu == None:
        cpu = cpu_count()
    for replacement in replacement_to_test:
        for weights in weights_to_test:
            for subsample_size in subsample_size_to_test:
                for pc_method in pc_method_to_test:
                    for precentage in precentages_to_test:
                        if verbose:
                            now = datetime.datetime.now()
                            print("starting:", replacement, weights, subsample_size, pc_method, precentage, "Current time: ", now.time())
                        if paral:
                            tmp_anomaly_list = Parallel(n_jobs=cpu)(
                                delayed(contamination_single_test)(normal_df = normal_df, anomaly_df = anomaly_df,
                                                        normals_subsample_size = normals_subsample_size, anomalies_subsample_size = anomalies_subsample_size,
                                                        anomaly_pre = precentage,
                                                        weights = weights, replacement = replacement, subsample_size = subsample_size, pc_method = pc_method,
                                                        number_of_trees = number_of_trees, max_depth = max_depth, min_samples_to_split = min_samples_to_split, splitting_method = splitting_method,
                                                        paral=paral, cpu=cpu, seed = i) for i in range(repeats))
                            for i in range(repeats):
                                tmp_anomaly_list[i]["iteration"] = i
                                anomalies_list.append(tmp_anomaly_list[i])
                        else:
                            for i in range(repeats):
                                anomalies_list.append(contamination_single_test(normal_df = normal_df, anomaly_df = anomaly_df,
                                                    normals_subsample_size = normals_subsample_size, anomalies_subsample_size = anomalies_subsample_size,
                                                    anomaly_pre=precentage,
                                                    weights = weights, replacement = replacement, subsample_size = subsample_size, pc_method = pc_method,
                                                    number_of_trees = number_of_trees, max_depth = max_depth, min_samples_to_split = min_samples_to_split, splitting_method = splitting_method,
                                                    paral=paral, cpu=cpu, seed = i))
                                anomalies_list[-1]["iteration"] = i


                        if verbose:
                            now = datetime.datetime.now()
                            print("finished:", replacement, weights, subsample_size, pc_method, "Current time: ", now.time())

    anomaly_summary = pd.concat(anomalies_list, axis  = 0)
    anomaly_summary.reset_index(drop = True, inplace = True)
    return anomaly_summary


def contamination_single_test(normal_df, anomaly_df, normals_subsample_size, anomalies_subsample_size, anomaly_pre,
                            weights, replacement, subsample_size, pc_method,
                            number_of_trees, max_depth, min_samples_to_split, splitting_method,
                            paral = True, cpu = None, seed = None):
    np.random.seed(seed)
    normal_subsample = normal_df.sample(normals_subsample_size + anomalies_subsample_size)
    anomaly_subsample = anomaly_df.sample(anomalies_subsample_size)

    normal_samples = normal_subsample.iloc[:normals_subsample_size]
    tmp_ind = "normal:" + normal_subsample.iloc[normals_subsample_size:].index.astype(str) + " anomaly:" + anomaly_subsample.index.astype(str)

    anomaly_samples = normal_subsample.iloc[normals_subsample_size:].reset_index(drop=True) * (1 - anomaly_pre) + anomaly_subsample.reset_index(drop=True) * anomaly_pre
    anomaly_samples.index = tmp_ind

    cur_comt_df = mislabeling_tests.mislabeling_single_test(normal_samples, anomaly_samples,
                                              normals_subsample_size, anomalies_subsample_size,
                                              weights, replacement, subsample_size, pc_method,
                                              number_of_trees, max_depth, min_samples_to_split, splitting_method,
                                              paral, cpu, seed)
    cur_comt_df["anomaly_percentage"] = anomaly_pre
    return cur_comt_df

