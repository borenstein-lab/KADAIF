import pandas as pd
import numpy as np
import datetime
import MicrobiomeIsolationForest
import CLOUD
from sklearn.ensemble import IsolationForest
from joblib import Parallel, delayed
from multiprocessing import cpu_count


#we assume that the index is the df name
def mislabeling_test(normal_df, anomaly_df, normals_subsample_size, anomalies_subsample_size,
                     weights_to_test, replacement_to_test, subsample_size_to_test, pc_method_to_test, repeats = 100, splitting_method = "pcoa",
                     number_of_trees = 100, max_depth = 100, min_samples_to_split = 2,
                     paral = True, cpu = None, verbose = True):
    anomalies_list = []
    if cpu == None:
        cpu = cpu_count()
    for replacement in replacement_to_test:
        for weights in weights_to_test:
            for subsample_size in subsample_size_to_test:
                for pc_method in pc_method_to_test:
                    if verbose:
                        now = datetime.datetime.now()
                        print("starting:", replacement, weights, subsample_size, pc_method, "Current time: ", now.time())
                    if paral:
                        tmp_anomaly_list = Parallel(n_jobs=cpu)(
                            delayed(mislabeling_single_test)(normal_df = normal_df, anomaly_df = anomaly_df,
                                                    normals_subsample_size = normals_subsample_size, anomalies_subsample_size = anomalies_subsample_size,
                                                    weights = weights, replacement = replacement, subsample_size = subsample_size, pc_method = pc_method,
                                                    number_of_trees = number_of_trees, max_depth = max_depth, min_samples_to_split = min_samples_to_split,
                                                             splitting_method = splitting_method,
                                                    paral=paral, cpu=cpu, seed = i) for i in range(repeats))
                        for i in range(repeats):
                            tmp_anomaly_list[i]["iteration"] = i
                            anomalies_list.append(tmp_anomaly_list[i])
                    else:
                        for i in range(repeats):
                            anomalies_list.append(mislabeling_single_test(normal_df = normal_df, anomaly_df = anomaly_df,
                                                normals_subsample_size = normals_subsample_size, anomalies_subsample_size = anomalies_subsample_size,
                                                weights = weights, replacement = replacement, subsample_size = subsample_size, pc_method = pc_method,
                                                number_of_trees = number_of_trees, max_depth = max_depth, min_samples_to_split = min_samples_to_split,
                                                                          splitting_method=splitting_method,
                                                paral=paral, cpu=cpu, seed = i))
                            anomalies_list[-1]["iteration"] = i


                    if verbose:
                        now = datetime.datetime.now()
                        print("finished:", replacement, weights, subsample_size, pc_method, "Current time: ", now.time())

    anomaly_summary = pd.concat(anomalies_list, axis  = 0)
    anomaly_summary.reset_index(drop = True, inplace = True)
    return anomaly_summary



def mislabeling_single_test(normal_df, anomaly_df, normals_subsample_size, anomalies_subsample_size,
                            weights, replacement, subsample_size, pc_method,
                            number_of_trees, max_depth, min_samples_to_split, splitting_method,
                            paral = True, cpu = None, seed = None):
    np.random.seed(seed)
    normal_subsample = normal_df.sample(normals_subsample_size)
    anomaly_subsample = anomaly_df.sample(anomalies_subsample_size)
    data_to_test = pd.concat([normal_subsample, anomaly_subsample], axis=0)
    mod = MicrobiomeIsolationForest.MicrobiomeIsolationForest(number_of_trees = number_of_trees, max_depth = max_depth,min_samples_to_split=min_samples_to_split,
                                                              pc_method = pc_method, subsample_size=subsample_size, replacement= replacement, weights = weights,
                                                              splitting_method = splitting_method,
                                                              paral = paral, cpu = cpu)
    anomalies = mod.fit_transform(data_to_test)
    anomalies_df = pd.DataFrame(anomalies, index = data_to_test.index)
    anomalies_df.reset_index(drop=False, inplace=True)
    anomalies_df.columns = ["SampleID", "score"]
    anomalies_df["depth"] = [mod.samples_dict_depth[sample] for sample in anomalies_df["SampleID"]]
    anomalies_df["anomaly"] = [False if j < normals_subsample_size else True for j in range(normals_subsample_size + anomalies_subsample_size)]
    anomalies_df["weights"] = weights
    anomalies_df["replacement"] = replacement
    anomalies_df["subsample_size"] = subsample_size
    anomalies_df["pc_method"] = pc_method

    iso_mod = IsolationForest()
    anomalies_df["iso"] = list(-iso_mod.fit(data_to_test).score_samples(data_to_test))

    ra_data = data_to_test.div(data_to_test.sum(axis=1), axis=0)
    cloud_res = CLOUD.CLOUD(ra_data)
    cloud_res_df = pd.DataFrame.from_dict(cloud_res[0], orient="index").reset_index(drop=False, inplace=False)
    cloud_res_df.columns = ["SampleID", "CLOUD_score"]
    anomalies_df = pd.merge(anomalies_df, cloud_res_df, on = "SampleID")


    return anomalies_df




######################NEW##############################
def mislabeling_test_subsampling(normal_df, anomaly_df, normals_subsample_size, anomalies_subsample_size,
                                weights_to_test, replacement_to_test, subsample_size_to_test, pc_method_to_test, repeats = 10,
                                number_of_trees = 100, max_depth = 100, min_samples_to_split = 2,
                                paral = True, cpu = None, verbose = True, seed = None,
                                different_forests_in_subsamples = 20, subsampling_frac_in_forest = 0.5):
    np.random.seed(seed)
    if paral:
        tmp_anomaly_list = Parallel(n_jobs=cpu)(
            delayed(mislabeling_test_subsampling_single_repeat)(normal_df =normal_df, anomaly_df = anomaly_df,
                                               normals_subsample_size = normals_subsample_size, anomalies_subsample_size = anomalies_subsample_size,
                                               different_forests_in_subsamples = different_forests_in_subsamples,
                                               subsampling_frac_in_forest = subsampling_frac_in_forest,
                                               weights_to_test = weights_to_test, replacement_to_test = replacement_to_test,
                                               subsample_size_to_test = subsample_size_to_test, pc_method_to_test = pc_method_to_test,
                                               number_of_trees = pc_method_to_test, max_depth = max_depth, min_samples_to_split = min_samples_to_split,
                                               paral = paral, cpu = cpu, verbose = verbose,
                                                                iteration = repeat) for repeat in range(repeats))
    else:
        tmp_anomaly_list = []
        for repeat in range(repeats):
            tmp_anomaly_list.append(mislabeling_test_subsampling_single_repeat(normal_df =normal_df, anomaly_df = anomaly_df,
                                               normals_subsample_size = normals_subsample_size, anomalies_subsample_size = anomalies_subsample_size,
                                               different_forests_in_subsamples = different_forests_in_subsamples,
                                               subsampling_frac_in_forest = subsampling_frac_in_forest,
                                               weights_to_test = weights_to_test, replacement_to_test = replacement_to_test,
                                               subsample_size_to_test = subsample_size_to_test, pc_method_to_test = pc_method_to_test,
                                               number_of_trees = pc_method_to_test, max_depth = max_depth, min_samples_to_split = min_samples_to_split,
                                               paral = paral, cpu = cpu, verbose = verbose,
                                                iteration = repeat))
    anomaly_summary = pd.concat(tmp_anomaly_list, axis=0)
    return anomaly_summary



def mislabeling_test_subsampling_single_repeat(normal_df, anomaly_df, normals_subsample_size, anomalies_subsample_size,
                                               different_forests_in_subsamples, subsampling_frac_in_forest,
                                               weights_to_test,
                                               replacement_to_test, subsample_size_to_test, pc_method_to_test, number_of_trees, max_depth,
                                               min_samples_to_split, paral, cpu, verbose,
                                               iteration):
    np.random.seed(iteration)
    normal_subsample = normal_df.sample(normals_subsample_size)
    anomaly_subsample = anomaly_df.sample(anomalies_subsample_size)

    samples_num_to_draw_in_each_forest = int(
        (normals_subsample_size + anomalies_subsample_size) * subsampling_frac_in_forest)
    normals_prop_in_subsample = float(normals_subsample_size) / (normals_subsample_size + anomalies_subsample_size)
    if paral:
        tmp_anomaly_list = Parallel(n_jobs=cpu)(
            delayed(mislabeling_test_subsampling_single_forest)(normal_subsample=normal_subsample, anomaly_subsample=anomaly_subsample,
                                             weights_to_test=weights_to_test, replacement_to_test=replacement_to_test,
                                             subsample_size_to_test=subsample_size_to_test, pc_method_to_test = pc_method_to_test,
                                             number_of_trees=number_of_trees, max_depth=max_depth, min_samples_to_split=min_samples_to_split,
                                             paral=paral, cpu=cpu, verbose = verbose,
                                             samples_num_to_draw_in_each_forest = samples_num_to_draw_in_each_forest,
                                             normals_prop_in_subsample = normals_prop_in_subsample, iteration = iteration,
                                              forest_num= i ) for i in range(different_forests_in_subsamples))
    else:
        tmp_anomaly_list = []
        for i in range(different_forests_in_subsamples):
            tmp_anomaly_list.append(mislabeling_test_subsampling_single_forest(normal_subsample=normal_subsample, anomaly_subsample=anomaly_subsample,
                                             weights_to_test=weights_to_test, replacement_to_test=replacement_to_test,
                                             subsample_size_to_test=subsample_size_to_test, pc_method_to_test = pc_method_to_test,
                                             number_of_trees=number_of_trees, max_depth=max_depth, min_samples_to_split=min_samples_to_split,
                                             paral=paral, cpu=cpu, verbose = verbose,
                                             samples_num_to_draw_in_each_forest = samples_num_to_draw_in_each_forest,
                                             normals_prop_in_subsample = normals_prop_in_subsample, iteration = iteration,
                                              forest_num= i ))
    anomaly_summary = pd.concat(tmp_anomaly_list, axis=0)
    return anomaly_summary

def mislabeling_test_subsampling_single_forest(normal_subsample, anomaly_subsample, weights_to_test,
                                               replacement_to_test, subsample_size_to_test, pc_method_to_test, number_of_trees, max_depth,
                                               min_samples_to_split, paral, cpu, verbose,
                                               samples_num_to_draw_in_each_forest, normals_prop_in_subsample, forest_num, iteration):
    np.random.seed(forest_num)
    normals_in_cur_forest = np.random.binomial(samples_num_to_draw_in_each_forest, normals_prop_in_subsample)
    anomaly_summary = mislabeling_test(normal_df=normal_subsample, anomaly_df=anomaly_subsample,
                    normals_subsample_size=normals_in_cur_forest,
                    anomalies_subsample_size=samples_num_to_draw_in_each_forest - normals_in_cur_forest,
                    weights_to_test=weights_to_test, replacement_to_test=replacement_to_test,
                    subsample_size_to_test=subsample_size_to_test, pc_method_to_test=pc_method_to_test,
                    repeats=1,
                    number_of_trees=number_of_trees, max_depth=max_depth, min_samples_to_split=min_samples_to_split,
                    paral=paral, cpu=cpu, verbose=verbose)
    anomaly_summary["forest_num"] = forest_num
    anomaly_summary["iteration"] = iteration
    return anomaly_summary

