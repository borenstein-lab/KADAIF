import os
import sys


import pandas as pd
import matplotlib.pyplot as plt
import mislabeling_tests
import MicrobiomeIsolationForest
print("importing IF")
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from IPython.display import display
# import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
from sklearn.manifold import MDS
import copy
from statsmodels.stats.multitest import fdrcorrection
from scipy.stats import ranksums
from scipy.stats import mannwhitneyu
from sklearn import ensemble
from sklearn.metrics import roc_auc_score
from scipy.spatial import distance
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import MDS
from sklearn.ensemble import IsolationForest
from datetime import datetime
import gc
print("imports finished")
warnings.simplefilter(action='ignore', category=FutureWarning)


all_res_df = pd.DataFrame(columns = ["SampleID", "anomaly", "normal_group", "anomaly_group",
             "iteration", "score"])

metacardis_path = "METACARDIS_original_processing/"
features_path = "hub.cellcount.motu.Genus.v2.data.frame.r"
metadata_path = "demographic_20201210.r"


features = pd.read_csv(metacardis_path+features_path, sep = "\t", index_col = 0)
features = features.pivot(index = "SampleID", columns = "Feature", values = "FeatureValue")
features = features.loc[:, features.sum() > 0]
features = features.div(features.sum(axis = 1), axis = 0)
metadata = pd.read_csv(metacardis_path + metadata_path, sep = "\t")
cur_normal_group_list = list(metadata["PATGROUPFINAL_C"].unique())


normal_to_sample = 25
anomaly_to_sample = 25
num_of_iterations = 50


for cur_normal_group in cur_normal_group_list:
    print(cur_normal_group, cur_normal_group_list)
    for cur_anomaly_group in cur_normal_group_list[cur_normal_group_list.index(cur_normal_group):]:
        now = datetime.now()
        print(cur_normal_group, cur_anomaly_group, now)
        if cur_normal_group == "7":
            continue
        elif cur_normal_group == cur_anomaly_group:
            continue
        elif cur_anomaly_group == "7":
            continue
        for iteration in range(num_of_iterations):
            cur_samples_normal = metadata[metadata["PATGROUPFINAL_C"] == cur_normal_group]["SampleID"]
            cur_normal = features[features.index.isin(cur_samples_normal)].sample(normal_to_sample)

            cur_samples_anomaly = metadata[metadata["PATGROUPFINAL_C"] == cur_anomaly_group]["SampleID"]
            cur_anomaly = features[features.index.isin(cur_samples_anomaly)].sample(anomaly_to_sample)

            tmp_all = pd.concat([cur_normal, cur_anomaly], axis = 0)
            tmp_all = tmp_all.loc[:, tmp_all.sum() > 0]

            mod = MicrobiomeIsolationForest.MicrobiomeIsolationForest(number_of_trees=100,
                max_depth=100,
                weights='equal',
                replacement=True,
                pc_method='proportion',
                subsample_size=100, paral = True)
            tmp_res = mod.fit_transform(tmp_all)

            tmp_res_df = pd.DataFrame(tmp_all.index)
            tmp_res_df["anomaly"] = tmp_res_df["SampleID"].isin(cur_anomaly.index)
            tmp_res_df["normal_group"] = cur_normal_group
            tmp_res_df["anomaly_group"] = cur_anomaly_group
            tmp_res_df["iteration"] = iteration
            tmp_res_df["score"] = tmp_res

            all_res_df = pd.concat([all_res_df, tmp_res_df])
            del cur_samples_normal
            del cur_normal
            del cur_samples_anomaly
            del cur_anomaly
            del tmp_all
            del mod
            del tmp_res
            del tmp_res_df
            gc.collect()



all_res_df.to_csv('metacardis/anna_karenina/results/all_res_equal_prop_df_%s.csv' % cur_normal_group_list[0])
