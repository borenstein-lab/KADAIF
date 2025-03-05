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

outlier_count = sys.argv[1]
try:
    outlier_count = int(outlier_count)
except Error:
    outlier_count = int(outlier_count[1:-1])




all_res_df = pd.DataFrame(columns = ["SampleID", "anomaly", "normal_group", "anomaly_group",
             "iteration", "score"])

features_path = "TCGA_breast/mrna_csv.csv"
metadata_path = "TCGA_breast/metadata_csv.csv"


franzosa_met_data = pd.read_csv(features_path, sep = ",", index_col = 0)
franzosa_metadata = pd.read_csv(metadata_path, sep = ",")
print(franzosa_metadata.columns)

ibd_series = franzosa_metadata[franzosa_metadata["TUMOR_STAGE_NUMERIC"] == 11]["SAMPLE_ID"]
ibd_metab_ibd = franzosa_met_data[franzosa_met_data.index.isin(ibd_series)]
healthy_series = franzosa_metadata[franzosa_metadata["TUMOR_STAGE_NUMERIC"] <= 3]["SAMPLE_ID"]
ibd_metab_healthy = franzosa_met_data[franzosa_met_data.index.isin(healthy_series)]



normal_to_sample = 49
anomaly_to_sample = 1
num_of_iterations = 100

results = mislabeling_tests.mislabeling_test(ibd_metab_healthy, ibd_metab_ibd, 50 - outlier_count, outlier_count,
                             ["equal"], ["True"], [100],
                             ["proportion"], repeats = 50, number_of_trees = 100, max_depth = 100, min_samples_to_split = 2, splitting_method = "pca",
                             paral = True, verbose = True, cpu = 150)

all_res_df = results

all_res_df.to_csv('TCGA_breast/results_mrna/new_tcga_breast_mrna_50_%s.csv' % outlier_count)
