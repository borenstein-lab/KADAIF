import os
import sys
import pandas as pd
import numpy as np
import importlib
#import seaborn as sns
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
import datetime
print("imports finished")

outlier_count = sys.argv[1]
try:
    outlier_count = int(outlier_count)
except Error:
    outlier_count = int(outlier_count[1:-1])


metadata = pd.read_csv("hmp/Human_Microbiome_Project_1_6_2017_metadata.tab", sep = "\t", index_col = 0)
data = pd.read_csv("hmp/HMP_data.txt", sep = "\t", index_col = 0)

data = data.div(data.sum(axis = 1), axis = 0)

normal_samples = metadata[metadata['body_site'] == 'UBERON:feces'].index
anomaly_samples = metadata[~(metadata['body_site'] == 'UBERON:feces')].index


healthy = data[data.index.isin(normal_samples)]
sick = data[data.index.isin(anomaly_samples)]

num_of_out = 25
np.random.seed(0)
results = mislabeling_tests.mislabeling_test(healthy, sick, 50 - outlier_count, outlier_count,
                             ["equal", "proportion"], ["True", "False"], [5, 10, 20, 35, 50, 75, 100, 125, 150, 200, 250, 300, 500, 750, 1000, 2000, 3000, 5000, 7500, 10000, 11000, "random"], ["proportion", "equal", "first"], repeats = 50, number_of_trees = 100, max_depth = 100, min_samples_to_split = 2,
                             paral = True, verbose = True, cpu = 150)

results.to_csv("hmp/results/full_hmp_results_50_%s.csv" % outlier_count )
