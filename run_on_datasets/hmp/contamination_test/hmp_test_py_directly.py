import os
import sys
import pandas as pd
import numpy as np
import importlib
import seaborn as sns
import matplotlib.pyplot as plt
import contamination_test
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
results = contamination_test.contamination_test(healthy, sick, 50 - outlier_count, outlier_count, [i*0.1 for i in range(11)],
                             ["equal"], ["True"], [150], 
                             [ "proportion"], repeats = 50, number_of_trees = 100, max_depth = 100, min_samples_to_split = 2,
                             paral = True, verbose = True, cpu = 150)


results.to_csv("hmp/contamination_test/results/mini_hmp_results_50_%s.csv" % outlier_count )
