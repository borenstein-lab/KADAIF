import os
import sys
import pandas as pd
import numpy as np
import importlib
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
#from statsmodels.stats.multitest import fdrcorrection
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


metadata = pd.read_csv("wang/metadata.tsv", sep = "\t")
data = pd.read_csv("wang/genera.tsv", sep = "\t", index_col = 0)


healthy = data.loc[metadata[metadata["Study.Group"] == "Control"]["Sample"]]
sick = data.loc[metadata[metadata["Study.Group"] == "ESRD"]["Sample"]]

num_of_out = 25
np.random.seed(0)
results = contamination_test.contamination_test(healthy, sick, 50 - outlier_count, outlier_count, [i*0.1 for i in range(11)],
                             ["equal" ], ["True"], [100], 
                             [ "proportion"], repeats = 50, number_of_trees = 100, max_depth = 100, min_samples_to_split = 2, splitting_method = "pcoa",
                             paral = True, verbose = True, cpu = 150)



results.to_csv("wang/contamination_test/results/mini_wang_results_50_%s.csv" % outlier_count )
