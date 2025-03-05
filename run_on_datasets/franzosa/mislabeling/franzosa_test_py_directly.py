import os
import sys
import pandas as pd
import numpy as np
import importlib
#import seaborn as sns
import matplotlib.pyplot as plt
import mislabeling_tests
import MicrobiomeIsolationForest
import CLOUD
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

metadata = pd.read_csv("franzosa/metadata.tsv", sep = "\t")
data = pd.read_csv("franzosa/genera.tsv", sep = "\t", index_col = 0)



healthy = data.loc[metadata[metadata["Study.Group"] == "Control"]["Sample"]]
sick = data.loc[metadata[metadata["Study.Group"] != "Control"]["Sample"]]

num_of_out = 25
np.random.seed(0)
results = mislabeling_tests.mislabeling_test(healthy, sick, 50 - outlier_count, outlier_count,
                             ["proportion","equal"], ["True", "False"], ["random", 5, 10, 20, 50, 100, 200, 300, 500, 750, 1000, 2000, 3000, 5000, 7500, 10000, 11000], 
                             ["first", "equal", "proportion"], repeats = 50, number_of_trees = 100, max_depth = 100, min_samples_to_split = 2,
                             paral = True, verbose = True, cpu = 150)


results.to_csv("franzosa/results/full_franzosa_results_50_%s.csv" % outlier_count )
