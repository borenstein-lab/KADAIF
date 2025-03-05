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
print(outlier_count)
try:
        outlier_count = int(outlier_count)
except Error:
        outlier_count = int(outlier_count[1:-1])

m3 = pd.read_csv("caporaso/M3_feces_L6.txt", index_col = 0)
m3 = (m3 / m3.sum()).T
m3.index = ["m3_" + time for time in m3.index]

f4 = pd.read_csv("caporaso/F4_feces_L6.txt", index_col = 0)
f4 = (f4 / f4.sum()).T
f4.index = ["f4_" + time for time in f4.index]



np.random.seed(0)
results_M3_normal = contamination_test.contamination_test(m3, f4 , 50 - outlier_count, outlier_count,[i * 0.1 for i in range(11)],
                             ["equal"], ["True"], [10], 
                             [ "proportion"], repeats = 50, number_of_trees = 100, max_depth = 100, min_samples_to_split = 2,
                             paral = True, verbose = True, cpu = 150)


results_M3_normal.to_csv("caporaso/contamination_test/results/mini_mp_results_M3_normal_50_%s.csv" % outlier_count )


results_F4_normal = contamination_test.contamination_test(f4, m3, 50 - outlier_count, outlier_count,[i * 0.1 for i in range(11)],
                             ["equal"], ["True"], [3, 5, 10],
                              ["proportion"], repeats = 50, number_of_trees = 100, max_depth = 100, min_samples_to_split = 2,
                             paral = True, verbose = True, cpu = 150)
results_F4_normal.to_csv("caporaso/contamination_test/results/mini_mp_results_F4_normal_50_%s.csv" % outlier_count )

