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
from scipy import stats
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


def get_informative_species(df, bacteria_unique_frac = 0.95):
    print(f"Data started with {len(df.columns)} taxa")
    print(f"Filtering out all non unique values (frac > {bacteria_unique_frac})")
    modes = df.apply(lambda x: stats.mode(x, keepdims=True).mode[0])
    mode_counts = (modes == df).sum()/len(df)
    column_df = pd.DataFrame(mode_counts).rename(columns={0:"mode_frac"})
    informative_columns = list(column_df[column_df['mode_frac'] < bacteria_unique_frac].index)
    print(f"Returning {len(informative_columns)} columns")
    return informative_columns


metadata = pd.read_csv("franzosa/metadata.tsv", sep = "\t")
data = pd.read_csv("franzosa/genera.tsv", sep = "\t", index_col = 0)

#data  = data.loc[:, get_informative_species(data )]


healthy = data.loc[metadata[metadata["Study.Group"] == "Control"]["Sample"]]
sick = data.loc[metadata[metadata["Study.Group"] != "Control"]["Sample"]]

num_of_out = 25
np.random.seed(0)
results = contamination_test.contamination_test(healthy, sick, 50 - outlier_count, outlier_count, [i * 0.1 for i in range(11)],
                                     ["equal"], ["True"], [ 100 ],
                                                                  ["proportion"], repeats = 50, number_of_trees = 50, max_depth = 100, min_samples_to_split = 2,
                                                                                               paral = True, verbose = True, cpu = 150)



results.to_csv("franzosa/contamination_test/results/mini_franzosa_results_filtered_50_%s.csv" % outlier_count )
