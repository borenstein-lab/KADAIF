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

data = pd.read_csv("david/host_lifestyle_affects.csv", index_col = 0, sep = "\t")
# data = data.applymap(lambda x: x + np.random.uniform(0, 0.1))
data = (data / data.sum()).T
metadata = pd.read_csv("david/sample_information_from_prep_203.tsv", index_col = 0, sep = "\t")
donor_a_metadata = metadata[(metadata["host_subject_id"] == "DonorA") & (metadata["host_body_product"] == "UBERON:feces")]
donor_a_data = data[data.index.isin(donor_a_metadata.index)]
donor_a_data = donor_a_data.loc[ : , donor_a_data.sum() > 0]
donor_a_data = (donor_a_data.T / donor_a_data.T.sum()).T
donor_a_data_counter = (donor_a_data > 0).T.sum()
donor_a_data = donor_a_data[donor_a_data.index.isin(donor_a_data_counter[donor_a_data_counter > 25].index)]


donor_b_metadata = metadata[(metadata["host_subject_id"] == "DonorB") & (metadata["host_body_product"] == "UBERON:feces")]
donor_b_data = data[data.index.isin(donor_b_metadata.index)]
donor_b_data = donor_b_data.loc[ : , donor_b_data.sum() > 0]
donor_b_data = (donor_b_data.T / donor_b_data.T.sum()).T
donor_b_data_counter = (donor_b_data > 0).T.sum()
donor_b_data = donor_b_data[donor_b_data.index.isin(donor_b_data_counter[donor_b_data_counter > 25].index)]


print("time for some KADAIF!")
mod = MicrobiomeIsolationForest.MicrobiomeIsolationForest(number_of_trees = 100, max_depth = 100,min_samples_to_split=2, 
                                                              subsample_size=50, replacement= True, weights = "equal",
                                                         pc_method = "proportion", paral = True)
mod.fit(donor_a_data)
print("finished CRISP")
mod.score()
scores_dict = {int(cur_timepoint[cur_timepoint.find("Stool") + 5:]): mod.scores[cur_timepoint] for cur_timepoint in mod.scores}

anomaly_score_df = pd.DataFrame.from_dict(scores_dict, orient = "index", columns = ["Score"])
anomaly_score_df["Method"] = "KADAIF"
anomaly_score_df["Day"] = anomaly_score_df.index



iso_mod = IsolationForest()
iso_mod.fit(donor_a_data)
iso_labels = donor_a_data.index.map(lambda x : int(x[x.find("Stool") + 5:]))

tmp_df = pd.DataFrame(-iso_mod.score_samples(donor_a_data), index = iso_labels, columns = ["Score"])
tmp_df["Method"] = "IF"
tmp_df["Day"] = tmp_df.index
anomaly_score_df = pd.concat([anomaly_score_df, tmp_df], axis = 0)

cloud_res = CLOUD.CLOUD(donor_a_data)
tmp_df = pd.DataFrame.from_dict(cloud_res[0], orient = "index", columns = ["Score"])
tmp_df["Method"] = "CLOUD"
tmp_df.index = iso_labels
tmp_df["Day"] = tmp_df.index
anomaly_score_df = pd.concat([anomaly_score_df, tmp_df], axis = 0)
anomaly_score_df.to_csv("david/results/all_without_filtering.csv")