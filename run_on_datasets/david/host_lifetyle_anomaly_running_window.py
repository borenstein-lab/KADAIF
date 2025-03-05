import os

import sys
import pandas as pd
import numpy as np
import importlib
#import seaborn as sns
import matplotlib.pyplot as plt
import MicrobiomeIsolationForest
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from IPython.display import display

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
from statsmodels.nonparametric.smoothers_lowess import lowess
import matplotlib.patches as patches
print("finished_imports")

data = pd.read_csv("david/Host lifestyle affects.csv.csv", index_col = 0, sep = "\t")
#data = data.applymap(lambda x: x + np.random.uniform(0, 0.1))
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

print("it's CRISP time!")

for time_dif in [10,20,30,40,50,60,70,80,90,100]:
    print(time_dif)
    running_window_results = {}
    running_window_results_iso = {}

    all_times = list(donor_a_data.index.map(lambda x : int(x[x.find("Stool") + 5:])))
    all_times = sorted(all_times)

    for cur_time in all_times:
    
        if cur_time > time_dif: 
            data_to_run = ["2202.Stool" +str(timepoint) for timepoint in all_times if ((timepoint <=cur_time) and (timepoint >= cur_time - time_dif))]
            if len(data_to_run) >= 5:
                print(len(data_to_run), cur_time)
                importlib.reload(MicrobiomeIsolationForest)
                mod = MicrobiomeIsolationForest.MicrobiomeIsolationForest(number_of_trees = 100, max_depth = 100,min_samples_to_split=2, 
                                                                          subsample_size=50, replacement= True, weights = "equal",
                                                                     pc_method = "proportion", paral = True, verbose = False)
                mod.fit(donor_a_data.loc[data_to_run])
                mod.score()
                running_window_results[cur_time] = mod.scores["2202.Stool" +str(cur_time)]
                
                iso_mod = IsolationForest()
                iso_mod.fit(donor_a_data.loc[data_to_run])
                running_window_results_iso[cur_time] =  iso_mod.score_samples(donor_a_data.loc[["2202.Stool" +str(cur_time)]])[-1]
                
            
            
    scores_df = pd.DataFrame.from_dict(running_window_results.items())
    print(running_window_results)
    
    print(scores_df)
    scores_df.columns = columns = ["Day", "Score"]
    scores_df = scores_df.sort_values(by = ["Day"])

#    scores_df["Rolling Score"] = scores_df["Score"].rolling(7).mean()
    scores_df["Method"] = "KADAIF"



    print(running_window_results_iso)
    scores_df_IF = pd.DataFrame.from_dict(running_window_results_iso.items())
    scores_df_IF.columns = columns = ["Day", "Score"]
    scores_df_IF = scores_df_IF.sort_values(by = ["Day"])

#    scores_df_IF["Rolling Score"] = scores_df_IF["Score"].rolling(7).mean()
    scores_df_IF["Method"] = "IF"
    scores_df = pd.concat([scores_df,scores_df_IF])

    scores_df.to_csv("david/results/res_%i_last_samples.csv" % time_dif)
