import pandas as pd
import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.manifold import MDS
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed
from multiprocessing import cpu_count
from scipy.spatial.distance import pdist, squareform

import warnings

# warnings.filterwarnings("error")

MDS_NUM = 20

class MicrobiomeIsolationForest:
    def __init__(self, number_of_trees=10, trees=None, min_samples_to_split = 2, max_depth = 100, weights = "proportion",replacement = True,
                 pc_method = "first", normalize = True, subsample_size = 100, splitting_method = "pcoa", paral = True, cpu = None, verbose = True):
        self.number_of_trees = number_of_trees
        if trees ==None:
            trees = []
        self.trees = trees
        self.min_samples_to_split = min_samples_to_split
        self.max_depth = max_depth
        self.subsample_size = subsample_size
        self.splitting_method = splitting_method
        self.weights = weights
        self.replacement = replacement
        self.normalize = normalize
        self.paral = paral
        self.cpu = cpu
        self.pc_method = pc_method
        if self.cpu == None:
            self.cpu = cpu_count()

        self.feature_matrix = None
        self.subsample_size_in_each_tree = None
        self.samples_dict_depth = {}
        self.scores = None
        self.verbose = verbose

    def fit(self, features_matrix):
        self.features_matrix = features_matrix
        self.subsample_size_in_each_tree = self.features_matrix.shape[0]
        if self.paral == False:
            for i in range(len(self.trees), self.number_of_trees):
                if self.verbose:
                    print("Starting tree %i" % i)
                cur_tree = MicrobiomeIsolationTree(min_samples_to_split = self.min_samples_to_split, max_depth = self.max_depth,
                                        subsample_size = self.subsample_size, splitting_method = self.splitting_method, pc_method = self.pc_method,
                                        weights = self.weights, normalize = self.normalize, replacement = self.replacement,
                                                   paral = self.paral, cpu=self.cpu)
                cur_tree.fit(self.features_matrix)
                self.trees.append(cur_tree)
                if self.verbose:
                    print("Finished tree %i" % i)
        else:
            trees = Parallel(n_jobs=self.cpu)(
            delayed(return_fitted_tree)(data = self.features_matrix, min_samples_to_split = self.min_samples_to_split,
                            max_depth = self.max_depth, subsample_size = self.subsample_size,
                            splitting_method = self.splitting_method, weights = self.weights,
                            normalize = self.normalize, pc_method = self.pc_method, replacement = self.replacement,
                                        paral = self.paral, cpu = self.cpu) for _ in range(self.number_of_trees - len(self.trees)))
            [self.trees.append(t) for t in trees]

    def score(self):
        harmonic_number = np.sum([1/ i for i in range(1, self.subsample_size_in_each_tree)])
        normalization_const = 2 * harmonic_number - 2 * (self.subsample_size_in_each_tree - 1) / self.subsample_size_in_each_tree
        for tree in self.trees:
            cur_depths = tree.samples_depth
            for cur_sample in cur_depths.keys():
                if cur_sample in self.samples_dict_depth.keys():
                    self.samples_dict_depth[cur_sample].append(cur_depths[cur_sample])
                else:
                    self.samples_dict_depth[cur_sample] = [cur_depths[cur_sample]]
            avg_depths = {key: np.mean(value) for key, value in self.samples_dict_depth.items()}
            calc_score = lambda x: 2 ** -(x / normalization_const)
            self.scores = {key: calc_score(value) for key, value in avg_depths.items()}



    def fit_transform(self, features_matrix):
        self.fit(features_matrix)
        self.score()
        return np.array(pd.DataFrame.from_dict(self.scores, orient='index').loc[self.features_matrix.index])




class MicrobiomeIsolationTree:
    def __init__(self, min_samples_to_split = 10, max_depth = 10, subsample_size = 100, splitting_method = "pcoa", weights = "proportion", pc_method = "first",
                replacement = True, normalize = True, depth = 0, left=None, right=None, split_att=None, split_val=None, features_matrix=None,
                 paral = True, cpu = None, parent = None):
        self.min_samples_to_split = min_samples_to_split
        self.max_depth = max_depth
        self.subsample_size = subsample_size
        self.depth = depth
        self.left = left
        self.right = right
        self.split_att = split_att
        self.split_val = split_val
        self.splitting_method = splitting_method
        self.features_matrix = features_matrix
        self.samples_names = None
        self.size = None
        self.samples_depth = {}
        self.weights = weights
        self.replacement = replacement
        self.normalize = normalize
        self.pc_method = pc_method

        self.paral = paral
        self.cpu = cpu
        if self.cpu == None:
            self.cpu = cpu_count()

        self.parent = parent


    def fit(self, features_matrix):
        self.features_matrix = features_matrix
        self.features_matrix.loc[:, self.features_matrix.sum() > 0]
        self.samples_names = list(features_matrix.index)
        self.size = len(self.samples_names)
        if self.size < self.min_samples_to_split:
            for sample in self.samples_names:
                self.samples_depth[sample] = self.depth
            return self.samples_depth
        elif self.depth >= self.max_depth:
            for sample in self.samples_names:
                self.samples_depth[sample] = self.depth
            return self.samples_depth

        counter = 0
        self.split_att = subsample(self.features_matrix, self.subsample_size, weights = self.weights,
                  replacement = self.replacement, normalize = self.normalize)
        self.split_att.fillna(value = 1 / self.split_att.shape[1], inplace = True)
        try:
            while pairwise_distances(self.split_att).max().max() <= 10 ** (-9) or self.split_att.var().max() == 0:
                self.split_att = subsample(self.features_matrix, self.subsample_size, weights=self.weights,
                                           replacement=self.replacement, normalize=self.normalize)
                self.split_att.fillna(value=1 / self.split_att.shape[1], inplace=True)
                counter += 1
                if counter == 100:
                    raise Microbiome_isolation_forest_error(self, Exception, [self.features_matrix,
                                                                              self.split_att, self.weights])
        except ValueError:
            print(self.split_att.sum(axis =1 ))
            raise ValueError
        except Microbiome_isolation_forest_error as tmp_error:
            print(self.features_matrix)
            raise Microbiome_isolation_forest_error
        except Exception as e:
            raise e

        # counter = 0
        # self.split_val, left_samples, right_samples = split_samples(self.split_att, method= self.splitting_method,
        #                                                             pc_method = self.pc_method)
        # while len(left_samples) == 0 or len(right_samples) == 0:
        #     self.split_att = subsample(self.features_matrix, self.subsample_size, weights=self.weights,
        #                                replacement=self.replacement, normalize=self.normalize)
        #     counter += 1
        #     if counter == 100:
        #         raise Microbiome_isolation_forest_error(self, Exception, [self.features_matrix,
        #                                                                       self.split_att])



        self.split_val, left_samples, right_samples = split_samples(self.split_att, method=self.splitting_method,
                                                                    pc_method = self.pc_method)
        self.left = MicrobiomeIsolationTree(min_samples_to_split = self.min_samples_to_split, max_depth = self.max_depth,
                                            subsample_size = self.subsample_size, splitting_method = self.splitting_method,
                                            depth = self.depth + 1, weights = self.weights, normalize = self.normalize, replacement = self.replacement,
                                            paral = self.paral, cpu = self.cpu, parent = self)

        self.right = MicrobiomeIsolationTree(min_samples_to_split=self.min_samples_to_split, max_depth=self.max_depth,
                                             subsample_size=self.subsample_size, splitting_method=self.splitting_method,
                                             depth=self.depth + 1, weights=self.weights, normalize=self.normalize, replacement=self.replacement,
                                             paral = self.paral, cpu = self.cpu, parent = self)

        if self.paral:
            samples_left_depths, samples_right_depths = Parallel(n_jobs=self.cpu)(
                                                                [delayed(self.left.fit)(input1) for input1 in [self.features_matrix.loc[left_samples]]] +
                                                                [delayed(self.right.fit)(input2) for input2 in [self.features_matrix.loc[right_samples]]]
)
        else:
            samples_left_depths = self.left.fit(self.features_matrix.loc[left_samples])
            samples_right_depths = self.right.fit(self.features_matrix.loc[right_samples])

        for sample in samples_left_depths:
            self.samples_depth[sample] = self.left.samples_depth[sample]



        for sample in samples_right_depths:
            self.samples_depth[sample] = self.right.samples_depth[sample]
        return self.samples_depth





def split_samples(feature_table, method = "pcoa", distance = "braycurtis", pc_method = "first"):
    if method == "pcoa":
        distance_table = pd.DataFrame(pairwise_distances(feature_table, metric=distance), index=feature_table.index,
                              columns=feature_table.index)

        if pc_method == "first":
            mod = MDS(n_components=1, dissimilarity="precomputed")
            mod_first_comp = mod.fit_transform(distance_table)
        else:
            mod = MDS(n_components=np.min([distance_table.shape[0], MDS_NUM]), dissimilarity="precomputed")
            all_comp = mod.fit_transform(distance_table)
            if pc_method == "equal":
                mod_first_comp = all_comp[:, np.random.randint(low=0, high=all_comp.shape[1])]
            elif pc_method == "proportion":
                # select according to stress decrease
                stress_value = []
                #stress in shuffled rows data and a single pc
                feature_table_shuffled = pd.DataFrame(
                    feature_table.apply(lambda row: np.random.permutation(row), axis=1).to_list())
                feature_table_shuffled.sum(axis=1)

                bray_curtis_distances_shuffled = pdist(feature_table_shuffled, metric=distance)
                bray_curtis_matrix_shuffled = squareform(bray_curtis_distances_shuffled)

                mod = MDS(n_components=1, dissimilarity='precomputed')
                mod.fit(bray_curtis_matrix_shuffled)
                stress_value.append(mod.stress_)

                for i in range(1, np.min([distance_table.shape[0], MDS_NUM]) + 1):
                    mod = MDS(n_components=i, dissimilarity='precomputed')
                    mod.fit(distance_table)
                    stress_value.append(mod.stress_)

                stress_value_difs = [np.max([stress_value[i - 1] - stress_value[i], 0]) for i in
                                     range(1, len(stress_value))]
                cur_probs = (stress_value_difs / np.sum(stress_value_difs))
                mod_first_comp = all_comp[:, np.random.choice([i for i in range(all_comp.shape[1])], p=cur_probs)]


    elif method == "pca":
        scaled_feature_table = pd.DataFrame(StandardScaler().fit_transform(feature_table), index=feature_table.index,
                                            columns=feature_table.columns)
        if pc_method == "first":
            mod = PCA(n_components=1)
            mod_first_comp = mod.fit_transform(scaled_feature_table).T[0]
        elif pc_method == "proportion":
            mod = PCA(n_components=np.min([scaled_feature_table.shape[0], MDS_NUM, scaled_feature_table.shape[1]]))
            all_comp = mod.fit_transform(scaled_feature_table)
            cur_probs = mod.explained_variance_ratio_ / mod.explained_variance_ratio_.sum()
            mod_first_comp = all_comp[:, np.random.choice([i for i in range(all_comp.shape[1])], p=cur_probs)]


    split_value = np.random.uniform(np.min(mod_first_comp), np.max(mod_first_comp))
    left_samples_list = list(feature_table[mod_first_comp >= split_value].index)
    right_samples_list = list(feature_table[mod_first_comp < split_value].index)
    return split_value, left_samples_list, right_samples_list

def subsample(feature_table, subsample_size, weights = "proportion", replacement = True, normalize = True):
    if subsample_size == "random":
        subsample_size = np.random.randint(1, feature_table.shape[1])
    if weights == "equal":
        weights = list((feature_table.sum(axis = 0) > 0)  / (feature_table.sum(axis = 0) > 0).sum())
    elif weights == "proportion":
        weights = list(feature_table.sum(axis = 0) / feature_table.sum(axis = 0).sum())
    elif weights == "None":
        return feature_table
    if replacement == False:
        subsample_size = np.min([subsample_size, np.sum([cur_weight != 0 for cur_weight in weights])])
    try:
        columns_to_select = np.random.choice(feature_table.columns, size = subsample_size, replace = replacement,p = weights)
    except ValueError:
        raise ValueError
    cur_feature_table = feature_table[columns_to_select]

    if normalize:
        real_columns = list(cur_feature_table.columns)
        tmp_columns = [i for i in range(subsample_size)]
        cur_feature_table.columns = tmp_columns
        cur_feature_table = cur_feature_table.div(cur_feature_table.sum(axis = 1), axis = 0)
        cur_feature_table.columns = list(real_columns)
    return cur_feature_table

def return_fitted_tree(data, min_samples_to_split, max_depth, subsample_size, splitting_method,
                       weights, normalize, replacement, pc_method, paral, cpu):
    t = MicrobiomeIsolationTree(min_samples_to_split = min_samples_to_split, max_depth = max_depth,
                                        subsample_size = subsample_size, splitting_method = splitting_method,
                                        weights = weights, normalize = normalize, replacement = replacement, pc_method = pc_method,
                                paral = paral, cpu = cpu)
    t.fit(data)
    return t
class Microbiome_isolation_forest_error(Exception):

    def __init__(self, t, e, features = None):
        self.t = t
        self.error = e
        self.features = features
