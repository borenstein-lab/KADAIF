# How to run KADAIF?

You can run KADAIF either directly from Python or from a BASH script that wraps the Python code. The Python interface provides full control and integration with custom pipelines, while the BASH approach is useful for command-line automation and reproducibility using shell scripts. See details bellow.

# Overview

The KADAIF algorithm is an anomaly detection method specifically designed for microbiome data. It is based on the concept of [Isolation Forest (liu et al., 2008)]([url](https://ieeexplore.ieee.org/abstract/document/4781136)), a tree-based unsupervised learning algorithm that isolates anomalies by recursively partitioning data.
The anomaly score is computed based on how the average depth on which a sample was isolated.

![Schematic illustration of KADAIF](KADAIF.jpg)

# Running KADAIF from Python
KADAIF code is fully provided in the KADAIF.py file

## Parameters
* **`number_of_trees`** (int, default=100): The number of isolation trees to create.
* **`trees`** (list, default=None): A list of pre-existing trees; if None, a new set of trees is created.
* **`min_samples_to_split`** (int, default=2): The minimum number of samples required to allow a split in a tree.
* **`max_depth`** (int, default=100): The maximum depth of each tree.
* **`weights`** (str, default="equal"): Weighting scheme for feature selection. Options: "equal" (all features are drawn over a uniform distribution), "proportion" (draw feature according to their mean abundance), or "None" (use all the features).
* **`replacement`** (bool, default=True): Whether to sample with replacement during feature selection.
* **`pc_method`** (str, default="proportion"): The method for selecting principal components. Options: "first" (use the first PC), "equal" (choose from the first few  PCs over the uniform distribution), "proportion" (choose from the first few PCs with probablity proportional to the variance explained).
* **`normalize`** (bool, default=True): Whether to normalize selected features.
* **`subsample_size`** (int, default=100): Number of features selected at each split.
* **`splitting_method`** (str, default="pcoa"): Method used for splitting. Options: "pcoa", "unifrac_unweighted_pcoa", "unifrac_weighted_pcoa" or "pca".
* **`paral`** (bool, default=True): Whether to parallelize tree building.
* **`cpu`** (int, default=None): Number of CPU cores used for parallelization. If None, all available cores are used.
* **`verbose`** (bool, default=True): Whether to print progress messages.
* **`unifrac_tree`** (skbio TreeNode, default=None): a phylogenetic tree in scikit-bio TreeNode format with tips corresponding to taxa in features_matrix. needed only if **`splitting_method`** is "unifrac_unweighted_pcoa" or "unifrac_weighted_pcoa".

## Methods

**`fit(features_matrix)`**
Trains KADAIF on the given feature matrix.
* **features_matrix (pd.DataFrame)**: A feature matrix where rows represent samples and columns represent features.
* **Returns: None.**

**`score()`**
Computes the anomaly scores for the dataset.
* **Returns: None.** The anomaly scores are stored in self.scores.

**`fit_transform(features_matrix)`**
Fits the model and computes anomaly scores in one step.
* **features_matrix (pd.DataFrame)**: Input feature matrix.
* **Returns: numpy.array**. An array of anomaly scores corresponding to input samples.

## Python Usage Example

```python
import pandas as pd
from KADAIF import KADAIF

# Load example dataset
data = pd.read_csv("microbiome_data.csv", index_col=0)

# Initialize model
model = KADAIF(number_of_trees=50, subsample_size=50)

# Fit and transform
anomaly_scores = model.fit_transform(data)
```

# Running KADAIF from BASH
If you prefer running KADAIF from the command line, you can use the wrapper provided in the "run_KADAIF.sh" files. The files needed to run the wrapper are run_KADAIF_directly.py and KADAIF.py.

## Explanation of BASH Script Parameters

### Input-Output Paths

* **`INPUT_MATRIX_PATH`**: Path to the input feature matrix in CSV format. Each row should represent a sample, and each column a feature (e.g., taxon).
* **`INPUT_MATRIX_SEP`**: Delimiter used in the input CSV file. Typically `","` for standard CSVs or `"\t"` for TSVs.
* **`OUTPUT_MATRIX_PATH`**: Path where the output anomaly scores will be saved as a CSV file.

### Input Parameters

* **`NUMBER_OF_TREES`**: Number of isolation trees to build. More trees usually increase robustness but also runtime.
* **`TREES`**: Path to a serialized set of trees (ignored in this script*based usage; set to `None`).
* **`MIN_SAMPLES_TO_SPLIT`**: Minimum number of samples required to allow a split in a tree node.
* **`MAX_DEPTH`**: Maximum allowed depth for each isolation tree.
* **`WEIGHTS`**: Strategy for sampling features.  
  * `"equal"`: All features are equally likely.  
  * `"proportion"`: Sampling is weighted by feature mean abundance.  
  * `"None"`: All features are used at each split.
* **`REPLACEMENT`**: Whether to sample features with replacement during selection.
* **`PC_METHOD`**: Method for selecting the principal component during splitting.  
  * `"first"`: Always use the first PC.  
  * `"equal"`: Randomly choose among the top PCs with equal probability.  
  * `"proportion"`: Choose among top PCs proportionally to explained variance.
* **`NORMALIZE`**: Whether to normalize feature values before PCA/PCoA splitting.
* **`SUBSAMPLE_SIZE`**: Number of features to subsample at each split.
* **`SPLITTING_METHOD`**: Method used to compute the principal coordinates or components.  
  * `"pcoa"`: Uses Bray*Curtis distance and PCoA.  
  * `"unifrac_unweighted_pcoa"` / `"unifrac_weighted_pcoa"`: Use UniFrac distances and PCoA (requires a phylogenetic tree).  
  * `"pca"`: Standard Principal Component Analysis.
* **`PARAL`**: Whether to parallelize tree construction.
* **`CPU`**: Number of CPU cores to use. If set to `None`, all available cores will be used.
* **`VERBOSE`**: Whether to print progress messages during execution.
* **`UNIFRAC_TREE`**: Path to a Newick*format phylogenetic tree used when `SPLITTING_METHOD` is `"unifrac_unweighted_pcoa"` or `"unifrac_weighted_pcoa"`. Tips must match taxa in the input matrix.

## BASH Usage Example

After editing the run_KADAIF.sh file according to your chosen parameters, make sure the files run_KADAIF_directly.py and KADAIF.py are located in the same folder and run the following command: 
```bash
python run_KADAIF.sh
```
