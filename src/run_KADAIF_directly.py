import argparse
import pandas as pd
import numpy as np
import joblib
import KADAIF
from skbio import TreeNode


parser = argparse.ArgumentParser()
parser.add_argument("--input_matrix", required=True)
parser.add_argument("--input_matrix_sep", required=True)
parser.add_argument("--output_matrix", required=True)
parser.add_argument("--number_of_trees", type=int, default=100)
parser.add_argument("--trees", default=None)
parser.add_argument("--min_samples_to_split", type=int, default=2)
parser.add_argument("--max_depth", type=int, default=100)
parser.add_argument("--weights", default="equal")
parser.add_argument("--replacement", type=lambda x: x.lower() == 'true', default=True)
parser.add_argument("--pc_method", default="proportion")
parser.add_argument("--normalize", type=lambda x: x.lower() == 'true', default=True)
parser.add_argument("--subsample_size", type=int, default=100)
parser.add_argument("--splitting_method", default="pcoa")
parser.add_argument("--paral", type=lambda x: x.lower() == 'true', default=True)
parser.add_argument("--cpu", type=lambda x: None if x == "None" else int(x), default=None)
parser.add_argument("--verbose", type=lambda x: x.lower() == 'true', default=True)
parser.add_argument("--unifrac_tree", default=None)

args = parser.parse_args()

if args.verbose:
    print("arguments are:", args)

data = pd.read_csv(args.input_matrix, index_col=0, header = 0, sep = args.input_matrix_sep, engine='python')


if args.splitting_method in ["unifrac_unweighted_pcoa", "unifrac_weighted_pcoa"]:
    tree_obj_for_unifrac = TreeNode.read(args.unifrac_tree)
else:
    tree_obj_for_unifrac = None
    
model = KADAIF.KADAIF(
    number_of_trees=args.number_of_trees,
    trees=None,
    min_samples_to_split=args.min_samples_to_split,
    max_depth=args.max_depth,
    weights=args.weights,
    replacement=args.replacement,
    pc_method=args.pc_method,
    normalize=args.normalize,
    subsample_size=args.subsample_size,
    splitting_method=args.splitting_method,
    paral=args.paral,
    cpu=args.cpu,
    verbose=args.verbose,
    unifrac_tree=tree_obj_for_unifrac
)
output = model.fit_transform(data)

output_df = pd.DataFrame(output, index = data.index, columns = ["Anomaly_scores"])
output_df.to_csv(args.output_matrix)