#!/bin/bash

# ==== Input-Output Paths ====
INPUT_MATRIX_PATH="path/to/input_matrix.csv"
INPUT_MATRIX_SEP="," # Separator between columns in the INPUT_MATRIX
OUTPUT_MATRIX_PATH="path/to/output_matrix.csv"

# ==== Input Parameters ====

####UNIFRAC_TREE_PATH is needed only if
NUMBER_OF_TREES=100
TREES=None  # Ignored in this running form
MIN_SAMPLES_TO_SPLIT=2
MAX_DEPTH=100
WEIGHTS="equal"  # Options: "equal", "proportion", "None"
REPLACEMENT=True
PC_METHOD="proportion"  # Options: "first", "equal", "proportion"
NORMALIZE=True
SUBSAMPLE_SIZE=100
SPLITTING_METHOD="pcoa"  # Options: "pcoa", "unifrac_unweighted_pcoa", "unifrac_weighted_pcoa", "pca"
PARAL=True
CPU=None  # Use None for all available cores
VERBOSE=True
UNIFRAC_TREE="path/to/tree.nwk" #needed only if SPLITTING_METHOD is either "unifrac_unweighted_pcoa" or "unifrac_weighted_pcoa"


# ==== Run KADAIF via Python ====
python run_KADAIF_directly.py \
    --input_matrix "$INPUT_MATRIX_PATH" \
    --input_matrix_sep "$INPUT_MATRIX_SEP" \
    --output_matrix "$OUTPUT_MATRIX_PATH" \
    --number_of_trees "$NUMBER_OF_TREES" \
    --trees "$TREES" \
    --min_samples_to_split "$MIN_SAMPLES_TO_SPLIT" \
    --max_depth "$MAX_DEPTH" \
    --weights "$WEIGHTS" \
    --replacement "$REPLACEMENT" \
    --pc_method "$PC_METHOD" \
    --normalize "$NORMALIZE" \
    --subsample_size "$SUBSAMPLE_SIZE" \
    --splitting_method "$SPLITTING_METHOD" \
    --paral "$PARAL" \
    --cpu "$CPU" \
    --verbose "$VERBOSE"\
    --unifrac_tree "$UNIFRAC_TREE" 