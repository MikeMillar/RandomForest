import pandas as pd
import numpy as np

# Entropy is measure of how impure the data set is.
# Computed as: negative of the summation for all classes (targets)
# of the data set with respect to their porportion in the data set.
# - sum_{i}^{c} p_{i} * log_{2}(p_{i})
# df : pandas dataframe which contains 'targets' column
# targets : string name of the target column in df
def entropy(df, targets):
    # Get total size of the data frame
    totalSize = len(df.index)
    # Get counts by targets column
    counts = df[targets].value_counts()
    # Calculate entropy for each
    entropy = 0.0
    for count in counts:
        # for each count, compute entropy and add to existing entropy
        proportion = count / totalSize
        entropy += proportion * np.log2(proportion)
    # return negative entropy value
    return -entropy

# Gini Index is a measure of how impure the data set is.
# Computed as: 1 - summation for all classes (targets) proportion squared
# of the data set. 
# 1 - sum_{c} P(c)^{2}
def giniIndex(df, targets):
    # Get total size of teh data frame
    totalSize = len(df.index)
    # Get counts by targets column
    counts = df[targets].value_counts()
    # Calculate gini index for each count
    gini = 0.0
    for count in counts:
        # for each count, compute gini index and add to existing gini index
        proportion = count / totalSize
        gini += proportion * proportion
    return 1 - gini

# Misclassification Index is a measure of how likely we are to 
# misclassify an instance based on the probability of each class (target).
# Computed as: 1 - max(p(c_1), p(c_2), ..., p(c_n))
def misclassificationIndex(df, targets):
    # Get total size of the data frame
    totalSize = len(df.index)
    # Get counts by targets column
    counts = df[targets].value_counts()
    # determine max count
    max = counts.max()
    return 1 - (max / totalSize)