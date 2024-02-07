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
# df : pandas dataframe which contains 'targets' column
# targets : string name of the target column in df
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
# df : pandas dataframe which contains 'targets' column
# targets : string name of the target column in df
def misclassificationIndex(df, targets):
    # Get total size of the data frame
    totalSize = len(df.index)
    # Get counts by targets column
    counts = df[targets].value_counts()
    # determine max count
    max = counts.max()
    return 1 - (max / totalSize)

# Information Gain is a measure of how much information a particular
# attribute (column) of our data set provides us towards our goal
# of classifying an instance.
# Computed as: entropy(s) - sum_{v in attr} |s_v| / |s| * entropy(s_v)
# df : pandas dataframe which contains 'targets' column
# attribute: string name of attribute column in df
# targets : string name of the target column in df
# func: impurity function to run
def informationGain(df, attribute, targets, func):
    # Get total size of the data frame
    totalSize = len(df.index)
    # Calculate entropy of entire data set
    impurity = func(df, targets)
    # Calculate information gain summation
    infoGain = 0.0
    # Split data frame based on categorical attribute values
    splits = [y for x, y in df.groupby(attribute)]
    for split in splits:
        # Get total size of split
        splitSize = len(split.index)
        # Get entropy of split
        splitImpurity = func(split, targets)
        # Calculate info gain of the split and add to existing info gain
        infoGain += (splitSize / totalSize) * splitImpurity
    return impurity - infoGain

# Function that finds and returns the attribute with the highest
# information gain.
# df : pandas dataframe which contains attributes and targets columns
# attributes : names of attribute columns
# targets : string name of target column
# func : impurity function to use
def findHighestGainAttribute(df, attributes, targets, func):
    print(df)
    # Set default best attribute to be first
    bestAttribute = attributes[0]
    bestGain = 0.0
    # Iterate through all attributes and find highest information gain
    for attribute in attributes:
        gain = informationGain(df, attribute, targets, func)
        if gain > bestGain:
            bestAttribute = attribute
            bestGain = gain
    return bestAttribute, bestGain