import pandas as pd
import numpy as np
import scipy.stats as sp

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

# Chi-Square Stop is a method of determining when a decision tree
# should turn the current value into a leaf node, when the outcome
# of a split is no better than random.
# We split on the given attribute and gather the following information:
#   - for each attribute value, count how many there are for each attribute
#     value-target pair (sunny, yes), (sunny, no), etc...
#   - for each attribute value, calculate expected amount for each
#     value-target pair using total number of attributes with that value
#     and the proportion of the target from the parent set
#   - Subtrack the actual from the expected and square the result, then
#     divide that by the expected amount.
#   - Add all values from the previous step together to get a "critical value"
#   - Calculate 'degrees of freedom' (# classes - 1) * (# attributes - 1)
#   - Calculate alpha value by subtracting our desired confidence level from 1.
#     Ex: 95% confidence, alpha = 1 - .95 = .05
#     (scipy.stats.chi2.ppf wants the condifence level before subtraction: 0.95)
#     scipy.stats.chi2.ppf(confidence_level, degrees_of_freedom)
#   - Look up chi-squared value using degrees of freedom and confidence value
#   - Compaire our critical value to our chi-squared value.
#       - critical >= chi -> Significant, continue to branch
#       - critical < chi -> Not Significant, create leaf and stop
# cv = sum_{all attribute-target pairs} (actual - expected)^2 / expected
# Returns True if the result is not better than random and we should not expand
def chiSquareStop(df, attribute, targets, confidence):
    # Get total size of data frame
    totalSize = len(df.index)
    # Get values of target
    targetValues = df[targets].unique()
    # Get target value counts
    targetCounts = df[targets].value_counts()
    # Calculate proportions
    targetProportions = []
    for i in range(0, len(targetValues)):
        targetProportions.append(targetCounts[targetValues[i]] / totalSize)
    # Split on attribute values
    splits = [y for x, y in df.groupby(attribute)]
    # Calculate critical value
    criticalValue = 0.0
    for split in splits:
        # Get total count in split
        splitSize = len(split.index)
        # Get target counts in split
        splitTargetCounts = split[targets].value_counts()
        # Calculate critical value part
        for i in range(0, len(targetValues)):
            actual = 0
            if targetValues[i] in splitTargetCounts:
                actual = splitTargetCounts[targetValues[i]]
            expected = splitSize * targetProportions[i]
            criticalValue += np.square(actual - expected) / expected
    # Calculate degrees of freedom
    df = (len(targetValues) - 1) * (len(splits) - 1)
    # Lookup chi2 value for confidence and df
    chi = sp.chi2.ppf(confidence, df)
    # Compare critical value to chi2 value
    print("Comparing c.v.", criticalValue, "to chi2", chi)
    if criticalValue >= chi:
        # Value is significant, do not stop
        return False
    # Value not significant, stop
    return True