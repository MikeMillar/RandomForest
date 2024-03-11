import pandas as pd
import numpy as np

def calculate_target_proportions(dataset: pd.DataFrame, target, target_values):
    target_counts = dataset[target].value_counts()
    target_proportions = {}
    for tval in target_values:
        target_proportions[tval] = target_counts[tval] / len(dataset)
    return target_proportions

def split_on_feature(dataset: pd.DataFrame, feature, drop_missing, fill, threshold=None):
    if fill and threshold != None:
        mean = dataset[feature].mean()
        dataset[feature].fillna(mean, inplace=True)
    elif fill:
        common_value = dataset[feature].value_counts().idxmax()
        dataset[feature].fillna(common_value, inplace=True)
    if threshold != None:
        mask = dataset[feature] <= threshold
        return [dataset[mask], dataset[~mask]]
    return [y for x, y in dataset.groupby(feature, dropna=drop_missing, axis=0)]

def largest_split_value(splits, feature):
    largest = 0
    value = None
    for split in splits:
        size = len(split)
        if size > largest:
            largest = size
            value = split[feature].iloc[0]
    return value

def index_of(columns, feature):
    index = -1
    for i in range(len(columns)):
        if columns[i] == feature:
            return i
        
# Undersample the target data by the given proportions.
# data -> dataframe to undersample
# target_col -> target column in dataframe
# targets -> array-like list of target values
# props -> array-like list of target proportions
def undersample(data: pd.DataFrame, target_col, targets, props):
    if len(targets) != len(props):
        print('Len of target and props does not match.')
        return
    total_size = len(data)
    resample_size: int = int(total_size)
    # without replacement, need to determine max size of output based on target count
    # and proproptions.
    for i in range(len(targets)):
        target = targets[i]
        prop = props[i]
        df1 = data[data[target_col] == target]
        tmp = len(df1) / prop
        resample_size = int(min(tmp, resample_size))
    # perform resampling
    output = pd.DataFrame()
    for i in range(len(targets)):
        target = targets[i]
        prop = props[i]
        maxSize = int(prop * resample_size)
        target_indices = data[data[target_col] == target].index
        random_indices = np.random.choice(target_indices, maxSize, False)
        sample = data.loc[random_indices]
        output = pd.concat([output, sample])
    return output

