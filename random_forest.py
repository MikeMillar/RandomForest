import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)

import pandas as pd
import numpy as np
import math
from sklearn.utils import resample
import classifier as cf
import utils

class RandomForestClassifier:
    def __init__(self, num_trees, max_depth, min_samples, confidence, impurity_function, features, target):
        self.trees = []
        self.num_trees = num_trees
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.confidence = confidence
        self.impurity_function = impurity_function
        self.features = features
        self.target = target

    def init_trees(self):
        for _ in range(self.num_trees):
            tree = cf.DecisionTreeClassifier(self.max_depth, self.min_samples, self.impurity_function,
                                             self.confidence)
            self.trees.append(tree)

    def init_samples(self, dataset):
        self.create_samples(dataset)

    def train(self, dataset):
        splits = self.create_samples(dataset)
        # for i in range(self.num_trees):
        for split in splits:
            # split = self.samples[i]
            split_features = list(split.columns)
            split_features.remove(self.target)
            target_proportions = utils.calculate_target_proportions(split, self.target, dataset[self.target].unique())
            # tree = self.trees[i]
            tree = cf.DecisionTreeClassifier(self.max_depth, self.min_samples, self.impurity_function,
                                             self.confidence)
            self.trees.append(tree)
            tree.train(split, split_features, self.target, target_proportions)

    def predict(self, dataset):
        count = 1
        prediction_columns = []
        for tree in self.trees:
            prediction_column = 'prediction_' + str(count)
            predictions = tree.classify(dataset)
            # predictions = tree.predict(dataset)
            # dataset[prediction_column] = predictions
            dataset[prediction_column] = predictions['prediction']
            count += 1
            prediction_columns.append(prediction_column)

        print(dataset)
        dataset['prediction'] = (dataset[prediction_columns].sum(axis=1) / len(prediction_columns)).apply(lambda x: 0 if x <= 0.5 else 1)
        dataset.drop(prediction_columns, axis=1, inplace=True)
        return dataset

    def create_samples(self, dataset, replacement=True):
        # features_per = math.ceil(math.log(len(self.features)))
        features_per = round(len(self.features) * .7)
        samples_per = round(len(dataset) * .7)
        print(f"features_per={features_per}, samples_per={samples_per}")
        self.samples = []
        for _ in range(self.num_trees):
            tree_features = resample(self.features, replace=False, n_samples=features_per)
            tree_features.append(self.target)
            sample = resample(dataset, replace=replacement, n_samples=samples_per, stratify=dataset[self.target])
            self.samples.append(sample[tree_features])
        return self.samples