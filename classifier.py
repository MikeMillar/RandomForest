import pandas as pd
import numpy as np
from dataclasses import dataclass
import utils
import impurity

class DecisionTreeClassifier():

    # Initialize the decision tree classifier.
    # impurity_function -> function to evaluate impurity of the dataset with. default: entropy
    # confidence_level -> how confident do we want to train the tree to be. default: 95% (0.95)
    def __init__(self, max_depth=None, min_samples=10,
                  imurity_function=impurity.entropy, confidence_level=0.95) -> None:
        self.root = None
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.impurity_function = imurity_function
        self.confidence_level = confidence_level
        self.counts = None

    def setAttributeStats(self, dataset):
        columns = dataset.columns
        self.counts = {}
        for col in columns:
            count = dataset[col].value_counts()
            self.counts[col] = count

    def getAttributeCount(self, attribute, value=None):
        # check if attribute is None
        if attribute == None:
            # invalid, return None to indicate
            return None
        if value == None:
            # return all counts for attribute
            return self.counts[attribute]
        else:
            # Assume that given attribute and value exists
            return self.counts[attribute][value]
        
    def start_training(self, queue, dataset, attributes, target, parent_target_proportions):
        print('Tree training starting...')
        root = self.train(dataset, attributes, target, parent_target_proportions)
        queue.put(root)
        print('Tree done training...')

    # Given a training dataset, target, and attributes, creates a decision tree
    # dataset -> Dataset to train with
    # target -> name of target attribute
    # attributes -> list of attributes to train with
    # depth -> current depth of the tree
    def train(self, dataset, attributes, target, parent_target_proportions, depth=0):
        if self.counts == None:
            # Populate stats of attributes
            self.setAttributeStats(dataset)

        # Create a new node
        nextNode = Node()

        # If root not yet set, set nextNode as root
        if self.root == None:
            self.root = nextNode

        # check if dataset has any data
        dataCount = len(dataset.index)
        if dataCount == 0: # No data
            return None
        
        # Get most common target value
        default_vault = dataset[target].value_counts().idxmax()

        # Verify minimum requirements to continue are met
        if dataCount <= self.min_samples or self.max_depth <= depth or len(attributes) == 0:
            # below min samples, max depth reached, or no remaining attributes
            return nextNode.set_leaf(default_value=default_vault)
        # check impurity of dataset with respect to target
        purity = self.impurity_function(dataset, target)
        if purity == 0: # No impurity in dataset
            return nextNode.set_leaf(default_value=default_vault)
        
        # For each attribute, calculate information gain and keep best
        bestAttribute, bestGain, bestThreshold = impurity.findHighestGainAttribute(dataset, attributes, target, self.impurity_function)

        if bestGain == 0:
            return nextNode.set_leaf(default_value=default_vault)
        
        # Split the data
        splits = None
        if dataset[bestAttribute].dtypes.name == 'category':
            splits = utils.split_on_feature(dataset, bestAttribute, True, True)
        else:
            splits = utils.split_on_feature(dataset, bestAttribute, True, True, bestThreshold)

        not_expand = False
        # Test if gain is better than random
        not_expand = impurity.chiSquareStop(splits, dataCount, bestThreshold, target, self.confidence_level,
                                            self.getAttributeCount(bestAttribute), 
                                            self.getAttributeCount(target),
                                            parent_target_proportions)
        # If we should not expand, set as leaf with most common target
        if not_expand:
            return nextNode.set_leaf(default_value=default_vault)
        
        # calculate current target proportions
        current_target_proportions = utils.calculate_target_proportions(dataset, target, self.counts[target].index)
        
        attributes = list(attributes)
        # Split and continue
        nextNode.attribute = bestAttribute
        nextNode.type = "decision"

        # Check if attribute is categorical
        if dataset[bestAttribute].dtypes.name == "category":
            bestSplitValue = utils.largest_split_value(splits, bestAttribute)
            # create list of attributes with missing used attribute
            remainingAttributes = attributes.copy()
            remainingAttributes.remove(bestAttribute)
            # For each split recurse and obtain child nodes
            for split in splits:
                if len(split) == 0:
                    continue
                # Get value of current split
                value = split[bestAttribute].iloc[0]
                # recurse and return child
                child = self.train(split, remainingAttributes, target, current_target_proportions, depth+1)
                # link parent and child
                child.parent = nextNode.attribute
                nextNode.children.append(child)
                # set the predicate value of the child
                child.predValue = value
                # If this is the largest split, set it as default split for missing values
                if bestSplitValue != None and value == bestSplitValue:
                    child.default = True
        else: # assume continuous numerical
            left = splits[0]
            right = splits[1]
            # Check if either split is empty
            if len(left) == 0 or len(right) == 0:
                # Create leaf node
                nextNode.set_leaf(default_value=default_vault)
            # Can re-use continous attributes again, so do not remove attribute
            # recurse and get left child
            leftChild = self.train(left, attributes, target, current_target_proportions, depth+1)
            if leftChild != None:
                leftChild.parent = nextNode.attribute
                leftChild.predValue = bestThreshold
                leftChild.default = True
                nextNode.children.append(leftChild)
            # recurse and get right child
            rightChild = self.train(right, attributes, target, current_target_proportions, depth+1)
            if rightChild != None:
                rightChild.parent = nextNode.attribute
                rightChild.predValue = bestThreshold
                nextNode.children.append(rightChild)

        return nextNode
    
    def predict(self, dataset):
        return self.classify(dataset)
        # check if node exists, or if tree has been trained
        if self.root == None:
            # No nodes to classify with
            print("No root node trained")
            return dataset
        
        predictions = [self.apply_prediction(row, self.root, dataset.columns, dataset.dtypes) for row in dataset.iloc[:,:].values]
        return predictions
        
    def apply_prediction(self, row, node, columns, data_types):
        # check if node is a leaf
        if node.type == "leaf":
            # return value
            return node.value
        
        # Get value of row for feature
        feature_index = utils.index_of(columns, node.attribute)
        feature_value = row[feature_index]
        data_type = data_types[feature_index]
        default = None
        if data_type == 'category':
            # Iterate through children to determine where node should go
            for child in node.children:
                # Set teh default child
                if child.default:
                    default = child
                if child.predValue == feature_value:
                    return self.apply_prediction(row, child, columns, data_types)
            # No match found, use default
            if default != None:
                return self.apply_prediction(row, default, columns, data_types)
        else:
            c1 = node.children[0]
            c2 = node.children[1]
            if c1.default and (feature_value <= c1.predValue or feature_value == np.nan):
                return self.apply_prediction(row, c1, columns, data_types)
            return self.apply_prediction(row, c2, columns, data_types)
        
    def start_classify(self, queue, dataset, root):
        print('Starting to classify...')
        prediction = self.classify(dataset, root)
        queue.put(prediction)
        print('Done prediction...')

    # Given a dataset, classifies the dataset based on the trained decision tree
    def classify(self, dataset, node=None):
        # check if a tree has been trained
        if self.root == None and node == None:
            # Tree not trained, cannot classify
            print("Classifier not trained, cannot classify.")
            return dataset
        # check if dataset has data
        dataCount = len(dataset.index)
        if dataCount == 0: # No data
            return dataset
        # classify data
        if node == None: # Start from the root node
            node = self.root

        # Check if current node is leaf
        if node.type == "leaf":
            # Set target column value for all data
            dataset = dataset.assign(prediction=node.value)
            return dataset
        
        # Check if attribute is categorical
        if dataset[node.attribute].dtypes.name == 'category':
            # split data categorically
            splits = utils.split_on_feature(dataset, node.attribute, False, True)
            data = pd.DataFrame()
            # Get the default child that missing values will fit into
            defaultChild = None
            for child in node.children:
                if child.default:
                    defaultChild = child
                    break
            total_length = 0
            for split in splits:
                splitSize = len(split)
                total_length += splitSize
                if splitSize == 0:
                    continue
                # Get value of attribute in split
                value = split[node.attribute].iloc[0]
                splitMatch = False
                # iterate through children to find matching child
                for child in node.children:
                    # child matched by value or missing value to default
                    if child.predValue == value or (child.default and value == np.nan):
                        splitMatch = True
                        # concatenate results
                        data1 = self.classify(split, child)
                        dataset = dataset.combine_first(data1)
                        # data = pd.concat([data, data1])
                        break
                # if split didn't get matched for some reason, match on default child
                if not splitMatch:
                    if defaultChild != None:
                        # concatenate results
                        data1 = self.classify(split, defaultChild)
                        dataset = dataset.combine_first(data1)
                        # data = pd.concat([data, data1])
            if total_length != len(dataset):
                print(f"Size mismatch, total_length={total_length}, expected={len(dataset)}")
                exit()
            # return data
            return dataset
        else:
            # Split and classify numerical data
            c1 = node.children[0]
            c2 = node.children[1]
            split = utils.split_on_feature(dataset, node.attribute, False, True, c1.predValue)
            left = split[0]
            right = split[1]

            data1 = None
            data2 = None
            # Recursively classify
            if c1.default:
                data1 = self.classify(left, c1)
                data2 = self.classify(right, c2)
            else:
                data1 = self.classify(dataset[right], c1)
                data2 = self.classify(dataset[left], c2)

            # merge results back together
            data = pd.concat([data1, data2])
            dataset = dataset.combine_first(data)
            if (len(left) + len(right)) != len(dataset) or len(data) != len(dataset):
                print(f"Size Mismatch, left={len(left)}, right={len(left)}, data={len(data)}, expected={len(dataset)}")
                exit()
            return data

    # Return the stats of the trained decision tree
    def getStats(self):
        # check if tree has been trained
        if self.root == None:
            print("Classifier not trained, no stats to collect.")
            return
        print("Getting stats...")
        # return stats
        self.stats = Stat()
        self.getStat(self.root, 0)
        avg_depth = self.stats.cum_leaf_depths / self.stats.num_leafs
        avg_child = self.stats.cum_children / self.stats.num_decision
        return self.stats.num_decision, self.stats.num_leafs, self.stats.max_depth, avg_depth, avg_child
    
    def getStat(self, node, depth):
        # Check if node is leaf or decision and increment respective count
        # If leaf, add current depth, set max depth so far, increment leaf count
        if node.type == 'leaf':
            self.stats.num_leafs += 1
            self.stats.cum_leaf_depths += depth
            self.stats.update_max_depth(depth)
            return
        
        # If decision, add its number of children to a count
        self.stats.num_decision += 1
        self.stats.cum_children += len(node.children)
        for child in node.children:
            # If decision, recurse with incremented depth
            self.getStat(child, depth+1)

    def printTree(self, node=None):
        if node == None:
            node = self.root
        print(node)
        for child in node.children:
            self.printTree(child)

    def save_classifier(self, filename):
        # Check if tree has been trained
        if self.root == None:
            print('Tree no classified, nothing to save.')
            return
        # Create file
        file = open(filename, "a")
        # Write header
        file.write("parent,attribute,predValue,type,value,default\n")
        self.save_tree(file, self.root)

    def save_tree(self, file, node):
        file.write(f"{node.parent},{node.attribute},{node.predValue},{node.type},{node.value},{node.default}\n")
        for child in node.children:
            self.save_tree(file, child)

    def load_classifier(self, filename):
        # read data from csv format
        data = pd.read_csv(filename)
        self.nodes = len(data)
        if self.nodes == 0:
            print("No data to load")
            return
        print(f"Loading Classifier({self.nodes})...")
        self.current = 0
        data = data.fillna('missing')
        # print(data)
        # print("=======================================")
        self.load_tree(data)

    def load_tree(self, data: pd.DataFrame, node=None):
        self.current += 1
        print(f"load_tree: {self.current} / {self.nodes}")
        print(f"Size of data: {len(data)}")
        if node == None and self.root == None:
            mask = data['parent'] == 'missing'
            df = data[mask]
            if len(df) == 0:
                print("No root node in data, cannot load")
                return
            root = df.iloc[0]
            value = root['value']
            rootVal = None if value == 'missing' else value
            rootNode = Node(attribute=str(root['attribute']), type=str(root['type']), value=rootVal)
            self.root = rootNode
            self.load_tree(data[~mask], self.root)
            return
        mask = data['parent'] == node.attribute
        df = data[mask]
        children = []
        for i in range(len(df)):
            row = df.iloc[i]
            attr = None if row['attribute'] == 'missing' else str(row['attribute'])
            pred = row['predValue']
            typ = str(row['type'])
            value = None if typ == 'decision' else row['value']
            default = bool(row['default'])
            child = Node(parent=node.attribute, attribute=attr, predValue=pred, type=typ, value=value, default=default)
            children.append(child)
        node.children = children
        for child in children:
            self.load_tree(data[~mask], child)

# A node of a decision tree which can be a subtree, or a leaf.
# feature -> Feature being evaluated if it is a decision node
# predicate -> Feature predicate which filters the dataset into 
#               node by.
# value -> If the node is a leaf node, this is the value it returns
class Node():
    def __init__(self, parent=None, attribute=None, predValue=None, default=False, type=None, value=None) -> None:
        # Assign passed in values, if given
        self.parent = parent
        self.attribute = attribute
        self.predValue = predValue
        self.default = default
        self.type = type
        self.value = value
        # Create empty array for children nodes
        self.children = []

    def set_leaf(self, default_value):
        self.type = "leaf"
        self.value = default_value
        return self

    def __str__(self) -> str:
        if self.type == 'leaf':
            return f"LeafNode(parent={self.parent}, predVal={self.predValue}, default={self.default}, value={self.value})"
        else:
            return f"DecisionNode(parent={self.parent}, attribute={self.attribute}, predVal={self.predValue}, default={self.default})"
        
@dataclass
class Stat:
    num_leafs: int = 0
    num_decision: int = 0
    cum_leaf_depths: int = 0
    cum_children: int = 0
    max_depth: int = 0

    def update_max_depth(self, depth):
        if depth > self.max_depth:
            self.max_depth = depth