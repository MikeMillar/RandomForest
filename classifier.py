import pandas as pd
import numpy as np
import impurity

class DecisionTreeClassifier():

    # Initialize the decision tree classifier.
    # impurity_function -> function to evaluate impurity of the dataset with. default: entropy
    # confidence_level -> how confident do we want to train the tree to be. default: 95% (0.95)
    def __init__(self, imurity_function=impurity.entropy, confidence_level=0.95) -> None:
        self.root = None
        self.impurity_function = imurity_function
        self.confidence_level = confidence_level
        pass

    # Given a training dataset, target, and attributes, creates a decision tree
    # dataset -> Dataset to train with
    # target -> name of target attribute
    # attributes -> list of attributes to train with
    def train(self, dataset, target, attributes):
        # Create a new node
        nextNode = Node()

        # If root not yet set, set nextNode as root
        if self.root == None:
            self.root = nextNode

        # check if dataset has any data
        dataCount = len(dataset.index)
        if dataCount == 0: # No data
            nextNode.type = "leaf"
            return nextNode
        elif dataCount == 1: # Only 1 data, no need to classify
            nextNode.type = "leaf"
            nextNode.value = dataset[0][target]
            return nextNode
        # check impurity of dataset with respect to target
        entropy = impurity.entropy(dataset, target)
        if entropy == 0: # No impurity in dataset
            nextNode.type = "leaf"
            nextNode.value = dataset[target].iloc[0]
            return nextNode
        # check if there are remaining attributes to train with
        if len(attributes) == 0: # No remaining attributes to train with
            # Get most common value of target dataset
            nextNode.type = "leaf"
            nextNode.value = dataset[target].value_counts().idxmax()
            return nextNode
        
        # For each attribute, calculate information gain and keep best
        bestAttribute, bestGain, bestThreshold = impurity.findHighestGainAttribute(dataset, attributes, target, self.impurity_function)

        # Test if gain is better than random
        not_expand = impurity.chiSquareStop(dataset, bestAttribute, target, self.confidence_level)
        # If we should not expand, set as leaf with most common target
        if not_expand:
            # Get most common value of target dataset
            nextNode.type = "leaf"
            nextNode.value = dataset[target].value_counts().idxmax()
            return nextNode
        
        attributes = list(attributes)
        # Split and continue
        nextNode.attribute = bestAttribute
        nextNode.type = "decision"
        # Check if attribute is categorical
        if dataset[bestAttribute].dtypes.name == "category":
            # Split dataset of values
            splits = [y for x, y in dataset.groupby(bestAttribute)]
            for split in splits:
                value = split[bestAttribute].value_counts().idxmax()
                pred = lambda x : x == value
                # create list of attributes with missing used attribute
                remainingAttributes = attributes.copy()
                remainingAttributes.remove(bestAttribute)
                # Set the child predicate, and recurse
                nextNode.predicates.append(pred)
                child = self.train(split, target, remainingAttributes)
                child.parent = nextNode.attribute
                nextNode.children.append(child)
        else: # assume continuous numerical
            # Split data based on best info gain threshold
            leftPred = lambda x : x <= bestThreshold
            rightPred = lambda x : x > bestThreshold
            nextNode.predicates.append(leftPred)
            nextNode.predicates.append(rightPred)
            left = np.array([y for x, y in dataset if y[bestAttribute] <= bestThreshold])
            right = np.array([y for x, y in dataset if y[bestAttribute] > bestThreshold])
            # Can re-use continous attributes again, so do not remove attribute
            leftChild = self.train(left, target, attributes)
            leftChild.parent = nextNode.attribute
            rightChild = self.train(right, target, attributes)
            rightChild.parent = nextNode.attribute
            nextNode.children.append(leftChild)
            nextNode.children.append(rightChild)

        return nextNode

    # Given a dataset, classifies the dataset based on the trained decision tree
    def classify(self):
        # check if a tree has been trained
        # check if dataset has data
        # classify data
        pass

    # Given a trained decision tree, save the tree to a file
    def save(self, filename):
        # check if tree has been trained
        # save to file
        pass

    # Given a decision tree file, load the decision tree
    def load(self, filename):
        # check if file exists
        # attempt to create decision tree
        pass

    # Return the stats of the trained decision tree
    def getStats(self):
        # check if tree has been trained
        # return stats
        pass

    def printTree(self, node=None):
        if node == None:
            node = self.root
        print(node)
        for child in node.children:
            self.printTree(child)


# A node of a decision tree which can be a subtree, or a leaf.
# feature -> Feature being evaluated if it is a decision node
# predicate -> Feature predicate which filters the dataset into 
#               node by.
# value -> If the node is a leaf node, this is the value it returns
class Node():
    def __init__(self, parent=None, attribute=None, type=None, value=None) -> None:
        # Assign passed in values, if given
        self.parent = parent
        self.attribute = attribute
        self.type = type
        self.value = value
        # Create empty array for children nodes
        self.children = []
        # Create predicate for each child node
        self.predicates = []

    def __str__(self):
        return f"Node(attribute={self.attribute}, parent={self.parent}, type={self.type}, value={self.value})"