class DecisionTreeClassifier():
    def __init__(self) -> None:
        self.root = None
        pass

    # Given a training dataset, target, and attributes, creates a decision tree
    # dataset -> Dataset to train with
    # target -> name of target attribute
    # attributes -> list of attributes to train with
    def train(dataset, target, attributes):
        # check if dataset has any data
        # check impurity of dataset with respect to target
        # check if there are remaining attributes to train with
        pass

    # Given a dataset, classifies the dataset based on the trained decision tree
    def classify():
        # check if a tree has been trained
        # check if dataset has data
        # classify data
        pass

    # Given a trained decision tree, save the tree to a file
    def save(filename):
        # check if tree has been trained
        # save to file
        pass

    # Given a decision tree file, load the decision tree
    def load(filename):
        # check if file exists
        # attempt to create decision tree
        pass

    # Return the stats of the trained decision tree
    def getStats():
        # check if tree has been trained
        # return stats
        pass

# A node of a decision tree which can be a subtree, or a leaf.
# feature -> Feature being evaluated if it is a decision node
# predicate -> Feature predicate which filters the dataset into 
#               node by.
# value -> If the node is a leaf node, this is the value it returns
class Node():
    def __init__(self, feature=None, predicate=None, type=None, value=None) -> None:
        # Assign passed in values, if given
        self.feature = feature
        self.predicate = predicate
        self.type = type
        self.value = value
        # Create empty array for children nodes
        self.children = []