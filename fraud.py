import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)


import pandas as pd
pd.options.mode.chained_assignment = None # Disable write on copy warning
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
from sklearn.utils import resample
import time
from datetime import datetime
import classifier as cf
import random_forest as rf
import impurity
import utils
from multiprocessing import Process, Queue

## Setup parameters
mode = 'PROD'  # Can be 'TEST' or 'PROD', test creates a test/train split and simply tests the models, prod will create an output file
model = 'FOREST' # Can be 'TREE' or 'FOREST', tree simple runs a single decision tree, forest will run a set number of trees in a random forest
sample_size = None # If you want to run on a smaller sample of the train data, set the total sample size here
save = False # If model should be saved
undersample = True # If you want to undersample to handle class imbalances
undersample_props = [0.7, 0.3] # Total proportion of target values [0, 1], should sum to 1

total_trees = 19 # Total number of trees to create in the random forest
total_depth = 25 # Max depth of any tree
sample_cutoff = 10 # Minimum number of samples required to branch
confidence_alpha = 0.95 # Can be between 0 and 1, indicates the confidence value when performing chi-square stop calculation
gain_func = impurity.entropy # Information Gain metric function to use, entropy, giniIndex, or misclassificationIndex

def current_time_milli():
    return round(time.time() * 1000)

def prepare_data(path, missing_target=False) -> pd.DataFrame:
    # load training data
    data = pd.read_csv(path, index_col=0)

    # Prep the data
    # Normalize missing values
    data = data.mask(data == 'NotFound', np.nan)

    # Set Categorical Columns
    # categorical = ['ProductCD', 'card1', 'card2', 'card3', 'card4', 'card5', 'card6',
    #             'addr1', 'addr2', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8',
    #             'C9', 'C10', 'C11', 'C12', 'C13', 'C14']
    categorical = ['ProductCD', 'card1', 'card2', 'card3', 'card4', 'card5',
                   'card6', 'addr1', 'addr2']
    if not missing_target:
        categorical.append('isFraud')
    # Set categorical columns to type category
    for col in categorical:
        data[col] = data[col].astype('category')

    numerical = ['TransactionAmt', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7',
                 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14']
    for col in numerical:
        data[col] = data[col].astype('float')

    # Columns to drop
    drop_cols = ['TransactionDT']
    for col in drop_cols:
        data.drop(drop_cols, axis=1, inplace=True)
    
    return data

def train(classifier, data, features, target, target_proportions) -> cf.DecisionTreeClassifier:
    # Train classifier
    print('Training on dataset of size:', len(data))
    start = current_time_milli()

    if isinstance(classifier, cf.DecisionTreeClassifier):
        classifier.train(data, features, target, target_proportions) 
    else:
        classifier.train(data)

    stop = current_time_milli()
    dur = stop - start
    print('Training completed in', dur, 'ms.')
    
    return classifier

def predict(classifier, test_split) -> pd.DataFrame:
    print('Starting predictions on dataset of size:', len(test_split))
    start = current_time_milli()

    predictions = classifier.predict(test_split)

    stop = current_time_milli()
    dur = stop - start
    print('Predictions took', dur, 'ms.')
    return predictions

if __name__ == '__main__':
    # Load the data
    train_data = prepare_data('data/train/train.csv')
    test_data = prepare_data('data/test/test.csv', True)
    target = 'isFraud'
    features = list(train_data.columns)
    features.remove(target)

    # If I defined a smaller size (used for testing)
    if sample_size != None:
        train_data = resample(train_data, replace=False, n_samples=sample_size, stratify=train_data[target])

    # If undersample, reduce training size
    if undersample:
        train_data = utils.undersample(train_data, target, [0, 1], undersample_props)

    # Split the data
    train_split, test_split = train_test_split(train_data, test_size=0.2, stratify=train_data[target])

    # Get info gain function symbol
    gain_symbol = 'E'
    if gain_func == impurity.giniIndex:
        gain_symbol = 'G'
    elif gain_func == impurity.misclassificationIndex:
        gain_symbol = 'M'

    print("Starting Fraud Prediction...")
    print(f"Paramters: trees={1 if model == 'TREE' else total_trees}, depth={total_depth}, min_samples={sample_cutoff}, confidence={confidence_alpha}, metric={gain_symbol}")    
    if mode == 'TEST':
        # calculate target proportions
        target_proportions = utils.calculate_target_proportions(train_split, target, train_data[target].unique())

        classifier = None
        if model == 'TREE':
            classifier = cf.DecisionTreeClassifier(max_depth=total_depth, min_samples=sample_cutoff, imurity_function=gain_func,
                                                confidence_level=confidence_alpha)
        elif model == 'FOREST':
            classifier = rf.RandomForestClassifier(num_trees=total_trees, max_depth=total_depth, min_samples=sample_cutoff,
                                                confidence=confidence_alpha, impurity_function=gain_func,
                                                features=features, target=target)

        # Train on the data
        train(classifier, train_split, features, target, target_proportions)
        # classifier.printTree()


        # Use classifier to predict
        predictions = predict(classifier, test_split)

        # Determine accuracy
        accuracy = balanced_accuracy_score(predictions[target], predictions['prediction']) * 100
        print('Accuracy:', accuracy)
    else:
        # calculate target proportions
        target_proportions = utils.calculate_target_proportions(train_split, target, train_data[target].unique())

        classifier = None
        if model == 'TREE':
            classifier = cf.DecisionTreeClassifier(max_depth=total_depth, min_samples=sample_cutoff, imurity_function=gain_func,
                                                confidence_level=confidence_alpha)
        elif model == 'FOREST':
            classifier = rf.RandomForestClassifier(num_trees=total_trees, max_depth=total_depth, min_samples=sample_cutoff,
                                                confidence=confidence_alpha, impurity_function=gain_func,
                                                features=features, target=target)
            
        # Train on the data
        train(classifier, train_split, features, target, target_proportions)

        # num_decision, num_leafs, max_depth, avg_depth, avg_child = classifier.getStats()
        # print(f"Decision#={num_decision}, Leaf#={num_leafs}, MaxDepth={max_depth}, AvgDepth={avg_depth}, AvgChild={avg_child}")

        # Test accuracy
        # test_pred = predict(classifier, test_split)

        # Determine accuracy
        # accuracy = balanced_accuracy_score(test_pred[target], test_pred['prediction']) * 100
        # print('Accuracy:', accuracy)

        # Predict on the test data
        predictions = predict(classifier, test_data)

        # Sort data by TransactionID
        output = predictions[['prediction']].sort_index()

        # Rename prediction column
        output.rename({'prediction': 'isFraud'}, axis=1, inplace=True)

        
        # Save output
        # Get current datetime
        now = datetime.now()
        dt_string = now.strftime("%m-%d-%Y_%H-%M")
        model_path = "data/model/"
        result_path = "data/result/"
        filename = f"{dt_string}_{1 if model == 'TREE' else total_trees}t_d{total_depth}_s{sample_cutoff}_{gain_symbol}.csv"
        # output.to_csv('data/result/1t_d3_s10_E.csv')
        output.to_csv(result_path + filename)
        if model == 'TREE' and save:
            classifier.save_classifier(model_path + filename)