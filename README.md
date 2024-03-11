# Random Forest - Fraud Detection
This project uses a collection of Decision Trees in a Random Forest configuration to detect fraudulent charges. It was created as a project for the UNM CS429 - Intro to Machine Learning course in Spring 2024.
---
# Developers
- Michael Millar: Sole developer

# How to use
## Setup
1) Have python 3 and pip installed on your computer.
2) With pip installed run the command `pip install -r requirements.txt` to install dependencies.

## Config
There are several configuration options you can specify before running the application. They can all be changed from `fraud.py` file.
- `mode`: either `'PROD'` or `'TEST'`. Test just runs on the training data and does not create an output file for kaggle submission.
- `model`: either `'TREE'` or `'FOREST'`. Tree will run a single tree regardless of number of trees specified, while forest will train and test multiple trees as specified.
- `sample_size`: If you want to reduce the total samples of the original data set to a set smaller number. Leave set to `None` otherwise.
- `save`: If the model should be saved to a file. Saved models with numerical data may have issues being reloaded (infinite size).
- `undersample`: either `True` or `False`. If true, will resample the training data based on the proportions specified in `undersample_props`.
- `undersample_props`: An array-like value which indicates the proportions of each target class. Should sum to 1.
- `total_trees`: Total number of trees to create in the random forest.
- `total_depth`: The maximum depth any tree can grow to.
- `sample_cutoff`: The minimum samples required to branch.
- `confidence_alpha`: Confidence level between 0 and 1.
- `gain_func`: Information gain metric to use, valid options are `impurity.entropy`, `impurity.giniIndex`, and `impurity.misclassificationIndex`.
- `train_file`: Path to the training data
- `test_file`: Path to the testing data

## Running the Program
To run the program you just need to have python run the `fraud.py` file. This can be done with the command: `python fraud.py`. You may need to use `python3` depending on your setup.

---
# Kaggle Competition Result
In the Kaggle competition, this algorithm had the highest balanced accuracy score of 77.06 on 3/9/24. This was achieved with a single decision tree using entropy as information gain metric, min samples of 10, max depth of 25, and a confidence value of 0.95.

The results from the random forest always produced lower accuracy ratings, despite the individual trees of the random forest having higher accuracy.

---
# File Manifest
- .gitignore: Hide all the unwanted files from the git repository.
- README.md: This description file.
- classifier.py: This contains the logic of the `DecisionTreeClassifier` and the `Node` class. This is the class that builds, trains, and predicts with.
- fraud.py: The starting point of the project. This file loads and prepares the data, initializes the classifiers, and starts the train and test processes.
- impurity.py: This file contains the entropy, gini index, misclassification index, information gain, and chi-square methods.
- random_forest.py: This file contains the `RandomForestClassifier` and creates the decision trees in the random forests.
- requirements.txt: The file that contains all the additional libraries required to run the script.
- utils.py: This file contains some utility helper functions that get used multiple times.