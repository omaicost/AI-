import numpy as np
import matplotlib.pyplot as plt

from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold


def evaluatePerformance():
    '''
    Evaluate the performance of decision trees,
    averaged over 100 trials of 10-fold cross validation
    
    Return:
      a matrix giving the performance that will contain the following entries:
      stats[0,0] = mean accuracy of decision tree
      stats[0,1] = std deviation of decision tree accuracy
      stats[1,0] = mean accuracy of decision stump
      stats[1,1] = std deviation of decision stump accuracy
      stats[2,0] = mean accuracy of 3-level decision tree
      stats[2,1] = std deviation of 3-level decision tree
      
    ** Note that your implementation must follow this API**
    '''
    
    # Load Data
    filename = 'data/SPECTF.dat'
    data = np.loadtxt(filename, delimiter=',')
    X = data[:, 1:]
    y = np.array([data[:, 0]]).T
    n, d = X.shape
    
    # Shuffle the data
    idx = np.arange(n)
    np.random.seed(13)
    np.random.shuffle(idx)
    X = X[idx]
    y = y[idx]
    
    # Define the number of trials and folds
    num_trials = 100
    num_folds = 10
    
    # Initialize lists to store accuracy scores
    accuracy_scores_decision_tree = []
    accuracy_scores_decision_stump = []
    accuracy_scores_dt3 = []
    
    # Perform trials of cross-validation
    for _ in range(num_trials):
        # Initialize the cross-validation splitter
        kf = KFold(n_splits=num_folds, shuffle=True)
        
        # Initialize variables to store accuracy scores for each model
        decision_tree_scores = []
        decision_stump_scores = []
        dt3_scores = []
        
        # Perform cross-validation
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # Train the decision tree
            clf = tree.DecisionTreeClassifier()
            clf = clf.fit(X_train, y_train)

            # Make predictions on the test set
            y_pred = clf.predict(X_test)

            # Compute the accuracy score for decision tree
            decision_tree_accuracy = accuracy_score(y_test, y_pred)
            decision_tree_scores.append(decision_tree_accuracy)

            # Compute the accuracy score for decision stump
            majority_class = np.argmax(np.bincount(y_train.flatten()))
            y_pred_stump = np.full(y_test.shape, majority_class)
            decision_stump_accuracy = accuracy_score(y_test, y_pred_stump)
            decision_stump_scores.append(decision_stump_accuracy)

            # Train a 3-level decision tree
            clf_dt3 = tree.DecisionTreeClassifier(max_depth=3)
            clf_dt3 = clf_dt3.fit(X_train, y_train)

            # Make predictions on the test set
            y_pred_dt3 = clf_dt3.predict(X_test)

            # Compute the accuracy score for 3-level decision tree
            dt3_accuracy = accuracy_score(y_test, y_pred_dt3)
            dt3_scores.append(dt3_accuracy)
        
        # Calculate mean accuracy scores for each model in the current trial
        decision_tree_mean = np.mean(decision_tree_scores)
        decision_stump_mean = np.mean(decision_stump_scores)
        dt3_mean = np.mean(dt3_scores)
        
        # Append mean accuracy scores to the main lists
        accuracy_scores_decision_tree.append(decision_tree_mean)
        accuracy_scores_decision_stump.append(decision_stump_mean)
        accuracy_scores_dt3.append(dt3_mean)
    
    # Calculate mean and standard deviation of accuracy scores across trials
    meanDecisionTreeAccuracy = np.mean(accuracy_scores_decision_tree)
    stddevDecisionTreeAccuracy = np.std(accuracy_scores_decision_tree)
    meanDecisionStumpAccuracy = np.mean(accuracy_scores_decision_stump)
    stddevDecisionStumpAccuracy = np.std(accuracy_scores_decision_stump)
    meanDT3Accuracy = np.mean(accuracy_scores_dt3)
    stddevDT3Accuracy = np.std(accuracy_scores_dt3)

    # Make certain that the return value matches the API specification
    stats = np.zeros((3, 2))
    stats[0, 0] = meanDecisionTreeAccuracy
    stats[0, 1] = stddevDecisionTreeAccuracy
    stats[1, 0] = meanDecisionStumpAccuracy
    stats[1, 1] = stddevDecisionStumpAccuracy
    stats[2, 0] = meanDT3Accuracy
    stats[2, 1] = stddevDT3Accuracy
    return stats


if __name__ == "__main__":
    stats = evaluatePerformance()
    print("Decision Tree Accuracy = ", stats[0, 0], " (", stats[0, 1], ")")
    print("Decision Stump Accuracy = ", stats[1, 0], " (", stats[1, 1], ")")
    print("3-level Decision Tree = ", stats[2, 0], " (", stats[2, 1], ")")
