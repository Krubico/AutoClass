from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from Format import final_scoring_system

Classifier_dict = {
    1: {"name": SVC, "fine_tuning": [
        {'C': [0.25, 0.5, 0.75, 1], 'kernel': ['linear']},
        {'C': [0.25, 0.5, 0.75, 1], 'kernel': ['rbf'], 'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}], "Scale": 1},
    2: {"name":KNeighborsClassifier , "fine_tuning": [{
        "n_neighbors": [5, 10, 15, 20], 'metric': ['minkowski'], "p": [2]}], "Scale": 1},
    3: {"name": CatBoostClassifier, "fine_tuning": {}, "Scale": 0},
    4: {"name": GaussianNB, "fine_tuning": {}, "Scale": 1},
    5: {"name":LogisticRegression, "fine_tuning": [{
        'random_state': [0,1]}], "Scale": 1},
}

Clustering_dict = {
    1: {"name": "k_means"}
}

#Adding precision & roc accuracy score 
#cv to change
#Kernel PCA Implementation
def fine_tune_scoring(key, X_train, Y_train):

    from sklearn.model_selection import GridSearchCV

    parameters = Classifier_dict[key]["fine_tuning"] 
    grid_search = GridSearchCV(estimator = Classifier_dict[key]["name"](),
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1,
                           )

    grid_search.fit(X_train, Y_train)
    best_accuracy = grid_search.best_score_
    best_parameters = grid_search.best_params_

    return best_accuracy*100, best_parameters


def cm_scoring(classifier, X_test, Y_test):

    from sklearn.metrics import confusion_matrix, accuracy_score

    Y_pred = classifier.predict(X_test)
    cm = confusion_matrix(Y_test, Y_pred)
    test_cm_score = accuracy_score(Y_test, Y_pred)   

    return cm, test_cm_score



