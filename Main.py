from Classifiers import Classifier_dict, fine_tune_scoring, cm_scoring
from Format import feature_scaling, dim_reduce, get_dataset

def main(dataset):
    print("Send me your CSV please")
    best_fine_tune_score = 0
    best_cm_score = 0

    X_train, X_test, Y_train, Y_test = get_dataset(dataset)
    #X_train, X_test = dim_reduce(X_train, X_test)


    for key in Classifier_dict:

        classifier = Classifier_dict[key]["name"]()

        if Classifier_dict[key]["Scale"] == 1:
            X_train, X_test = feature_scaling(X_train, X_test)

        if Classifier_dict[key]["Scale"] == 1:
            fine_tuned_score, best_params = fine_tune_scoring(key, X_train, Y_train)
            classifier = classifier.set_params(**best_params)
            classifier.fit(X_train, Y_train)
            cm, test_cm_score = cm_scoring(classifier, X_test, Y_test)
        else:
            classifier.fit(X_train, Y_train)
            cm, test_cm_score = cm_scoring(classifier, X_test, Y_test)
            
        if best_fine_tune_score < fine_tuned_score: 
            best_fine_tune_score = fine_tuned_score
            train_classifier_name = "{}".format(classifier)
            best_train_parameters = best_params
            
        
        if best_cm_score < test_cm_score:
            best_cm_score = test_cm_score
            best_cm = cm
            test_classifier_name = "{}".format(classifier)
            best_test_parameters = best_params
        
        #If best_fine_tune_score == fine_tuned_score
        #if best_cm_score == test_cm_score
          
        cm = 0
        test_cm_score = 0
        fine_tuned_score = 0
        best_params = 0


    if (train_classifier_name == test_classifier_name) & (best_cm_score == best_fine_tune_score):
        print("Best test classifier: {} {} \n Parameters: {} Confusion Matrix {}".format(train_classifier_name, best_cm_score, best_test_parameters, best_cm))
    else: 
        print("Best train classifier: {} {}% \n Parameters: {}".format(train_classifier_name, best_fine_tune_score, best_train_parameters))
        print("Best test classifier: {} {} \n Parameters: {} Confusion Matrix {}".format(test_classifier_name, best_cm_score, best_test_parameters, best_cm))

main('PUT YOUR DATASET HERE PLEASE!')