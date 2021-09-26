import pandas as pd
import numpy as np


#To check the validity of your dataset
#sklearn.model_selection.permutation_test_score()

from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score

#To check the validity of your dataset
#sklearn.model_selection.permutation_test_score()

def get_dataset(dataset):

    train_test_vars = [
        0.25,
        0
    ]
    Train_test_vars = {
        "test_size": 0.25,
        "random_split": 0,
        "NotInt_Columns": []
    }

    # Remember to put your dependent variable in the last column of your csv file
    dataset = pd.read_csv("{}".format(dataset))
    X = dataset.iloc[:, :-1].values
    Y = dataset.iloc[:, -1].values

    
    #imputer = SimpleImputer(missing_values = np.nan, strategy = "mean")
    #Index doesn't include end value
    #imputer.fit(X[:, 1:-1])
    #X[:, 1:-1] = imputer.transform(X[:, 1:-1])

    #Add this Encoder if you have non-int values

    #try:
     #ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), Train_test_vars["NotInt_Columns"])], remainder='passthrough')
     #try:
       #X = np.array(ct.fit_transform(X))
     #except:
       
    #except TypeError:
       #return "Couldn't encode your not int dependent variable"


    try:
     le = LabelEncoder()
     Y = le.fit_transform(Y)
    except TypeError:
      print("Couldn't encode your independent variable")

    
    #test_size, random_state,  = train_test_vars.values(), train_test_vars.values()

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = Train_test_vars["test_size"], random_state = Train_test_vars["random_split"])
    
    return X_train, X_test, Y_train, Y_test

#def kernel_SVM(X_train, X_test, Y_train, Y_test):
    
def feature_scaling(X_train, X_test):

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    return X_train, X_test

def final_scoring(X_train, Y_train):

    #cv var is variable
    accuracies = cross_val_score(estimator = classifier, X = X_train, Y = Y_train, cv = 10)
    mean_accuracy, std_accuracy = accuracies.mean()*100, accuracies.std()*100
    
    return mean_accuracy, std_accuracy

#def dim_reduce(X_train, X_test):
  #from sklearn.decomposition import PCA
  #pca = PCA(n_components = 2)
  #X_train = pca.fit_transform(X_train)
  #X_test = pca.transform(X_test)

  #return X_train, X_test

#def dim_reduce(X_train, X_test):
  #from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
  #lda = LDA(n_components = 2)
  #X_train = lda.fit_transform(X_train, Y_train)
  #X_test = lda.transform(X_test)

  #return X_train, X_test

#def dim_reduce(X_train, X_test):
  #from sklearn.decomposition import KernelPCA  
  #kpca = KernelPCA(n_components = 2, kernel = 'rbf')
  #X_train = kpca.fit_transform(X_train)
  #X_test = kpca.transform(X_test)

  #return X_train, X_test