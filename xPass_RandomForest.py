import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error
import math

trainX="train_inputs.txt"#training data file
trainY="train_pass_success.txt"#training data success/failure

testX="test_inputs.txt"#test data file
testY="test_pass_success.txt"#test data success/failure

savename="RandFor_results.txt"
num_estimators=100#The number of trees in the forest

def calc_xP_RandomForest(trainX,trainY,testX,testY,num_estimators):#calculate expected passes from training and test data

    X_train = load_data_from_file(trainX)#get regression inputs
    y_train = load_data_from_file(trainY)#Success or fail data

    X_test = load_data_from_file(testX)#get regression inputs
    y_test = load_data_from_file(testY)#Success or fail data

    y_naive = np.empty(len(y_test))
    y_naive.fill(np.mean(y_train))#Calculate naive xPass based on average pass completion rate
    print "Naive data RMSE:",math.sqrt(mean_squared_error(y_test, y_naive))

    xP_RandForest = RandForest(X_train, y_train, X_test, num_estimators)#xP calculated via random forest on training data set
    print "RMSE using Random Forest:",math.sqrt(mean_squared_error(y_test, xP_RandForest))
    np.savetxt(savename, xP_RandForest, delimiter=',')  # save text file with results

def RandForest(X, y, pred_data, num_estimators):#random forest function
    clf = RandomForestClassifier(n_estimators=num_estimators) #define random forest parameters
    clf.fit(X, y)
    print clf.feature_importances_
    y_pred = clf.predict_proba(pred_data)[:, 1]
    return y_pred

def load_data_from_file(file):#load data from file
    #file is defined below
    f = open(file, 'r')
    data = np.genfromtxt(f, delimiter=',')#assumes delimiter is a comma
    data = np.delete(data, 0, 0)  # Erases the first row (i.e. the header)
    f.close()
    return data
