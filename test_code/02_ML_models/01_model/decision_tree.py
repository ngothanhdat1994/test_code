import pandas as pd
import numpy as np
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from sklearn import svm, metrics
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns


#---- Get data by using numpy
data =np.genfromtxt('./../02_clear_data/new_train_1.csv',delimiter=',')
data = np.delete(data, 0 ,0) #delete header
row, col = np.shape(data)
x = data[:,0:col-1]
y = data[:,col-1]
row, col = np.shape(x)


#------ Scaled
from sklearn.preprocessing import StandardScaler

#------- 5-fold cross validation

roc_score_dt       = 0
f1_score_weight_dt = 0
f1_score_macro_dt  = 0
f1_score_acc_dt    = 0
main_feature_dt = np.zeros((1, col)) 


fold_num = 5
kf = KFold(n_splits=fold_num)
n_fold = 0
for train_index, test_index in kf.split(x):
    n_fold = n_fold+1
    x_train_pre, x_test = x[train_index], x[test_index]
    y_train_pre, y_test = y[train_index], y[test_index]
    
    
    x_train, y_train = RandomOverSampler(random_state=1).fit_resample(x_train_pre, y_train_pre)
    #x_train, y_train = SMOTE().fit_resample(x_train_pre, y_train_pre)
    #x_train, y_train = ADASYN().fit_resample(x_train_pre, y_train_pre)
    
    #------ Scaler 
   # scaler = StandardScaler()
   # x_train_scaled = scaler.fit_transform(x_train)

        
    print("\n ============================================ PROCESS FOLD {}...".format(n_fold))
    train_num = len(x_train)
    test_num = len(x_test)

    print("\n ---------- Classifier: Decision Tree, train {} and test {}".format(len(x_train), len(x_test)))
    classifier = DecisionTreeClassifier(criterion="entropy", max_depth=10)
    classifier.fit(x_train, y_train)
 
    predicted = classifier.predict(x_train)
    expected = y_train

    cm = metrics.confusion_matrix(expected, predicted)
    print("Traing Confusion Matrix:\n{}".format(cm))
    print("Traing Acc.={}".format(round(metrics.accuracy_score(expected, predicted),2) ))

    predicted = classifier.predict(x_test)
    expected = y_test
    cm = metrics.confusion_matrix(expected, predicted)
    print("\nTesting Confusion Matrix:\n{}".format(cm))
    print("Testing Acc.={}".format(round(metrics.accuracy_score(expected, predicted),2) ))

    roc_score_dt   = roc_score_dt + roc_auc_score(expected, predicted)
    f1_score_weight_dt = f1_score_weight_dt + f1_score(expected, predicted, average='weighted')
    f1_score_macro_dt  = f1_score_macro_dt + f1_score(expected, predicted, average='macro')
    f1_score_acc_dt    = f1_score_acc_dt + metrics.accuracy_score(expected, predicted)


roc_score_dt        = roc_score_dt/fold_num
f1_score_weight_dt  = f1_score_weight_dt/fold_num
f1_score_macro_dt   = f1_score_macro_dt/fold_num
f1_score_acc_dt     = f1_score_acc_dt/fold_num

print('\n\n ================================== FINAL REPORT: Decision Tree...')
print(' +ROC_AUC_SCORE:{}\n +ACC:{}\n +F1_SCORE_WEIGHTED:{}\n +F1_SCORE_MACRO:{}\n'.format(roc_score_dt, f1_score_acc_dt, f1_score_weight_dt, f1_score_macro_dt))



