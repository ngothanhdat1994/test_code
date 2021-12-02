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
data =np.genfromtxt('/Users/thanhdat/EMBC_2022/Genrerate_MFCC_26_11_21/02_dl_for_dat/test_code/trill_embeddings.csv',delimiter=',')


data = np.delete(data, 0 ,0) #delete header
row, col = np.shape(data)
x = data[:,1:col-1]
y = data[:,col-1]
row, col = np.shape(x)


#factor_name=['gender','age','hypertension','heart_disease','ever_married','work_type','Residence_type','avg_glucose_level','bmi','smoking_status']

#------ Scaler 
from sklearn.preprocessing import StandardScaler
#scaler = StandardScaler()
#x = scaler.fit_transform(x)

#------- 5-fold cross validation
roc_score_rf       = 0
f1_score_weight_rf = 0
f1_score_macro_rf  = 0
f1_score_acc_rf    = 0
main_feature_rf = np.zeros((1, col))

fold_num = 5
kf = KFold(n_splits=fold_num, random_state=5, shuffle=True)
n_fold = 0
for train_index, test_index in kf.split(x):
    n_fold = n_fold+1
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    # x_train_pre, x_test = x[train_index], x[test_index]
    # y_train_pre, y_test = y[train_index], y[test_index]
    #x_train, y_train = RandomOverSampler(random_state=1).fit_resample(x_train_pre, y_train_pre)
    #x_train, y_train = SMOTE().fit_resample(x_train_pre, y_train_pre)
    #x_train, y_train = ADASYN().fit_resample(x_train_pre, y_train_pre)
    
   #------ Scaler 
   # scaler = StandardScaler()
   # x_train_scaled = scaler.fit_transform(x_train)
 
    print("\n ============================================ PROCESS FOLD {}...".format(n_fold))
    train_num = len(x_train)
    test_num = len(x_test)

    print("\n ---------- Classifier: Random Forest, train {} and test {}".format(len(x_train), len(x_test)))
    classifier = RandomForestClassifier(n_estimators=100, max_depth = 20)
    classifier.fit(x_train, y_train)
 
    predicted = classifier.predict(x_train)
    expected = y_train
 
    cm = metrics.confusion_matrix(expected, predicted)
    print("Training Confusion Matrix:\n{}".format(cm))
    print("Training Acc.={}".format(round(metrics.accuracy_score(expected, predicted),2) ))
    
    # Metrics in Train
    tn, fp, fn, tp = cm.ravel()
    
    specificity    = tn/(tn+fp)
    sensitivity    = tp/(tp+fn)
    
    fpr, tpr, thresholds = metrics.roc_curve(expected, predicted)
    auc = metrics.auc(fpr, tpr)
    
    print("\nTraining True Negative: {}".format(tn))
    print("\nTraining False Positive: {}".format(fp))
    print("\nTraining False Negative: {}".format(fn))
    print("\nTraining True Positive: {}".format(tp))
    print("\nTraining Specificity: {}".format(specificity))
    print("\nTraining Sensitivity: {}".format(sensitivity))
    print("\nTraining AUC: {}".format(auc))
    
    #------------------------------------------------------
    
    predicted = classifier.predict(x_test)
    expected = y_test
    cm = metrics.confusion_matrix(expected, predicted)
    print("\nTesting Confusion Matrix:\n{}".format(cm))
    print("Testing Acc.={}".format(round(metrics.accuracy_score(expected, predicted),2) ))
    
    # Metrics in Test
    tn, fp, fn, tp = cm.ravel()
    
    specificity    = tn/(tn+fp)
    sensitivity    = tp/(tp+fn)
    
    fpr, tpr, thresholds = metrics.roc_curve(expected, predicted)
    auc = metrics.auc(fpr, tpr)
    
    print("\nTesting True Negative: {}".format(tn))
    print("\nTesting False Positive: {}".format(fp))
    print("\nTesting False Negative: {}".format(fn))
    print("\nTesting True Positive: {}".format(tp))
    print("\nTesting Specificity: {}".format(specificity))
    print("\nTesting Sensitivity: {}".format(sensitivity))
    print("\nTesting AUC: {}".format(auc))
 
    roc_score_rf   = roc_score_rf + roc_auc_score(expected, predicted)
    f1_score_weight_rf = f1_score_weight_rf + f1_score(expected, predicted, average='weighted')
    f1_score_macro_rf  = f1_score_macro_rf + f1_score(expected, predicted, average='macro')
    f1_score_acc_rf    = f1_score_acc_rf + metrics.accuracy_score(expected, predicted)

roc_score_rf       = roc_score_rf/fold_num
f1_score_weight_rf = f1_score_weight_rf/fold_num
f1_score_macro_rf  = f1_score_macro_rf/fold_num
f1_score_acc_rf    = f1_score_acc_rf/fold_num


#print(x_train)
#print(y_train)
 

print('\n\n ================================== FINAL REPORT: Random Forest')
print(' +ROC_AUC_SCORE:{}\n +ACC:{}\n +F1_SCORE_WEIGHTED:{}\n +F1_SCORE_MACRO:{}\n'.format(roc_score_rf, f1_score_acc_rf, f1_score_weight_rf, f1_score_macro_rf))

# feature_imp = pd.Series(classifier.feature_importances_,index=factor_name).sort_values(ascending=False)
# print(feature_imp)
# sns.barplot(x=feature_imp, y=feature_imp.index)

# # Add labels the graph
# plt.xlabel('Feature Importance Score')
# plt.ylabel('Features')
# plt.title("Visualizing Important Features")
# plt.legend()
# plt.show()
