#!/usr/bin/env python

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix,classification_report, f1_score, roc_auc_score,\
                recall_score
#from sklearn.metrics import classification_report
#from sklearn.metrics import confusion_matrix
#from sklearn.cluster import KMeans
from sklearn.mixture import GMM
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
#from sklearn import metrics
#from sklearn.metrics import roc_auc_score
#from sklearn.metrics import recall_score
from sklearn.feature_selection import RFE
from sklearn.linear_model import Ridge
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

# %matplotlib inline
names = ['id','age', "gender", "education", "country","ethnicity","neuroticism", "extraversion",
"openness", "agreeableness", "conscient", "impulsuv","SS", "alco_consp", "amphet_consp", "amly_consp",
"benzos_consp", "caff_consp", "cannabis_consp","choc_consp", "cocaine_consp", "crack_consp",
"ecstasy_consp", "heroin_consp", "ketamine_consp","legalh_consp", "lsd_consp", "meth_consp",
"mmushroom_consp","nicotine_consp", "semeron_consp", "vsa_consp"]

# reading drug consumption data
drug_consumption = pd.read_csv(
    "http://archive.ics.uci.edu/ml/machine-learning-databases/00373/drug_consumption.data",
                               header = None,index_col = None, names = names)
# writing data
drug_consumption.to_csv("data/drug_consumption.csv")
drug_consumption.head()

# some data wrangling
# coding drug consumption variables into binary consumption variable based on frequecy of use
drug_data = drug_consumption.copy()
for i in names:
    if drug_data[i].dtype == 'object':
        #drug_data[i + "_ever"] = drug_data[i].map({"CL0": 0, "CL1": 1, "CL2": 1, "CL3": 1,
        #                                "CL4": 1, "CL5":1, "CL6":1})
        drug_data[i + "_decade"] = drug_data[i].map({"CL0": 0, "CL1": 0, "CL2": 1, "CL3": 1,
                                        "CL4": 1, "CL5":1, "CL6":1})
        #drug_data[i + "_year"] = drug_data[i].map({"CL0": 0, "CL1": 0, "CL2": 0, "CL3": 1,
        #                                "CL4": 1, "CL5":1, "CL6":1})
        #drug_data[i + "_month"] = drug_data[i].map({"CL0": 0, "CL1": 0, "CL2": 0, "CL3": 0,
        #                                "CL4": 1, "CL5":1, "CL6":1})
        #drug_data[i + "_week"] = drug_data[i].map({"CL0": 0, "CL1": 0, "CL2": 0, "CL3": 0,
        #                                "CL4": 0, "CL5":1, "CL6":1})
        #drug_data[i + "_day"] = drug_data[i].map({"CL0": 0, "CL1": 0, "CL2": 0, "CL3": 0,
        #                                "CL4": 0, "CL5":0, "CL6":1})
        drug_data.drop([i], axis =1, inplace = True)

# coding gender variable into categorical variable: female == 1, male == 0
drug_data["gender"] = np.where(drug_data['gender']>0, 1, 0)
drug_data.drop('id', axis =1, inplace = True)
print(drug_data.shape)
#print(drug_data.columns)
print(drug_data.head())

# creating personal traits data
drug_traits = drug_data.loc[:,['age', "gender", "education", "country", "ethnicity","neuroticism",
        "extraversion", "openness", "agreeableness", "conscient", "impulsuv","SS"]]

print(drug_traits.shape)
# drug_traits.head()
print("================================================")

# Logistic regression with 'L1' penalty + Oversampling on drug_traits data
print("A : LogisticRegression with penalty l1 with personal traits data")

X = drug_traits

# drug analyzed
y = drug_data.loc[:,["lsd_consp_decade", "crack_consp_decade", "amly_consp_decade"]]

for i in [0, 1, 2]:
    X_train, X_test, y_train, y_test = train_test_split(X, y.iloc[:,i], test_size = 0.2, random_state = 233)
    # oversampling
    sm = SMOTE(random_state=12, ratio = 1.0)
    x_res, y_res = sm.fit_sample(X_train, y_train)
    #print(y_train.value_counts(), np.bincount(y_res))

    param_grid = {"C" : 10.0**np.arange(-3,3)}
    grid = GridSearchCV(LogisticRegression(penalty = 'l1'), param_grid = param_grid, scoring = 'recall', cv =10)
    grid.fit(x_res, y_res)

    print("best param C: {}".format(grid.best_params_['C']))
    print("best score: {}".format(grid.best_score_))
    logit = LogisticRegression(penalty = 'l1', C = grid.best_params_['C'])
    logit.fit(x_res, y_res)

    L1_selec_feat = np.where(np.absolute(logit.coef_/np.max(logit.coef_))> 10**-6)
    print("\nNumber of features chosen by 'L1' regularization: {}".format(len(L1_selec_feat[0])))
    print("Features chosen by L1 regularization:\n{}".format(X.columns[L1_selec_feat[1]]))

    print("Method: Logistic Regression (l1)")
    #print("X_test.shape {}".format(X_test.shape))
    #print("x_res.shape {}".format(x_res.shape))
    print("{}".format(y.columns[i]))
    #print("training accuracy with oversampled data is {:.3f}".format(logit.score(x_res, y_res)))
    print("test data recall with oversampled data is {:.3f}".format(recall_score(y_test, logit.predict(X_test))))
    print("test data accuracy is {:.3f}".format(logit.score(X_test, y_test)))
    print("Classification report:\n {}".format(classification_report(y_test,logit.predict(X_test),
                                        target_names = ['not used', 'used'])))
    print("Confussion matrix: \n {}".format(confusion_matrix(y_test, logit.predict(X_test))))

    print("--------------------------------")
print("================================================")
print("B : KNN algorithm with personal traits data")
# KNN algorithm + Oversampling on drug_traits data

X = drug_traits
y = drug_data.loc[:,["lsd_consp_decade", "crack_consp_decade", "amly_consp_decade"]]

for i in [0, 1, 2]:
    X_train, X_test, y_train, y_test = train_test_split(X, y.iloc[:,i], test_size = 0.2, random_state = 233)
    # oversampling
    sm = SMOTE(random_state=12, ratio = 1.0)
    x_res, y_res = sm.fit_sample(X_train, y_train)

    param_grid = {"n_neighbors" : np.arange(10, 80, 2)}
    grid = GridSearchCV(KNeighborsClassifier(),param_grid = param_grid, scoring = 'recall', cv =10)
    grid.fit(x_res, y_res)

    print("best param n_neighbors: {}".format(grid.best_params_['n_neighbors']))
    print("best score: {}".format(grid.best_score_))
    knn = KNeighborsClassifier(n_neighbors = grid.best_params_['n_neighbors'])
    knn.fit(x_res, y_res)

    print("method : Kneighbors")
    print("{}".format(y.columns[i]))
    # print("X_test.shape {}".format(X_test.shape))
    # print("x_res.shape {}".format(x_res.shape))
    print("test data recall is {:.3f}".format(recall_score(y_test, knn.predict(X_test))))
    print("Classification report:\n {}".format(classification_report(y_test,knn.predict(X_test),
                                        target_names = ['not used', 'used'])))
    print("Confussion matrix: \n {}".format(confusion_matrix(y_test, knn.predict(X_test))))

    print("--------------------------------")

print("================================================")


print("C: Random Forest with personal traits data")

# Random Forest on drug_traits data
# Over sampling is done before RFE

drug_traits = drug_data.loc[:,['age', "gender", "education", "country",
"ethnicity","neuroticism", "extraversion",
"openness", "agreeableness", "conscient", "impulsuv",
"SS"]]
drug_traits_copy = drug_traits.copy()

X = drug_traits_copy.values
y = drug_data.loc[:,["lsd_consp_decade", "crack_consp_decade", "amly_consp_decade"]]

for j in [0, 1, 2]:

    X_train, X_test, y_train, y_test = train_test_split(X, y.iloc[:,j], test_size = 0.2, random_state = 22345)

    sm = SMOTE(random_state=12, ratio = 1.0)
    X_train_res, y_train_res = sm.fit_sample(X_train, y_train)

    no_feature = np.arange(1, 10, 1)
    recall_scores = []
    selected_features = []
    best_params = []
    for i in no_feature:
        rfe = RFE(estimator = LogisticRegression(), n_features_to_select = i)
        rfe.fit(X_train_res, y_train_res)
        X_train_sel = X_train_res[:, rfe.support_]
        X_test_sel = X_test[:, rfe.support_]
        selected_features.append(rfe.support_)

        param_grid = {"n_estimators" : np.arange(10, 100, 5)}
        grid = GridSearchCV(RandomForestClassifier(), param_grid = param_grid, scoring = 'recall', cv =10)
        grid.fit(X_train_sel, y_train_res)
        rforest = RandomForestClassifier(n_estimators = grid.best_params_['n_estimators'])
        rforest.fit(X_train_sel, y_train_res)
        best_params.append(grid.best_params_['n_estimators'])
        recall_scores.append(recall_score(y_test, rforest.predict(X_test_sel)))

    opt_selected_features = selected_features[np.argmax(recall_scores)]
    opt_n = best_params[np.argmax(recall_scores)]

    print(y.columns[j])
    print("Optimum number of features is {}".format(no_feature[np.argmax(recall_scores)]))
    print(selected_features[np.argmax(recall_scores)])
    print("name of selected_features {}".format(drug_traits_copy.columns[opt_selected_features]))
    #print(recall_scores)
    #plt.plot(no_feature, recall_scores,"-o", color = 'red');

    X_test_sel = X_test[:, opt_selected_features]
    rforest = RandomForestClassifier(n_estimators= opt_n)
    rforest.fit(X_train_res, y_train_res)
    print()
    print("training recall is {:.3f}".format(recall_score(y_train_res, rforest.predict(X_train_res))))
    print("test recall is {:.3f}".format(recall_score(y_test, rforest.predict(X_test))))
    print("Classification report:\n {}".format(classification_report(y_test,rforest.predict(X_test),
                                        target_names = ['not used', 'used'])))
    print("Confussion matrix: \n {}".format(confusion_matrix(y_test, rforest.predict(X_test))))


print("================================================")


print("D: RFE() and Logistic Regression(penalty = l2) with personal traits data")

# RFE + Logistic Regression. Over sampling is done before RFE

drug_traits = drug_data.loc[:,['age', "gender", "education", "country",
"ethnicity","neuroticism", "extraversion",
"openness", "agreeableness", "conscient", "impulsuv",
"SS"]]
drug_traits_copy = drug_traits.copy()

X = drug_traits_copy.values
y = drug_data.loc[:,["lsd_consp_decade", "crack_consp_decade", "amly_consp_decade"]]

for j in [0, 1, 2]:

    X_train, X_test, y_train, y_test = train_test_split(X, y.iloc[:,j], test_size = 0.2, random_state = 22345)

    sm = SMOTE(random_state=12, ratio = 1.0)
    X_train_res, y_train_res = sm.fit_sample(X_train, y_train)

    no_feature = np.arange(1, 10, 1)
    recall_scores = []
    selected_features = []
    best_params = []
    for i in no_feature:
        rfe = RFE(estimator = LogisticRegression(), n_features_to_select = i)
        rfe.fit(X_train_res, y_train_res)
        X_train_sel = X_train_res[:, rfe.support_]
        X_test_sel = X_test[:, rfe.support_]
        selected_features.append(rfe.support_)

        param_grid = {"C" : 10.0**np.arange(-3,3)}
        grid = GridSearchCV(LogisticRegression(), param_grid = param_grid, scoring = 'recall', cv =10)
        grid.fit(X_train_sel, y_train_res)
        logit = LogisticRegression(C = grid.best_params_['C'])
        logit.fit(X_train_sel, y_train_res)
        best_params.append(grid.best_params_['C'])
        recall_scores.append(recall_score(y_test, logit.predict(X_test_sel)))

    opt_selected_features = selected_features[np.argmax(recall_scores)]
    opt_c = best_params[np.argmax(recall_scores)]


    print("Method: RFE + Logistic Regression l2")
    print(y.columns[j])
    print("Optimum number of features is {}".format(no_feature[np.argmax(recall_scores)]))
    print(selected_features[np.argmax(recall_scores)])
    print("name of selected_features {}".format(drug_traits_copy.columns[opt_selected_features]))
    #print(recall_scores)
    #plt.plot(no_feature, recall_scores,"-o", color = 'red');

    X_test_sel = X_test[:, opt_selected_features]
    logit = LogisticRegression(C = opt_c)
    logit.fit(X_train_res, y_train_res)
    print()
    #print("training recall is {:.3f}".format(recall_score(y_train_res, logit.predict(X_train_res))))
    print("test recall is {:.3f}".format(recall_score(y_test, logit.predict(X_test))))
    print("Classification report:\n {}".format(classification_report(y_test,logit.predict(X_test),
                                        target_names = ['not used', 'used'])))
    print("Confussion matrix: \n {}".format(confusion_matrix(y_test, logit.predict(X_test))))


print("================================================")

print("E: RFE() and KNN algorithm with personal traits data")

# KNN + RFE. Over sampling is done before RFE

drug_traits = drug_data.loc[:,['age', "gender", "education", "country",
"ethnicity","neuroticism", "extraversion",
"openness", "agreeableness", "conscient", "impulsuv",
"SS"]]
drug_traits_copy = drug_traits.copy()

X = drug_traits_copy.values
y = drug_data.loc[:,["lsd_consp_decade", "crack_consp_decade", "amly_consp_decade"]]

for j in [0, 1, 2]:

    X_train, X_test, y_train, y_test = train_test_split(X, y.iloc[:,j], test_size = 0.2, random_state = 22345)

    sm = SMOTE(random_state=12, ratio = 1.0)
    X_train_res, y_train_res = sm.fit_sample(X_train, y_train)

    no_feature = np.arange(1, 10, 1)
    recall_scores = []
    selected_features = []
    best_params = []
    for i in no_feature:
        rfe = RFE(estimator = LogisticRegression(), n_features_to_select = i)
        rfe.fit(X_train_res, y_train_res)
        X_train_sel = X_train_res[:, rfe.support_]
        X_test_sel = X_test[:, rfe.support_]
        selected_features.append(rfe.support_)

        param_grid = {"n_neighbors" : np.arange(4,20,2)}
        grid = GridSearchCV(KNeighborsClassifier(), param_grid = param_grid, scoring = 'recall', cv =10)
        grid.fit(X_train_sel, y_train_res)
        knn = KNeighborsClassifier(n_neighbors = grid.best_params_['n_neighbors'])
        knn.fit(X_train_sel, y_train_res)
        best_params.append(grid.best_params_['n_neighbors'])
        recall_scores.append(recall_score(y_test, knn.predict(X_test_sel)))

    opt_selected_features = selected_features[np.argmax(recall_scores)]
    opt_neighbors = best_params[np.argmax(recall_scores)]

    print("Method: RFE + KNN")
    print(y.columns[j])
    print("Optimum number of features is {}".format(no_feature[np.argmax(recall_scores)]))
    print(selected_features[np.argmax(recall_scores)])
    print("name of selected_features {}".format(drug_traits_copy.columns[opt_selected_features]))

    X_test_sel = X_test[:, opt_selected_features]
    knn = KNeighborsClassifier(n_neighbors = opt_neighbors)
    knn.fit(X_train_res, y_train_res)

    print("test recall is {:.3f}".format(recall_score(y_test, knn.predict(X_test))))
    print("Classification report:\n {}".format(classification_report(y_test,knn.predict(X_test),
                                        target_names = ['not used', 'used'])))
    print("Confussion matrix: \n {}".format(confusion_matrix(y_test, knn.predict(X_test))))


print("================================================")

print("F: RFE() and Logistic Regression(penalty = l2) with personal traits + drug consumption data")

# Other Drugs' Consumptions Added as explanatory variables
# RFE + Logistic Regression(l2)

# Over sampling is done before RFE

drug_copy = drug_data.copy()

X = drug_copy
drug_analyzed = ["lsd_consp_decade", "crack_consp_decade", "amly_consp_decade"]


y = drug_data.loc[:,drug_analyzed]

for j in [0, 1, 2]:

    X_dropped = X.drop(str(drug_analyzed[j]), axis = 1).values
    X_train, X_test, y_train, y_test = train_test_split(X_dropped, y.iloc[:,j], test_size = 0.2, random_state = 22345)

    sm = SMOTE(random_state=12, ratio = 1.0)
    X_train_res, y_train_res = sm.fit_sample(X_train, y_train)

    no_feature = np.arange(1, 10, 1)
    recall_scores = []
    selected_features = []
    best_params = []
    for i in no_feature:
        rfe = RFE(estimator = LogisticRegression(), n_features_to_select = i)
        rfe.fit(X_train_res, y_train_res)
        X_train_sel = X_train_res[:, rfe.support_]
        X_test_sel = X_test[:, rfe.support_]
        selected_features.append(rfe.support_)
        param_grid = {"C" : 10.0**np.arange(-3,3)}
        grid = GridSearchCV(LogisticRegression(), param_grid = param_grid, scoring = 'recall', cv =10)
        grid.fit(X_train_sel, y_train_res)
        logit = LogisticRegression(C = grid.best_params_['C'])
        logit.fit(X_train_sel, y_train_res)
        best_params.append(grid.best_params_['C'])
        recall_scores.append(recall_score(y_test, logit.predict(X_test_sel)))
    opt_selected_features = selected_features[np.argmax(recall_scores)]
    opt_c = best_params[np.argmax(recall_scores)]

    print(y.columns[j])
    print("Optimum number of features is {}".format(no_feature[np.argmax(recall_scores)]))
    print(selected_features[np.argmax(recall_scores)])
    print("name of selected_features {}".format(drug_copy.columns[opt_selected_features]))
    #print(recall_scores)
    #plt.plot(no_feature, recall_scores,"-o", color = 'red');

    X_test_sel = X_test[:, opt_selected_features]
    logit = LogisticRegression(C = opt_c)
    logit.fit(X_train_res, y_train_res)
    #print("training recall is {:.3f}".format(recall_score(y_train_res, logit.predict(X_train_res))))
    print("test recall is {:.3f}".format(recall_score(y_test, logit.predict(X_test))))
    print("Classification report:\n {}".format(classification_report(y_test,logit.predict(X_test),
                                        target_names = ['not used', 'used'])))
    print("Confussion matrix: \n {}".format(confusion_matrix(y_test, logit.predict(X_test))))


print("================================================")

print("G: RFE() and Logistic Regression(penalty = l1) with personal traits + drug consumption data")

# Other Drugs' Consumptions Added as explanatory variables
# RFE + Logistic Regression(l1)

# Over sampling is done before RFE

drug_copy = drug_data.copy()

X = drug_copy
drug_analyzed = ["lsd_consp_decade", "crack_consp_decade", "amly_consp_decade"]


y = drug_data.loc[:,drug_analyzed]

for j in [0, 1, 2]:

    X_dropped = X.drop(str(drug_analyzed[j]), axis = 1).values
    X_train, X_test, y_train, y_test = train_test_split(X_dropped, y.iloc[:,j], test_size = 0.2, random_state = 22345)

    sm = SMOTE(random_state=12, ratio = 1.0)
    X_train_res, y_train_res = sm.fit_sample(X_train, y_train)

    no_feature = np.arange(1, 10, 1)
    recall_scores = []
    selected_features = []
    best_params = []
    for i in no_feature:
        rfe = RFE(estimator = LogisticRegression(), n_features_to_select = i)
        rfe.fit(X_train_res, y_train_res)
        X_train_sel = X_train_res[:, rfe.support_]
        X_test_sel = X_test[:, rfe.support_]
        selected_features.append(rfe.support_)

        param_grid = {"C" : 10.0**np.arange(-3,3)}
        grid = GridSearchCV(LogisticRegression(penalty = 'l1'), param_grid = param_grid,
                            scoring = 'recall', cv =10)
        grid.fit(X_train_sel, y_train_res)
        logit = LogisticRegression(C = grid.best_params_['C'])
        logit.fit(X_train_sel, y_train_res)
        best_params.append(grid.best_params_['C'])
        recall_scores.append(recall_score(y_test, logit.predict(X_test_sel)))

    opt_selected_features = selected_features[np.argmax(recall_scores)]
    opt_c = best_params[np.argmax(recall_scores)]

    print(y.columns[j])
    print("Optimum number of features is {}".format(no_feature[np.argmax(recall_scores)]))
    print(selected_features[np.argmax(recall_scores)])
    print("name of selected_features {}".format(drug_copy.columns[opt_selected_features]))
    #print(recall_scores)
    #plt.plot(no_feature, recall_scores,"-o", color = 'red');

    X_test_sel = X_test[:, opt_selected_features]
    logit = LogisticRegression(C = opt_c)
    logit.fit(X_train_res, y_train_res)
    print()
    #print("training recall is {:.3f}".format(recall_score(y_train_res, logit.predict(X_train_res))))
    print("test recall is {:.3f}".format(recall_score(y_test, logit.predict(X_test))))
    print("Classification report:\n {}".format(classification_report(y_test,logit.predict(X_test),
                                        target_names = ['not used', 'used'])))
    print("Confussion matrix: \n {}".format(confusion_matrix(y_test, logit.predict(X_test))))


print("================================================")


print("H: RFE() and KNN with personal traits + drug consumption data")

# KNN + RFE + Other Drug Consumptions

# Other Drugs' Consumptions Added as explanatory variables

drug_copy = drug_data.copy()

X = drug_copy
drug_analyzed = ["lsd_consp_decade", "crack_consp_decade", "amly_consp_decade"]


y = drug_data.loc[:,drug_analyzed]

for j in [0, 1, 2]:

    X_dropped = X.drop(str(drug_analyzed[j]), axis = 1).values
    X_train, X_test, y_train, y_test = train_test_split(X_dropped, y.iloc[:,j], test_size = 0.2, random_state = 22345)

    sm = SMOTE(random_state=12, ratio = 1.0)
    X_train_res, y_train_res = sm.fit_sample(X_train, y_train)

    no_feature = np.arange(1, 10, 1)
    recall_scores = []
    selected_features = []
    best_params = []
    for i in no_feature:
        rfe = RFE(estimator = LogisticRegression(), n_features_to_select = i)
        rfe.fit(X_train_res, y_train_res)
        X_train_sel = X_train_res[:, rfe.support_]
        X_test_sel = X_test[:, rfe.support_]
        selected_features.append(rfe.support_)

        param_grid = {"n_neighbors" : np.arange(2,20,2)}
        grid = GridSearchCV(KNeighborsClassifier(), param_grid = param_grid, scoring = 'recall', cv =10)
        grid.fit(X_train_sel, y_train_res)
        knn = KNeighborsClassifier(n_neighbors = grid.best_params_['n_neighbors'])
        knn.fit(X_train_sel, y_train_res)
        best_params.append(grid.best_params_['n_neighbors'])
        recall_scores.append(recall_score(y_test, knn.predict(X_test_sel)))

    opt_selected_features = selected_features[np.argmax(recall_scores)]
    opt_neighbors = best_params[np.argmax(recall_scores)]

    print(y.columns[j])
    print("Optimum number of features is {}".format(no_feature[np.argmax(recall_scores)]))
    print(selected_features[np.argmax(recall_scores)])
    print("name of selected_features {}".format(drug_copy.columns[opt_selected_features]))

    X_test_sel = X_test[:, opt_selected_features]
    knn = KNeighborsClassifier(n_neighbors = opt_neighbors)
    knn.fit(X_train_res, y_train_res)

    print("test recall is {:.3f}".format(recall_score(y_test, knn.predict(X_test))))
    print("Classification report:\n {}".format(classification_report(y_test,knn.predict(X_test),
                                        target_names = ['not used', 'used'])))
    print("Confussion matrix: \n {}".format(confusion_matrix(y_test, knn.predict(X_test))))
