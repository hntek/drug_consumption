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
import matplotlib
matplotlib.use('Agg')
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

print("Exploratory Data Analysis")

# coding categorical variables for vizualization

statistical_data = drug_data.copy()
statistical_data['age'] = statistical_data['age'].astype('category').cat.\
        rename_categories(["18-24", "25-34", "35-44", "45-54", "55-64", "65+"])

statistical_data['gender'] = statistical_data['gender'].astype('category').cat.\
        rename_categories(["Male", "Female"])

statistical_data['education'] = statistical_data['education'].astype('category').cat.\
        rename_categories(["left < 16", "left = 16", "left = 17", "left = 18", "no degree",
                          "certificate", "university", "master", "doctorate"])

statistical_data['country'] = statistical_data['country'].astype('category').cat.\
        rename_categories(["Australia", "Canada", "New Zealand", "Other", "Ireland","UK", "USA"])

statistical_data['ethnicity'] = statistical_data['ethnicity'].astype('category').cat.\
        rename_categories(["Asian", "Black", "Mixed-B/A", "Mixed-W/A", "Mixed-W/B","Other", "White"])

statistical_data['lsd_consp_decade'] = statistical_data['lsd_consp_decade'].astype("category").cat.\
        rename_categories(["not used", "used/using"])

statistical_data['crack_consp_decade'] = statistical_data['crack_consp_decade'].astype("category").\
            cat.rename_categories(["not used", "used/using"])

statistical_data['amly_consp_decade'] = statistical_data['amly_consp_decade'].astype("category").cat.\
        rename_categories(["not used","used/using"])

# Value Counts of drug consumption
print("LSD consumption \n {}".format(statistical_data['lsd_consp_decade'].value_counts()))
print("=============")
print("Crack consumption \n {}".format(statistical_data['crack_consp_decade'].value_counts()))
print("============")
print("Amly consumption \n {}".format(statistical_data['amly_consp_decade'].value_counts()))

# statistical summary of personal traits measurements
stat_smry_traits =drug_data.iloc[:,0:12].describe().to_csv("data/stat_smry_traits.csv")
drug_data.iloc[:,0:12].describe()

# statistical summary of drug consumption
stat_smry_consp = drug_data.loc[:,['lsd_consp_decade', 'crack_consp_decade', 'amly_consp_decade']]\
.describe().to_csv("data/stat_smry_consp.csv")

#  Bar graph for drug consumption count by gender

fig, ((ax1, ax2, ax3)) = plt.subplots(nrows=1, ncols=3)
g1 = sns.countplot(x=statistical_data['lsd_consp_decade'],hue = statistical_data['gender'],
                   palette = "hls", ax=ax1)
g2 = sns.countplot(x= statistical_data['crack_consp_decade'],hue = statistical_data['gender'],
                   palette = "hls", ax=ax2)
g3 = sns.countplot(x= statistical_data['amly_consp_decade'],hue = statistical_data['gender'],
                   palette = "hls", ax=ax3)

g1.set_ylabel("count", fontsize = 18)
g1.set_xlabel('LSD consumption', fontsize = 18)
g1.tick_params(labelsize=12)
g2.set_ylabel("count", fontsize = 18)
g2.set_xlabel('Crack consumption', fontsize = 18)
g2.tick_params(labelsize=12)
g3.set_ylabel("count", fontsize = 18)
g3.set_xlabel('Amyl consumption', fontsize = 18)
g3.tick_params(labelsize=12)

plt.gcf().set_size_inches(10,6)
plt.tight_layout()
#plt.show()
plt.savefig("images/drugconsp_bygender.png");

# bar graph for drug consumption by country
fig, ((ax1, ax2, ax3)) = plt.subplots(nrows = 1, ncols = 3)
g1 = sns.countplot(x=statistical_data['country'],hue = statistical_data['lsd_consp_decade'],
                   palette = "hls", ax=ax1)
g2 = sns.countplot(x= statistical_data['country'],hue = statistical_data['crack_consp_decade'],
                   palette = "hls", ax=ax2)
g3 = sns.countplot(x= statistical_data['country'],hue = statistical_data['amly_consp_decade'],
                   palette = "hls", ax=ax3)

g1.set_ylabel("count", fontsize = 18)
g1.set_xlabel('LSD consumption', fontsize = 18)
g1.tick_params(labelsize=12)
g2.set_ylabel("count", fontsize = 18)
g2.set_xlabel('Crack consumption', fontsize = 18)
g2.tick_params(labelsize=12)
g3.set_ylabel("count", fontsize = 18)
g3.set_xlabel('Amyl consumption', fontsize = 18)
g3.tick_params(labelsize=12)

ax1.set_xticklabels(ax1.get_xticklabels(), rotation = 90)
ax2.set_xticklabels(ax2.get_xticklabels(), rotation = 90)
ax3.set_xticklabels(ax3.get_xticklabels(), rotation = 90)
ax1.legend(loc = 2)
ax2.legend(loc = 2)
ax3.legend(loc = 2)

plt.gcf().set_size_inches(12,6)
plt.tight_layout()
plt.savefig("images/drugconsp_bycountry.png");
