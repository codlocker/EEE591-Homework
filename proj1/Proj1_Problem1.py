#!/usr/bin/env python
# coding: utf-8

# Problem 1: 
# 
# - Read the database in from this heart1.csv file and analyze the data.
# - Your analysis should include a statistical study of each variable: correlation of each variable, dependent  or independent, with all the other variables. Determine which variables are most highly correlated with each other and also which are highly correlated with the variable you wish to predict. 
# - Create a cross covariance matrix to show which variables are not independent of each other and which ones are best predictors of heart disease. Create a pair plot.
# - Based on this analysis you must determine what you think you will be able to do and which variables you think are most likely to play a significant roll in predicting the dependent variable, in this case occurrence of heart disease. 
# 
# - Your management at AMAPE want to be kept constantly updated on your progress. Write one paragraph based on these results indicating what you have learned from this analysis. We are looking for specific observations


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Read the dataset
heart_df = pd.read_csv('./heart1.csv')


# ## CORRELATION ANALYSIS
# Get the absolute correlation values for the dataset
heart_corr = heart_df.corr().abs()


tri_corr = heart_corr * np.tri(*heart_corr.values.shape, k=-1).T
correlation_array = tri_corr.unstack() # Unstack the correlation results into a Pandas series
correlation_array = correlation_array.sort_values(ascending=False)


# This returns a pandas dataframe that shows the covariance
# and correlation of the labels with each other 
def get_relation_with_heart_disease(corr_array: pd.Series, cov_corr: int, is_target: bool):
    label_1 = pd.DataFrame([x[0] for x in corr_array.axes[0]])
    label_2 = pd.DataFrame([x[1] for x in corr_array.axes[0]])
    values = pd.Series(corr_array.values)
    
    relation_data = pd.concat([label_1, label_2, values], axis=1)
    relation_data.columns = ["label1", "label2", "Correlation" if cov_corr == 2 else "Covariance"]
    
    heart_data = relation_data[relation_data["label1"] == "a1p2"] if is_target else relation_data
    heart_data.reset_index(inplace=True, drop=True)
    return heart_data


# ## Correlation with all features other than the target feature a1p2


# Get the top 10 correlation data points
cor_with_features = get_relation_with_heart_disease(correlation_array, 2, False)
print("=====================")
print("Top 10 Correlation Values")
print(cor_with_features[:10])


# ## Correlation with target feature


# Get correlation with the target feature of a1p2
print("=====================")
print("Top 10 Correlation Values with a1p2 (the target variable)")
cor_with_disease = get_relation_with_heart_disease(correlation_array, 2, True)
print(cor_with_disease[:10])


# ## COVARIANCE ANALYSIS
# Covariance calculation
heart_cov = heart_df.cov().abs()


tri_cov_df = heart_cov * np.tri(*heart_cov.values.shape, k=-1).T
cov_unstack = tri_cov_df.unstack() # Unstack the covariance results into a Pandas series
cov_unstack = cov_unstack.sort_values(ascending=False)


# Get the covariance of the top 10 related to the target feature
cov_with_disease = get_relation_with_heart_disease(cov_unstack, 1, True)
print("=====================")
print("Top 10 Covariance Values with a1p2 (the target variable)")
print(cov_with_disease[:10])


# Get the covariance of the top 10 related and dependant variables overall.
cov_with_other_features = get_relation_with_heart_disease(cov_unstack, 1, False)
print("=====================")
print("Top 10 Covariance Values")
print(cov_with_other_features[:10])


# ## PAIR PLOT 
sns.set() # set the appearance
sns.pairplot(heart_df,height=1.5) # create the pair plots
plt.show() # and show them

