#!/usr/bin/env python
# coding: utf-8

# ## <u>**Use the "Text" blocks to provide explanations wherever you find them necessary. Highlight your answers inside these text fields to ensure that we don't miss it while grading your HW.**</u> 

# ## **Setup**
# 
# - Code to download the data directly from the colab notebook.
# - If you find it easier to download the data from the kaggle website (and uploading it to your drive), you can skip this section.

# ## **Section 1: Library and Data Imports (Q1)**
# 
# - Import your libraries and read the data into a dataframe. Print the head of the dataframe. 

# In[206]:


use_cols = ["MachineIdentifier", "SmartScreen", "AVProductsInstalled", "AppVersion", "CountryIdentifier", "Census_OSInstallTypeName", "Wdft_IsGamer",
           "EngineVersion", "AVProductStatesIdentifier", "Census_OSVersion", "Census_TotalPhysicalRAM", "Census_ActivationChannel",
           "RtpStateBitfield", "Census_ProcessorModelIdentifier", "Census_PrimaryDiskTotalCapacity", 
            "Census_InternalPrimaryDiagonalDisplaySizeInInches", "Wdft_RegionIdentifier", "LocaleEnglishNameIdentifier",
           "AvSigVersion", "IeVerIdentifier", "IsProtected", "Census_InternalPrimaryDisplayResolutionVertical", "Census_PrimaryDiskTypeName",
            "Census_OSWUAutoUpdateOptionsName", "Census_OSEdition", "Census_GenuineStateName", "Census_ProcessorCoreCount", 
           "Census_OEMNameIdentifier", "Census_MDC2FormFactor", "Census_FirmwareManufacturerIdentifier", "OsBuildLab", "Census_OSBuildRevision", 
            "Census_OSBuildNumber", "Census_IsPenCapable", "Census_IsTouchEnabled", "Census_IsAlwaysOnAlwaysConnectedCapable", "Census_IsSecureBootEnabled", 
            "Census_SystemVolumeTotalCapacity", "Census_PrimaryDiskTotalCapacity", "HasDetections"
           ]
dtypes = {
        'MachineIdentifier':                                    'category',
        'ProductName':                                          'category',
        'EngineVersion':                                        'category',
        'AppVersion':                                           'category',
        'AvSigVersion':                                         'category',
        'IsBeta':                                               'int8',
        'RtpStateBitfield':                                     'float64',
        'IsSxsPassiveMode':                                     'int8',
        'DefaultBrowsersIdentifier':                            'float64',
        'AVProductStatesIdentifier':                            'float64',
        'AVProductsInstalled':                                  'float64',
        'AVProductsEnabled':                                    'float16',
        'HasTpm':                                               'int8',
        'CountryIdentifier':                                    'int16',
        'CityIdentifier':                                       'float32',
        'OrganizationIdentifier':                               'float64',
        'GeoNameIdentifier':                                    'float16',
        'LocaleEnglishNameIdentifier':                          'int8',
        'Platform':                                             'category',
        'Processor':                                            'category',
        'OsVer':                                                'category',
        'OsBuild':                                              'int16',
        'OsSuite':                                              'int16',
        'OsPlatformSubRelease':                                 'category',
        'OsBuildLab':                                           'category',
        'SkuEdition':                                           'category',
        'IsProtected':                                          'float64',
        'AutoSampleOptIn':                                      'int8',
        'PuaMode':                                              'category',
        'SMode':                                                'float16',
        'IeVerIdentifier':                                      'float64',
        'SmartScreen':                                          'category',
        'Firewall':                                             'float16',
        'UacLuaenable':                                         'float32',
        'Census_MDC2FormFactor':                                'category',
        'Census_DeviceFamily':                                  'category',
        'Census_OEMNameIdentifier':                             'float64',
        'Census_OEMModelIdentifier':                            'float64',
        'Census_ProcessorCoreCount':                            'float64',
        'Census_ProcessorManufacturerIdentifier':               'float64',
        'Census_ProcessorModelIdentifier':                      'float64',
        'Census_ProcessorClass':                                'category',
        'Census_PrimaryDiskTotalCapacity':                      'float64',
        'Census_PrimaryDiskTypeName':                           'category',
        'Census_SystemVolumeTotalCapacity':                     'float64',
        'Census_HasOpticalDiskDrive':                           'int8',
        'Census_TotalPhysicalRAM':                              'float64',
        'Census_ChassisTypeName':                               'category',
        'Census_InternalPrimaryDiagonalDisplaySizeInInches':    'float64',
        'Census_InternalPrimaryDisplayResolutionHorizontal':    'float16',
        'Census_InternalPrimaryDisplayResolutionVertical':      'float64',
        'Census_PowerPlatformRoleName':                         'category',
        'Census_InternalBatteryType':                           'category',
        'Census_InternalBatteryNumberOfCharges':                'float32',
        'Census_OSVersion':                                     'category',
        'Census_OSArchitecture':                                'category',
        'Census_OSBranch':                                      'category',
        'Census_OSBuildNumber':                                 'int16',
        'Census_OSBuildRevision':                               'int32',
        'Census_OSEdition':                                     'category',
        'Census_OSSkuName':                                     'category',
        'Census_OSInstallTypeName':                             'category',
        'Census_OSInstallLanguageIdentifier':                   'float16',
        'Census_OSUILocaleIdentifier':                          'int16',
        'Census_OSWUAutoUpdateOptionsName':                     'category',
        'Census_IsPortableOperatingSystem':                     'int8',
        'Census_GenuineStateName':                              'category',
        'Census_ActivationChannel':                             'category',
        'Census_IsFlightingInternal':                           'float64',
        'Census_IsFlightsDisabled':                             'float64',
        'Census_FlightRing':                                    'category',
        'Census_ThresholdOptIn':                                'float16',
        'Census_FirmwareManufacturerIdentifier':                'float64',
        'Census_FirmwareVersionIdentifier':                     'float64',
        'Census_IsSecureBootEnabled':                           'int8',
        'Census_IsWIMBootEnabled':                              'float64',
        'Census_IsVirtualDevice':                               'float64',
        'Census_IsTouchEnabled':                                'int8',
        'Census_IsPenCapable':                                  'int8',
        'Census_IsAlwaysOnAlwaysConnectedCapable':              'float64',
        'Wdft_IsGamer':                                         'float64',
        'Wdft_RegionIdentifier':                                'float64'        
        }


# In[207]:


len(use_cols),len(dtypes)


# In[208]:


import pandas as pd
import os


# In[95]:


fileName ='/Users/sbvaranasi/Documents/Fall21/DataScienceFundamentals/microsoft-malware-prediction/train.csv'
df=pd.read_csv(fileName, usecols=use_cols, dtype=dtypes)


# In[96]:


df.head()


# In[97]:


df.shape


# ## **Section 2: Measure of Power (Q2a & 2b)**

# In[98]:


df[['Census_ProcessorCoreCount', 'Census_TotalPhysicalRAM', 'Wdft_IsGamer', 'HasDetections']].describe()


# In[28]:


#Replacing NaN values with mode/mean
df['Wdft_IsGamer'].fillna(value = df['Wdft_IsGamer'].mode()[0], inplace = True)
df['Census_ProcessorCoreCount'].fillna(value = df['Census_ProcessorCoreCount'].mean(), inplace=True)


# In[29]:


#Normalizing the data
from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()
df['Census_ProcessorCoreCount']=min_max_scaler.fit_transform(df[['Census_ProcessorCoreCount']])
df['Census_TotalPhysicalRAM']=min_max_scaler.fit_transform(df[['Census_TotalPhysicalRAM']])


# In[30]:


df[['Census_ProcessorCoreCount', 'Census_TotalPhysicalRAM', 'Wdft_IsGamer', 'HasDetections']].describe()


# Assigning weights to the selected features : ['Census_ProcessorCoreCount', 'Census_TotalPhysicalRAM', 'Wdft_IsGamer'] as [0.4, 0.35, 0.25]
# 
# Defining Computing Power function as ,
# computing_power = 0.4 * processor_core_count + 0.35 * physical_ram + 0.25 * is_gamer

# In[144]:


df['ComputingPower'] = (0.4*df['Census_ProcessorCoreCount'] + 0.35*df['Census_TotalPhysicalRAM'] + 0.25*df['Wdft_IsGamer'])


# In[145]:


df[['ComputingPower','HasDetections']].describe()


# In[152]:


import matplotlib.pyplot as plt

df.hist(column='ComputingPower', bins=500)


# Computing power as a weighted function of Processor core count, RAM and if it's a gaming laptop.
# We could see a bimodal distribution of the machines using this metric computing power.
# 
# Intuition : Machines might be classified into high-power gaming laptops with high RAM/Processor cores and low-power non-gaming laptops.

# In[154]:


ranges = [0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4]
comp_power_bin_data = df[["ComputingPower","HasDetections"]]
row_info = pd.cut(comp_power_bin_data["ComputingPower"], ranges);

sns.set(rc={'figure.figsize':(20,10)})
gph = sns.countplot(x=row_info, hue="HasDetections",data=comp_power_bin_data)
for patch in gph.patches:
    wd = patch.get_width()
    ht = patch.get_height()
    x, y = patch.get_xy() 
    gph.annotate(f'{ht}', (x + wd/2, y + ht*1.02), ha='center')


# Based on the above countplot, nothing solid can be said about computing power vs malware vulnerability.
# 
# But if we compare the ratio of malware detections, it's increasing as the computing power increases gradually. We can 
# see the spike in ratio after 0.3 computing power.

# ## **Section 3: OS version vs Malware detected (Q3)**

# In[155]:


import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


# In[110]:


df[['Census_OSBuildNumber', 'Census_OSBuildRevision', 'HasDetections']].sample(15)


# In[111]:


set(df.HasDetections)


# In[121]:


group_df = df.groupby(['Census_OSBuildNumber']).agg({'HasDetections': ['sum', 'mean','count']})
group_df.columns = ['detections_sum', 'detections_mean', 'detections_count']
group_df = group_df.reset_index()

#Ignoring the points with occurrences less than 100
group_df = group_df.loc[group_df['detections_count'] > 50]

plt.plot(group_df['Census_OSBuildNumber'], group_df['detections_sum'])
plt.xlabel('Census_OSBuildNumber')
plt.ylabel('Sum_of_Detections')
plt.show()

plt.plot(group_df['Census_OSBuildNumber'], group_df['detections_mean'])
plt.xlabel('Census_OSBuildNumber')
plt.ylabel('Percentage_Detections')
plt.show()

group_df = df.groupby(['Census_OSBuildRevision']).agg({'HasDetections': ['sum', 'mean', 'count']})
group_df.columns = ['detections_sum', 'detections_mean', 'detections_count']
group_df = group_df.reset_index()


#Ignoring the points with occurrences less than 100
group_df = group_df.loc[group_df['detections_count'] > 50]

plt.plot(group_df['Census_OSBuildRevision'], group_df['detections_sum'])
plt.xlabel('Census_OSBuildRevision')
plt.ylabel('Sum_of_Detections')
plt.show()

plt.plot(group_df['Census_OSBuildRevision'], group_df['detections_mean'])
plt.xlabel('Census_OSBuildRevision')
plt.ylabel('Percentage_Detections')
plt.show()


# Census_OSBuildNumber vs HasDetections: 
# 
# 1. Number of malware detections increased with varying slopes till 17000 OS_BuildNumber but we see a very iregular trend leading upto 17000 and till 18000 which is evident even in the % malware detections plot.
# 2. Probably a major release went very bad and they had to undergo frequent minor version releases where few are stable and few are not.
# 
# 
# Census_OSBuildRevision vs HasDetections:
# 1. We see huge activity(high malware detections) in the initial revisions till 2500 and again we see a spike in activity > 16000 which might account to the behaviour we saw in the Census_OSBuildNumber above.
# 

# ## **Section 4: Effect of Number of AV Products Installed (Q4)**

# In[76]:


df.AVProductsInstalled.describe()


# In[120]:


group_df = df.groupby(['AVProductsInstalled']).agg({'HasDetections': ['sum', 'mean', 'count']})
group_df.columns = ['detections_sum', 'detections_mean', 'detections_count']
group_df = group_df.reset_index()
print(group_df)


# Dropping the rows at indices 0,7 as they have  too low detection_count

# In[115]:


groupdf=group_df.drop(index=[0,7])


# In[118]:


plt.plot(groupdf['AVProductsInstalled'], groupdf['detections_sum'])
plt.xlabel('AVProductsInstalled')
plt.ylabel('Detections_Sum')
plt.show()

plt.plot(groupdf['AVProductsInstalled'], groupdf['detections_mean'])
plt.xlabel('AVProductsInstalled')
plt.ylabel('Detections_Percentage')
plt.show()


# As we see from the above two graphs, as the number of Antivirus products Installed increases, total malware detections and detection percentage decreases gradually.
# 
# So yes as evident from the above graphs, the number of antivirus products matter and the more products the lesser the system is vulnerable to malwares.

# ## **Section 5: Interesting findings (Q5)**

# In[89]:


df['HasDetections'].describe()


# In[176]:


new_df = pd.read_csv(fileName, usecols=['Census_OSArchitecture','HasDetections', 'Census_IsVirtualDevice', 
                                        'Census_MDC2FormFactor', 'Platform', 'CountryIdentifier'])


# In[177]:


df_OSArchitecture = pd.DataFrame(new_df)


# In[178]:


df_OSArchitecture.head()


# In[179]:


df_OSArchitecture.Census_IsVirtualDevice.describe()


# In[181]:


arch=df_OSArchitecture.groupby(['Census_OSArchitecture']).agg({'HasDetections' :['sum', 'mean']})
arch.columns=['detections_sum','detections_mean']
arch=arch.reset_index()
print(arch), print(type(arch))


plt.plot(arch['Census_OSArchitecture'], arch['detections_sum'])
plt.xlabel('Census_OSArchitecture')
plt.ylabel('Detections_Sum')
plt.show()


# FINDING 1 : AMD64 devices are comparatively more vulnerable to malware while arm64 are the least.

# In[183]:


group_df = df_OSArchitecture.groupby(['Platform']).agg({'HasDetections': ['count']})
group_df.columns = ['detections_count']
group_df = group_df.reset_index()
group_df


# In[171]:


fig = plt.gcf()
x = np.arange(len(group_df['Platform']))
ax = plt.bar(x, group_df['detections_count'])

plt.xlabel('Platform Name')
plt.ylabel('Number of malwares detected')

plt.xticks(x, group_df['Platform'])
plt.show()


# FINDING 2 : Surprisingly most of the malware detected is found in windows 10 computers.

# In[186]:


group_df = df.groupby(['AVProductsInstalled']).agg({'HasDetections': ['sum', 'mean', 'count']})
group_df.columns = ['detections_sum', 'detections_mean', 'detections_count']
group_df = group_df.reset_index()

plt.plot(groupdf['AVProductsInstalled'], groupdf['detections_sum'])
plt.xlabel('AVProductsInstalled')
plt.ylabel('Detections_Sum')
plt.show()


# FINDING 3 : The more the number of antivirus products installed on a machine, the lesser is the number of malware detected.

# ## **Section 6: Baseline modelling (Q6)**

# In[187]:


import seaborn as sns
from matplotlib import pyplot as plt


# In[188]:


df.columns
df.isnull().sum()


# In[123]:


df_droppedna = df.dropna(how='any', inplace=False)
print(df_droppedna.shape)


# In[190]:


columns_list = list(df_droppedna.columns)

#Columns which are IsTouchCapable, IsProtected are named as Boolean Columns
BooleanColumns = [x for x in columns_list if 'Is' in x]

#For Baseline model Model0 I chose Numerical columns by describing the dataframe.
NumericalColumns = ['AVProductsInstalled','Census_ProcessorCoreCount','Census_PrimaryDiskTotalCapacity',
                'Census_SystemVolumeTotalCapacity','Census_TotalPhysicalRAM','Census_OSBuildNumber','Census_OSBuildRevision', 'RtpStateBitfield']

#Ignoring categorical features for baseline model as they are not encoded yet. 

Final_Features = BooleanColumns + NumericalColumns


# In[189]:


from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder


#Running logistic regression model=> Model0 without any preprocessing of the data.


# In[195]:


X=df_droppedna[Final_Features]
y=df_droppedna['HasDetections']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 0)
Model0 = LogisticRegression(max_iter=800)
Model0.fit(X_train, y_train)

y_pred = Model0.predict(X_test)
# print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))

print('Accuracy : {:.2f}'.format(metrics.accuracy_score(y_test, y_pred)))
print('AUC score : {:.2f}'.format(metrics.roc_auc_score(y_test, Model0.predict_proba(X_test)[:,1])))


# Error rate : 0.49 (Computed as 1-Accuracy = 1-0.51 = 0.49)

# ## **Section 7: Feature Cleaning and Additional models (Q7a & 7b)**

# # Plan for feature cleaning:
# 
# 
# 1.   Instead of dropping null values as I did for logistic regression, I will try to compute mean, mode and accordingly replace for numerical, categorical data respectively.
# 2.   Convert categorical features(_Identifiers, _Version) from string -> int by doing some preprocessing. (Later realized this is what label encoding does).
# 3.   Normalize numerical values using min-max scaling or standard scaling.
# 4.   Using groupby->counts for each categorical variable, I will try to introduce few one-hot encoding columns and compare the model's performance. (Couldn't execute this plan because introducing one-hot encoding involved an explosion of features).   
# 
# 

# In[137]:


#Computing correlation matrix
corr = df_droppedna.corr()

plt.figure(figsize=(20,15))
sns.heatmap(corr, cmap="Greens",annot=True)

c1 = corr[(corr < 0.3) & (corr > -0.3)].abs().unstack().transpose().sort_values(ascending=True).drop_duplicates()
#Picking values that lie between -0.3 and 0.3 from correlation matrix
print(c1[1:25])

#Used these values from correlation matrix to check if there's any correlation between the features that I selected.


# In[133]:


# Adding categorical columns in Model1 model.
IdentifierColumns = [x for x in columns_list if 'Identifier' in x]
IdentifierColumns.remove('MachineIdentifier')

CategoricalColumns = IdentifierColumns + [x for x in columns_list if 'Version' in x]

# df_droppedna.isnull().sum() 


# In[134]:


print(CategoricalColumns)
print(BooleanColumns)
print(NumericalColumns)

# print(set(CategoricalColumns).intersection(BooleanColumns))


# df.astype({'IeVerIdentifier': 'float64', 'Census_OEMNameIdentifier': 'float64', 'Census_ProcessorModelIdentifier': 'float64', 'Census_FirmwareManufacturerIdentifier': 'float64', 'Wdft_RegionIdentifier': 'float64', ...}).dtypes
# 
# 
# Changing datatypes of the above columns to float64 because of overflow while computing mean => Made this change while loading the dataset itself as a part of use_cols.

# In[135]:


df.isnull().sum()


# In[193]:


# Filling nan values in the original dataframe df with mean for numerical features 
# and mode for categorical and boolean features.

def replaceNan(df):
    for item in NumericalColumns:
        df[item].fillna(value = df[item].mean(), inplace = True)
        
    for item in CategoricalColumns:
        df[item].fillna(value = df[item].mode()[0], inplace = True)

    for item in BooleanColumns:
        df[item].fillna(value = df[item].mode()[0], inplace = True)
    print('Replace Nan execution successful')
        
def labelEncoding(df):
    #Label encoding for categorical features
    labelencoder = LabelEncoder()
    for item in CategoricalColumns:
        df[item] = labelencoder.fit_transform(df[item])
    print('Label encoding execution successful')
        
#Min-max scaling for numerical features
def min_max_scaling(df):
    for column in NumericalColumns:
        df[column] = (df[column] - df[column].min())/(df[column].max() - df[column].min())
    print('Min-max scaling execution successful')
        
# TransformDataFrame function which does the dataframe transformation by replacing Nan values following by Label Encoding
# and Min_Max_Scaling.
def transformDataFrame(df):
    replaceNan(df)
    labelEncoding(df)
    min_max_scaling(df)


# In[130]:


transformDataFrame(df)


# In[19]:


df[NumericalColumns].describe()


# In[20]:


Final_Features_1 = Final_Features + CategoricalColumns
print(Final_Features_1,  len(Final_Features_1))


# In[64]:


df[Final_Features_1].describe()


# In[26]:


X=df[Final_Features_1]
y=df['HasDetections']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 0)
# print(X_train.shape), print(X_test.shape), print(y_train.shape)
Model1 = LogisticRegression(max_iter=1600)
Model1.fit(X_train, y_train)

y_pred = Model1.predict(X_test)

# print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))
print('Accuracy score : {:.2f}'.format(metrics.accuracy_score(y_test, y_pred)))
print('AUC score : {:.2f}'.format(metrics.roc_auc_score(y_test, Model1.predict_proba(X_test)[:,1])))


# Error_Rate : 0.45 (Computed as 1- Accuracy score = 1 - 0.55 = 0.45)

# In[21]:


# Using RandomForestClassifier for Model2
from sklearn.ensemble import RandomForestClassifier

X=df[Final_Features_1]
y=df['HasDetections']

# Instantiate model with 100 decision trees
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 0)
Model2 = RandomForestClassifier(n_estimators=50)
Model2.fit(X_train, y_train)                                    # Train the model on training data

y_predict = Model2.predict(X_test)
# print('Accuracy of random forest regressor on test set: {:.2f}'.format(rf.score(X_test, y_test)))

print('Accuracy : {:.2f}'.format(metrics.accuracy_score(y_test, y_predict)))
print('AUC score : {:.2f}'.format(metrics.roc_auc_score(y_test, Model2.predict_proba(X_test)[:,1])))


# Error_Rate : 0.38 (Computed as 1 - Accuracy score = 1 - 0.62 = 0.38)

# # Comparing Models

# | Model  | Error_Rate  |
# |---|---|
# | Model0  | 0.49  |
# | Model1  | 0.45  |
# | Model2  | 0.38  |

# Comparing Models:
# 1. Model0 : LogisticRegression model without any preprocessing. 
# 2. Model1 : LogisticRegression model<br />
#     a. Features selected as numerical and categorical columns by describing the data and understanding the data.<br />
#     b. Compared the selected with correlation matrix table and none of them are highly-correlated (filterd the pearson correlation matrix between -0.3 <-> 0.3 for making sure to choose features which are not highly-correlated), hence went ahead with the chosen 26 features list.<br />
#     c. Handling Nan values with mean/mode of the respective Pandas Series(based on if it's numerical/categorical/boolean).<br />
#     d. Label encoding for categorical variables.<br />
#     e. Min-max scaling for normalizing the numerical features.<br />
# 3. Model2 : Random forest classifier model<br />
#     a. Preprocessing similar to Model0.<br />
#     b. Conducted exploratory data analysis with different estimators and the model seems to converge starting n_estimators=50. (Also tested with n_jobs = -1 for parallel processing).

# # Testing models on test.csv

# In[22]:


use_cols_test=use_cols.remove('HasDetections')


# In[23]:


testFile ='/Users/sbvaranasi/Documents/Fall21/DataScienceFundamentals/microsoft-malware-prediction/test.csv'
test_df=pd.read_csv(testFile, usecols=use_cols_test, dtype=dtypes)


# In[53]:


transformDataFrame(test_df)
X_test=test_df[Final_Features_1]
# Machine_ID=test_df['MachineIdentifier']

my_submission=pd.DataFrame({'MachineIdentifier': Machine_ID, 'HasDetections':Model2.predict_proba(X_test)[:,1]})
my_submission


# In[54]:


my_submission_model1=pd.DataFrame({'MachineIdentifier': Machine_ID, 'HasDetections':Model1.predict_proba(X_test)[:,1]})
my_submission_model1


# In[39]:


# my_submission.to_csv('model2_submission.csv', index=False)
my_submission_model1.to_csv('model1_submission.csv', index=False)


# # Section 8: Screenshots (Q8)

# Public Score: 0.56812
# 
# Private Score: 0.55044
# 
# Kaggle profile link: https://www.kaggle.com/saibvara/account
# 
# Screenshot(s): 

# In[205]:


from IPython.display import Image
Image("Screen Shot 2021-09-23 at 9.10.28 AM.png")


# In[ ]:




