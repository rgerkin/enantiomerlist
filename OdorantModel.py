#!/usr/bin/env python
# coding: utf-8

# ## Preliminaries

# In[34]:


# TODO: Recursive feature elimination to improve model and identify features
# TODO: Use harmonic mean (within pair) detection threshold as predictor
# TODO: Make detection threshold model (using features).  Use predicted threshold as predictor


# In[1]:


#Gets the updates from the development files you imported
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[23]:


#Imports the pands library, the math library, and the init class python file
#Last line: updates matlab library that is used in init file
import __init__ as a
import math
import numpy as np
import matplotlib.pyplot as plt
import pyrfume 
import seaborn as sns
sns.set(font_scale=1.2)
#a.mpl.rcParams['font.size'] = 12


# ## Start of Project

# In[4]:


# This section loads in the data and groups each pair of enantiomers by making them have the same number associated with column "N"
# We also set the Normalized Detection Thresholds to be of the same type as a way of trying to avoid type errors later on
# This section also takes each pair of enantiomers and computes the ratio of the Normalized Detection Thresholds between the two
# The ratio is saved and will be used when the dataset gets cut in half
coleman_smiles = a.load_other_smiles(coleman=True)
coleman_data = a.load_data("coleman")#.iloc[1:]
coleman_data['N'] = np.arange(0, coleman_data.shape[0]/2, 0.5).astype(int)
coleman_data['Normalized Detection Threshold'] = coleman_data['Normalized Detection Threshold'].astype('float')
coleman_data.head()


# ## Cutting the Dataframe in Half

# In[47]:


def harmonic(x):
    return np.log10(1/np.mean(1/x['Normalized Detection Threshold'].values))


# In[48]:


# This section creates a new data frame with just one odorant of each enantiomeric pair from the original dataset 
# Adds the ratio of absolute log values
# half_log_abs = coleman_data.groupby('N').apply(log_abs)
half_log_abs = coleman_data.groupby('N').apply(a.log_abs)
half_det = coleman_data.groupby('N').apply(harmonic)
half_coleman_data = coleman_data.iloc[::2].copy()
half_coleman_data.loc[:, 'log_abs'] = half_log_abs.values
half_coleman_data.loc[:, 'det'] = half_det.values


# In[49]:


# This line makes sure that the rest of the exsisting null values are equal in the new data frame and in the new data frame's 'log_abs' column
assert half_log_abs.isnull().sum() == half_coleman_data['log_abs'].isnull().sum()


# In[50]:


# This line checks that the half dataset was created properly
half_coleman_data.head()


# In[51]:


# This section gets rid of all the invalid SMILES Strings, specifically the duplicates because we don't want to count their perceptual features twice and the "nan" values 
half_coleman_data = half_coleman_data.drop_duplicates(subset=['SMILES String'])
half_coleman_data = half_coleman_data[~half_coleman_data['SMILES String'].str.contains('NaN', na=True)]
half_coleman_data = half_coleman_data[~half_coleman_data['SMILES String'].str.contains('nan', na=True)]


# In[52]:


# These two assert statements are to ensure that we only have unqiue smiles strings and that no smiles strings are nan values
assert half_coleman_data['SMILES String'].shape == half_coleman_data['SMILES String'].unique().shape, "Number of SMILES strings should equal number of unique SMILES strings at this stage"


# In[53]:


# This line makes sure that are no more nan values in the smiles string column
assert sum(half_coleman_data['SMILES String']=='nan') == 0, "There should be no NaN SMILES strings at this point"


# In[54]:


#This line gets rid of the rows with a null log_abs value
half_coleman_data = half_coleman_data[~half_coleman_data['log_abs'].isnull()]


# In[55]:


#This line makes sure there are no more log_abs values with the value null
assert not sum(half_coleman_data['log_abs'].isnull())
assert not sum(half_coleman_data['det'].isnull())


# ## Computing the Features

# In[13]:


# These lines calculate the mordred and morgan features
mordred_data = a.calculate_features(half_coleman_data, "mordred")
morgan_data = a.calculate_features(half_coleman_data, "morgan")


# In[14]:


#These lines make sure that we only use the molecules that have mordred and morgan features computed
common_index = mordred_data.index.intersection(morgan_data.index)
mordred_data = mordred_data.loc[common_index]
morgan_data = morgan_data.loc[common_index]


# In[15]:


#Data frame that has both the mordred and morgan features
both = mordred_data.join(morgan_data.iloc[:,10:], how="inner")
both.head()


# In[16]:


#Gets all Mordred or mogan features that have numeric values and not Null values
#The last line joins the final mordred and morgan features 
finite_mordred = a.finite_features(mordred_data)
finite_morgan = a.finite_features(morgan_data)
both_features = finite_mordred | finite_morgan


# ## Creating the Models

# In[24]:


import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")
#sns.set_context('poster')
#plt.figure(figsize=(7,7))
stuff = half_coleman_data[half_coleman_data["log_abs"]<50]
plt.hist(stuff["log_abs"], bins=25, color="#800000", alpha=0.8);
plt.xticks([0,1,2,3], ["1x", "10x", "100x", "1000x"])
plt.xlabel("\n Fold Difference in Detection Threshold between Enantiomers")
plt.ylabel("Number of Enantiomeric Pairs")


# In[196]:


#X = both[both_features]; y = both['log_abs']
#X = morgan_data[finite_morgan]; y = morgan_data['log_abs']
X = mordred_data[finite_mordred]; y = mordred_data['log_abs']
X = X.join(half_coleman_data.set_index('SMILES String')['det'])
X = X[y < 10]
y = y[y < 10]
Xn = pd.DataFrame(StandardScaler().fit_transform(X), index=X.index, columns=X.columns)


# In[67]:


import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import cross_val_predict, LeaveOneOut
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler, StandardScaler 
from sklearn.linear_model import ElasticNet
from tqdm.auto import tqdm
#Xn = pd.DataFrame(MinMaxScaler().fit_transform(X), index=X.index, columns=X.columns)
#enr = ElasticNet(alpha=0, l1_ratio=1)
#rfr = RandomForestRegressor(n_estimators=100)
Cs = np.logspace(-3, 3, 13)
rs_in = pd.Series(index=Cs, dtype=float)
rs_out = pd.Series(index=Cs, dtype=float)
rhos_out = pd.Series(index=Cs, dtype=float)
for C in tqdm(Cs):
    svr = SVR(C=C, kernel='rbf')
    clf = svr
    #y_predict = cross_val_predict(rfr, Xn, y, cv=LeaveOneOut(), n_jobs=-1)
    clf.fit(Xn, y)
    y_predict_in = clf.predict(Xn)
    y_predict_out = cross_val_predict(clf, Xn, y, cv=LeaveOneOut(), n_jobs=-1)
    y_predict_in = np.clip(y_predict_in, 0, np.inf)
    y_predict_out = np.clip(y_predict_out, 0, np.inf)
    rs_in[C] = pearsonr(y, y_predict_in)[0]
    rs_out[C] = pearsonr(y, y_predict_out)[0]
    rhos_out[C] = spearmanr(y, y_predict_out)[0]


# In[68]:


rs_in.plot(label='In-sample prediction R')
rs_out.plot(label='Out-of-sample prediction R')
rhos_out.plot(label=r'Out-of-sample prediction $\rho$')
plt.xscale('log')
plt.ylabel('Correlation\n(predicted vs observed)')
plt.xlabel('C (SVR hyperparameter)')
plt.legend(fontsize=10)


# In[212]:


from sklearn.feature_selection import RFE, RFECV
svr = SVR(C=10, kernel='linear')
rfe = RFE(svr, n_features_to_select=1, step=10)
rfe.fit(Xn, y)


# In[215]:


rfe.ranking_.max()


# In[217]:


svr = SVR(C=10, kernel='rbf')
ns = range(1, 109, 1)
rs = pd.Series(index=ns, dtype=float)
for n in tqdm(ns):
    Xn_ = Xn[Xn.columns[rfe.ranking_ <= n]]
    y_predict = cross_val_predict(svr, Xn_, y, cv=LeaveOneOut(), n_jobs=-1)
    rs[n] = np.corrcoef(y, y_predict)[0, 1]


# In[218]:


rs.plot()


# In[219]:


Cs = np.logspace(-3, 3, 13)
rs_in = pd.Series(index=Cs, dtype=float)
rs_out = pd.Series(index=Cs, dtype=float)
rhos_out = pd.Series(index=Cs, dtype=float)
Xn_ = Xn[Xn.columns[rfe.ranking_ <= 10]]
for C in tqdm(Cs):
    svr = SVR(C=C, kernel='rbf')
    clf = svr
    #y_predict = cross_val_predict(rfr, Xn, y, cv=LeaveOneOut(), n_jobs=-1)
    clf.fit(Xn_, y)
    y_predict_in = clf.predict(Xn_)
    y_predict_out = cross_val_predict(clf, Xn_, y, cv=LeaveOneOut(), n_jobs=-1)
    y_predict_in = np.clip(y_predict_in, 0, np.inf)
    y_predict_out = np.clip(y_predict_out, 0, np.inf)
    rs_in[C] = pearsonr(y, y_predict_in)[0]
    rs_out[C] = pearsonr(y, y_predict_out)[0]
    rhos_out[C] = spearmanr(y, y_predict_out)[0]


# In[220]:


rs_in.plot(label='In-sample prediction R')
rs_out.plot(label='Out-of-sample prediction R')
rhos_out.plot(label=r'Out-of-sample prediction $\rho$')
plt.xscale('log')
plt.ylabel('Correlation\n(predicted vs observed)')
plt.xlabel('C (SVR hyperparameter)')
plt.legend(fontsize=10)


# In[223]:


Xn_ = Xn[Xn.columns[rfe.ranking_ <= 10]]
rfr = RandomForestRegressor(n_estimators=100)
y_predict = cross_val_predict(rfr, Xn_, y, cv=LeaveOneOut(), n_jobs=-1)
np.corrcoef(y, y_predict)[0, 1]


# In[228]:


rfr.fit(Xn_, y)
pd.Series(rfr.feature_importances_, Xn_.columns).sort_values(ascending=False).head(25)


# In[197]:


svr = SVR(C=10, kernel='rbf')
svr.fit(Xn, y)
y_predict = cross_val_predict(svr, Xn, y, cv=LeaveOneOut(), n_jobs=-1)
y_predict = np.clip(y_predict, 0, np.inf)
plt.scatter(y, y_predict, alpha=0.3)
plt.plot([0, 3], [0, 3], '--')
plt.title('R = %.2g' % np.corrcoef(y, y_predict)[0, 1])
ticks = range(4)
plt.xticks(ticks, ['%d' % 10**x for x in ticks])
plt.yticks(ticks, ['%d' % 10**x for x in ticks])
plt.xlabel('Actual Detection Threshold Ratio')
plt.ylabel('Predicted Detection\nThreshold Ratio')


# In[203]:


sns.heatmap(svr.support_vectors_, cmap='RdBu_r')


# In[78]:


abraham = pyrfume.load_data('abraham_2011/abraham-2011-with-CIDs.csv')
abraham = abraham.dropna(subset=['SMILES'])
from pyrfume.features import smiles_to_mordred
abraham_mordred = smiles_to_mordred(abraham['SMILES'].values)


# In[180]:


#X = abraham_mordred.join(abraham.set_index('SMILES').drop(['Substance', 'Group'], axis=1), how='inner', rsuffix='abr_')
abe_ok = ['MW',
 'Log (1/ODT)',
 'E',
 'S',
 'A',
 'B',
 'L',
 'V',
 'M',
 'AL',
 'AC',
 'ES',
 'C1',
 'C1AC',
 'C1AL',
 'HS',
 'C2',
 'C2Al',
 'C2AC']
X = abraham_mordred.join(abraham.set_index('SMILES')[['Log (1/ODT)']], how='inner', rsuffix='abr_')
y = X['Log (1/ODT)']
X = X.astype('float').drop('Log (1/ODT)', axis=1)
#finite_all = X.dropna(axis=1).columns.intersection(finite_mordred)
X = X[finite_mordred].fillna(0)
X = X.dropna(axis=1)
Xn = X.copy()
Xn[:] = StandardScaler().fit_transform(X)


# In[167]:


Cs = np.logspace(-3, 3, 13)
rs_in = pd.Series(index=Cs, dtype=float)
rs_out = pd.Series(index=Cs, dtype=float)
rhos_out = pd.Series(index=Cs, dtype=float)
for C in tqdm(Cs):
    svr = SVR(C=C, kernel='rbf')
    clf = svr
    #y_predict = cross_val_predict(rfr, Xn, y, cv=LeaveOneOut(), n_jobs=-1)
    clf.fit(Xn, y)
    y_predict_in = clf.predict(Xn)
    y_predict_out = cross_val_predict(clf, Xn, y, cv=LeaveOneOut(), n_jobs=-1)
    y_predict_in = np.clip(y_predict_in, 0, np.inf)
    y_predict_out = np.clip(y_predict_out, 0, np.inf)
    rs_in[C] = pearsonr(y, y_predict_in)[0]
    rs_out[C] = pearsonr(y, y_predict_out)[0]
    rhos_out[C] = spearmanr(y, y_predict_out)[0]


# In[168]:


rs_in.plot(label='In-sample prediction R')
rs_out.plot(label='Out-of-sample prediction R')
rhos_out.plot(label=r'Out-of-sample prediction $\rho$')
plt.xscale('log')
plt.ylabel('Correlation\n(predicted vs observed)')
plt.xlabel('C (SVR hyperparameter)')
plt.legend(fontsize=10)


# In[184]:


svr = SVR(C=1, kernel='rbf')
svr.fit(Xn, y)
X_ = mordred_data[finite_mordred]; y_ = mordred_data['log_abs']
X_ = X_.join(half_coleman_data.set_index('SMILES String')['det'])
X_ = X_[y_ < 10]
y_ = y_[y_ < 10]
Xn_ = pd.DataFrame(StandardScaler().fit_transform(X_), index=X_.index, columns=X_.columns)
X_['det'] = svr.predict(Xn_.drop('det', axis=1))
#y_predict = cross_val_predict(svr, Xn, y, cv=LeaveOneOut(), n_jobs=-1)


# In[188]:


Xn_ = pd.DataFrame(StandardScaler().fit_transform(X_), index=X_.index, columns=X_.columns)
Cs = np.logspace(-3, 3, 13)
rs_in = pd.Series(index=Cs, dtype=float)
rs_out = pd.Series(index=Cs, dtype=float)
rhos_out = pd.Series(index=Cs, dtype=float)
for C in tqdm(Cs):
    svr = SVR(C=C, kernel='rbf')
    clf = svr
    #y_predict = cross_val_predict(rfr, Xn, y, cv=LeaveOneOut(), n_jobs=-1)
    clf.fit(Xn_, y_)
    y_predict_in = clf.predict(Xn_)
    y_predict_out = cross_val_predict(clf, Xn_, y_, cv=LeaveOneOut(), n_jobs=-1)
    y_predict_in = np.clip(y_predict_in, 0, np.inf)
    y_predict_out = np.clip(y_predict_out, 0, np.inf)
    rs_in[C] = pearsonr(y_, y_predict_in)[0]
    rs_out[C] = pearsonr(y_, y_predict_out)[0]
    rhos_out[C] = spearmanr(y_, y_predict_out)[0]


# In[189]:


rs_in.plot(label='In-sample prediction R')
rs_out.plot(label='Out-of-sample prediction R')
rhos_out.plot(label=r'Out-of-sample prediction $\rho$')
plt.xscale('log')
plt.ylabel('Correlation\n(predicted vs observed)')
plt.xlabel('C (SVR hyperparameter)')
plt.legend(fontsize=10)


# In[ ]:




