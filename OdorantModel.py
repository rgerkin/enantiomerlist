#!/usr/bin/env python
# coding: utf-8

# ## Preliminaries

# In[1]:


#Gets the updates from the development files you imported
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


import pyrfume 
pyrfume.__file__


# In[4]:


#Imports the pands library, the math library, and the init class python file
#Last line: updates matlab library that is used in init file
import __init__ as a
import math
import numpy as np
import matplotlib.pyplot as plt
a.mpl.rcParams['font.size'] = 12


# ## Demo of the Libraries

# In[5]:


# Demo for morgan features
from pyrfume.features import smiles_to_mordred, smiles_to_morgan, smiles_to_morgan_sim
from pyrfume.odorants import all_smiles


# In[6]:


smiles_to_morgan_sim(["CCC","CC(=O)[Si](C)(C(C)(C)C)C(C)(C)C"],all_smiles())


# In[7]:


#Demo for mordred features
a.test_rdkit_mordred() # Should show features for two molecules below.  


# In[8]:


#Demo
#Takes a pair of entantiomers' smiles strings and computes the 
#mordred features from those smiles strings and creates an array of those features for each enantiomer
#Then the two arrays are compared to see how similar the mordred features are in one pairr
plus_carvone = 'CC1=CC[C@@H](CC1=O)C(=C)C'
minus_carvone = 'CC1=CC[C@H](CC1=O)C(=C)C'
features = a.smiles_to_mordred([plus_carvone,minus_carvone])
plus_features = features.values[0,:]
minus_features = features.values[1,:]
print("%.1f percent of Mordred features are identical between (+)-carvone and (-)-carvone." % 
      (100.0*(plus_features==minus_features).mean()))


# ## Start of Project

# In[86]:


# This section loads in the data and groups each pair of enantiomers by making them have the same number associated with column "N"
# We also set the Normalized Detection Thresholds to be of the same type as a way of trying to avoid type errors later on
# This section also takes each pair of enantiomers and computes the ratio of the Normalized Detection Thresholds between the two
# The ratio is saved and will be used when the dataset gets cut in half
coleman_smiles = a.load_other_smiles(coleman=True)
coleman_data = a.load_data("coleman")#.iloc[1:]
coleman_data['N'] = np.arange(0, coleman_data.shape[0]/2, 0.5).astype(int)
coleman_data['Normalized Detection Threshold'] = coleman_data['Normalized Detection Threshold'].astype('float')
coleman_data.head()


# In[143]:


names = []
for i in range(0,464,2):
    value = float(coleman_data.loc[i+1, "Normalized Detection Threshold"]) - float(coleman_data.loc[i, "Normalized Detection Threshold"])
    if abs(value) < 0.03:
        names.append(coleman_data.loc[i, "Molecule Name"])
print([x + "\n" for x in names])


# ## Cutting the Dataframe in Half

# In[10]:


# This section creates a new data frame with just one odorant of each enantiomeric pair from the original dataset 
# Adds the ratio of absolute log values
# half_log_abs = coleman_data.groupby('N').apply(log_abs)
half_log_abs = coleman_data.groupby('N').apply(a.log_abs)
half_coleman_data = coleman_data.iloc[::2]
half_coleman_data.loc[:, 'log_abs'] = half_log_abs.values


# In[11]:


# This line makes sure that the rest of the exsisting null values are equal in the new data frame and in the new data frame's 'log_abs' column
half_log_abs.isnull().sum(), half_coleman_data['log_abs'].isnull().sum()


# In[12]:


# This line checks that the half dataset was created properly
half_coleman_data.head()


# In[13]:


# This section gets rid of all the invalid SMILES Strings, specifically the duplicates because we don't want to count their perceptual features twice and the "nan" values 
half_coleman_data = half_coleman_data.drop_duplicates(subset=['SMILES String'])
half_coleman_data = half_coleman_data[~half_coleman_data['SMILES String'].str.contains('NaN', na=True)]
half_coleman_data = half_coleman_data[~half_coleman_data['SMILES String'].str.contains('nan', na=True)]


# In[14]:


# These two assert statements are to ensure that we only have unqiue smiles strings and that no smiles strings are nan values
assert half_coleman_data['SMILES String'].shape == half_coleman_data['SMILES String'].unique().shape, "Number of SMILES strings should equal number of unique SMILES strings at this stage"


# In[15]:


# This line makes sure that are no more nan values in the smiles string column
assert sum(half_coleman_data['SMILES String']=='nan') == 0, "There should be no NaN SMILES strings at this point"


# In[16]:


#This line gets rid of the rows with a null log_abs value
half_coleman_data = half_coleman_data[~half_coleman_data['log_abs'].isnull()]


# In[31]:


#This line makes sure there are no more log_abs values with the value null
assert sum(half_coleman_data['log_abs'].isnull())
# print([x for x in half_coleman_data["log_abs"] if x ])


# ## Computing the Features

# In[17]:


# These lines calculate the mordred and morgan features
mordred_data = a.calculate_features(half_coleman_data, "mordred")
morgan_data = a.calculate_features(half_coleman_data, "morgan")


# In[18]:


#These lines make sure that we only use the molecules that have mordred and morgan features computed
common_index = mordred_data.index.intersection(morgan_data.index)
mordred_data = mordred_data.loc[common_index]
morgan_data = morgan_data.loc[common_index]


# In[19]:


#Data frame that has both the mordred and morgan features
both = mordred_data.join(morgan_data.iloc[:,10:], how="inner")
both.head()


# In[20]:


#Gets all Mordred or mogan features that have numeric values and not Null values
#The last line joins the final mordred and morgan features 
finite_mordred = a.finite_features(mordred_data)
finite_morgan = a.finite_features(morgan_data)
both_features = finite_mordred | finite_morgan


# In[21]:


half_coleman_data


# ## Creating the Models

# In[113]:


#This model tests on the average of the predicted values of the mordred and morgan feature dataframes
# average_model = a.model_average(mordred_data, finite_mordred, morgan_data, finite_morgan)


# In[114]:


#This model tests on a predicted values list that contains data from a dataframe that has both morgan and mordred features
# combined = a.model(both, both_features)


# In[176]:


#Two models are represented here, one shows a test on mordred features and the other shows a test on morgan features
#mordred_rs = a.model(mordred_data, finite_mordred)
combined_features = a.model(both, both_features)
morgan_rs, predicted_values = a.model(morgan_data, finite_morgan)


# In[84]:


plt.scatter(predicted_values["log_abs"], predicted_values["predict"])
#plt.xticks([0,1,2,3], ["1x", "10x", "100x", "1000x"])


# In[260]:


import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")
sns.set_context('poster')
#plt.figure(figsize=(7,7))
stuff = half_coleman_data[half_coleman_data["log_abs"]<50]
plt.hist(stuff["log_abs"], bins=25, color="#800000", alpha=0.8)
#plt.scatter(stuff["log_abs"], stuff["log_abs"])
plt.xticks([0,1,2,3], ["1x", "10x", "100x", "1000x"])
plt.xlabel("\n Fold Difference in Detection Threshold between Enantiomers")
plt.ylabel("Number of Enantiomeric Pairs")


# In[173]:


#print(x for x in half_coleman_data["log_abs"] if x < 0.4)
for x in predicted_values["log_abs"]:
    if x < 0.3:
        print(x)
        


# In[174]:


predicted_values.reset_index()


# In[178]:


columns = {"old model": [0], "new model": [0], "actual ratio": [0]}
predicted_models = pd.DataFrame(data=columns, index=predicted_values["Molecule Name"])
predicted_models.iloc[:,0] = 1
predicted_models.iloc[:,1] = predicted_values["predict"].values
predicted_models.iloc[:,2] = ["%.4g"%(10**x) for x in predicted_values["log_abs"]]
predicted_models


# In[172]:


names = ["(4R)-(-)-carvone/(4S)-(+)-carvone", "(3R)-(-)-linalool/(3S)-(-)linalool"]
condensed_predicted_models = pd.DataFrame(data=columns, index=names ) 
condensed_predicted_models.iloc[0,:] = predicted_models.iloc[1,:]
condensed_predicted_models.iloc[1,:] = predicted_models.iloc[30,:]
condensed_predicted_models


# In[23]:


#This plot shows the correlation between the predicted mordred values and predicted morgan values used in the model
plt.scatter(mordred_rs, morgan_rs)
plt.title("Relationship between Morgan and Mordred Correlational Values")
plt.xlabel("Modred Correlation Values")
plt.ylabel("Morgan Correlation Values")
plt.plot()


# In[111]:


plt.figure(figsize=(5,5))
a.cumulativeHistogram(morgan_rs, 'green', 'Morgan Features',)
a.cumulativeHistogram(mordred_rs, 'blue', 'Mordred Features',)
a.cumulativeHistogram(combined, 'red', 'Mordred & Morgan Features')
a.cumulativeHistogram(average_model, 'orange', 'Average of Mordred & Morgan Features')


# In[117]:


#This line gets rid of the mordred feature columns where the log abs values are 0 or greater than 1e100
#good_data = all_data[(all_data['log_abs']>=0) & (all_data['log_abs']<=1e100)]
#halfColeman_data['log_abs'].isnull().sum()

#NOTE: NOT SURE IF WE STILL NEED THE LINES ABOVE

