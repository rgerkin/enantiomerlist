import os
import string
import sys
import requests
import io

import numpy as np
import pandas as pd
import scipy as py
import matplotlib.pyplot as plt
import matplotlib as mpl  # Need this line
from sklearn.ensemble import RandomForestRegressor
from IPython.display import display
from scipy.stats import pearsonr
from scipy.stats import spearmanr
import seaborn as sns
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import LeaveOneOut
from sklearn.svm import SVR
from tqdm.auto import tqdm

from rdkit import Chem
from rdkit.Chem import AllChem, SaltRemover
from rdkit import DataStructs
from mordred import Calculator
from mordred import descriptors as mordred_descriptors

from pyrfume.odorants import all_smiles


DATA_HOME = '.'  # Where the files downloaded from e.g. http://gdb.unibe.ch/downloads/ are located
HOME = os.path.expanduser('~')  # Use this HOME Line if you are running on windows, otherwise use line 25
# HOME = os.environ['HOME']
HERE = os.path.realpath(os.path.dirname(__file__))
print(HERE)
OP_HOME = os.path.join(HERE, 'olfaction-prediction')  # On GitHub at dream-olfaction/olfaction-prediction
# OP_HOME = '/home/jovyan/olfaction-prediction' # On GitHub at dream-olfaction/olfaction-prediction

paths = [OP_HOME]
for path in paths:
    if path not in sys.path:
        sys.path.append(path)


# Random SMILES Strings used for demonstration
smile1 = 'OC[C@@H](O1)[C@@H](O)[C@H](O)[C@@H]2[C@@H]1c3c(O)c(OC)c(O)cc3C(=O)O2'
smile2 = 'CC(=O)NCCC1=CNc2c1cc(OC)cc2'

def test_rdkit_mordred():
    """[This function shows the mordred features for two random smiles string; The purpose here is to demonstrate how the mordred library works]
    """
    result = smiles_to_mordred([smile1, smile2])
    display(result)


def load_data(source: str) -> pd.DataFrame:
    """Gets the data and saves it as a data frame

    Args:
        source (str): The key word cooresponding to the correct dataset

    Returns:
        pd.DataFrame: returns the data as a pandas data frame
    """
    if source == 'coleman':
        # Base URL of Google spreadsheets (for API access)
        url = "https://spreadsheets.google.com/feeds/download/spreadsheets/Export"
        # ID of the enantiomers spreadsheet
        key = "1yo6iRZ8f6meKpdsjD4UAQ5CQrMFlwFDhBUKk0DFspIQ"
        # Make a full URL from the above two things
        full_url = '%s?key=%s&exportFormat=csv&gid=0' % (url, key)
        # Get the data
        response = requests.get(full_url)
        # Make sure it worked
        assert response.status_code == 200, 'Wrong status code'
        # Turn that returned data into a string buffer that will behave like a file, so we can load it into Pandas
        f = io.StringIO(str(response.content).replace("\\r\\n", "\r").replace("b'", ""))
        # Load it into a Pandas dataframe
        df = pd.read_csv(f)
        # put that code block from the notebook that reads the spreadsheet and gets df
    else:
        df = None
    return df

def load_other_smiles(gdb11=False,
                      gdb13_fr=False,
                      fragranceDB=False,
                      coleman=False,
                      min_atoms=3,  # Minimum number of heavy atoms (1-11)
                      max_atoms=8,  # Maximum number of heavy atoms (1-11)
                      smiles_remove=[]):
    """Load SMILES strings from other sources"""
    smiles = []
    if gdb11:
        # Load GDB-11 data
        for n in range(min_atoms, max_atoms+1):
            print("Loading molecules with %d heavy atoms..." % n)
            df = pd.read_csv(os.path.join(DATA_HOME,'gdb11/gdb11_size%02d.smi') % n, delimiter = '\t', header = None)
            smiles += list(df[0])
    if gdb13_fr:
        # Load GDB-13 fragrance-like data
        print("Loading fragrance-like molecules from GDB-13...")
        df = pd.read_csv(os.path.join(DATA_HOME,'GDB-13.FL.smi'), delimiter='\t', header = None)
        smiles += list(df[0])
    if fragranceDB:
        # Load fragranceDB data
        print("v molecules from FragranceDB (fragrance-like)...")
        df = pd.read_csv(os.path.join(DATA_HOME,'FragranceDB.FL.smi'), delimiter = ' ', header = None)
        smiles += list(df[0])
    if coleman:
        df=load_data('coleman')
        # May need to change the df part below, I just know it will not be the same as above where it says "df.index"
        smiles += list(df['SMILES String'])
        
    # Ensure a standard SMILES format by converting to Mol and back
    # lines 261 to 267 disregard anything that is not a smile string, aka the trace of any molecule that does not have a smile string
    newSmiles = []
    # now everything that says newSmiles has been put in replacement of just "smiles"
    if "SMILES String" in smiles:
        smiles.remove('SMILES String')
    for smile in smiles:
        if str(smile) == "nan":
            continue
        else:
            # print(smile)
            result = Chem.MolFromSmiles(smile)
            # print(type(result))
            newSmiles.append(Chem.MolToSmiles(result))      
#     smiles = [Chem.MolToSmiles(Chem.MolFromSmiles(smile),isomericSmiles=True) \
#               for smile in smiles]
    # newSmiles = list(set(newSmiles)) # Remove duplicates
    print("Loaded %d molecules" % len(newSmiles))
    wildcard_smiles = [s for s in newSmiles if '*' in s]
    if wildcard_smiles:
        print("Removing %d molecules with wildcards" % len())
        newSmiles = list(set(newSmiles).difference(wildcard_smiles))
    # Remove molecules that are also in the DREAM data, and make sure that both data sets 
    if smiles_remove:
        newSmiles = list(set(newSmiles).difference(smiles_remove))
        print("After removing excluded molecules, there are now %d molecules" % len(newSmiles))
    return newSmiles


def log_abs(enantiomer):
    """Creates the log_abs values for each enantiomer"""
    return np.abs(np.log10(enantiomer['Normalized Detection Threshold'].values[1]/enantiomer['Normalized Detection Threshold'].values[0]))

def harmonic(enantiomer):
    """Compuates harmonic value for each enantiomer"""
    return np.log10(1/np.mean(1/enantiomer['Normalized Detection Threshold'].values))


# Using the SMILES Strings, this section extracts the mordred features if package is True, morgan features otherwise
def calculate_features(half_dataset: pd.DataFrame, feature_library: str) -> pd.DataFrame:
    """Caluculates the morgan or modred features on enantiomers dataframe

    Args:
        half_dataset (pd.DataFrame): The dataframe with just the enantiomer molecules
        feature_library (str): Accepts either "mordred" or "morgan"

    Raises:
        Exception: If a valid feature_library is not given

    Returns:
        pd.DataFrame: Returns a dataframe with the enatiomer molecules and their features computed from the feature library 
    """
    if feature_library.lower() == "mordred":
        all_features = smiles_to_mordred(half_dataset["SMILES String"].values)
    elif feature_library.lower() == "morgan":
        smiles = all_smiles()
        assert all([isinstance(s, str) for s in smiles]), "there are non string values"
        all_features = smiles_to_morgan_sim(half_dataset['SMILES String'].values, all_smiles())
    else:
        raise Exception("No such Library : %s "%feature_library)
    combined_dataset = half_dataset.set_index('SMILES String').join(all_features, how="inner", rsuffix="molecule") # Combines that half data frame with the features and sets the index to be by smiles strings
    good_data = combined_dataset.loc[combined_dataset["log_abs"].notnull()] # Creates a data frame with all the data from the combined_data data frame minus the lines where log abs column has a null value
    return good_data


def finite_features(whole_dataset : pd.DataFrame) -> pd.DataFrame:
    """Gets rid of the features that have null values

    Args:
        whole_dataset (pd.DataFrame): The whole dataset of enantiomers and features

    Returns:
        pd.DataFrame: Returns all the feature columns that did not have null values
    """
    whole_dataset = whole_dataset[list(whole_dataset.iloc[:, 10:])].drop(columns=['Resources'])
    finite_feature = whole_dataset[list(whole_dataset.iloc[:, 10:])].astype(float).isnull().sum() == 0 # checks that the number of nulls in the features columns is zero
    finite_feature = finite_feature[finite_feature].index # takes the columns where the condition in the line above is true
    return finite_feature

def smiles_to_mordred(smiles: str, features: list = None) -> pd.DataFrame:
    """[This function accepts SMILES strings and calculates their respective mordred features]

    Args:
        smiles (str): [The SMILE String(s) for a molecule(s)]
        features (list, optional): [description]. Defaults to None.

    Returns:
        pd.DataFrame: [This function returns a pandas dataframe where rows are the smiles strings used when calling this function and columns are all the mordred features for each SMILES string]
    """
    # create descriptor calculator with all descriptors
    calc = Calculator(mordred_descriptors)
    print("Convering SMILES string to Mol format...")
    mols = [Chem.MolFromSmiles(smi) for smi in smiles]
    print("Computing 3D coordinates...")
    s = SaltRemover.SaltRemover()
    failed_embeddings = []
    for i, mol in enumerate(mols):
        try:
            mol = s.StripMol(mol, dontRemoveEverything=True)
            mol = Chem.AddHs(mol)
            AllChem.Compute2DCoords(mol)
            result = AllChem.EmbedMolecule(mol, maxAttempts=10)
            if result == 0:  # Successful embedding
                AllChem.UFFOptimizeMolecule(mol)  # Is this deterministic?
            else:
                print("Removing %s due to failed embedding" % smiles[i])
                failed_embeddings.append(i)
        except:
            print(mol)
            print(smiles[i])
    smiles = [smile for i, smile in enumerate(smiles) if i not in failed_embeddings]
    mols = [mol for i, mol in enumerate(mols) if i not in failed_embeddings]
    print("Computing Mordred features...")
    df = calc.pandas(mols)
    if features is not None:
        df = df[features]  # Retain only the specified features
    # Getting rid of as_matrix() below
    mordred = pd.DataFrame(df.to_numpy(), index=smiles, columns=df.columns)
    print("There are %d molecules and %d features" % mordred.shape)
    return mordred


def smiles_to_morgan(smiles: str, radius: int = 5, features: list = None) -> pd.DataFrame:
    """Takes a list of SMILES strings and generates their respective morgan features

    Args:
        smiles (str): list of SMILES strings
        radius (int, optional): The number of atoms away from the center that can be considered apart of the fingerprint. Defaults to 5.
        features (list, optional): Defaults to None.

    Returns:
        [pd.DataFrame]: Returns the dataframe of morgan features. Rows are the SMILES Strings and columns are the features for each SMILES String
    """
    mols = [Chem.MolFromSmiles(smi) for smi in smiles]
    fps = [AllChem.GetMorganFingerprint(mol, radius) for mol in mols]
    features = []
    fp_ids = list()
    for fp in fps:
        fp_ids += list(fp.GetNonzeroElements().keys())
    fp_ids = list(set(fp_ids))
    morgan = np.empty((len(fps), len(fp_ids)))
    for i, fp in enumerate(fps):
        for j, fp_id in enumerate(fp_ids):
            morgan[i, j] = fp[fp_id]
    morgan = pd.DataFrame(morgan, index=smiles, columns=fp_ids)
    if features is not None:
        morgan = morgan[features]  # Retain only the specified features
    return morgan


def smiles_to_morgan_sim(smiles: str, ref_smiles: str, radius: int = 5, features: list = None) -> pd.DataFrame:
    """Takes a list of SMILES strings and generates their respective morgan features

    Args:
        smiles (str): List of SMILES Strings 
        ref_smiles (str): A list of SMILES Strings
        radius (int, optional): The number of atoms away from the center that can be considered apart of the fingerprint. Defaults to 5.
        features (list, optional): Defaults to None.

    Returns:
        pd.DataFrame: [description]
    """
    mols = [Chem.MolFromSmiles(smi) for smi in smiles]
    fps = [AllChem.GetMorganFingerprint(mol, radius) for mol in mols]
    morgan = np.empty((len(smiles), len(ref_smiles)))
    for i, ref_smile in enumerate(ref_smiles):
        ref = Chem.MolFromSmiles(ref_smile)
        if ref is not None:
            fp_ref = AllChem.GetMorganFingerprint(ref, radius)
            morgan[:,i] = np.array([DataStructs.DiceSimilarity(fp, fp_ref) for fp in fps])
    morgan = pd.DataFrame(morgan, index = smiles, columns=ref_smiles)
    print("%d similarity features for %d molecules" % morgan.shape)
    return morgan

def fold_difference_of_enantiomers(enantiomers_df):
    """Relational graph to show the magnitude difference between all enantiomeric pairs in the dataset.

    Args:
        enantiomers_df (df): Dataframe with all enantiomeric pairs

    Returns:
        N/a: Plots fold difference versus enantiomeric pairs in the set
    """
    sns.set_style("whitegrid")
    plt.figure(figsize=(4,5.5))
    df_with_log_abs = enantiomers_df[enantiomers_df["log_abs"]<50]
    plt.hist(df_with_log_abs["log_abs"], bins=25, alpha=0.8);
    plt.xticks([0,1,2,3], ["1x", "10x", "100x", "1000x"])
    plt.xlabel("\nFold Difference in Detection\nThreshold between Enantiomers")
    plt.ylabel("Number of Enantiomeric Pairs")
    plt.plot()

def create_model(Xn, y):
    """Creating a Support Vector Regression Model to display predictings

    Args:
        Xn (df): Features 
        y (array): Resulting y values from model predicting

    Return: 
        N/a: Plot model results
    """
    Cs = np.logspace(-3, 3, 13)
    rs_in = pd.Series(index=Cs, dtype=float)
    rs_out = pd.Series(index=Cs, dtype=float)
    rhos_out = pd.Series(index=Cs, dtype=float)
    
    for C in tqdm(Cs):
        svr = SVR(C=C, kernel='rbf')
        clf = svr
        clf.fit(Xn, y)
        y_predict_in = clf.predict(Xn)
        y_predict_out = cross_val_predict(clf, Xn, y, cv=LeaveOneOut(), n_jobs=-1)
        y_predict_in = np.clip(y_predict_in, 0, np.inf)
        y_predict_out = np.clip(y_predict_out, 0, np.inf)
        rs_in[C] = pearsonr(y, y_predict_in)[0]
        rs_out[C] = pearsonr(y, y_predict_out)[0]
        rhos_out[C] = spearmanr(y, y_predict_out)[0]

    plt.figure(figsize=(5, 5))
    rs_in.plot(label='In-sample prediction R')
    rs_out.plot(label='Out-of-sample prediction R')
    rhos_out.plot(label=r'Out-of-sample prediction $\rho$')
    plt.xscale('log')
    plt.ylim(0, 1)
    plt.ylabel('Correlation\n(predicted vs observed)')
    plt.xlabel('C (SVR hyperparameter)')
    plt.legend(fontsize=10)

def cross_val(Xn, y):
    """Creates cross validation model to measure model performance

    Args:
        Xn (df): Features 
        y (array): Resulting y values from model predicting

    Returns:
        N/a: Plots validation metrics
    """
    sns.set_style('whitegrid')
    svr = SVR(C=10, kernel='rbf')
    svr.fit(Xn, y)
    y_predict = cross_val_predict(svr, Xn, y, cv=LeaveOneOut(), n_jobs=-1)
    y_predict = np.clip(y_predict, 0, np.inf)
    plt.figure(figsize=(5, 5))
    plt.scatter(y, y_predict, alpha=0.3)

    plt.plot([0, 4], [0, 4], '--')
    plt.xlim(0, 4)
    plt.ylim(0, 4)

    plt.title('R = %.2g' % np.corrcoef(y, y_predict)[0, 1])
    plt.xticks([1,2,3,4], ["1x", "10x", "100x", "1000x"])
    plt.yticks([1,2,3,4], ["1x", "10x", "100x", "1000x"])

    plt.xlabel('Actual Detection Threshold Ratio')
    plt.ylabel('Predicted Detection\nThreshold Ratio')