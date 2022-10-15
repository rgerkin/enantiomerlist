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
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import LeaveOneOut
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
    # print("aqui")
    # assert all([isinstance(s, str) for s in ref_smiles]), "there are non string values"
    # print("should be")
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


def log_abs(x):
    """This section creates the log_abs values for each enantiomer
     """
    return np.abs(np.log10(x['Normalized Detection Threshold'].values[1]/x['Normalized Detection Threshold'].values[0]))


# Using the SMILES Strings, this section extracts the mordred features if package is True, morgan features otherwise
def calculate_features(half_dataset: pd.DataFrame, feature_library: str) -> pd.DataFrame:
    """Caluculates the morgan or modred features on the dataframe with just the enantiomers

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
        print("here")
        smiles = all_smiles()
        assert all([isinstance(s, str) for s in smiles]), "there are non string values"
        print("there")
        all_features = smiles_to_morgan_sim(half_dataset['SMILES String'].values, all_smiles())
    else:
        raise Exception("No such Library : %s "%feature_library)
    combined_dataset = half_dataset.set_index('SMILES String').join(all_features, how="inner", rsuffix="molecule") #This part combines that half data frame with the features and sets the index to be by smiles strings
    good_data = combined_dataset.loc[combined_dataset["log_abs"].notnull()] # This line creates a data frame with all the data from the combined_data data frame minus the lines where log abs column has a null value
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


def model(whole_dataset: pd.DataFrame, features_dataset: pd.DataFrame):
    """Generates a Random Forest Model for the dataframe of enantiomers and their features

    Args:
        whole_dataset (pd.DataFrame): The dataframe of enantiomers and their features
        features_dataset (pd.DataFrame): The dataframe with the features columns that do not have null values

    Returns:
        [Numpy Array]: returns the mean, standarad deviation, and histogram for the model
    """
    dataset = ""
    # Tells us which features dataset we used
    if whole_dataset.size < 400000:
        dataset = "Mordred Features"
    elif whole_dataset.size < 2000000:
        dataset = "Morgan Features"
    else:
        dataset = "Both Morgan and Mordred Features"

    # The Y variable holds all the correct classification values
    # The X variable holds all the data that will be used learn the classification problem
    Y = whole_dataset["log_abs"].values
    X = whole_dataset[features_dataset].astype(float).values  
    # This will help create a random split for training and testing data
    rs = np.zeros(100)
    ss = ShuffleSplit(n_splits=len(rs), random_state=0)

    counting_carvone = 0
    counting_glutamate = 0
    for i, (train, test) in enumerate(ss.split(X)):
        rfr = RandomForestRegressor(n_estimators=100, max_features=25)
        rfr.fit(X[train, :], Y[train])
        predicted = rfr.predict(X[test, :])
        rs[i] = np.corrcoef(predicted, Y[test])[0, 1]
        counting_carvone += predicted[1]
        counting_glutamate  += predicted[9]
    print(counting_carvone/100)
    print(counting_glutamate/100)
    print("The mean is ", np.mean(rs), "The Standard Error is ", np.std(rs)/np.sqrt(len(rs)))
    plt.hist(rs, alpha=0.5, label=dataset)
    plt.title("Correlation of Predicted Odor Divergence")
    plt.xlabel("Correlational Value (r) ")
    plt.ylabel("Number of Enantiomeric Pairs")
    plt.legend()
    return rs

def model_average(whole_dataset1: pd.DataFrame, features_dataset1: pd.DataFrame, whole_dataset2: pd.DataFrame, features_dataset2: pd.DataFrame):
    """Generates a Random Forest Model on the average of the predicted values of the mordred and morgan dataframes

    Args:
        whole_dataset1 (pd.DataFrame): The dataframe with the enantiomer and their mordred(or mogran) features
        features_dataset1 (pd.DataFrame): The dataframe with the features (mordred or morgan) columns that do not have null values
        whole_dataset2 (pd.DataFrame): The dataframe with the enantiomer and their morgan(or mordred) features
        features_dataset2 (pd.DataFrame): The dataframe with the features (mordred or morgan) columns that do not have null values

    Returns:
        [Numpy Array]: returns the mean, standarad deviation, and histogram for the model
    """
    # The Y variable holds all the correct classification values
    # The X variable holds all the data that will be used learn the classification problem
    Y = whole_dataset1["log_abs"].values
    X = whole_dataset1[features_dataset1].astype(float).values

    y = whole_dataset2["log_abs"].values
    x = whole_dataset2[features_dataset2].astype(float).values
    # This will help create a random split for training and testing data
    rs = np.zeros(100)
    ss = ShuffleSplit(n_splits=len(rs), random_state=0)

    for i, (train, test) in enumerate(ss.split(X)):
        rfr1 = RandomForestRegressor(n_estimators=100, max_features=25)
        rfr1.fit(X[train, :], Y[train])
        rfr2 = RandomForestRegressor(n_estimators=100, max_features=25)
        rfr2.fit(x[train, :], y[train])
        predicted = rfr1.predict(X[test, :])
        predicted2 = rfr2.predict(x[test, :])
        averaged = (predicted+predicted2)/2
        rs[i] = np.corrcoef(averaged, Y[test])[0, 1]
    print("The mean is ", np.mean(rs), " The Standard Error is ", py.stats.sem(rs))
    plt.hist(rs, alpha=0.5, label="The average of Mordred and Morgan")
    plt.title("Correlation of Predicted Odor Divergence")
    plt.xlabel("Correlational Value (r) ")
    plt.ylabel("Number of Enantiomeric Pairs")
    plt.legend()
    return rs 


def cumulativeHistogram(predicted_array: np.array, color: str, label_val: str):
    """Creates a cumulative histogram of all of the results

    Args:
        predicted_array (np.array): The predicted results in a numpy array
        color (str): Color of the line
        label_val (str): Tells use which test results each line is \
    
    return: No return because it will show the plots as the function is called
    """
    values, base = np.histogram(predicted_array)
    cumulative = np.cumsum(values)
    plt.plot(base[:-1], cumulative, c=color, label=label_val)
    plt.title("Correlation of Predicted Odor Divergence")
    plt.xlabel("Correlational Value (r) ")
    plt.ylabel("Number of Enantiomeric Pairs")
    plt.legend()
    return

def leave_one_out(whole_dataset: pd.DataFrame, features_dataset: pd.DataFrame):
    """Generates a Random Forest Model for the dataframe of enantiomers and their features

    Args:
        whole_dataset (pd.DataFrame): The dataframe of enantiomers and their features
        features_dataset (pd.DataFrame): The dataframe with the features columns that do not have null values

    Returns:
        [Numpy Array]: returns the mean, standarad deviation, and histogram for the model
    """
    dataset = ""
    # Tells us which features dataset we used
    if whole_dataset.size < 400000:
        dataset = "Mordred Features"
    elif whole_dataset.size < 2000000:
        dataset = "Morgan Features"
    else:
        dataset = "Both Morgan and Mordred Features"

    # The Y variable holds all the correct classification values
    # The X variable holds all the data that will be used learn the classification problem
    Y = whole_dataset["log_abs"].values
    X = whole_dataset[features_dataset].astype(float).values  
    # This will help create a random split for training and testing data
    predictedValues = np.zeros(X.shape[0])
    loo = LeaveOneOut()

    for i, (train, test) in enumerate(tqdm(loo.split(X))):
        rfr = RandomForestRegressor(n_estimators=100, max_features=25)
        rfr.fit(X[train, :], Y[train])
        predictedValues[test[0]] = rfr.predict(X[test, :])
    correlationCoefficient = np.corrcoef(predictedValues, Y)[0,1]
    plt.hist(correlationCoefficient, alpha=0.5, label=dataset)
    # plt.title("Correlation of Predicted Odor Divergence")
    # plt.xlabel("Correlational Value (r) ")
    # plt.ylabel("Number of Enantiomeric Pairs")
    plt.legend()
    return correlationCoefficient, predictedValues