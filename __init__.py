import os
import sys
import collections
import requests
import io

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor,ExtraTreesRegressor
from IPython.display import display,HTML

from rdkit import Chem
from rdkit.Chem.Descriptors import MolWt
from rdkit.Chem import inchi,AllChem,SaltRemover
from rdkit import DataStructs
from rdkit.ML.Descriptors import MoleculeDescriptors
from mordred import Calculator, all_descriptors

try:
    import obabel
    from eden.converter.molecule.obabel import obabel_to_eden
    from eden.graph import Vectorizer
    import pybel
    HAS_OBABEL = True
except ImportError:
    HAS_OBABEL = False

DATA_HOME = '.' # Where the files downloaded from e.g. http://gdb.unibe.ch/downloads/ are located
OP_HOME = '/Users/rgerkin/Dropbox/science/olfaction-prediction' # On GitHub at dream-olfaction/olfaction-prediction
#OP_HOME = '/home/jovyan/olfaction-prediction' # On GitHub at dream-olfaction/olfaction-prediction

paths = [OP_HOME]
for path in paths:
    if path not in sys.path:
        sys.path.append(path)

from opc_python.gerkin import dream
from opc_python.utils import loading
# Load descriptors, CIDs, and dilutions for DREAM molecules
descriptors = loading.get_descriptors(format=True)
descriptors_short = [x[:5] for x in descriptors]
USE_DRAGON = False # If set to False, uses Mordred features instead of Dragon features
SHADMANY_FILE = 'EnantiomerList - Sheet1.csv'

smile1 = 'OC[C@@H](O1)[C@@H](O)[C@H](O)[C@@H]2[C@@H]1c3c(O)c(OC)c(O)cc3C(=O)O2'
smile2 = 'CC(=O)NCCC1=CNc2c1cc(OC)cc2' 

def test_eden_pybel():
    # Test Code to make sure EdEN and pybel are works=ing
    vectorizer = Vectorizer(min_r=0,min_d=0,r=1,d=2)
    eden_graph_generator = obabel_to_eden([smile1,smile2],file_format='smi')
    graphs = [graph for graph in eden_graph_generator]
    X = vectorizer.transform(graphs)
    result1 = pybel.readstring("smi", smile1)
    result2 = pybel.readstring("smi", smile2)
    display(result1,result2)
    # Should display the molecules corresponding to 'smile1'

def test_rdkit_mordred():
    result = smiles_to_mordred([smile1,smile2])
    display(result)

def smiles_to_mordred(smiles,features=None):
    # create descriptor calculator with all descriptors
    calc = Calculator(all_descriptors())
    print("Convering SMILES string to Mol format...")
    mols = [Chem.MolFromSmiles(smi) for smi in smiles]
    print("Computing 3D coordinates...")
    s = SaltRemover.SaltRemover()
    for i,mol in enumerate(mols):
        mol = s.StripMol(mol,dontRemoveEverything=True)
        mol = Chem.AddHs(mol)
        AllChem.Compute2DCoords(mol)
        AllChem.EmbedMolecule(mol)
        AllChem.UFFOptimizeMolecule(mol) # Is this deterministic?  
    print("Computing Mordred features...")
    df = calc.pandas(mols)
    if features is not None:
        df = df[features] # Retain only the specified features
    mordred = pd.DataFrame(df.as_matrix(),index=smiles,columns=df.columns)
    print("There are %d molecules and %d features" % mordred.shape)
    return mordred

def make_graphs(smiles):
    eden_graph_generator = obabel_to_eden(smiles,file_format='smi') # Convert from SMILES to EdEN format
    graphs = [graph for graph in eden_graph_generator] # Compute graphs for each molecule
    vectorizer = Vectorizer(min_r=0,min_d=0,r=1,d=2)
    sparse = vectorizer.transform(graphs) # Compute the NSPDK features and store in a sparse array
    return sparse
  
def smiles_to_nspdk(smiles,features=None):
    nspdk_sparse = make_graphs(smiles) # Compute the NSPDK features and store in a sparse array
    n_molecules,n_features = nspdk_sparse.shape
    print("There are %d molecules and %d potential features per molecule" % (n_molecules,n_features))
    # Extract the indices of NSPDK features where at least one molecules is non-zero
    if features is None:
        original_indices = sorted(list(set(nspdk_sparse.nonzero()[1])))
        n_used_features = len(original_indices)
        print('Only %d of these features are used (%.1f features per molecule; %.1f molecules per feature)' % \
              (n_used_features,nspdk_sparse.size/n_molecules,nspdk_sparse.size/n_used_features))
        # Create a dense array from those non-zero features
        nspdk_dense = nspdk_sparse[:,original_indices].todense()
        indices = original_indices
    else:
        n_used_features = len(features)
        nspdk_sparse = nspdk_sparse[:,features]
        print('Only %d of these features will be used (%.1f features per molecule; %.1f molecules per feature)' % \
              (n_used_features,nspdk_sparse.size/n_molecules,nspdk_sparse.size/n_used_features))
        nspdk_dense = nspdk_sparse.todense() # Include only the desired features
        indices = features
    # Create a Pandas DataFrame
    nspdk = pd.DataFrame(nspdk_dense,index=smiles,columns=indices)
    mols = [Chem.MolFromSmiles(smi) for smi in smiles]
    return nspdk

def smiles_to_nspdk_sim(smiles,ref_smiles,features=None):
    sparse = make_graphs(smiles)
    ref_sparse = make_graphs(ref_smiles)
    sim = sparse.dot(ref_sparse.T).todense()
    sim = pd.DataFrame(sim,index=smiles,columns=ref_smiles)
    return sim

def smiles_to_dragon(smiles,features=None):
    smiles_df = pd.read_csv(os.path.join(DATA_HOME,'all_SMILES.txt'),delimiter=' ',names=['SMILES'])
    smiles_list = list(smiles_df['SMILES'])
    dragon = pd.read_csv(os.path.join(DATA_HOME,'all_SMILES_Dragon.txt'),index_col=1,delimiter='\t').iloc[:,1:]
    dragon.index = smiles_list
    if features is not None:
        dragon = dragon[features]    
    return dragon.loc[smiles,:]

def smiles_to_morgan(smiles,radius=5,features=None):
    mols = [Chem.MolFromSmiles(smi) for smi in smiles]
    fps = [AllChem.GetMorganFingerprint(mol,radius) for mol in mols]
    features = []
    for fp in fps:
        fp_ids += list(fp.GetNonzeroElements().keys())
    fp_ids = list(set(fp_ids))
    morgan = np.empty((len(fps),len(fp_ids)))
    for i,fp in enumerate(fps):
        for j,fp_id in enumerate(fp_ids):
            morgan[i,j] = fp[fp_id]
    morgan = pd.DataFrame(morgan,index=smiles,columns=fp_ids)
    if features is not None:
        morgan = morgan[features] # Retain only the specified features
    return morgan

def smiles_to_morgan_sim(smiles,ref_smiles,radius=5,features=None):
    mols = [Chem.MolFromSmiles(smi) for smi in smiles]
    fps = [AllChem.GetMorganFingerprint(mol,radius) for mol in mols]
    morgan = np.empty((len(smiles),len(ref_smiles)))
    for i,ref_smile in enumerate(ref_smiles):
        ref = Chem.MolFromSmiles(ref_smile)
        fp_ref = AllChem.GetMorganFingerprint(ref,radius)
        morgan[:,i] = np.array([DataStructs.DiceSimilarity(fp,fp_ref) for fp in fps])
    morgan = pd.DataFrame(morgan,index=smiles,columns=ref_smiles)
    print("%d similarity features for %d molecules" % morgan.shape)
    return morgan

def mol_file_to_smiles(mol_file_path):
    # Takes a path to a .mol file and returns a SMILES string
    x = Chem.MolFromMolFile(mol_file_path)
    result = Chem.MolToSmiles(a,isomericSmiles=True)
    return result

def smiles_to_image(smiles):
    # Takes a SMILES string and renders a 2D image of the molecule
    result = pybel.readstring("smi",smiles)
    display(result)

def get_dream_smiles(CIDs):
    # Get SMILES strings for DREAM molecules
    df = pd.read_csv(os.path.join(OP_HOME,'data/CIDs_to_SMILES.txt'),header=None,
                                   delimiter='\t',names=['SMILES'])
    smiles_dream = list(df['SMILES'][CIDs])
    # Ensure that isomeric information is consistent.  
    smiles_dream = [Chem.MolToSmiles(Chem.MolFromSmiles(smile),isomericSmiles=True) \
                    for smile in smiles_dream]
    return smiles_dream

def find_isomers(CIDs,smiles_list):
    smiles_list2 = [Chem.MolToSmiles(Chem.MolFromSmiles(smile),isomericSmiles=False) \
                   for smile in smiles_list]  
    duplicates = [item for item, count in collections.Counter(smiles_list2).items() \
                  if count > 1]
    isomeric_CIDs = []
    for duplicate in duplicates:
        isomeric_CIDs.append([CIDs[i] for i,x in enumerate(smiles_list2) \
                     if x==duplicate])
    return isomeric_CIDs

def show_isomers(isomeric_CIDs,CIDs,smiles_list):
    for CID1,CID2 in isomeric_CIDs:
        print(CID1,CID2)
        index1 = CIDs.index(CID1)
        index2 = CIDs.index(CID2)
        display(pybel.readstring("smi",smiles_list[index1]))
        display(pybel.readstring("smi",smiles_list[index2]))

def plot_isomer_ratings(Y_dream,CIDs,labels):
    means = dream.filter_Y_dilutions(Y_dream,'gold').mean(axis=1).unstack('Descriptor')
    std = dream.filter_Y_dilutions(Y_dream,'gold').std(axis=1).unstack('Descriptor')
    plt.errorbar(range(21),means.loc[CIDs[0]],std.loc[CIDs[0]],label='racemic')
    plt.errorbar(range(21),means.loc[CIDs[1]],std.loc[CIDs[1]],label='L')
    plt.xticks(range(21),sorted(descriptors_short),rotation=75)
    plt.xlim(-0.5,20.5)
    plt.legend()
    plt.title('%s vs %s' % labels)
    plt.show()


def load_other_smiles(gdb11=False,
                      gdb13_fr=False,
                      fragranceDB=False,
                      shadmany=False,
                      min_atoms=3, # Minimum number of heavy atoms (1-11)
                      max_atoms=8, # Maximum number of heavy atoms (1-11)
                      smiles_remove=[]):
    """Load SMILES strings from other sources"""

    smiles = []
    if gdb11:
        # Load GDB-11 data
        for n in range(min_atoms,max_atoms+1):
            print("Loading molecules with %d heavy atoms..." % n)
            df = pd.read_csv(os.path.join(DATA_HOME,'gdb11/gdb11_size%02d.smi') % n,delimiter='\t',header=None)
            smiles += list(df[0])
    if gdb13_fr:
        # Load GDB-13 fragrance-like data
        print("Loading fragrance-like molecules from GDB-13...")
        df = pd.read_csv(os.path.join(DATA_HOME,'GDB-13.FL.smi'),delimiter='\t',header=None)
        smiles += list(df[0])
    if fragranceDB:
        # Load fragranceDB data
        print("v molecules from FragranceDB (fragrance-like)...")
        df = pd.read_csv(os.path.join(DATA_HOME,'FragranceDB.FL.smi'),delimiter=' ',header=None)
        smiles += list(df[0])
    if shadmany:
        # Load Shadmany data
        print("Loading molecules from the Shadmany enantiomer collection...")
        df = load_data('shadmany')
        smiles += list(df.index)
        
    # Ensure a standard SMILES format by converting to Mol and back
    smiles = [Chem.MolToSmiles(Chem.MolFromSmiles(smile),isomericSmiles=True) \
              for smile in smiles]
    smiles = list(set(smiles)) # Remove duplicates
    print("Loaded %d molecules" % len(smiles))
    wildcard_smiles = [s for s in smiles if '*' in s]
    if wildcard_smiles:
        print("Removing %d molecules with wildcards" % len())
        smiles = list(set(smiles).difference(wildcard_smiles))
    # Remove molecules that are also in the DREAM data, and make sure that both data sets 
    if smiles_remove:
        smiles = list(set(smiles).difference(smiles_remove))
        print("After removing excluded molecules, there are now %d molecules" % len(smiles))
    return smiles

def load_data(source):
    if source == 'shadmany':
        url = "https://spreadsheets.google.com/feeds/download/spreadsheets/Export"
        key = "1yo6iRZ8f6meKpdsjD4UAQ5CQrMFlwFDhBUKk0DFspIQ"
        response = requests.get('%s?key=%s&exportFormat=csv&gid=0' % (url,key))
        assert response.status_code == 200, 'Wrong status code'
        x = io.StringIO(str(response.content).replace("\\r\\n","\r").replace("b'",""))
        df = pd.read_csv(x)
        #df = pd.read_csv(os.path.join(DATA_HOME,SHADMANY_FILE),header=0)
        df = df.dropna(subset=['SMILES String','Detection Threshold'])
        df = df.set_index('SMILES String')
        df.index = [Chem.MolToSmiles(Chem.MolFromSmiles(smile),isomericSmiles=True) \
                    for smile in df.index]
        df = df.groupby(df.index).first() # Drop duplicates
        df['Pubchem ID #'] = df['Pubchem ID #'].astype(int)
        df['Detection Threshold'] = df['Detection Threshold'].astype(float)
        
        # Work with nonisomeric smiles in order to group isomers
        x = df.reset_index()
        x['index'] = [Chem.MolToSmiles(Chem.MolFromSmiles(smile),isomericSmiles=False) \
                      for smile in x['index']]
        counts = x['index'].value_counts()
        nonisomers = counts[counts!=2]
        nonisomer_CIDs = x[x['index'].isin(nonisomers.index)]['Pubchem ID #']
        df = df[~df['Pubchem ID #'].isin(nonisomer_CIDs)]

        # Now sort by SMILES to get isomers to be adjacent
        x = df.reset_index()
        df['smiles_temp'] = [Chem.MolToSmiles(Chem.MolFromSmiles(smile),isomericSmiles=False) \
                      for smile in x['index']]
        df = df.sort_values('smiles_temp')
        df = df.drop('smiles_temp',axis=1)

        # Now make sure that (+) isomers come before (-) isomers
        for i in range(0,df.shape[0],2):
            if ('(-)' in df.iloc[i]['Molecule Name']):
                x = df.copy().iloc[i+1]
                df.iloc[i+1] = df.iloc[i]
                df.iloc[i] = x
    else:
        df = None
    return df

def compare_smiles_lengths(smiles1, smiles2, labels):
    # Display comparison of molecular weights
    def len_hist(strings,color='k',label='lengths'):
        plt.hist([len(x) for x in list(set(strings))],cumulative=False,histtype='step',color=color,
                normed=True,bins=np.linspace(0,50,50),label=label)
    len_hist(smiles1,color='b',label=labels[0])
    len_hist(smiles2,color='r',label=labels[1])
    plt.xlim(0,50)
    plt.xlabel('Length')
    plt.ylabel('Probability')
    plt.legend(loc=1)
    plt.title('SMILES string lengths')
    plt.figure()
    plt.show()

def compare_molecular_weights(X1, X2, labels):
    # Display comparison of molecular sizes (based on SMILES strings) for DREAM and GDB molecules
    def len_hist(X,color='k',label='lengths'):
        plt.hist(X['MW'].as_matrix(),cumulative=False,histtype='step',color=color,
             normed=True,bins=np.linspace(0,500,50),label=label)
    len_hist(X1,color='b',label=labels[0])
    len_hist(X2,color='r',label=labels[1])
    plt.xlim(0,500)
    plt.xlabel('Molecular Weight')
    plt.ylabel('Probability')
    plt.legend(loc=1)
    plt.title('Molecular Weights')
    plt.show()

def create_smiles_file(smiles_lists):
    # Create file to use with PubChem Identifier Exchange to get CIDs
    with open(os.path.join(DATA_HOME,'all_SMILES.txt'),'w') as f:
        for smiles_list in smiles_lists:
            for smiles in smiles_list:
                f.write(smiles+'\r')

def make_predictions(clfs,X,descriptors):
    # Compute descriptor rating predictions for other molecules
    Y_predicted = pd.DataFrame(index=X.index,columns=descriptors)
    for descriptor in descriptors:
        Y_predicted[descriptor] = clfs[descriptor].predict(X)
    return Y_predicted

def get_ranks(predictions):
    n_molecules = predictions.shape[0]
    ranks = np.empty((n_molecules,len(descriptors))) # -1 to remove CIDs
    for i,descriptor in enumerate(descriptors_short):
        p = predictions.sort_values(descriptor,ascending=False)
        ranks[:,i] = p['CID'].astype('int') # +1 to skip CID column
    pd.set_option('display.max_columns', 500)
    url = '<a href="https://pubchem.ncbi.nlm.nih.gov/compound/{0}">{0}</a>'
    ranks = pd.DataFrame(ranks,index=np.arange(1,n_molecules+1),columns=descriptors_short,dtype='int')
    for i,descriptor in enumerate(descriptors_short):
        ranks[descriptor] = ranks[descriptor].apply(lambda x: url.format(x))
    return ranks

def extra_gdb_stuff1():
    # Assumes that the all_CIDs.txt file has been converted to CIDs on PubChem Identifier Exchange
    path = os.path.join(DATA_HOME,'all_SMILES_to_CIDs.txt')
    CIDs_all = pd.read_csv(path,delimiter='\t',index_col=0,header=None,names=['CID']).fillna(0).astype('int')
    CIDs_gdb = CIDs_all.loc[smiles_gdb]
    predictions = CIDs_gdb.join(Y_gdb_predicted)

def extra_gdb_stuff2():
    pd.set_option('display.max_colwidth',100)
    pd.set_option('display.float_format',lambda x:'%.1f'%x)
    HTML("""<!--Some CSS-->
    <style>
    table.dataframe {
    font-size:70%;
    }
    </style>""")

def extra_gdb_stuff3():
    ranks = a.get_ranks()
    HTML(ranks.head(25).to_html(escape=False)) # Render HTML to get CID links to PubChem (ignore 0 values)

def extra_gdb_stuff4():
    predictions['CID'] = predictions['CID'].apply(lambda x: url.format(x))

def sort_by(descriptor):
    p = predictions.sort_values(descriptor,ascending=False).head(10)
    HTML(p.to_html(escape=False)) # Render HTML to get CID links to PubChem

'''
def get_CIDs(sets, target_dilution=None):
    CIDs = []; CID_dilutions = []
    for set_ in sets:
        CIDs += loading.get_CIDs(set_)
        CID_dilutions += loading.get_CID_dilutions(set_,target_dilution=target_dilution)
    print("Loaded CIDs for %d molecules from the DREAM challenge (%s)" % (len(CIDs),'+'.join(sets)))
    return CIDs,CID_dilutions

# Load CIDs and dilutions for the DREAM molecules
CIDs_train,CID_dilutions_train = get_CIDs(['training'])
CIDs_test,CID_dilutions_test = get_CIDs(['leaderboard'],target_dilution='gold')
CIDs,CID_dilutions = get_CIDs(['training','leaderboard','testset'])
'''
# Load SMILES strings for the DREAM molecules

#smiles_dream_train = list(CIDs_to_SMILES_dream['SMILES'][CIDs_train])
#smiles_dream_test = list(CIDs_to_SMILES_dream['SMILES'][CIDs_test])
#print("Converted DREAM CIDs to SMILES strings")
    
