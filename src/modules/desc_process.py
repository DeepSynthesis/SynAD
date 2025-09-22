import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from tqdm import tqdm

desc_path = Path(__file__).parent / Path("../../descriptor")
ULD_selector = {"QPOC": True, "solv_param": True, "SPOC": True, "other_info": True}


def remove_desc_nonexist_mol(data_df, target_idx, desc_type):
    mask = data_df[desc_type].isin(target_idx)
    if not all(mask.tolist()):
        logger.warning(f"{data_df[~mask][desc_type].drop_duplicates().to_list()} is not in {desc_type} qmdesc!")
        return data_df[mask].reset_index(drop=True)
    return data_df


def map_descriptors(smiles, feature_dict, desc_type=None):
    """Mapping descriptors from feature dictionary"""
    current_columns = [f"{desc_type}_{f}" for f in feature_dict.columns]
    desc = pd.DataFrame(np.zeros((len(smiles), feature_dict.shape[1])), index=smiles, columns=current_columns).astype(float)
    for idx in desc.index.unique():
        mask = desc.index == idx
        desc.loc[mask] = feature_dict.loc[idx].values  # broadcast

    desc.reset_index(inplace=True, drop=True)
    return desc


def read_all_dataframes(file_path, key="all", index_col="smiles"):
    """Read dataframes from hdf5 file."""
    data = {}
    with pd.HDFStore(file_path, "r") as store:
        for key in store.keys():
            data[key.replace("/", "")] = pd.read_hdf(file_path, key=key, index_col=index_col, dtype=object)
    for key in data.keys():
        data[key].index = [x.replace("\\\\", "\\") if ("\\\\") in x else x for x in data[key].index]
    if key != "all":
        data_temp = data[key]
        return data[key].astype(float)
    else:
        return data


def get_rdkit_descriptor(smiles_list):
    smiles_list = [smiles for smiles in smiles_list if smiles == smiles]
    smiles_list = [smiles for smiles in smiles_list if smiles != "blank_cell" and smiles != "neat"]
    smiles_list = pd.Series(smiles_list).drop_duplicates().tolist()
    try:
        mol_list = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]
    except:
        raise Exception(f"Some SMILES is not validate SMILES!")
    logger.info("generating descriptors...")
    descriptor_names = [desc[0] for desc in Descriptors.descList]
    calculator = MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names)  # calculate each molecule descriptors
    descriptors = [calculator.CalcDescriptors(mol) for mol in tqdm(mol_list)]
    df = pd.DataFrame(descriptors, columns=descriptor_names, index=smiles_list)
    df.loc["blank_cell"] = np.zeros(len(df.columns))
    df.loc["neat"] = np.zeros(len(df.columns))
    return df


class ReactionDesc:
    """Reaction descriptor process."""

    def __init__(
        self,
        data_df: pd.DataFrame,
        reagent_columns: list,
        data_columns: list,
        dataset_type: str = "ULD",
    ):
        self.dataset_type = dataset_type
        self.data_df = data_df
        self.reagent_columns = reagent_columns
        self.data_columns = data_columns
        assert all(r in data_df.columns.tolist() for r in reagent_columns), f"some reagents are not in data_df.columns"
        assert all(d in data_df.columns.tolist() for d in data_columns), f"some data are not in data_df.columns"

    def generate_spoc_descriptor(self, spoc_desc_type="RDKitDescriptors", recalc=False):
        desc_file_path = desc_path / Path(f"{self.dataset_type}/SPOC.csv")
        if desc_file_path.exists() and not recalc:
            return

        desc_file_path.parent.mkdir(parents=True, exist_ok=True)
        all_smiles_list = self.data_df[self.reagent_columns].values.flatten().tolist()

        if spoc_desc_type == "RDKitDescriptors":
            desc_df = get_rdkit_descriptor(all_smiles_list)
        else:
            raise NotImplementedError(f"{spoc_desc_type} is not implemented!")

        desc_df.fillna(0.0, inplace=True)
        desc_df.to_csv(desc_path / Path(f"{self.dataset_type}/SPOC.csv"))

    def load_reaction_desc(self, selector=ULD_selector):
        if self.dataset_type == "ULD":
            self.desc_dict = load_ULD_reaction_desc(selector)
        else:
            self.desc_dict = load_reaction_desc(desc_df_path=desc_path / Path(f"{self.dataset_type}/SPOC.csv"))

    def generate_descriptor_matrix(self, target_prop=None, skip_no_desc_data=True, verbose=False):
        self.drop_no_desc_data() if skip_no_desc_data else None

        X = []
        wrap_func = tqdm if verbose else lambda x: x
        for reagent in wrap_func(self.reagent_columns):
            X.append(map_descriptors(self.data_df[reagent], self.desc_dict["SPOC"].copy(), f"{reagent}_SPOC"))

        for k in wrap_func(self.desc_dict.keys()):
            if "QM" in k or "param" in k:
                X.append(map_descriptors(self.data_df[k.split("_")[0]], self.desc_dict[k].copy(), k))

        for data in self.data_columns:
            X.append(self.data_df[[data]])

        X = pd.concat(X, axis=1)
        X = X.loc[:, (X != X.iloc[0]).any()]
        X = X.astype(float)
        assert not np.isnan(X.values).any(), f"there is nan value in X!, localtion is {np.argwhere(np.isnan(X.values))}"

        if target_prop == None:
            logger.warning("Not generate y data!")
            y = None
        else:
            y = self.data_df[target_prop].astype(float)

        logger.info(f"X shape is {X.shape}.")
        return X, y

    def drop_no_desc_data(self):
        judgement_reagents = [r.split("_")[0] for r in self.desc_dict.keys()]
        for r, key in zip(judgement_reagents, self.desc_dict.keys()):
            if "QM" not in key and "param" not in key:
                continue
            self.data_df = remove_desc_nonexist_mol(self.data_df, self.desc_dict[key].index, r)

        logger.info(f"after qmdesc clean, rest data size is {len(self.data_df)}.")
        self.data_df.reset_index(drop=True, inplace=True)
        return self.data_df


def load_reaction_desc(desc_df_path):
    desc_df = {}
    desc_df["SPOC"] = pd.read_csv(desc_df_path, index_col=0).astype(float)
    return desc_df


def load_ULD_reaction_desc(selector=ULD_selector):
    assert type(selector) == dict, "selector must be a dictionary"
    assert selector.keys() == ULD_selector.keys(), "selector must be a subset of ULD_selector"
    desc_dict = {}
    if selector["QPOC"]:
        for qm_type in ["ligand1", "reactant", "product", "add1", "add2"]:
            if qm_type == "reactant":
                desc_dict[f"{qm_type}1_QM"] = read_all_dataframes(desc_path / Path(f"ULD/QMdesc_{qm_type}.h5"), key="global")
                desc_dict[f"{qm_type}2_QM"] = read_all_dataframes(desc_path / Path(f"ULD/QMdesc_{qm_type}.h5"), key="global")
            else:
                desc_dict[f"{qm_type}_QM"] = read_all_dataframes(desc_path / Path(f"ULD/QMdesc_{qm_type}.h5"), key="global")

    if selector["solv_param"]:
        solv_param_file = desc_path / Path(f"ULD/solvent_param.csv")
        desc_dict["solv_param"] = pd.read_csv(solv_param_file, index_col="smiles").astype(float)

    if selector["SPOC"]:
        spoc_file = desc_path / Path(f"ULD/SPOC.csv")
        desc_dict["SPOC"] = pd.read_csv(spoc_file, index_col="smiles").astype(float)

    return desc_dict


def get_fingerprint(mol, fp_type):
    if fp_type == "avalon":
        from rdkit.Avalon import pyAvalonTools

        return pyAvalonTools.GetAvalonFP(mol, nBits=512)
    elif fp_type == "atom_pairs":
        return Chem.rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(mol, nBits=512)
    elif fp_type == "topological_torsions":
        return Chem.rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect(mol, nBits=512)
    elif fp_type == "maccs":
        return AllChem.GetMACCSKeysFingerprint(mol)
    elif fp_type == "rdkit":
        return Chem.RDKFingerprint(mol)
    elif fp_type == "morgan":  # ECFP
        return AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=512)
    elif fp_type == "mordred":
        from mordred import Calculator, descriptors

        calc = Calculator(descriptors, ignore_3D=True)
        df = calc.pandas([mol]).iloc[0]
        arr = np.array([float(x) if str(x) != "nan" else 0.0 for x in df.values])
        return arr
    else:
        raise ValueError(f"Unsupported fingerprint type: {fp_type}")
