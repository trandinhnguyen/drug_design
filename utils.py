from drugex.logs import logger

import numpy as np
import pandas as pd
import logging
import os
import copy

from rdkit import Chem
from rdkit.Chem import RDConfig
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdDepictor, rdMolDraw2D

from qsprpred.data.sources.papyrus import Papyrus
from qsprpred.data.data import QSPRDataset
from qsprpred.models.tasks import ModelTasks

opts = Draw.DrawingOptions()
Draw.SetComicMode(opts)


def initLogger(filename, dir_name="data/logs/"):
    """
    Initializes a logging directory if necessary and places all DrugEx outputs in the specified file.

    Args:
        filename: name of the log file for DrugEx outputs
        dir_name: directory where the log file will be placed
    """

    filename = os.path.join(dir_name, filename)

    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    if os.path.exists(filename):
        os.remove(filename)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s", "%m-%d-%Y %H:%M:%S"
    )
    fh = logging.FileHandler(filename)
    fh.setFormatter(formatter)

    logger.addHandler(fh)


# different grid visualizations
standard_grid = Chem.Draw.MolsToGridImage


def interactive_grid(mols, *args, molsPerRow=5, **kwargs):
    """
    install mols2grid with pip to use
    """

    import mols2grid

    return mols2grid.display(mols, *args, n_cols=molsPerRow, **kwargs)


# show molecules as grid
make_grid = interactive_grid  # change this to 'standard_grid' if you do not have the mols2grid package


def smilesToGrid(smiles, *args, molsPerRow=5, **kwargs):
    mols = []
    for smile in smiles:
        try:
            m = Chem.MolFromSmiles(smile)
            if m:
                AllChem.Compute2DCoords(m)
                mols.append(m)
            else:
                raise Exception(f"Molecule empty for SMILES: {smile}")
        except Exception as exp:
            pass

    return make_grid(mols, *args, molsPerRow=molsPerRow, **kwargs)


def convert_smiles(smiles: str) -> str:
    """Convert to canonical SMILES"""
    return Chem.MolToSmiles(Chem.MolFromSmiles(smiles))


def A1R(data_dir="data/datasets", quality="high", th=[6.5]):
    """
    A classification dataset that contains activity data on the adenosine A1 receptor loaded from the Papyrus database
    using the built-in Papyrus wrapper.

    Params:
        data_dir: data location
        quality: choose minimum quality from {"high", "medium", "low"}
        th: threshold of classification task
    Returns:
        a `QSPRDataset` instance with the loaded data
    """

    acc_keys = ["P30542"]  # https://www.uniprot.org/uniprotkb/P30542/entry
    dataset_name = "A1R_LIGANDS"
    papyrus_version = "05.6"  # Papyrus database version

    papyrus = Papyrus(data_dir=data_dir, stereo=False, version=papyrus_version)

    dataset = papyrus.getData(acc_keys, quality, name=dataset_name, use_existing=True)

    print(f"Number of samples loaded: {len(dataset.getDF())}")
    return QSPRDataset.fromMolTable(
        dataset, "pchembl_value_Median", task=ModelTasks.CLASSIFICATION, th=th
    )


def get_dataset(
    dataset_name, acc_keys, data_dir="data/datasets", quality="high", th=[6.5]
):
    """
    A classification dataset that contains activity data on the *acc_keys param loaded from the Papyrus database
    using the built-in Papyrus wrapper.

    Params:
        dataset_name: dataset's name
        data_dir: data location
        quality: choose minimum quality from {"high", "medium", "low"}
        th: threshold of classification task
        *acc_keys: list of accession key
    Returns:
        a `QSPRDataset` instance with the loaded data
    """
    papyrus_version = "05.6"  # Papyrus database version

    papyrus = Papyrus(data_dir=data_dir, stereo=False, version=papyrus_version)

    dataset = papyrus.getData(acc_keys, quality, name=dataset_name, use_existing=True)

    print(f"Number of samples loaded: {len(dataset.getDF())}")
    return QSPRDataset.fromMolTable(
        dataset, "pchembl_value_Median", task=ModelTasks.CLASSIFICATION, th=th
    )


def sort_by_prediction(file_name):
    path = os.path.join("data", "result", file_name + ".csv")
    new_path = os.path.join("data", "result", "sorted_" + file_name + ".csv")

    df = pd.read_csv(path)
    df = df.sort_values(by=["Prediction"], ascending=False)
    df.to_csv(new_path)
    return df


def merge_AR_papyrus_cortellis():
    DATASETS_PATH = "data/datasets"

    # read AR_LIGANDS of Papyrus dataset
    df_papyrus = pd.read_csv(
        f"{DATASETS_PATH}/AR_LIGANDS.tsv",
        sep="\t",
        header=0,
        na_values=("NA", "nan", "NaN"),
    )

    # read AR ligands of cortellis dataset
    df_cortellis = pd.read_csv(
        f"{DATASETS_PATH}/cortellis_P30542_same.csv",
        header=0,
        na_values=("NA", "nan", "NaN"),
    )

    # remove compound whose target is not AR family from df_cortellis
    idx = df_cortellis[
        (df_cortellis["Target"] != "P0DMS8")
        & (df_cortellis["Target"] != "P29274")
        & (df_cortellis["Target"] != "P29275")
        & (df_cortellis["Target"] != "P30542")
    ].index
    df_cortellis.drop(idx, inplace=True)

    # keep SMILES and pchembl_value_Median columns
    df_papyrus = pd.concat(
        [df_papyrus["SMILES"], df_papyrus["pchembl_value_Median"]], axis=1
    )
    df_cortellis = pd.concat(
        [df_cortellis["Drug"], df_cortellis["regression_label"]], axis=1
    )
    df_cortellis.rename(
        columns={"regression_label": "pchembl_value_Median", "Drug": "SMILES"},
        inplace=True,
    )

    # merge 2 datasets
    df = pd.concat([df_papyrus, df_cortellis])

    # remove duplicate rows
    df.drop_duplicates(inplace=True)

    # save merged dataset
    df.to_csv(f"{DATASETS_PATH}/AR_papyrus_cortellis.csv")


if __name__ == "__main__":
    # A1R()
    # get_dataset("AR_LIGANDS", ["P0DMS8", "P29274", "P29275", "P30542"])
    merge_AR_papyrus_cortellis()
