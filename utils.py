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


if __name__ == "__main__":
    A1R()
