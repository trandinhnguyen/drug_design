from drugex.training.scorers.qsprpred import QSPRPredScorer
from rdkit import Chem
import os


def get_scorer(model):
    return QSPRPredScorer(model)


def get_score(model, smiles_list):
    return get_scorer(model).getScores(smiles_list)


def get_DDB28K_scores(model, file_name):
    with Chem.MultithreadedSDMolSupplier("data/datasets/DDB28K.sdf") as sdSupl:
        smiles_list = [Chem.MolToSmiles(mol) for mol in sdSupl if mol is not None]

    pred = get_score(model, smiles_list)
    path = os.path.join("data/result", file_name + ".csv")
    with open(path, "w") as f:
        f.write("SMILES,Prediction\n")
        for smile, p in zip(smiles_list, pred):
            f.write(f"{smile},{p}\n")
            
    return pred