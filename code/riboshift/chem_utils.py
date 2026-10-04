from __future__ import annotations

from functools import lru_cache

from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import Crippen, Descriptors, Lipinski
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.error")


@lru_cache(maxsize=50000)
def standardize_smiles(smiles: str) -> str:
    # This is intentionally conservative: if standardization fails, we return
    # an empty string and let the caller decide whether to drop the record.
    smiles = str(smiles or "").strip()
    if not smiles or smiles.lower() in {"nan", "none", "null"}:
        return ""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    try:
        mol = rdMolStandardize.FragmentParent(mol)
        uncharger = rdMolStandardize.Uncharger()
        mol = uncharger.uncharge(mol)
    except Exception:
        pass
    try:
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return ""


@lru_cache(maxsize=50000)
def smiles_to_mol(smiles: str):
    text = standardize_smiles(smiles)
    if not text:
        return None
    return Chem.MolFromSmiles(text)


@lru_cache(maxsize=50000)
def inchikey_of(smiles: str) -> str:
    mol = smiles_to_mol(smiles)
    if mol is None:
        return ""
    try:
        return Chem.MolToInchiKey(mol)
    except Exception:
        return ""


@lru_cache(maxsize=50000)
def scaffold_of(smiles: str) -> str:
    mol = smiles_to_mol(smiles)
    if mol is None:
        return ""
    try:
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
        if scaffold:
            return scaffold
        return f"ACYCLIC::{standardize_smiles(smiles)}"
    except Exception:
        return ""


def descriptor_dict(smiles: str) -> dict[str, float | int | str]:
    mol = smiles_to_mol(smiles)
    if mol is None:
        return {
            "canonical_smiles": "",
            "inchikey": "",
            "murcko_scaffold": "",
            "mol_wt": 0.0,
            "logp": 0.0,
            "hbd": 0,
            "hba": 0,
            "tpsa": 0.0,
            "ring_count": 0,
        }
    canonical = standardize_smiles(smiles)
    return {
        "canonical_smiles": canonical,
        "inchikey": inchikey_of(canonical),
        "murcko_scaffold": scaffold_of(canonical),
        "mol_wt": float(Descriptors.MolWt(mol)),
        "logp": float(Crippen.MolLogP(mol)),
        "hbd": int(Lipinski.NumHDonors(mol)),
        "hba": int(Lipinski.NumHAcceptors(mol)),
        "tpsa": float(rdMolDescriptors.CalcTPSA(mol)),
        "ring_count": int(rdMolDescriptors.CalcNumRings(mol)),
    }
