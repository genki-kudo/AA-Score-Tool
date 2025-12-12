from interaction_components.plinteraction import get_interactions
import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem
from openbabel import pybel

from utils.hbonds import calc_hbond_strength
from utils.hydrophobic import calc_hydrophobic
from utils.vdw import calc_vdw
from utils.electrostatic import Electrostatic
import os
import sys
import argparse

import fast_elec

residue_names = [
    "HIS",
    "ASP",
    "ARG",
    "PHE",
    "ALA",
    "CYS",
    "GLY",
    "GLN",
    "GLU",
    "LYS",
    "LEU",
    "MET",
    "ASN",
    "SER",
    "TYR",
    "THR",
    "ILE",
    "TRP",
    "PRO",
    "VAL"]


def is_sidechain(atom):
    res = atom.GetPDBResidueInfo()
    atom_name = res.GetName().strip(" ")
    if atom_name in ("C", "CA", "N", "O", "H"):
        return False
    else:
        return True


def create_dict():
    interaction_dict = {}
    for name in residue_names:
        interaction_dict.update({name + "_side": 0})
        interaction_dict.update({name + "_main": 0})
    return interaction_dict


def calc_hb(hbonds, hb_dict):
    for hb in hbonds:
        restype = hb.restype
        sidechain = hb.sidechain
        energy = calc_hbond_strength(hb)
        if restype == "HIN":
            restype = "HIS"
        if restype == "ACE":
            continue
        if sidechain:
            key = restype + "_side"
        else:
            key = restype + "_main"
        hb_dict[key] += energy
    return


def calc_hbonds_descriptor(interactions):
    hb_dict = create_dict()
    calc_hb(interactions.all_hbonds_ldon, hb_dict)
    calc_hb(interactions.all_hbonds_pdon, hb_dict)
    return hb_dict


def calc_hydrophybic_descriptor(interactions):
    hc_dict = create_dict()
    for hc in interactions.hydrophobic_contacts:
        restype = hc.restype
        sidechain = hc.sidechain
        energy = calc_hydrophobic(hc)
        if restype[:2] == "HI" and restype not in residue_names:
            restype = "HIS"
        if restype == "ACE":
            continue
        if sidechain:
            key = restype + "_side"
        else:
            key = restype + "_main"
        hc_dict[key] += energy
    return hc_dict


def calc_vdw_descriptor(result, mol_lig):
    prot = result.prot
    residues = prot.residues
    vdw_dict = create_dict()
    for res in residues:
        main_vdw, side_vdw = calc_vdw(res, mol_lig)
        restype = res.residue_name
        if restype[:2] == "HI" and restype not in residue_names:
            restype = "HIS"
        if restype not in residue_names:
            continue
        vdw_dict[restype + "_side"] += main_vdw
        vdw_dict[restype + "_main"] += side_vdw
    return vdw_dict


def calc_ele_descriptor(result, mol_lig, mol_prot):
    """元の Python 実装による電荷相互作用"""
    prot = result.prot
    residues = prot.residues
    ele_same_dict = create_dict()
    ele_opposite_dict = create_dict()
    for res in residues:
        ele = Electrostatic(res, mol_lig, mol_prot)
        restype = res.residue_name
        if restype[:2] == "HI" and restype not in residue_names:
            restype = "HIS"
        if restype not in residue_names:
            continue
        ele_same_dict[restype + "_side"] += ele.side_ele_same
        ele_same_dict[restype + "_main"] += ele.main_ele_same

        ele_opposite_dict[restype + "_side"] += ele.side_ele_opposite
        ele_opposite_dict[restype + "_main"] += ele.main_ele_opposite
    return ele_same_dict, ele_opposite_dict

def calc_desolvation_descriptor(result, mol_prot, mol_lig):
    dehyd = Dehydration(mol_prot, mol_lig)
    prot = result.prot

    dehyd_energy = 0
    origin = "protein"
    for at in mol_prot.GetAtoms():
        dehyd_energy += dehyd.calc_atom_dehyd(at, origin)
    origin = "ligand"
    for at in mol_lig.GetAtoms():
        dehyd_energy += dehyd.calc_atom_dehyd(at, origin)
    return dehyd_energy


def calc_metal_complexes(metal):
    dist = metal.distance
    if dist < 2.0:
        return -1.0
    elif 2.0 <= dist < 3.0:
        return -3.0 + dist
    else:
        return 0.0

# C++ 用の関数
def get_coords_charges(mol):
    """RDKit Mol から
       - 座標: RDKit の Conformer
       - 電荷: Pybel (OpenBabel) の partialcharge
    を使って配列を作る。

    Electrostatic クラスの get_partial_charge と同じ発想：
    一度 PDB に変換してから Pybel で partialcharge を読む。
    """

    # 座標は RDKit から
    conf = mol.GetConformer()
    N = mol.GetNumAtoms()

    coords = np.empty((N, 3), dtype=np.float32)
    for i, at in enumerate(mol.GetAtoms()):
        p = conf.GetAtomPosition(i)
        coords[i] = (p.x, p.y, p.z)

    # 電荷は Pybel から取る
    pdb_block = Chem.MolToPDBBlock(mol)
    pmol = pybel.readstring("pdb", pdb_block)

    charges = np.zeros((N,), dtype=np.float32)
    for at in mol.GetAtoms():
        idx = at.GetIdx()          
        ob_atom = pmol.atoms[idx]  

        q = ob_atom.partialcharge

        # H は実質無視：電荷0に潰す
        if at.GetAtomicNum() == 1:
            charges[idx] = 0.0
        else:
            charges[idx] = 0.0 if (q != q) else float(q)  # NaN ガード

    return coords, charges


def get_residue_ids(mol_prot):
    """mol_prot の各原子に「0..原子数-1 の残基ID」を振る。

    戻り値:
      res_ids: shape (Np,), int32
               i 番目の原子が属する残基ID (0..Nres-1)。情報なしは -1。
      id_to_key: list of (chain, resno, icode)
                 残基ID → (チェーンID, 残基番号, 挿入コード) の対応表
    """
    Np = mol_prot.GetNumAtoms()
    res_ids = np.full(Np, -1, dtype=np.int32)

    key_to_id = {}
    id_to_key = []

    next_id = 0
    for i, atom in enumerate(mol_prot.GetAtoms()):
        info = atom.GetPDBResidueInfo()
        if info is None:
            continue

        chain = (info.GetChainId() or "").strip() or "_"
        resno = info.GetResidueNumber()
        icode = (info.GetInsertionCode() or "").strip() or "-"

        key = (chain, resno, icode)

        if key not in key_to_id:
            key_to_id[key] = next_id
            id_to_key.append(key)
            next_id += 1

        res_ids[i] = key_to_id[key]

    return res_ids, id_to_key


def get_sidechain_mask(mol_prot):
    """各原子が sidechain(1) か mainchain(0) かのフラグ配列を返す。"""
    Np = mol_prot.GetNumAtoms()
    flags = np.zeros(Np, dtype=np.int32)
    for i, atom in enumerate(mol_prot.GetAtoms()):
        if is_sidechain(atom):
            flags[i] = 1  # sidechain
    return flags


# C++ fast_elec をその場で使う版
def cpp_calc_ele_descriptor(result, mol_lig, mol_prot, r_cut=999.0):
    """C++のfast_elec で side/main、same/opp を計算し、
    calc_ele_descriptor と同じ形式の dict を返す。

    ※protein 側も ligand 側もこの中で毎回計算する版。
      単発計算や検証用。多数の ligand なら後述の
      precompute_protein_for_cpp + calc_score_with_precomputed_prot を推奨。
    """

    # 1) protein / ligand 両方の座標・電荷
    coords_p, q_p = get_coords_charges(mol_prot)
    coords_l, q_l = get_coords_charges(mol_lig)

    # 2) 残基ID ＆ sidechainフラグ
    res_ids, _ = get_residue_ids(mol_prot)
    side_flags = get_sidechain_mask(mol_prot)

    coords_p = np.asarray(coords_p, dtype=np.float32, order="C")
    coords_l = np.asarray(coords_l, dtype=np.float32, order="C")
    q_p = np.asarray(q_p, dtype=np.float32, order="C")
    q_l = np.asarray(q_l, dtype=np.float32, order="C")
    res_ids = np.asarray(res_ids, dtype=np.int32, order="C")
    side_flags = np.asarray(side_flags, dtype=np.int32, order="C")

    # 3) C++拡張呼び出し
    (
        total,
        per_res_side_same,
        per_res_side_opp,
        per_res_main_same,
        per_res_main_opp,
    ) = fast_elec.electrostatic_sum(
        coords_p,
        q_p,
        res_ids,
        side_flags,
        coords_l,
        q_l,
        float(r_cut),
        1.0,
    )

    per_res_side_same = np.asarray(per_res_side_same, dtype=np.float32)
    per_res_side_opp  = np.asarray(per_res_side_opp,  dtype=np.float32)
    per_res_main_same = np.asarray(per_res_main_same, dtype=np.float32)
    per_res_main_opp  = np.asarray(per_res_main_opp,  dtype=np.float32)

    # 4) dict 化（元の calc_ele_descriptor と同じ形式）
    ele_same_dict     = create_dict()
    ele_opposite_dict = create_dict()

    residues = result.prot.residues
    for ridx, res in enumerate(residues):
        restype = res.residue_name
        if restype[:2] == "HI" and restype not in residue_names:
            restype = "HIS"
        if restype not in residue_names:
            continue

        if ridx >= len(per_res_side_same):
            continue

        side_same = float(per_res_side_same[ridx])
        side_opp  = float(per_res_side_opp[ridx])
        main_same = float(per_res_main_same[ridx])
        main_opp  = float(per_res_main_opp[ridx])

        ele_same_dict[restype + "_side"] += side_same
        ele_same_dict[restype + "_main"] += main_same

        ele_opposite_dict[restype + "_side"] += side_opp
        ele_opposite_dict[restype + "_main"] += main_opp

    return ele_same_dict, ele_opposite_dict


# protein 側を事前計算する
def precompute_protein_for_cpp(mol_prot):
    """C++ fast_elec 用に、protein 側の情報を一度だけ前計算しておく。

    戻り値:
      coords_p   : (Np, 3) float32 C-contiguous
      q_p        : (Np,)   float32 C-contiguous
      res_ids    : (Np,)   int32   C-contiguous
      side_flags : (Np,)   int32   (1=sidechain, 0=main)
    """
    coords_p, q_p = get_coords_charges(mol_prot)

    res_ids, _ = get_residue_ids(mol_prot)
    side_flags = get_sidechain_mask(mol_prot)

    coords_p   = np.asarray(coords_p,   dtype=np.float32, order="C")
    q_p        = np.asarray(q_p,        dtype=np.float32, order="C")
    res_ids    = np.asarray(res_ids,    dtype=np.int32,   order="C")
    side_flags = np.asarray(side_flags, dtype=np.int32,   order="C")

    return coords_p, q_p, res_ids, side_flags


def calc_score_with_precomputed_prot(
    mol_lig,
    mol_prot,
    clf,
    coords_p,
    q_p,
    res_ids,
    side_flags,
    r_cut=999.0,
):
    """protein 側を事前計算済みの情報で差し替えた calc_score。

    - mol_prot: RDKit Mol（Pocket付き）
    - mol_lig : RDKit Mol（sdf/mol2などから読んだもの）
    - clf     : すでに load 済みの Model
    """

    # 1) 相互作用を取る（hb/vdw/picationなど用）
    result = get_interactions(mol_prot, mol_lig)
    interactions = result.interactions

    hb_dict = calc_hbonds_descriptor(interactions)
    hc_dict = calc_hydrophybic_descriptor(interactions)
    vdw_dict = calc_vdw_descriptor(result, mol_lig)

    # 2) ligand 側だけ座標・電荷を作る
    coords_l, q_l = get_coords_charges(mol_lig)
    coords_l = np.asarray(coords_l, dtype=np.float32, order="C")
    q_l      = np.asarray(q_l,      dtype=np.float32, order="C")

    # 3) C++ で side/main × same/opp を残基ごとに集計
    (
        total,
        per_res_side_same,
        per_res_side_opp,
        per_res_main_same,
        per_res_main_opp,
    ) = fast_elec.electrostatic_sum(
        coords_p,
        q_p,
        res_ids,
        side_flags,
        coords_l,
        q_l,
        float(r_cut),
        1.0,
    )

    per_res_side_same = np.asarray(per_res_side_same, dtype=np.float32)
    per_res_side_opp  = np.asarray(per_res_side_opp,  dtype=np.float32)
    per_res_main_same = np.asarray(per_res_main_same, dtype=np.float32)
    per_res_main_opp  = np.asarray(per_res_main_opp,  dtype=np.float32)

    # 4) dict 化
    ele_same_dict     = create_dict()
    ele_opposite_dict = create_dict()

    residues = result.prot.residues
    for ridx, res in enumerate(residues):
        restype = res.residue_name
        if restype[:2] == "HI" and restype not in residue_names:
            restype = "HIS"
        if restype not in residue_names:
            continue

        if ridx >= len(per_res_side_same):
            continue

        side_same = float(per_res_side_same[ridx])
        side_opp  = float(per_res_side_opp[ridx])
        main_same = float(per_res_main_same[ridx])
        main_opp  = float(per_res_main_opp[ridx])

        ele_same_dict[restype + "_side"] += side_same
        ele_same_dict[restype + "_main"] += main_same

        ele_opposite_dict[restype + "_side"] += side_opp
        ele_opposite_dict[restype + "_main"] += main_opp

    # 5) 残りの descriptor は元の calc_score と同じ
    metal_ligand = calc_metal_descriptor(interactions)
    tpp_energy, ppp_energy = calc_pistacking_descriptor(interactions)
    ppc_energy, pic_dict = calc_pication_descriptor(interactions)
    rotat = Chem.rdMolDescriptors.CalcNumRotatableBonds(mol_lig)

    descriptors = merge_descriptors(
        hb_dict,
        hc_dict,
        vdw_dict,
        ele_same_dict,
        ele_opposite_dict,
        pic_dict,
        metal_ligand,
        tpp_energy,
        ppp_energy,
        ppc_energy,
        rotat,
    )
    score = clf.predict(descriptors)
    return score


def calc_desolvation_descriptor(result, mol_prot, mol_lig):
    dehyd = Dehydration(mol_prot, mol_lig)
    dehyd_energy = 0.0
    origin = "protein"
    for at in mol_prot.GetAtoms():
        dehyd_energy += dehyd.calc_atom_dehyd(at, origin)
    origin = "ligand"
    for at in mol_lig.GetAtoms():
        dehyd_energy += dehyd.calc_atom_dehyd(at, origin)
    return dehyd_energy


def calc_metal_complexes(metal):
    dist = metal.distance
    if dist < 2.0:
        return -1.0
    elif 2.0 <= dist < 3.0:
        return -3.0 + dist
    else:
        return 0.0


def calc_metal_descriptor(interactions):
    ml_energy = 0
    for ml in interactions.metal_complexes:
        if ml.target.location != "ligand":
            continue
        energy = calc_metal_complexes(ml)
        ml_energy += energy
    return ml_energy


def calc_pistacking_descriptor(interactions):
    T_pistacking_energy, P_pistacking_energy = 0, 0
    for pis in interactions.pistacking:
        if pis.type == "T":
            T_pistacking_energy += -1
        else:
            P_pistacking_energy += -1
    return T_pistacking_energy, P_pistacking_energy


def calc_pication_laro(interactions):
    pic_dict = create_dict()
    for pic in interactions.pication_laro:
        restype = pic.restype
        sidechain = is_sidechain(pic.charge.atoms[0])
        energy = -1
        if restype[:2] == "HI" and restype not in residue_names:
            restype = "HIS"
        if restype == "ACE":
            continue
        if sidechain:
            key = restype + "_side"
        else:
            key = restype + "_main"
        pic_dict[key] += energy
    return pic_dict


def calc_pication_descriptor(interactions):
    paro_pication_energy, laro_pication_energy = 0, 0
    for pic in interactions.pication_paro:
        paro_pication_energy += -1
    pic_dict = calc_pication_laro(interactions)
    return paro_pication_energy, pic_dict


class Model:
    def __init__(self, arr):
        self.arr = arr

    def predict(self, data):
        data = np.array(data)
        return np.sum(self.arr * data)-0.999


def load_model():
    param = np.load("models/model-final.npy")
    clf = Model(param)
    return clf


def merge_descriptors(
        hb_dict,
        hc_dict,
        vdw_dict,
        ele_same_dict,
        ele_opposite_dict,
        pic_dict,
        metal_ligand,
        tpp_energy,
        ppp_energy,
        ppc_energy,
        rotat):
    line = []
    descriptors = [hb_dict, vdw_dict, ele_same_dict, ele_opposite_dict]
    for des in descriptors:
        for v in des.values():
            line.append(v)
    line.append(sum(hc_dict.values()))
    line.append(sum(pic_dict.values()))
    line.append(metal_ligand)
    line.append(tpp_energy)
    line.append(ppp_energy)
    line.append(ppc_energy)
    line.append(rotat)
    descriptors = np.array(line)
    return descriptors


def calc_score(mol_lig, mol_prot, clf):
    result = get_interactions(mol_prot, mol_lig)
    interactions = result.interactions

    hb_dict = calc_hbonds_descriptor(interactions)
    hc_dict = calc_hydrophybic_descriptor(interactions)
    vdw_dict = calc_vdw_descriptor(result, mol_lig)

    ## オリジナル版
    # ele_same_dict, ele_opposite_dict = calc_ele_descriptor(
    #     result, mol_lig, mol_prot)

    # CPP置き換え版
    ele_same_dict, ele_opposite_dict = cpp_calc_ele_descriptor(
        result, mol_lig, mol_prot
    )

    metal_ligand = calc_metal_descriptor(interactions)
    tpp_energy, ppp_energy = calc_pistacking_descriptor(interactions)
    ppc_energy, pic_dict = calc_pication_descriptor(interactions)
    rotat = Chem.rdMolDescriptors.CalcNumRotatableBonds(mol_lig)

    descriptors = merge_descriptors(
        hb_dict,
        hc_dict,
        vdw_dict,
        ele_same_dict,
        ele_opposite_dict,
        pic_dict,
        metal_ligand,
        tpp_energy,
        ppp_energy,
        ppc_energy,
        rotat)
    score = clf.predict(descriptors)
    return score

def get_format(ligand_file):
    file_format = os.path.basename( ligand_file ).split(".")[1]
    return file_format

# def calc_batch(mol_prot, mol_ligs, output_file, clf):
#     for mol_lig in mol_ligs:
#         name = mol_lig.GetProp("_Name")
#         score = calc_score(mol_lig, mol_prot, clf)

#         if output_file:
#             with open(output_file, "a") as f:
#                 f.write(name + "\t" + str(score) + "\n")
#         else:
#             print( name, score )
#     return

def calc_batch(mol_prot, mol_ligs, output_file, clf):
    # タンパク質を先に処理してしまう
    coords_p, q_p, res_ids, side_flags = precompute_protein_for_cpp(mol_prot)

    for mol_lig in mol_ligs:
        if mol_lig is None:
            continue
        name = mol_lig.GetProp("_Name")
        score = calc_score_with_precomputed_prot(
            mol_lig,
            mol_prot,
            clf,
            coords_p,
            q_p,
            res_ids,
            side_flags,
        )
        if output_file:
            with open(output_file, "a") as f:
                f.write(name + "\t" + str(score) + "\n")
        else:
            print( name, score )
    return

def calc_single(mol_prot, mol_lig, output_file, clf):
    name = mol_lig.GetProp("_Name")
    score = calc_score(mol_lig, mol_prot, clf)

    if output_file:
        with open(output_file, "a") as f:
            f.write(name + "\t" + str(score) + "\n")
    else:
        print( name, score )
    return

def calc_single(mol_prot, mol_lig, output_file, clf):
    if not mol_lig:
        raise RuntimeError("RDKit parse the file error")
    name = mol_lig.GetProp("_Name")
    score = calc_score(mol_lig, mol_prot, clf)

    if output_file:
        with open(output_file, "a") as f:
            f.write(name + "\t" + str(score) + "\n")
    else:
        print( name, score )
    return

def predict_dG(mol_prot, mol_lig, output_file=None):
    clf = load_model()
    
    name = mol_lig.GetProp("_Name")
    score = calc_score(mol_lig, mol_prot, clf)

    if output_file:
        with open(output_file, "a") as f:
            f.write(name + "\t" + str(score) + "\n")
    else:
        return name, score

def func():
    parser = argparse.ArgumentParser(description='parse AA Score prediction parameters')
    parser.add_argument('--Rec', type=str, help='the file of binding pocket, only support PDB format')
    parser.add_argument('--Lig', type=str, help='the file of ligands, support mol2, mol, sdf, PDB')
    parser.add_argument('--Out', type=str, help='the output file for recording scores', default=None)
    args = parser.parse_args()
    protein_file = args.Rec
    ligand_file = args.Lig
    output_file = args.Out
    
    clf = load_model()
    mol_prot = Chem.MolFromPDBFile(protein_file, removeHs=False)
    lig_format = get_format(ligand_file)
    if lig_format not in ["sdf", "mol2", "mol", "pdb"]:
        raise RuntimeError("ligand format {} is not supported".format(lig_format))

    if lig_format == "sdf":
        mol_ligs = Chem.SDMolSupplier(ligand_file, removeHs=False)
        calc_batch(mol_prot, mol_ligs, output_file, clf)
    elif lig_format == "mol2":
        mol_lig = Chem.MolFromMol2File(ligand_file, removeHs=False)
        calc_single(mol_prot, mol_lig, output_file, clf)
    elif lig_format == "mol":
        mol_lig = Chem.MolFromMolFile(ligand_file, removeHs=False)
        calc_single(mol_prot, mol_lig, output_file, clf)
    elif lig_format == "pdb":
        mol_lig = Chem.MolFromPDBFile(ligand_file, removeHs=False)
        calc_single(mol_prot, mol_lig, output_file, clf)
    return

if __name__ == "__main__":
    func()
    print('DONE.')