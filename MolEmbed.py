## input  : results.csv (obtained from ChemTS)
## output : lead_NNN/conformers.sdf

import sys, os, shutil, subprocess, re
import numpy as np
import pandas as pd
import yaml

from scipy.cluster.hierarchy import linkage, dendrogram
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import squareform

from rdkit import Chem, DataStructs
from rdkit.Chem import rdMolAlign, Descriptors, AllChem, PandasTools, rdMolDescriptors, Draw, rdmolops
from rdkit.Chem.AllChem import AlignMol, EmbedMolecule, EmbedMultipleConfs
from rdkit.Chem.rdForceFieldHelpers import UFFGetMoleculeForceField
from openbabel import pybel
from glob import glob

from Bio.PDB import PDBParser, PDBIO, Select
from Bio.PDB.NeighborSearch import NeighborSearch
import concurrent.futures

from IPython.core.debugger import Pdb
from concurrent.futures import ThreadPoolExecutor, as_completed

class Embed_Mols:
    def __init__(self, generate_lead, config):
        self.gl = generate_lead
        self.trajectory_dirs = self.gl.trajectory_dirs
        self.conf = config
        self.outdir = self.conf['OUTPUT']['directory']
        self.workdir = os.path.join(self.outdir, self.conf['AAScore']['working_directory'])
        self.sinchodir = os.path.join(self.outdir, self.conf['SINCHO']['working_directory'])
        self.logger = self.gl.cm.setup_custom_logger('AAScore', os.path.join(self.outdir, self.conf['OUTPUT']['logs_dir'], 'Embed.log'))

    def run(self):
        num_threads = int(self.conf['GENERAL']['use_num_threads'])
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(self._run_a_rank, rank_dir) for rank_dir in self.gl.rank_output_dirs]
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    self.logger.error(f"Thread failed: {e}")
        """
        for rank_dir in self.gl.rank_output_dirs:
            self._run_a_rank(rank_dir)
        """
        

    
    def _run_a_rank(self, rank_dir):
        # ディレクトリ・ファイルの定義
        self.logger.info(rank_dir)
        trajectory_name = rank_dir.split("/")[-2].split("_")[0]
        trajectory_num = rank_dir.split("/")[-2].split("_")[1]
        tdir = rank_dir.split("/")[-2]
        rdir = rank_dir.split("/")[-1]
        csv_file = os.path.join(rank_dir, 'results.csv')
        parent_dir = os.path.join(self.workdir, tdir, rdir)
        os.makedirs(parent_dir, exist_ok=True)
        output_path_prefix = os.path.join(parent_dir, 'lead')
                
        # yaml記述の条件下で、results.csvからlead.xlsxを作成,df生成
        df_choice = self._compounds_select(csv=csv_file, output_path_prefix=output_path_prefix)

        ncpd_scale = int(len(str(int(len(df_choice))))+1)
        
        # 各compoundについて3次元構造生成
        # まずhitのセットアップ # 反応点チェック
        core_pdb = os.path.join(self.sinchodir, tdir, 'lig_'+trajectory_num+'.pdb')
        with open(os.path.join(self.sinchodir, tdir, 'sincho_result.yaml'),'r')as f:
            sincho_res = yaml.safe_load(f)
        anchor_atomname = str(sincho_res['SINCHO_result'][rdir]['atom_num']).split('_')[-1]

        # 構造マッチのためのコア定義＆ワイルドカード化&SMARTS化
        core_mol, core_wc_mol, smarts = self._core_def(core_pdb, anchor_atomname)
        #self.logger.info(f"Core definition: {smarts}")

        # 各構造でコンフォーマー生成のループ
        for idx, row in df_choice.iterrows():
            confgen_output_path = output_path_prefix+'_'+str(idx).zfill(ncpd_scale)
            self._conf_gen(idx=idx, row=row, output_path=confgen_output_path,
                            core_mol=core_mol, core_wc_mol=core_wc_mol, smarts=smarts, confgen_output_path=confgen_output_path)

        # 中性化していた場合chargeを付与して戻す
        # 途中からの計算時に落ちる。全部やっても構わないため分岐無に変更(2025/06/05 kudo)
        #if not self.gl.is_neutral:
        #self.add_charge(output_path_prefix)

    def _compounds_select(self, csv, output_path_prefix):
        ## input  : results.csv (obtained from ChemTS)
        ## output : choice_to_docking.csv, lead.xlsx (with 2D images)
        choice_method = self.conf['AAScore']['method']
        if choice_method == 'rand':
            num_of_cpd = self.conf['AAScore']['num_of_cpd']
        reward_cutoff = self.conf['AAScore']['reward_cutoff']
        df = pd.read_csv(csv)
        if len(df)==0:
            return
        # 物性値をdfに追記
        df = self._calc_df_properties(df)
        df_indep = df.drop_duplicates(subset='canonical_smiles', keep='first')
        if reward_cutoff:
            df_rew = df_indep[df_indep['reward'] >= reward_cutoff]
        else:
            df_rew = df_indep
        if choice_method == "rand" and len(df_rew)>num_of_cpd:
            df_choice = df_rew.sample(frac=1)[:num_of_cpd].reset_index(drop=True)
        else:
            df_choice = df_rew.reset_index(drop=True)
        df_choice['Choice_idx'] = [ i for i in range(len(df_choice)) ]
        df_choice.drop('mols', axis=1).reset_index(drop=False).to_csv(os.path.join(os.path.dirname(csv), 'choice_to_docking.csv'))
        
        if 'mols' in df_choice.columns and df_choice['mols'].notna().any():
            PandasTools.SaveXlsxFromFrame(df_choice, output_path_prefix+'.xlsx', molCol='mols', size=(150,150))
        df_choice['mhs'] = [ Chem.AddHs(m) for m in df_choice['mols'] ]
        return df_choice
        
    def _calc_df_properties(self, df):
        ## input  : dataframe
        ## output : dataframe with properties
        if 'Unnamed: 0' in df.columns:
            df = df.drop('Unnamed: 0', axis=1)

        df['mols'] = [Chem.MolFromSmiles(smi) for smi in df['smiles']]
        df['canonical_smiles'] = [Chem.MolToSmiles(m) for m in df['mols']]
        df['MW'] = [Descriptors.ExactMolWt(m) for m in df['mols']]
        df['LogP'] = [Descriptors.MolLogP(m) for m in df['mols']]
        df['donor'] = [rdMolDescriptors.CalcNumLipinskiHBD(m) for m in df['mols']]
        df['acceptor'] = [rdMolDescriptors.CalcNumLipinskiHBA(m) for m in df['mols']]
        return df
    
    def _core_def(self, pdb_path, label):
        mol = Chem.MolFromPDBFile(pdb_path, removeHs=False, sanitize=False)
        pdbmol = mol
        for atom in mol.GetAtoms():
            atom.SetProp("pdb_label", atom.GetPDBResidueInfo().GetName())
        Chem.SanitizeMol(mol, Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE)
        # ワイルドカードを追加するIndex取得
        idx = None
        for atom in mol.GetAtoms():
            info = atom.GetPDBResidueInfo()
            if info and info.GetName().strip() == label:
                idx = atom.GetIdx()
                break
        rw = Chem.RWMol(mol)
        target_atom = rw.GetAtomWithIdx(idx)
        # ターゲット原子に結合している水素を1つ削除
        for nbr in target_atom.GetNeighbors():
            if nbr.GetAtomicNum() == 1:
                rw.RemoveBond(idx, nbr.GetIdx())
                rw.RemoveAtom(nbr.GetIdx())
                break
        # ワイルドカード（*）を追加
        dummy_idx = rw.AddAtom(Chem.Atom(0))  # AtomicNum=0 = *
        rw.AddBond(idx, dummy_idx, Chem.BondType.SINGLE)
        # 水素を削除して出力用に整形
        mol_final = rw.GetMol()
        #xyzの再帰
        for atom in mol_final.GetAtoms():
            for a in pdbmol.GetAtoms():
                if atom.HasProp("pdb_label")==True:
                    if a.GetPDBResidueInfo().GetName() == atom.GetProp("pdb_label"):
                        ind = a.GetIdx()
                        xyz = pdbmol.GetConformer().GetAtomPosition(ind)
                        atom.SetProp("original_pos", f'{xyz.x},{xyz.y},{xyz.z}')
                        break
        mol_final = Chem.RemoveHs(mol_final)
        smarts = Chem.MolToSmarts(mol_final)
        smarts = smarts.replace("#0","*")
        return pdbmol, mol_final, smarts
 
    def _conf_gen(self, idx, row, output_path, core_mol, core_wc_mol, smarts, confgen_output_path):
        # input  : row and idx in dataframe
        # output : conformations
        os.makedirs(output_path, exist_ok=True)
        smiles_b = row['smiles']
        #object立ち上げ
        query = Chem.MolFromSmarts(smarts)
        target = Chem.MolFromSmiles(smiles_b, sanitize=True)
        target = Chem.AddHs(target)
        AllChem.Compute2DCoords(target)
        #部分構造マッチ
        matches = target.GetSubstructMatches(query)
        if not matches:
            self.logger.warning(f"No substructure match found for {smiles_b} with core {smarts}. Skipping embedding.")
            return
        match_pairs = []
        for match in matches:
            pair = list(enumerate(match))
            match_pairs.append(pair)
        #マッチ結果の検証(２箇所ある時用の処理)
        if len(match_pairs) > 1:
            #基本ChemTSのつづりはSMILES先頭が母核ー＞Molオブジェクト生成時に小さいindexが付与される
            min_b_values = [min(pair_list, key=lambda x: x[1])[1] for pair_list in match_pairs]
            min_index = min(range(len(min_b_values)), key=lambda i: min_b_values[i])
            pairs = match_pairs[min_index]
        else:
            pairs = match_pairs[0]
        #マッチ結果の検証(正しいマッチングかどうかチェック)
        #checker = self._is_only_modified_at_wildcard(query, target, pairs)
        #if checker== False:
        #    self.logger.warning(f"Invalid match for {smiles_b} with core {smarts}. Skipping embedding.")
        #    return
        
        ref = core_wc_mol.GetConformer()
        #拘束する原子のindexとその座標を格納
        coord_map = { target_idx: ref.GetAtomPosition(ref_idx) 
                     for ref_idx, target_idx in pairs 
                     if core_wc_mol.GetAtomWithIdx(ref_idx).GetAtomicNum() != 0 }
        #構造を立ち上げていく
        #立ち上げオプション
        n_conf = int(self.conf['AAScore']['conf_per_cpd']) #最終的に欲しい構造数
        max_attempts = int(self.conf['AAScore']['embed_details']['max_attempts']) # 最大試行回数
        if max_attempts< n_conf*1.5: 
            max_attempts = n_conf * 1.5
        rms_thresh = float(self.conf['AAScore']['embed_details']['rms_thresh'])  # RMSD閾値(オングscale)
        target.RemoveAllConformers()  # 既存のコンフォーマーを削除

        self._embed_confs(target, coord_map, n_conf, max_attempts, rms_thresh, confgen_output_path)
        return
    
    def _is_only_modified_at_wildcard(self, query, target, match):
        # query中のワイルドカードのindex（AtomicNum=0）
        wildcard_qidx = [a.GetIdx() for a in query.GetAtoms() if a.GetAtomicNum() == 0]
        if not wildcard_qidx:
            raise ValueError("SMARTSに[*] or [#0] ワイルドカードがありません")
        wildcard_qidx = wildcard_qidx[0]
        wildcard_tid = match[wildcard_qidx]
        # 除去対象のインデックス = match中の他のインデックス
        remove_ids = [tid for i, tid in enumerate(match) if i != wildcard_qidx]
        #print(remove_ids)
        # target Mol を EditableMol に変換
        rw = Chem.EditableMol(Chem.Mol(target))
        #for iii in sorted(remove_ids, reverse=True):  # 高いindexから削除
        #    rw.RemoveAtom(iii)
        for _, tgt_idx in sorted(remove_ids, key=lambda x: x[1], reverse=True):
            rw.RemoveAtom(tgt_idx)
        try:
            remaining = rw.GetMol()
            fragments = Chem.GetMolFrags(remaining, asMols=True)
            return len(fragments) <= 1
        except:
            return False


    def _embed_confs(self, target, coord_map, n_conf, max_attempts, rms_thresh, confgen_output_path):
        generated, attempts = 0,0
        heavy_atoms = [atom.GetIdx() for atom in target.GetAtoms() if atom.GetAtomicNum()>1]
        while generated < n_conf and attempts < max_attempts:
            attempts += 1
            #print(attempts)
            tmpmol = Chem.Mol(target)
            res = AllChem.EmbedMolecule(tmpmol, coordMap=coord_map, useRandomCoords=True, randomSeed=-1, maxAttempts=1, useExpTorsionAnglePrefs=True, useBasicKnowledge=True) 
            if res != 0:
                #print("EmbedMolecule was failed")
                continue
            #ここでminimize
            fixed_atoms = coord_map.keys()
            uff = AllChem.UFFGetMoleculeForceField(tmpmol, confId=0)
            for a_idx in fixed_atoms:
                uff.AddFixedPoint(a_idx)
            uff.Initialize()
            uff.Minimize()
 
            new_conf = tmpmol.GetConformer()
            new_conf.SetId(generated)
            target.AddConformer(new_conf, assignId=True)
            new_conf_id = target.GetNumConformers() -1

            is_unique = True
            for conf in target.GetConformers():
                rms = self._rms_no_align(target, conf.GetId(), new_conf_id, atom_indices=heavy_atoms)
                if rms < rms_thresh and rms != 0:
                    is_unique = False
                    target.RemoveConformer(new_conf_id)
                    break
            if is_unique:
                generated+=1

        writer = Chem.SDWriter(os.path.join(confgen_output_path, 'conformers.sdf'))
        for cid,conf in enumerate(target.GetConformers()):
            conf_id = conf.GetId()
            conf_mol = Chem.Mol(target)
            conf_mol.RemoveAllConformers()
            conf_mol.AddConformer(conf, assignId=True)
            conf_mol.SetProp("_Name", "confid_"+str(cid))
            #writer.write(target, confId=conf.GetId())
            writer.write(conf_mol)
        writer.close()

        return

    def _rms_no_align(self, mol, confId1, confId2, atom_indices=None):
        conf1 = mol.GetConformer(confId1)
        conf2 = mol.GetConformer(confId2)
        if atom_indices is None:
            atom_indices = range(mol.GetNumAtoms())
        sq_diffs = []
        for i in atom_indices:
            p1 = conf1.GetAtomPosition(i)
            p2 = conf2.GetAtomPosition(i)
            diff = p1 - p2
            sq_diffs.append(diff.LengthSq())
        return (sum(sq_diffs) / len(sq_diffs) **0.5)






    """legacy, but useful for protonated molecules
    def add_charge(self, conformers_path):
        conformers = sorted(glob(os.path.join(conformers_path + '*', 'conformers_*.mol2')))
        for conf in conformers:
            # for debug->pdb,mol2は不要だよね。
            # obabel -ipdb ${input}.pdb -opdb -O ${output}.pdb -ph 7.4
            subprocess.run(['obabel', '-imol2', conf,'-omol2', '-O', conf, '-ph', '7.4'])
            # rdkitではmol2扱いにくいのでpdbにしておく 
            subprocess.run(['obabel', '-imol2', conf,'-opdb', '-O', conf.replace('mol2','pdb')])
    """
