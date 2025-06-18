import sys, os, shutil, subprocess, re
import numpy as np
import pandas as pd

from scipy.cluster.hierarchy import linkage, dendrogram
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import squareform
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, AllChem, PandasTools, rdMolDescriptors, Draw, rdmolops
from rdkit.Chem.AllChem import AlignMol, EmbedMolecule, EmbedMultipleConfs
from rdkit.Chem.rdForceFieldHelpers import UFFGetMoleculeForceField
from openbabel import pybel
from glob import glob
from Bio.PDB import PDBParser, PDBIO, Select
from Bio.PDB.NeighborSearch import NeighborSearch
import concurrent.futures
from IPython.core.debugger import Pdb

class ResidueSelect(Select):
    def __init__(self, residues):
        self.residues = residues

    def accept_residue(self, residue):
        return residue in self.residues
        

class AA_Score:
    def __init__(self, generate_lead, config):
        self.gl = generate_lead
        self.conf = config
        self.output_dir = self.conf['OUTPUT']['directory'] 
        
        self.aascore_output_dirname = os.path.join(self.output_dir, self.conf['AAScore']['working_directory'])
        self.aascore_output_filename = 'scores.txt'
        self.generate_dir = os.path.join(self.conf['OUTPUT']['directory'],
                                         self.conf['ChemTS']['working_directory']
                                         )
        self.aascore_outdir = self.aascore_output_dirname
        self.output_files = []
        self.rank_output_dir = []
        self.max_workers = self.conf['AAScore']['parallel_num']
        self.SDF_output_num = self.conf['AAScore']['output_num']           
        self.SDF_name_prefix = self.conf['AAScore']['OUTPUT']['sdf_name_prefix']
        self.logger = self.gl.cm.setup_custom_logger('AAScore', os.path.join(self.output_dir, self.conf['OUTPUT']['logs_dir'], 'AAScore.log'))

    def run(self):
        self.aascore_outdir = self.aascore_output_dirname
        os.makedirs(self.aascore_outdir, exist_ok = True)

        for trajectory_dir in self.gl.trajectory_dirs:
            trajectory_name, trajectory_num = self._parse_trajectory_name(trajectory_dir)

            # ChemTS生成結果のパスを取得
            generate_tra_dir = self._get_generate_dir(trajectory_name)

            # AAScoreの結果出力先を設定
            aascore_tra_dir = self._get_aascore_tra_dir(self.aascore_outdir, trajectory_name)
            os.makedirs(aascore_tra_dir, exist_ok = True)

            # トラジェクトリーごとにタンパク質とリガンドを取得し、ポケットを計算
            protein_pdb, compound_pdb = self._get_pdb_files(trajectory_dir, trajectory_num)
            self._copy_files([protein_pdb, compound_pdb], aascore_tra_dir)
            pocket_pdb_path = os.path.join(aascore_tra_dir, f'pocket_{trajectory_num}.pdb')
            distance = self.conf['AAScore']['protein_range']
            self._extract_residues_within_distance(protein_pdb, compound_pdb, pocket_pdb_path, distance=distance)

            rank_num = len(os.listdir(generate_tra_dir))
            order_scale = int(len(str(int(rank_num)))+1)
            for r in range(rank_num):
                rank = str(r).zfill(order_scale)
                self._process_rank(generate_tra_dir, aascore_tra_dir, rank, pocket_pdb_path)

    def _parse_trajectory_name(self, trajectory_dir):
        """トラジェクトリ名と番号を抽出"""
        trajectory_name = trajectory_dir.split(os.sep)[-1]
        trajectory_num = str(trajectory_name.split('_')[-1])
        return trajectory_name, trajectory_num

    def _get_generate_dir(self, trajectory_name):
        """生成ディレクトリを取得"""
        return os.path.join(self.generate_dir, trajectory_name)

    def _get_aascore_tra_dir(self, aascore_outdir, trajectory_name):
        """AAScoreトラジェクトリディレクトリを取得"""
        return os.path.join(aascore_outdir, trajectory_name)

    def _get_pdb_files(self, trajectory_dir, trajectory_num):
        """PDBファイルを取得"""
        protein_pdb = os.path.join(trajectory_dir, f'prot_{trajectory_num}.pdb')
        compound_pdb = os.path.join(trajectory_dir, f'lig_{trajectory_num}.pdb')
        return protein_pdb, compound_pdb

    def _copy_files(self, files, dest_dir):
        """複数のファイルをコピー"""
        for file in files:
            shutil.copy(file, dest_dir)

    def _extract_residues_within_distance(self, protein_pdb, compound_pdb, output_pdb, distance=13.0):
        parser = PDBParser(QUIET=True)
        protein_structure = parser.get_structure('protein', protein_pdb)
        compound_structure = parser.get_structure('compound', compound_pdb)
        protein_atoms = list(protein_structure.get_atoms())
        compound_atoms = list(compound_structure.get_atoms())
        neighbor_search = NeighborSearch(protein_atoms)
        close_residues = set()
        for atom in compound_atoms:
            close_atoms = neighbor_search.search(atom.coord, distance)
            for close_atom in close_atoms:
                close_residues.add(close_atom.get_parent())
        accept_residues = ["ALA","GLY","VAL","LEU","ILE","MET","PHE","TYR","TRP","PRO",
                        "SER","THR","CYS","ASN","GLN","ASP","GLU","LYS","ARG","HIS"]
        convert_residues = {"HIE":"HIS", "HID":"HIS", "HIP":"HIS", "CYX":"CYS", "CYM":"CYS",
                            "ASH":"ASP", "GLH":"GLU"}

        for residue in close_residues:
            if residue.get_resname() in convert_residues:
                residue.resname = convert_residues[residue.get_resname()]
            if residue.get_resname() not in accept_residues:
                close_residues.remove(residue)
        close_residues = {
            residue for residue in close_residues
            if residue.get_resname() in accept_residues
        }
        io = PDBIO()
        io.set_structure(protein_structure)
        io.save(output_pdb, ResidueSelect(close_residues))
        print('save at', output_pdb)

    def _process_rank(self, generate_tra_dir, aascore_tra_dir, rank, pocket_pdb_path):
        rank_name = f'rank_{str(rank)}'
        generate_tra_rank_dir = os.path.join(generate_tra_dir, rank_name)
        aascore_tra_rank_dir = os.path.join(aascore_tra_dir, rank_name)
        os.makedirs(aascore_tra_rank_dir, exist_ok=True)

        self.rank_output_dir.append(aascore_tra_rank_dir)

        lead_paths = glob(os.path.join(aascore_tra_rank_dir, 'lead_*'))
        #lead_paths = glob(os.path.join(generate_tra_rank_dir, 'lead_*'))
        lead_num = len(lead_paths)
        lead_order_scale = int(len(str(int(lead_num)))+1) 
        aascore_output_file_paths = []

        task_list = []
        for lead in range(lead_num):
            lead_name = f'lead_{str(lead).zfill(lead_order_scale)}'
            generate_tra_rank_lead_dir = os.path.join(generate_tra_rank_dir, lead_name)
            aascore_tra_rank_lead_dir = os.path.join(aascore_tra_rank_dir, lead_name)
            os.makedirs(aascore_tra_rank_lead_dir, exist_ok=True)
            aascore_output_file_path = os.path.join(aascore_tra_rank_lead_dir, self.aascore_output_filename)
            aascore_output_file_paths.append(aascore_output_file_path)

            # lead.sdfのパスを取得
            sdf_paths = glob(os.path.join(aascore_tra_rank_lead_dir, '*.sdf'))
            if not sdf_paths:
                self.logger.error(f'No conf sdf in {generate_tra_rank_lead_dir}.')
                continue

            lead_sdf_path = sdf_paths[0]
            #shutil.copy(lead_sdf_path, aascore_tra_rank_lead_dir)

            task_list.append((pocket_pdb_path, lead_sdf_path, aascore_output_file_path))
        self.output_files.append(aascore_output_file_paths)

        # 並列処理の実行
        max_workers = self.max_workers
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self._process_lead, *args) for args in task_list]
            # すべてのタスクが完了するのを待つ（エラーチェック含む）
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()  # 例外があればここでキャッチされる
                except Exception as e:
                    self.logger.error(f"Error in parallel execution: {e}")

    def _process_lead(self, pocket_pdb_path, lead_sdf_path, aascore_output_file_path):
        if not os.path.exists(lead_sdf_path):
            self.logger.error(f'No conf sdf at {lead_sdf_path}.')
            return

        self._run_AAScore(pocket_pdb_path, lead_sdf_path, aascore_output_file_path)

    def _run_AAScore(self, pocket_pdb_path, lead_sdf_path, aascore_output_file_path):
        self.logger.info(f'start , {lead_sdf_path}')
        cwd = os.getcwd()
        poc = os.path.abspath(pocket_pdb_path)
        lig = os.path.abspath(lead_sdf_path)
        log = os.path.abspath(aascore_output_file_path)
        os.chdir('/AA-Score-Tool')
        os.system("python AA_Score.py --Rec "+poc+" --Lig "+lig+" --Out "+log)
        self.logger.info(f'{lead_sdf_path} done.')
        os.chdir(cwd)
        return log

    def _get_trajectory_rank_number(self, path):
        split_path = path.split(os.sep)
        trajectory_name = split_path[[i for i, item in enumerate(split_path) if re.match(r"trajectory_\d+$", item)][0]]
        trajectory_num = str(trajectory_name.split('_')[-1])
        rank_name = split_path[[i for i, item in enumerate(split_path) if re.match(r"rank_\d+$", item)][0]]
        rank_num = str(rank_name.split('_')[-1])
        return trajectory_num, rank_num
        
    def result_output(self):
        df_all, df_all_top = [], []
        for n, rank_results in enumerate(self.output_files):
            mols, aascores = [], []
            for lead_result in rank_results:
                df_aascore = pd.read_csv(lead_result, names=['Name', 'score'], sep="\t").sort_values('score')
                top_conformer_name , top_conformer_score = df_aascore.loc[df_aascore["score"].idxmin(), ['Name','score']]
                top_conformer_num = top_conformer_name.split('_')[-1]
                aascores.append(top_conformer_score)

                sdf_path = lead_result.replace(self.aascore_output_filename, 'conformers.sdf')
                sup = Chem.SDMolSupplier(sdf_path)
                mols_sdf = [mol for mol in sup if mol is not None]
                mol = mols_sdf[int(top_conformer_num)]
                mols.append(mol)

            trajectory_num, rank_num = self._get_trajectory_rank_number(lead_result)
            df = pd.DataFrame({'ROMol':mols, 'AAScore':aascores})
            df['trajectory_num'], df['rank_num'], df['lead_num'] = trajectory_num, rank_num, range(len(df))
            # 分子からプロパティを取得してカラムに追加
            properties = []
            for mol in mols:
                if mol is not None:
                    props = mol.GetPropsAsDict() #プロパティを辞書型で取得
                    properties.append(props)
            df_props = pd.DataFrame(properties)
            df = pd.concat([df, df_props], axis=1).sort_values('AAScore')
            df_top = df[:self.SDF_output_num]

            df_all.append(df)
            df_all_top.append(df_top)

            sdf_out_dir = self.rank_output_dir[n]
            all_sdf_out_path = os.path.join(sdf_out_dir, self.SDF_name_prefix + '_all.sdf')
            top_sdf_out_path = os.path.join(sdf_out_dir, self.SDF_name_prefix + '_top.sdf')
            
            PandasTools.WriteSDF(df, all_sdf_out_path, molColName='ROMol' ,properties=list(df.columns))
            PandasTools.WriteSDF(df_top, top_sdf_out_path, molColName='ROMol' ,properties=list(df_top.columns))

        df_all_concat = pd.concat(df_all, ignore_index=True).sort_values('AAScore')
        df_all_top_concat = pd.concat(df_all_top, ignore_index=True).sort_values('AAScore')
                    
        all_trajectory_sdf_out_path= os.path.join(self.aascore_outdir, self.SDF_name_prefix + '_all_traj.sdf')
        top_trajectory_sdf_out_path= os.path.join(self.aascore_outdir, self.SDF_name_prefix + '_all_traj_top.sdf')
        PandasTools.WriteSDF(df_all_concat, all_trajectory_sdf_out_path, 
                             molColName='ROMol' ,properties=list(df_all_concat.columns))
        PandasTools.WriteSDF(df_all_top_concat, top_trajectory_sdf_out_path,
                             molColName='ROMol' ,properties=list(df_all_top_concat.columns))

