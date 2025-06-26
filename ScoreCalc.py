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
        
        self.aascore_output_dirname = os.path.join(self.output_dir, 
                                                   self.conf['AAScore']['working_directory'])
        self.aascore_output_filename = 'scores.txt'
        self.generate_dir = os.path.join(self.conf['OUTPUT']['directory'],
                                         self.conf['ChemTS']['working_directory']
                                         )
        self.sincho_dir = os.path.join(self.conf['OUTPUT']['directory'],
                                       self.conf['SINCHO']['working_directory']
                                       )
        self.aascore_outdir = self.aascore_output_dirname
        self.output_files = []
        self.rank_output_dir = []
        self.max_workers = int(self.conf['GENERAL']['use_num_threads'])
        self.SDF_output_num = self.conf['AAScore']['output_num']           
        self.SDF_name_prefix = self.conf['AAScore']['OUTPUT']['sdf_name_prefix']
        self.logger = self.gl.cm.setup_custom_logger('AAScore', os.path.join(self.output_dir, self.conf['OUTPUT']['logs_dir'], 'AAScore.log'))

    def run(self):
        os.makedirs(self.aascore_outdir, exist_ok = True)

        #生成(及びEmbed)のtrajectory-rankの候補を取得
        chemts_trial_dirs = self.gl.rank_output_dirs
        scores = []
        for ct_dir in chemts_trial_dirs:
            #ct_dir ex. 'out_6Z0R/03_CompGen/trajectory_006/rank_01'
            aa_wdir = ct_dir.replace(self.generate_dir, self.aascore_outdir)
            #aa_wdir ex. 'out_6Z0R/04_DeltaGEst/trajectory_006/rank_01'
            aa_wdir_par = os.path.dirname(aa_wdir)
            #sc_wdir_par ex. 'out_6Z0R/02_MakeDec/trajectory_006'
            sc_wdir_par = aa_wdir_par.replace(self.aascore_outdir, self.sincho_dir)

            prot_name = f'prot_{ct_dir.split("/")[-2].split("_")[-1]}.pdb'
            lig_name  = f'lig_{ct_dir.split("/")[-2].split("_")[-1]}.pdb'
            prot_pdb   = os.path.join(aa_wdir_par, f'prot_{ct_dir.split("/")[-2].split("_")[-1]}.pdb')
            lig_pdb    = os.path.join(aa_wdir_par, f'lig_{ct_dir.split("/")[-2].split("_")[-1]}.pdb')
            pocket_pdb = os.path.join(aa_wdir_par, f'pocket_{ct_dir.split("/")[-2].split("_")[-1]}.pdb')

            if not os.path.isfile(prot_pdb):
                shutil.copy(os.path.join(sc_wdir_par, prot_name), aa_wdir_par)
            if not os.path.isfile(lig_pdb):
                shutil.copy(os.path.join(sc_wdir_par, lig_name), aa_wdir_par)
            if not os.path.isfile(pocket_pdb):
                distance = self.conf['AAScore']['protein_range']
                self._extract_residues_within_distance(prot_pdb, lig_pdb, pocket_pdb, distance=distance)
            #AA実行
            self._processess(aa_wdir, pocket_pdb, scores)
        #結果まとめ
        self._summary(scores)

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
        #print('save at', output_pdb)

    def _processess(self, aa_wdir, pocket_pdb, scores):
        lead_paths = glob(os.path.join(aa_wdir, 'lead_*'))
        task_list = []

        for lead in lead_paths:
            scores_file_path = os.path.join(lead, self.aascore_output_filename)
            input_sdf = os.path.join(lead, 'conformers.sdf')
            if os.path.isfile(input_sdf) and os.stat(input_sdf).st_size>0: #conformer.sdfが存在し、空(Embed失敗)でないもの
                task_list.append([pocket_pdb, input_sdf, scores_file_path])
                scores.append(scores_file_path)

        # 並列処理の実行
        max_workers = self.max_workers
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self._run_AAScore, *args) for args in task_list]
            # すべてのタスクが完了するのを待つ（エラーチェック含む）
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()  # 例外があればここでキャッチされる
                except Exception as e:
                    self.logger.error(f"Error in parallel execution: {e}")

    def _run_AAScore(self, pocket_pdb, input_sdf, scores_file_path):
        #self.logger.info(f'start , {input_sdf}')
        cwd = os.getcwd()
        poc = os.path.abspath(pocket_pdb)
        lig = os.path.abspath(input_sdf)
        log = os.path.abspath(scores_file_path)
        os.chdir('/AA_Score_Tool')
        os.system("python AA_Score.py --Rec "+poc+" --Lig "+lig+" --Out "+log)
        self.logger.info(f'{input_sdf} done.')
        os.chdir(cwd)
        return log

    def _summary(self, scores):
        df_all = pd.DataFrame(columns=['ROMol', 'AAScore', 'trajectory_num', 'rank_num', 'lead_num', 'conf_num'])
        
        for n, each_cpd_log_file in enumerate(scores): #each_cpd_log_file: out_6Z0R/04_DeltaGEst/trajectory_006/rank_01/lead_01/scores.txt

            df_aascore = pd.read_csv(each_cpd_log_file, names=['Name', 'score'], sep="\t").sort_values('score')
            #一番いいポーズのNameとスコアを取得
            top_conformer_name , top_conformer_score = df_aascore.loc[df_aascore["score"].idxmin(), ['Name','score']]
            top_conformer_num = top_conformer_name.split('_')[-1]
            input_sdf = each_cpd_log_file.replace(self.aascore_output_filename, 'conformers.sdf')
            best_pose = Chem.SDMolSupplier(input_sdf)[int(top_conformer_num)]
            best_pose.SetProp('AAScore', str(top_conformer_score))
            best_pose.SetProp('RootName', os.path.dirname(each_cpd_log_file))
            writer = Chem.SDWriter(each_cpd_log_file.replace(self.aascore_output_filename, 'best_pose.sdf'))
            writer.write(best_pose)
            writer.close()

            #df_allに追加
            sp_dir = each_cpd_log_file.split("/")
            df_all.loc[str(n)] = [best_pose, float(top_conformer_score), sp_dir[-4], sp_dir[-3], sp_dir[-2], str(top_conformer_name)]
        df_all_sorted = df_all.sort_values('AAScore')
        df_top = df_all_sorted[:self.SDF_output_num]

        #sdf出力
        all_sdf_out_file = os.path.join(self.aascore_outdir, 'all.sdf')
        PandasTools.WriteSDF(df_all_sorted, all_sdf_out_file, molColName='ROMol' ,properties=list(df_all_sorted.columns))
        top_sdf_out_file = os.path.join(self.aascore_outdir, 'top_'+str(self.SDF_output_num)+'.sdf')
        PandasTools.WriteSDF(df_top, top_sdf_out_file, molColName='ROMol' ,properties=list(df_top.columns))


