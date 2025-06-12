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

def _parse_trajectory_name(trajectory_dir):
    """トラジェクトリ名と番号を抽出"""
    trajectory_name = trajectory_dir.split(os.sep)[-1]
    trajectory_num = str(trajectory_name.split('_')[-1])
    return trajectory_name, trajectory_num

class ResidueSelect(Select):
    def __init__(self, residues):
        self.residues = residues

    def accept_residue(self, residue):
        return residue in self.residues
        
def extract_residues_within_distance(protein_pdb, compound_pdb, output_pdb, distance=13.0):
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

class Embed_Mols:
    def __init__(self, generate_lead, config):
        self.gl = generate_lead
        self.trajectory_dirs = self.gl.trajectory_dirs
        self.conf = config
        self.outdir = self.conf['OUTPUT']['directory']
        self.logger = self.gl.cm.setup_custom_logger('AAScore', os.path.join(self.conf['OUTPUT']['directory'], self.conf['OUTPUT']['logs_dir'], 'AAScore.log'))

    def run(self):
        #for trajectory_dir in self.trajectory_dirs:
        #    trajectory_name, trajectory_num = _parse_trajectory_name(trajectory_dir)
        
        for rank_dir in self.gl.rank_output_dirs:
            print(rank_dir)
            output_rank_dir = rank_dir
            trajectory_name = output_rank_dir.split("/")[-2].split("_")[0]
            trajectory_num = output_rank_dir.split("/")[-2].split("_")[1]
            rdir = output_rank_dir.split("/")[-1]
            csv_path = os.path.join(output_rank_dir, 'results.csv')
            output_path_prefix = os.path.join(self.conf['OUTPUT']['directory'], self.conf['AAScore']['working_directory'], trajectory_name+'_'+trajectory_num, rdir, 'lead')
            os.makedirs(os.path.join(self.conf['OUTPUT']['directory'], self
.conf['AAScore']['working_directory'], trajectory_name+'_'+trajectory_num, rdir), exist_ok=True)

            #output_path_prefix = os.path.join(output_rank_dir, 'lead')
            input_compound_file = os.path.join(self.outdir, self.conf['SINCHO']['working_directory'], trajectory_name+'_'+trajectory_num, 'lig_'+trajectory_num+'.pdb')
            print(input_compound_file)
            #input_compound_file = self.gl.input_compound_files[int(trajectory_num)]

            # 全生成結果から化合物を任意の数選択し、さらにそれらのコンフォーマーを生成
            # コンフォーマーはmol2形式で保存
            self.csv_to_mol2(csv = csv_path, 
                             output_path_prefix = output_path_prefix,
                             ligand_pdb = input_compound_file)

            # 中性化していた場合chargeを付与して戻す
            # 途中からの計算時に落ちる。全部やっても構わないため分岐無に変更(2025/06/05 kudo)
            #if not self.gl.is_neutral:
            #self.add_charge(output_path_prefix)

    def csv_to_mol2(self, csv, output_path_prefix, ligand_pdb):
        aascore_conf = self.conf['AAScore']
        choise_method = aascore_conf['method']
        reward_cutoff = aascore_conf['reward_cutoff']
        num_of_cpd = aascore_conf['num_of_cpd']
        noc_order_scale = int(len(str(int(num_of_cpd)))+1)
        conf_per_cpd = aascore_conf['conf_per_cpd']
        cpc_order_scale = int(len(str(int(conf_per_cpd)))+1)
        
        # 立ち上げの基準となるligand
        lig = Chem.MolFromPDBFile(ligand_pdb)
        self.lig_morgan_fp = AllChem.GetMorganFingerprintAsBitVect(lig, 2, 1024)

        # read csv
        df = pd.read_csv(csv)
        if len(df)==0:
            return
        df = self.calc_df_properties(df)

        # reward_cutoffが0以外ならその値以上の化合物のみを対象にする
        if reward_cutoff > 0:
            df = df[df['reward'] >= reward_cutoff]
        
        # randならシャッフルして指定個数
        # all(rand以外)なら順に全て
        if choise_method == 'rand':
            if len(df) >= num_of_cpd:
                df = df.sample(frac=1)[:num_of_cpd].reset_index(drop=True)
            else:
                df = df.reset_index(drop=True)
        else:
            df = df.reset_index(drop=True)

        df.drop('mols', axis=1).reset_index(drop=False).to_csv(os.path.join(os.path.dirname(csv), 'choise_to_docking.csv'))

        # create ID
        df_rows_scale = int(len(str(len(df)))+1)
        if choise_method == 'all':
            noc_order_scale = df_rows_scale
        df['ChemTS_idx'] = ['ChemTS_'+str(i).zfill(df_rows_scale) for i in range(len(df))]

        # Add hidrogen
        df['mhs'] = [Chem.AddHs(m) for m in df['mols']]
        
        mols = list(df['mols'])
        # AllChem.Compute2DCoords(lig)
        # for mol in mols:
        #     AllChem.GenerateDepictionMatching2DStructure(mol, lig)
        # legends = ['reward=' + str(round(rw, 3)) for rw in df['reward']]
        # img = Draw.MolsToGridImage(mols, molsPerRow=5, subImgSize=(400, 400), legends=legends)
        # img.save(os.path.join(os.path.dirname(csv), 'mols.png'))

        # write structure
        out_cols = list(df.columns)
        out_cols.remove('mhs')
        if 'mols' in df.columns and df['mols'].notna().any():
            PandasTools.SaveXlsxFromFrame(df[out_cols], output_path_prefix + '.xlsx', molCol='mols', size=(150, 150))
        else:
            print("Mol column is missing or empty – skipping SaveXlsxFromFrame")

        # calc 3D structures
        for idx, row in df.iterrows():
            mh = row['mhs']
            try:
                # AllChem.ConstrainedEmbed(mh, lig)
                if int(conf_per_cpd * 2) > 50:
                    numConfs = int(conf_per_cpd * 2)
                else:
                    numConfs = 50
                embedded_mols, energies = self.ConstrainedEmbedMultipleConfs(mh, lig, numConfs = numConfs )
                                                                            
                conformers = []
                for cid in range(embedded_mols.GetNumConformers()):
                    conf_mol = Chem.Mol(embedded_mols)  # 元の分子をコピー
                    conf_mol.RemoveAllConformers()  # 全てのコンフォーマーを削除
                    conf_mol.AddConformer(embedded_mols.GetConformer(cid))  # 指定したコンフォーマーのみ追加
                    conformers.append(conf_mol)

                # クラスタリングによる代表コンフォーマーの選択
                conformers, energies = self.remove_similar_conformers_from_clustering(conformers, energies,
                                                                                      conf_per_cpd = conf_per_cpd)
            except Exception as e:
                # AllChem.EmbedMolecule(mh, randomSeed=0)
                self.logger.error(f"エラーが発生しました: {e}")
                self.logger.error(f"embed error.")
                continue
            for n, mol in enumerate(conformers):
                mol.SetProp('_Name', row['ChemTS_idx'] + '_' + 'conformers_' + str(n).zfill(cpc_order_scale))

            df_conformers = pd.DataFrame({'conformers':conformers, 'energy':energies})
            df_conformers = pd.concat([df_conformers, pd.concat([row.to_frame().T ] * len(df_conformers), ignore_index=True)], axis=1)
            df_conformers['_Name'] =  [f"{value}_{str(n).zfill(cpc_order_scale)}" for n, value in enumerate(df_conformers['ChemTS_idx'], start=0)]
            
            # 1分子に対して、複数コンフォーマーのmol2保存
            # write to args.sdf
            output_dir = f'{output_path_prefix}_{str(idx).zfill(noc_order_scale)}'
            os.makedirs(output_dir, exist_ok = True)
            sdf_name = os.path.join(output_dir, 'conformers.sdf')
            PandasTools.WriteSDF(df_conformers, sdf_name, molColName='conformers',
                    properties=['generated_id', 'smiles', 'MW', 'LogP', 'donor', 'acceptor', 'energy', '_Name'], idName='ChemTS_idx')

            conformers_output_dir = os.path.join(output_dir, 'conformers')
            mol2_paths = []
            for i, pbmol in enumerate(pybel.readfile('sdf', sdf_name)):
                mol2_name = f'{conformers_output_dir}_{str(i).zfill(cpc_order_scale)}.mol2'
                pbmol.write('mol2', mol2_name, overwrite=True)
                #pdbは全く使用していないのでCO
                #subprocess.run(['obabel', '-imol2', mol2_name,'-opdb', '-O', mol2_name.replace('mol2','pdb')])
                subprocess.run(['obabel', '-imol2', mol2_name,'-omol2', '-O', mol2_name, '-ph', '7.4'])
                mol2_paths.append(mol2_name)
            charged_mols = []
            for mol2 in mol2_paths:
                mol = Chem.MolFromMol2File(mol2, sanitize=False, removeHs=False)
                if mol is not None:
                    charged_mols.append(mol)
            for index, mol in enumerate(charged_mols):
                mol.SetProp('_Name', f"{row['ChemTS_idx']}_{str(index).zfill(cpc_order_scale)}")
            
            charged_sdf_name = sdf_name
            writer = Chem.SDWriter(charged_sdf_name)
            for mol in charged_mols:
                writer.write(mol)
            writer.close()

    def calc_df_properties(self, df):
        if 'Unnamed: 0' in df.columns:
            df = df.drop('Unnamed: 0', axis=1)

        df = df.drop_duplicates(['smiles'])

        # add canonical smiles
        df['mols'] = [Chem.MolFromSmiles(smi) for smi in df['smiles']]
        df['canonical_smiles'] = [Chem.MolToSmiles(m) for m in df['mols']]

        # Add mw logp donner acceptor
        df['MW'] = [Descriptors.ExactMolWt(m) for m in df['mols']]
        df['LogP'] = [Descriptors.MolLogP(m) for m in df['mols']]
        df['donor'] = [rdMolDescriptors.CalcNumLipinskiHBD(m) for m in df['mols']]
        df['acceptor'] = [rdMolDescriptors.CalcNumLipinskiHBA(m) for m in df['mols']]
        
        morgan_fps = [AllChem.GetMorganFingerprintAsBitVect(m, 2, 1024) for m in df['mols']]
        df['Tanimoto_sim'] = DataStructs.BulkTanimotoSimilarity(self.lig_morgan_fp, morgan_fps)
        
        return df

    def remove_similar_conformers(self, conformers, energies, threshold=0.8):
        rmsd_matrix = self.calc_rmsd_matrix(conformers)
        n = len(conformers)
        to_keep = np.ones(n, dtype=bool)

        for i in range(n):
            if not to_keep[i]:
                continue
            for j in range(i + 1, n):
                if rmsd_matrix[i, j] <= threshold:
                    to_keep[j] = False

        return [conformers[i] for i in range(n) if to_keep[i]], [energies[i] for i in range(n) if to_keep[i]]

    def remove_similar_conformers_from_clustering(self, conformers, energies, conf_per_cpd):
        rmsd_mtx = self.calc_rmsd_matrix(conformers)
        centers, labels = self.make_dendrogram(rmsd_mtx, n_clusters=conf_per_cpd)
        choise_conformers = [conformers[i] for i in centers]
        choise_conformers_energy = [energies[i] for i in centers]
        # choise_conformers_label = [labels[i] for i in centers]

        return choise_conformers, choise_conformers_energy

    def make_dendrogram(self, rmsd_mtx, n_clusters=10):  
        upper_triangle_values = rmsd_mtx[np.triu_indices(rmsd_mtx.shape[0], k=1)]  # k=1 で対角成分を除外
        # 階層クラスタリングの実行 (Ward法)
        linkage_matrix = sch.linkage(upper_triangle_values, method='ward')
        
        # クラスタを割り当て
        labels = sch.fcluster(linkage_matrix, n_clusters, criterion='maxclust')
        
        # 各クラスタの中心点を選ぶ（クラスタごとに最も代表的な点を選択）
        cluster_centers = []
        for cluster_id in range(1, n_clusters + 1):
            cluster_indices = np.where(labels == cluster_id)[0]  # クラスタに属する点のインデックス
            
            # クラスタ内の中心を決める（合計距離が最小の点）
            intra_cluster_distances = rmsd_mtx[cluster_indices][:, cluster_indices].sum(axis=1)
            representative_index = cluster_indices[np.argmin(intra_cluster_distances)]  # 最も中心的な点
            cluster_centers.append(representative_index)
        
        return cluster_centers, labels

    def calc_rmsd_matrix(self, conformers):
        rmsd_matrix = np.zeros((len(conformers), len(conformers)))
        for i in range(len(conformers)):
            for j in range(i):
                rmsd_matrix[i, j] = rmsd_matrix[j, i] = AlignMol(conformers[i], conformers[j])
    
        return rmsd_matrix

    # from https://github.com/hesther/espsim/blob/master/espsim/electrostatics.py
    def ConstrainedEmbedMultipleConfs(self,
                                    mol,
                                    core,
                                    numConfs = 10,
                                    useTethers=True,
                                    coreConfId = -1,
                                    randomSeed = 2342,
                                    getForceField = UFFGetMoleculeForceField,
                                    **kwargs,
    ):
        """
        Function to obtain multiple constrained embeddings per molecule. This was taken as is from:
        from https://github.com/rdkit/rdkit/issues/3266
        :param mol: RDKit molecule object to be embedded.
        :param core: RDKit molecule object of the core used as constrained. Needs to hold at least one conformer coordinates.
        :param numCons: Number of conformations to create
        :param useTethers: (optional) boolean whether to pull embedded atoms to core coordinates, see rdkit.Chem.AllChem.ConstrainedEmbed
        :param coreConfId: (optional) id of the core conformation to use
        :param randomSeed: (optional) seed for the random number generator
        :param getForceField: (optional) force field to use for the optimization of molecules
        :return: RDKit molecule object containing the embedded conformations.
        """

        confs = self.conf['AAScore']['conf_minimize_threshold']
        total_num = confs['total_num']
        energyTol = confs['energyTol']
        forceTol = confs['forceTol']

        match = mol.GetSubstructMatch(core)
        if not match:
            raise ValueError("molecule doesn't match the core")
        coordMap = {}
        coreConf = core.GetConformer(coreConfId)
        for i, idxI in enumerate(match):
            corePtI = coreConf.GetAtomPosition(i)
            coordMap[idxI] = corePtI

        cids = EmbedMultipleConfs(mol, numConfs=numConfs, coordMap=coordMap, randomSeed=randomSeed, **kwargs)
        cids = list(cids)
        if len(cids) == 0:
            raise ValueError('Could not embed molecule.')

        algMap = [(j, i) for i, j in enumerate(match)]

        energies = []
        
        if not useTethers:
            # clean up the conformation
            for cid in cids:
                ff = getForceField(mol, confId=cid)
                for i, idxI in enumerate(match):
                    for j in range(i + 1, len(match)):
                        idxJ = match[j]
                        d = coordMap[idxI].Distance(coordMap[idxJ])
                        ff.AddDistanceConstraint(idxI, idxJ, d, d, 100.)
                ff.Initialize()
                n = total_num
                more = ff.Minimize()
                while more and n:
                    more = ff.Minimize()
                    n -= 1
                # rotate the embedded conformation onto the core:
                rms = AlignMol(mol, core, atomMap=algMap)
        else:
            # rotate the embedded conformation onto the core:
            for cid in cids:
                rms = AlignMol(mol, core, prbCid=cid, atomMap=algMap)
                ff = getForceField(mol, confId=cid)
                conf = core.GetConformer()
                for i in range(core.GetNumAtoms()):
                    p = conf.GetAtomPosition(i)
                    pIdx = ff.AddExtraPoint(p.x, p.y, p.z, fixed=True) - 1
                    ff.AddDistanceConstraint(pIdx, match[i], 0, 0, 100.)
                ff.Initialize()
                n = total_num
                more = ff.Minimize(energyTol=energyTol, forceTol=forceTol)
                while more and n:
                    more = ff.Minimize(energyTol=energyTol, forceTol=forceTol)
                    n -= 1
                # realign
                rms = AlignMol(mol, core, prbCid=cid, atomMap=algMap)
                energy = ff.CalcEnergy()
                energies.append(energy)
        return mol, energies

    def add_charge(self, conformers_path):
        conformers = sorted(glob(os.path.join(conformers_path + '*', 'conformers_*.mol2')))
        for conf in conformers:
            # for debug->pdb,mol2は不要だよね。
            # obabel -ipdb ${input}.pdb -opdb -O ${output}.pdb -ph 7.4
            subprocess.run(['obabel', '-imol2', conf,'-omol2', '-O', conf, '-ph', '7.4'])
            # rdkitではmol2扱いにくいのでpdbにしておく 
            subprocess.run(['obabel', '-imol2', conf,'-opdb', '-O', conf.replace('mol2','pdb')])

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

    def run(self):
        self.aascore_outdir = self.aascore_output_dirname
        os.makedirs(self.aascore_outdir, exist_ok = True)

        for trajectory_dir in self.gl.trajectory_dirs:
            trajectory_name, trajectory_num = _parse_trajectory_name(trajectory_dir)

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
            extract_residues_within_distance(protein_pdb, compound_pdb, pocket_pdb_path, distance=distance)

            rank_num = len(os.listdir(generate_tra_dir))
            order_scale = int(len(str(int(rank_num)))+1)
            for r in range(rank_num):
                rank = str(r).zfill(order_scale)
                self._process_rank(generate_tra_dir, aascore_tra_dir, rank, pocket_pdb_path)
            
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

        self.run_AAScore(pocket_pdb_path, lead_sdf_path, aascore_output_file_path)

    def run_AAScore(self, pocket_pdb_path, lead_sdf_path, aascore_output_file_path):
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

