import os
import sys
import shutil
import glob
import subprocess
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from Bio.PDB import MMCIFParser, PDBIO, PDBParser
from Bio.SeqUtils import seq1

# Configuration
BASE_DIR = "/home/yangyicheng/27.evolution/20.AGT-RMSD-entropy"
FOLDX_BIN = os.path.join(BASE_DIR, "03.FoldX/foldx_20270131")
OUTPUT_DIR = os.path.join(BASE_DIR, "05.whole_prot_foldx")

# Protein Information - removed entropy_file as we scan everything
PROTEINS = {
    'psbA': {'cif_path': os.path.join(BASE_DIR, "01.PSII------/00.data/psbA/WP_006101755.1_GCF_000155555/wp_006101755.1_gcf_000155555/seed-1_sample-0/model.cif")},
    'psbC': {'cif_path': os.path.join(BASE_DIR, "01.PSII------/00.data/psbC/WP_002739004.1_GCF_001264245/wp_002739004.1_gcf_001264245/seed-1_sample-0/model.cif")},
    'psbD': {'cif_path': os.path.join(BASE_DIR, "01.PSII------/00.data/psbD/WP_006100727.1_GCF_000155555/wp_006100727.1_gcf_000155555/seed-1_sample-0/model.cif")},
    'petC': {'cif_path': os.path.join(BASE_DIR, "02.Plastocyanin/petC/WP_017285669.1_GCF_002142495/wp_017285669.1_gcf_002142495/seed-1_sample-2/model.cif")},
    'petE': {'cif_path': os.path.join(BASE_DIR, "02.Plastocyanin/petE/WP_023175298.1_GCF_000484535/wp_023175298.1_gcf_000484535/seed-1_sample-2/model.cif")},
    'petJ': {'cif_path': os.path.join(BASE_DIR, "02.Plastocyanin/petJ/WP_011318044.1_GCF_000204075/wp_011318044.1_gcf_000204075/seed-1_sample-1/model.cif")},
}

def convert_cif_to_pdb(cif_path, pdb_path):
    print(f"Converting {cif_path} to PDB...")
    try:
        parser = MMCIFParser(QUIET=True)
        structure = parser.get_structure('struct', cif_path)
        io = PDBIO()
        io.set_structure(structure)
        io.save(pdb_path)
        return True
    except Exception as e:
        print(f"Error converting CIF to PDB: {e}")
        return False

def run_repair(gene, work_dir, pdb_path):
    # Copy PDB to work dir if needed
    local_pdb = os.path.join(work_dir, os.path.basename(pdb_path))
    if os.path.abspath(pdb_path) != os.path.abspath(local_pdb):
        shutil.copy(pdb_path, local_pdb)
        
    pdb_name = os.path.basename(local_pdb)
    
    # Check if repair already done
    repaired_1 = "Repair_" + pdb_name
    repaired_2 = pdb_name.replace(".pdb", "_Repair.pdb")
    
    if os.path.exists(os.path.join(work_dir, repaired_1)):
        return repaired_1
    if os.path.exists(os.path.join(work_dir, repaired_2)):
        return repaired_2
        
    print(f"Running RepairPDB for {gene}...")
    cmd = [FOLDX_BIN, "--command=RepairPDB", f"--pdb={pdb_name}", "--water=IGNORE"]
    try:
        subprocess.run(cmd, cwd=work_dir, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError as e:
        print(f"RepairPDB failed for {gene}: {e}")
        return None
        
    if os.path.exists(os.path.join(work_dir, repaired_1)):
        return repaired_1
    elif os.path.exists(os.path.join(work_dir, repaired_2)):
        return repaired_2
    return None

def get_residue_mapping(pdb_path):
    # Map ResID to AA code
    mapping = {}
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('struct', pdb_path)
    model = structure[0]
    chain = list(model.get_chains())[0] # Assume first chain
    chain_id = chain.id
    
    for res in chain:
        if res.id[0] == ' ': # Standard residue
            try:
                aa = seq1(res.get_resname())
                mapping[res.id[1]] = aa
            except:
                pass
    return mapping, chain_id

def process_gene(gene):
    info = PROTEINS[gene]
    work_dir = os.path.join(OUTPUT_DIR, gene)
    os.makedirs(work_dir, exist_ok=True)
    
    # 1. CIF to PDB
    pdb_path = os.path.join(work_dir, f"{gene}.pdb")
    if not os.path.exists(pdb_path):
        if not convert_cif_to_pdb(info['cif_path'], pdb_path):
            return f"{gene}: Failed CIF conversion"
            
    # 2. Repair PDB
    repaired_pdb = run_repair(gene, work_dir, pdb_path)
    if not repaired_pdb:
        return f"{gene}: Failed Repair"
        
    # 3. Get Mapping & Targets (ALL residues)
    mapping, chain_id = get_residue_mapping(os.path.join(work_dir, repaired_pdb))
    targets = sorted(list(mapping.keys()))
    
    # 4. Check completed
    completed = set()
    for f in glob.glob(os.path.join(work_dir, "energies_*_*.txt")):
        try:
            # energies_155_psbA_Repair.txt
            parts = os.path.basename(f).split('_')
            resid = int(parts[1])
            completed.add(resid)
        except:
            pass
            
    remaining = [r for r in targets if r not in completed]
    if not remaining:
        return f"{gene}: All {len(targets)} residues completed."
        
    # 6. Build positions arg
    pos_list = []
    for rid in remaining:
        if rid in mapping:
            aa = mapping[rid]
            # Format: <WT><CHAIN><RES>a
            pos_list.append(f"{aa}{chain_id}{rid}a")
            
    if not pos_list:
        return f"{gene}: No valid mapping for remaining residues."
        
    # 7. Run PositionScan
    # For whole protein, the list might be too long for a single command line call on some systems.
    # But usually Linux supports very long args. 
    # Max residues ~350 (psbC), string length ~350 * 5 = 1750 chars. This is fine.
    
    print(f"{gene}: Running PositionScan for {len(pos_list)} residues...")
    
    positions_arg = ",".join(pos_list)
    
    cmd = [FOLDX_BIN, "--command=PositionScan", f"--pdb={repaired_pdb}", f"--positions={positions_arg}"]
    try:
        subprocess.run(cmd, cwd=work_dir, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # Cleanup
        generated_pdbs = glob.glob(os.path.join(work_dir, "*_*_*.pdb"))
        cleaned = 0
        for gp in generated_pdbs:
            if os.path.basename(gp) == repaired_pdb: continue
            try:
                os.remove(gp)
                cleaned += 1
            except:
                pass
        return f"{gene}: Completed {len(pos_list)} residues. Cleaned {cleaned} PDBs."
        
    except subprocess.CalledProcessError as e:
        return f"{gene}: PositionScan failed: {e}"

def main():
    print(f"Starting Whole Protein FoldX Analysis in {OUTPUT_DIR}")
    # Reduce workers if needed, FoldX can be CPU intensive
    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = {executor.submit(process_gene, gene): gene for gene in PROTEINS}
        
        for future in as_completed(futures):
            gene = futures[future]
            try:
                result = future.result()
                print(result)
            except Exception as e:
                print(f"{gene} generated an exception: {e}")

if __name__ == "__main__":
    main()
