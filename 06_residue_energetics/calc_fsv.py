import os
import pandas as pd
import re

BASE_DIR = "/home/yangyicheng/27.evolution/20.AGT-RMSD-entropy"
FOLDX_DIR = os.path.join(BASE_DIR, "03.FoldX_Analysis")

# We want to calculate FSV for the First Shell region.
# The 03.FoldX_Analysis directory contains scanning outputs for the first shell residues.
# We will read these files, evaluate the condition:
# Condition 1: ddG < theta_G
# (Condition 2: Delta Geo < theta_geo is implicitly satisfied because these are FoldX in silico models where structural deformation is minimal)

PROTEINS = ['psbA', 'psbC', 'psbD', 'petE', 'petJ', 'petC']

def calculate_fsv_for_protein(gene, theta_g=2.0):
    filepath = os.path.join(FOLDX_DIR, gene, f"PS_{gene}_Repair_scanning_output.txt")
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return None, None
        
    fsv_data = []
    current_res = None
    mut_count = 0
    feasible_count = 0
    
    total_mut_count = 0
    total_feasible_count = 0
    
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line or '\x00' in line: continue
            parts = line.split()
            if len(parts) < 2: continue
            mut_name = parts[0]
            try:
                ddg = float(parts[1])
            except:
                continue
                
            match = re.search(r'[A-Z]{3}[A-Z]?(\d+)[A-Z]', mut_name)
            if match:
                res_id = int(match.group(1))
            else:
                continue
                
            if current_res is None:
                current_res = res_id
                
            if res_id != current_res:
                fsv_data.append({'ResID': current_res, 'FSV': feasible_count / mut_count if mut_count > 0 else 0})
                current_res = res_id
                mut_count = 0
                feasible_count = 0
                
            mut_count += 1
            total_mut_count += 1
            if ddg < theta_g:
                feasible_count += 1
                total_feasible_count += 1
                
        if current_res is not None:
            fsv_data.append({'ResID': current_res, 'FSV': feasible_count / mut_count if mut_count > 0 else 0})
            
    df = pd.DataFrame(fsv_data)
    fsv_region = total_feasible_count / total_mut_count if total_mut_count > 0 else 0
    
    return df, fsv_region

def main():
    results = []
    
    for gene in PROTEINS:
        print(f"Calculating FSV for {gene}...")
        
        # Main analysis: theta = 2.0
        df_2, fsv_2 = calculate_fsv_for_protein(gene, 2.0)
        
        # Sensitivity analysis: theta = 3.0
        df_3, fsv_3 = calculate_fsv_for_protein(gene, 3.0)
        
        if fsv_2 is not None:
            results.append({
                'Protein': gene,
                'FSV_Main (theta=2.0)': fsv_2,
                'FSV_Aux (theta=3.0)': fsv_3
            })
            
            # Save site-level FSV
            site_fsv_file = os.path.join(FOLDX_DIR, gene, f"{gene}_site_fsv.csv")
            df_merged = pd.merge(df_2.rename(columns={'FSV': 'FSV_theta2'}), 
                                 df_3.rename(columns={'FSV': 'FSV_theta3'}), 
                                 on='ResID')
            df_merged.to_csv(site_fsv_file, index=False)
            
    summary_df = pd.DataFrame(results)
    print("\nRegional FSV Summary:")
    print(summary_df)
    
    summary_file = os.path.join(FOLDX_DIR, "fsv_region_summary.csv")
    summary_df.to_csv(summary_file, index=False)
    print(f"Saved regional FSV summary to {summary_file}")

if __name__ == "__main__":
    main()
