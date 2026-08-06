# Hard-coded paths

Every script in this release was run against a single analysis tree at
`/home/yangyicheng/27.evolution`. Those paths are left in place so that the code here is
byte-identical to the code that produced the published numbers. Before running anything,
point them at your own copy of the data:

```bash
bash tools/set_project_root.sh /path/to/your/analysis/tree
```

The table below lists every occurrence (30 lines in 21 files) so the rewrite can be audited.

| file | line | statement |
|---|---|---|
| `01_panel_and_orthology/build_analysis_units_351.py` | 22 | `ROOT = "/home/yangyicheng/27.evolution"` |
| `02_structure_prediction/summarise_plddt.py` | 34 | `ROOT = Path("/home/yangyicheng/27.evolution")` |
| `03_mr_cr/prepare_layout.py` | 23 | `ROOT     = Path("/home/yangyicheng/27.evolution")` |
| `04_structural_partition/aggregate_core_surface_351.py` | 7 | `ROOT = "/home/yangyicheng/27.evolution"` |
| `04_structural_partition/aggregate_core_surface_351.py` | 8 | `OUTDIR = "/home/yangyicheng/27.evolution/40.whole_CR-MR"` |
| `04_structural_partition/compute_buried_fraction_351.py` | 41 | `/home/yangyicheng/mambaforge/envs/mrcr/bin/python compute_buried_fraction_351.py` |
| `04_structural_partition/compute_buried_fraction_351.py` | 58 | `ROOT = "/home/yangyicheng/27.evolution"` |
| `05_functional_sites/build_upb_masks_351.py` | 10 | `ROOT = "/home/yangyicheng/27.evolution"` |
| `05_functional_sites/build_upb_masks_351.py` | 11 | `OUT = "/home/yangyicheng/27.evolution/40.whole_CR-MR"` |
| `05_functional_sites/compute_binding_cr_351.py` | 9 | `ROOT = "/home/yangyicheng/27.evolution"` |
| `05_functional_sites/compute_binding_cr_351.py` | 10 | `OUT = "/home/yangyicheng/27.evolution/40.whole_CR-MR"` |
| `06_residue_energetics/calc_fsv.py` | 5 | `BASE_DIR = "/home/yangyicheng/27.evolution/20.AGT-RMSD-entropy"` |
| `06_residue_energetics/run_foldx_whole_protein.py` | 12 | `BASE_DIR = "/home/yangyicheng/27.evolution/20.AGT-RMSD-entropy"` |
| `08_metabolic_flux/aggregate_to_og.py` | 22 | `ROOT = Path("/home/yangyicheng/27.evolution/39.flux_CR")` |
| `08_metabolic_flux/run_native_fba.py` | 35 | `ROOT = Path("/home/yangyicheng/27.evolution/39.flux_CR")` |
| `10_biosynthetic_cost/build_cost_master.py` | 52 | `ROOT = Path("/home/yangyicheng/27.evolution")` |
| `10_biosynthetic_cost/compute_multi_gem_costs.py` | 21 | `PY=/home/yangyicheng/mambaforge/envs/orthofinder/bin/python   # or py12` |
| `10_biosynthetic_cost/compute_multi_gem_costs.py` | 48 | `GEM_DIR = Path("/home/yangyicheng/27.evolution/17.flux_CR/08.models")` |
| `10_biosynthetic_cost/fix_og_namespace.py` | 23 | `ROOT = Path("/home/yangyicheng/27.evolution")` |
| `11_composition/compute_aa_enrichment.py` | 18 | `ROOT = Path("/home/yangyicheng/27.evolution")` |
| `11_composition/compute_residue_properties.py` | 34 | `ROOT = Path("/home/yangyicheng/27.evolution")` |
| `12_robustness_tests/compute_strain_mean_CR_351.py` | 53 | `ROOT = Path("/home/yangyicheng/27.evolution")` |
| `12_robustness_tests/test_rc_surface_ceiling.py` | 35 | `ROOT = Path("/home/yangyicheng/27.evolution")` |
| `lib/cyano_cost_lib.py` | 12 | `ROOT = Path("/home/yangyicheng/27.evolution/38.protein_cost")` |
| `lib/cyano_cost_lib.py` | 16 | `DATA_ROOT = Path("/home/yangyicheng/27.evolution/00.data")` |
| `lib/cyano_cost_lib.py` | 18 | `ABUNDANCE = Path("/home/yangyicheng/27.evolution/19.pepMutic_CR/abundance_merged.csv")` |
| `lib/cyano_cost_lib.py` | 20 | `CORE_SURFACE_DIR = Path("/home/yangyicheng/27.evolution/15.coresiteAminoAcid/02.plot2/101.CRcore_surfacee")` |
| `lib/plot_mr_cr_jointplot.py` | 183 | `parser.add_argument("--dir", default="/home/yangyicheng/27.evolution/02.alphafold3/04.single_mrcr_1112/Result",` |
| `lib/residue_cr_accessibility.py` | 49 | `ROOT = "/home/yangyicheng/27.evolution"` |
| `lib/residue_cr_accessibility.py` | 53 | `MAFFT = "/home/yangyicheng/mambaforge/envs/orthofinder/bin/mafft"` |
