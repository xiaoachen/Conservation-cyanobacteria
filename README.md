# Code for *Functional geometry complements metabolic demand in shaping the evolution of photosynthetic molecular machines*

Analysis code for a structural-evolution survey of 36 cyanobacterial genomes: 351 gene families, 12,017 AlphaFold 3 models, and a per-residue structural conservation ratio crossed with metabolic demand, spatial organisation, substitution energetics and biosynthetic cost.

This repository holds the **analysis** code. Figure drawing and manuscript-table assembly are not included; they are distributed with the paper as Supplementary Code.

## Layout

Directories run in the order they are numbered; each is one stage of the pipeline.

```
lib/                     modules imported by several stages; put on PYTHONPATH
01_panel_and_orthology/  Genome panel, OrthoFinder orthogroups and the 351 analysis units.
02_structure_prediction/ AlphaFold 3 prediction of every gene in the panel, and the confidence summary.
03_mr_cr/                Reference selection and the pairwise structural superposition that defines MR and CR.
04_structural_partition/ Solvent burial and secondary structure, and CR recomputed within each partition.
05_functional_sites/     UniProt functional-site transfer and CR at annotated positions.
06_residue_energetics/   FoldX saturation mutagenesis: per-residue ddG and fraction of surviving variants.
07_positional_geometry/  Per-residue positional rigidity and the geometry of functional-site constellations.
08_metabolic_flux/       Parsimonious FBA over six genome-scale models, mapped to orthogroups.
09_diel_demand/          Diel amplitude and mean magnitude against conservation, five datasets.
10_biosynthetic_cost/    Per-amino-acid biosynthetic cost, protein-level cost and the family cost table.
11_composition/          Amino-acid composition, module enrichment and residue-property correlations.
12_robustness_tests/     Controls: per-genome conservation, the antenna divergence confound, the ceiling tests.
```

## Pipeline

### panel and orthology

Genome panel, OrthoFinder orthogroups and the 351 analysis units.

| script | what it does |
|---|---|
| `build_analysis_units_351.py` | Assembles the whole-project panorama: collects every computed unit across the four prediction trees, applies the canonical aggregation, attaches pathway labels and the single-copy core flag, and writes the 351-unit master table. |

### structure prediction

AlphaFold 3 prediction of every gene in the panel, and the confidence summary.

| script | what it does |
|---|---|
| `run_alphafold3_batch.py` *(not in the Supplementary Code set)* | Drives AlphaFold 3 over the panel: writes one single-chain ligand-free JSON per sequence, distributes jobs over four GPUs with one concurrent job each, skips existing output directories so the run is resumable, and records sequences above 3,000 residues as not predicted. |
| `summarise_plddt.py` | Traverses every model predicted for the panel and computes the confidence summaries: residue-level pLDDT distribution, per-model means, and the per-strain medians used to show that model quality does not track phylogenetic position, genome size or habitat. |

### mr cr

Reference selection and the pairwise structural superposition that defines MR and CR.

| script | what it does |
|---|---|
| `prepare_layout.py` *(not in the Supplementary Code set)* | Lays out one directory per analysis unit and selects each unit's reference structure by the three-tier rule (PCC 6803, else BP-1, else the member with the most resolved Ca atoms), writing the choice and its tier to alt_ref_map.tsv. |
| `mrcr_one.sh` *(not in the Supplementary Code set)* | Per-target worker invoked by the pairwise driver: activates the environment, passes the reference chosen by prepare_layout.py explicitly, and skips targets already computed so the run is resumable. |
| `compute_mr_cr_ChimeraX.py` | The core measurement. Superposes each orthologue on the reference with ChimeraX matchmaker, matches Ca atoms one-to-one within 2.0 A, applies the pLDDT >= 70 mask to both structures, and returns the mapping ratio and the conservation ratio per pair. |

### structural partition

Solvent burial and secondary structure, and CR recomputed within each partition.

| script | what it does |
|---|---|
| `aggregate_core_surface_351.py` | Aggregate existing per-target core/surface CR (Result_Structure/*_mrcr_results.csv) to the 351 panorama families. complete_CR = panorama whole-protein CR. Reports coverage per category. |
| `compute_buried_fraction_351.py` | Buried-residue fraction for every reference structure in the 351-family panel. |

### functional sites

UniProt functional-site transfer and CR at annotated positions.

| script | what it does |
|---|---|
| `build_functional_sites.py` | Independent functional-site annotations per enzyme from UniProt (CR-independent), for all 35 genes. For each gene: query reviewed cyanobacterial (taxon 1117) entries, pick the candidate whose sequence best matches our FoldX reference, align it to the reference, and map UniProt functional sites (Active site / Binding site / Metal binding / Site) onto our reference residue numbering. |
| `build_upb_masks_351.py` | Step B1 (fast, no ChimeraX): for each UPb-covered gene-named family, pick the reference AF3 structure, extract its sequence+resids, fetch reviewed-cyano UniProt sites, align, and map sites onto reference residue numbering. Writes upb_sites_351.csv (unit, ref_path, ResID, is_site) + upb_sites_summary_351.csv. Run in mrcr env (Bio.Align, gemmi). |
| `compute_binding_cr_351.py` | Step B2 (heavy, ChimeraX): for each UPb-covered family, compute MR/CR for four masks {complete, core, surface, binding} against every member, using the canonical structure_mr_cr pipeline (2.0 A Ca mapping, RSA<0.25 = core). Reference + core/surface reproduced here so all four categories are self-consistent. Checkpointed (skips existing outputs). Run in mrcr env. |

### residue energetics

FoldX saturation mutagenesis: per-residue ddG and fraction of surviving variants.

| script | what it does |
|---|---|
| `run_foldx_whole_protein.py` | Drives FoldX 5: RepairPDB with water ignored, then PositionScan over every residue and all 19 alternative amino acids. |
| `calc_fsv.py` | Computes the fraction of surviving variants at the 2 kcal/mol threshold, and at 1 and 3 kcal/mol for the threshold robustness check. |

### positional geometry

Per-residue positional rigidity and the geometry of functional-site constellations.

| script | what it does |
|---|---|
| `functional_constellation_rigidity.py` | Functional-constellation geometric rigidity (generalisable P) — all genes with reliable UniProt sites. |

### metabolic flux

Parsimonious FBA over six genome-scale models, mapped to orthogroups.

| script | what it does |
|---|---|
| `run_native_fba.py` | Runs parsimonious FBA on the six genome-scale models under light and dark, with the per-model photon lower bounds and the closure of heterotrophic carbon uptake applied as documented in the script. |
| `aggregate_to_og.py` | Collapses the per-reaction flux table from (strain, model, orthogroup, reaction, gene) granularity to one row per orthogroup, deriving light flux, dark flux, diel amplitude and mean magnitude, per model and as a cross-model consensus. |

### diel demand

Diel amplitude and mean magnitude against conservation, five datasets.

| script | what it does |
|---|---|
| `diel_build_and_analyze_knoop2013.py` | Builds the curated diel flux series of Knoop et al. 2013 for PCC 6803 and tests diel amplitude and mean magnitude against conservation. |

### biosynthetic cost

Per-amino-acid biosynthetic cost, protein-level cost and the family cost table.

| script | what it does |
|---|---|
| `compute_cyano_costs.py` | Computes per-amino-acid costs on iJN678: at fixed biomass flux, adds 1 mmol gDW^-1 h^-1 through the amino-acid sink and subtracts the growth-only baseline at the same growth rate, giving the five model-derived cost metrics. |
| `compute_multi_gem_costs.py` | Multi-GEM amino-acid cost calculation (extends single-iJN678 pipeline). |
| `build_cost_master.py` | Build the R6 master table: biosynthetic cost and conservation on the same protein set. |
| `fix_og_namespace.py` *(not in the Supplementary Code set)* | Corrects an orthogroup-namespace error in the cost table: the 222 single-copy core families kept identifiers from the intermediate clustering while the photosynthetic families were re-keyed to the final assignment, so costs had been looked up against the wrong families. Rebuilds the translation from sequence membership and recomputes. |

### composition

Amino-acid composition, module enrichment and residue-property correlations.

| script | what it does |
|---|---|
| `compute_aa_enrichment.py` | Amino-acid composition of each module class against the non-photosynthetic background. |
| `compute_residue_properties.py` | Which residues are over-used by the more conserved families, and where in the structure. |

### robustness tests

Controls: per-genome conservation, the antenna divergence confound, the ceiling tests.

| script | what it does |
|---|---|
| `compute_strain_mean_CR_351.py` | Per-genome mean conservation ratio computed on the 351 analysis units. |
| `check_divergence_confound.py` | Does divergence regression explain the antenna's negative cost-CR slope? |
| `test_rc_surface_ceiling.py` | Is the reaction-centre surface more conserved than its own overall CR already implies? |

### Shared modules (`lib/`)

Each of these is both importable and runnable on its own.

| module | what it does |
|---|---|
| `structure_mr_cr_ChimeraX.py` | The class-resolved pairwise measurement: the same superposition as compute_mr_cr, but MR and CR returned separately for buried core, surface, helix, extended strand and coil. Also supplies the mmCIF reader and the maximum-solvent-accessibility table used by the residue-level stages. |
| `residue_cr_accessibility.py` | Residue-level quantification: mutational ACCESSIBILITY (FoldX) vs CONSERVATION. |
| `add_lrmsd.py` | Per-residue lRMSD (structural drift across orthologs) for all 35 genes — 37's structural dimension, brought to residue level. For each reference residue: RMS of its Cα deviation across ortholog AF3 models, after Kabsch superposition using the MSA-derived residue correspondence. |
| `plot_mr_cr_jointplot.py` *(not in the Supplementary Code set)* | Supplies load_mr_cr_from_dir, the canonical aggregation from per-pair rows to one MR and one CR per analysis unit: orthogroup-median and invalid rows dropped, ratios required to lie in [0, 1], a pair kept only when the mapped length reaches 30% of the reference length, then the mean over surviving pairs. |
| `cyano_cost_lib.py` *(not in the Supplementary Code set)* | Shared constants for the cost calculations: the amino-acid to metabolite mapping used to place the demand flux, residue molecular weights, canonical residue order and the model file locations. |

## Orthology

Orthogroups were inferred outside this repository with a single OrthoFinder call:

```bash
orthofinder -f protein_sequences_folder -t 144 -a 36 -M msa -T iqtree3
```

OrthoFinder v3.1.0, with DIAMOND for the all-versus-all search, FAMSA for alignment and IQ-TREE 3 for gene and species trees. `01_panel_and_orthology/build_analysis_units_351.py` starts from its `Orthogroups.tsv`.

## Running it

```bash
# 1. environment
conda env create -f environment.yml          # statistics, metabolic models
conda env create -f environment-mrcr.yml     # structural stages

# 2. shared modules on the path
source setup_env.sh

# 3. point the scripts at your copy of the data
bash tools/set_project_root.sh /path/to/your/analysis/tree
```

Every script carries the absolute path of the tree it was run against. They are left in place so that this code is byte-identical to the code that produced the published numbers; `PATHS.md` lists every such line, and the helper above rewrites them all.

### Software that must be installed separately

| tool | version | needed by |
|---|---|---|
| AlphaFold 3 | released weights | `02_structure_prediction` |
| UCSF ChimeraX | 1.7.1 | `03_mr_cr`, `04_structural_partition` |
| FoldX | 5 | `06_residue_energetics` |
| OrthoFinder | 3.1.0 | orthology, run outside this repository |

AlphaFold 3 weights, ChimeraX and FoldX are obtained under their own licences.

## Data

Input structures and intermediate tables are not in this repository. The processed tables that every analysis here consumes or produces are published as Supplementary Data with the paper (52 files, workflow-ordered, MD5-verified). Genomes are the NCBI RefSeq protein sets of the 36 assemblies listed in Supplementary Table 1.

## Provenance

`MANIFEST.tsv` lists all 31 files with their MD5 and their Supplementary Code identifier. Files marked *not in the Supplementary Code set* are upstream steps added here so the chain from genome to conservation ratio can be followed end to end: the AlphaFold 3 driver, the reference-selection rule, the pairwise driver, the orthogroup namespace correction and two shared modules.

Scripts here keep their original module names; the submission set prefixes them `SC##_`, which breaks the imports between them. Otherwise the two sets are the same bytes, as `MANIFEST.tsv` records.

## Citing

Please cite the paper. This code is released under the MIT License; see `LICENSE`.
