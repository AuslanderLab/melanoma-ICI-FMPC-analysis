# melanome-ICI-FMPC-analysis

This repository contains supplemental data and analysis source code for Taxonomy-free fecal microbial predictors of immune checkpoint inhibitor benefit and adverse events in melanoma [ref]. 

### Repo structure
- `data/`: Contains clinical and metagenomic summary data used for analysis, which were published as supplementary data files in the original paper
- `src/`: Contains source code for major reported analyses
- `results/`: Scratch directory that `src/` writes to if code is rerun

### Data files

The following files are included in the `data/` folder.

- `supp_dataset_1.*` : Clinical information for publicly available cohorts utilized in this study, Pittsburgh and New York (training), Dallas (validation) and Houston (test).
- `supp_dataset_2.csv` : Clinical information for the RadVax cohort (this study)
- `supp_dataset_3.*` : FMPC database providing protein IDs, cluster information and functional description for each FMPC 
- `supp_dataset_4.*` : Normalized FMPC counts for the training and validation cohorts
- `supp_dataset_5.csv` : AUROCs assigned to each FMPC for predicting immunotherapy response in the training and validation cohorts
- `supp_dataset_6.csv` : Permutation testing results for the 49 selected FMPCs for prediction of immunotherapy response based on the training and validation cohorts.
- `supp_dataset_7.csv` : Houston test dataset (for immunotherapy response prediction) normalized FMPC abundances for the 49 FMPCs tested for response prediction
- `supp_dataset_8.csv` : Immunotherapy response prediction AUROCs across the cohorts for the 49 selected response-predictive FMPCs
- `supp_dataset_9.csv` : AUROCs predicting irAE for each FMPC in the Pittsburgh cohort
- `supp_dataset_10.csv` : RadVax prospective testing cohort (for irAE prediction) normalized abundances of 5 tested irAE-predictive FMPCs 
- `supp_dataset_11.csv` : AUROCs for irAE prediction for the 5 selected irAE FMPCs across Pittsburgh and RadVax cohorts, with adjustment for covariates.
- `supp_dataset_12.txt` : Assembled regions of bacteria encoding the identified 4Fe-4S irAE predictive FMPCs
- `supp_dataset_13.csv` : HLA typing of patients in the RadVax cohort
- `supp_dataset_14.csv` : Peptide MHC binding prediction results for the RadVax cohort with paired HLA typing
- `supp_dataset_15.txt` : 4Fe-4S DNA sequences derived for quantification as a proxy for a qPCR test

This study also uses publicly available metagnomic data from the following studies: PRJNA762360, PPRJNA41981, PRJNA397906, PRJEB22893

### Code & scripts

The following scripts are included in the `src/` folder. Scripts which require additional resources not provided in this repository (e.g. fasta files) are denoted

Scripts can be run in the order specified

- `01_process_data.py` : Creates normalized FMPC count table from BlastX output *requires BlastX output & files containing number of lines in the fasta file for each sample
- `02_compute_aucs.py` : Computes AUROCs for normalized count matrices. The script assumes the matrices have been subsetted by cohort
- `03_filter_aucs.R` : Extracts signigicant FMPCs for downstream analysis
- `04_train_mrs.R` : Trains MRS betas
- `05_validate_mrs.R` : Validates MRS model
- `06_multi_FMPC_model_train.py` : Training/feature selection for the multi-FMPC model
- `07_multi_FMPC_model_validate.py` : Validation & exploratory analysis of the muti-FMPC model
- `08_extract_peptides.py` : Extract peptides from SPAdes contigs *expects sample level SPAdes contigs as input

Additional code notes:

The original database used to cluster `WP_*` proteins into FMPCs can be constructed using the following commands
```
wget 'ftp://ftp.ncbi.nlm.nih.gov/blast/db/nr.*.tar.gz'
cat nr.*.tar.gz | tar -zxvi -f - -C .
blastdbcmd -db path_to_nr_db/nr -taxids 2 -outfmt %f -out fasta_file_name.(fa or fasta)

# Note: path_to_nr_db/ is where the .tar.gz files are extracted.
```



