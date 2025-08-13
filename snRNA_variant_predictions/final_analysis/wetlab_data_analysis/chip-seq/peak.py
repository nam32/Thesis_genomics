import pandas as pd
import glob
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
import pyranges as pr

# Load metadata
metadata = pd.read_csv("/storage/research/dbmr_rubin_lab/projects/minor_splicing_anke/chip-seq_histone_polymerase/files_summary_new.tsv", sep="\t")

# Build metadata dictionary (key = file_id)
metadata_dict = {}
for idx, row in metadata.iterrows():
    file_id = row['file_id']
    metadata_dict[file_id] = {
        'Target': row['Target of assay'],
        'CellLine': row['Biosample term name'],
        'Condition': row['Biosample summary'],
        'BiosampleClassification': row['Biosample classification'],
        'LinkedAntibody': row['Linked antibody']
    }

# Load signal matrix .tab files
his_signal_matrix_dir = "/storage/research/dbmr_rubin_lab/projects/minor_splicing_anke/chip-seq_analysis_histones"
pol_signal_matrix_dir = "/storage/research/dbmr_rubin_lab/projects/minor_splicing_anke/chip-seq_analysis_pol"
his_tab_files = glob.glob(os.path.join(his_signal_matrix_dir, "*_signal_matrix.tab"))
pol_tab_files = glob.glob(os.path.join(pol_signal_matrix_dir, "*_signal_matrix.tab"))

all_data = []

# Combine both histone and pol files
for tab_file in his_tab_files + pol_tab_files:
    accession = os.path.basename(tab_file).replace("_snRNA_signal_matrix.tab", "")
    
    if accession not in metadata_dict:
        print(f"⚠️ Warning: {accession} not found in metadata!")
        continue

    info = metadata_dict[accession]

    df = pd.read_csv(tab_file, sep="\t")
    df.columns = ['chr', 'start', 'end', 'signal']

    # Add metadata
    df['mark'] = info['Target']
    df['cell_line'] = info['CellLine']
    df['condition'] = info['Condition']
    df['biosample_classification'] = info['BiosampleClassification']
    df['linked_antibody'] = info['LinkedAntibody']
    df['accession'] = accession
    df['tab_file'] = os.path.basename(tab_file)
    df['chip_type'] = 'histone' if tab_file in his_tab_files else 'pol'

    all_data.append(df)

# Combine all DataFrames
if all_data:
    full_df = pd.concat(all_data, ignore_index=True)
    print("✅ Successfully merged histone and pol signal matrices!")
else:
    raise ValueError("❌ No valid data loaded — check filenames or metadata!")

full_df.to_csv("./all_chip_seq_data.csv", index=False)

# Load ChIP-seq signal data
full_df = pd.read_csv("./all_chip_seq_data.csv")

# Define BED file paths for each chip type
chip_beds = {
    "histone": "/storage/homefs/tj23t050/snRNA_variant_predictions/results/0_results/rfam_ensembl_infernal/final_snRNA_combined.TSSm500_TESp0.bed",
    "pol": "/storage/homefs/tj23t050/snRNA_variant_predictions/results/0_results/rfam_ensembl_infernal/final_snRNA_combined.TSSm100_TESp20.bed"
}

# Prepare signal PyRanges once
signal_ranges = pr.PyRanges(full_df.rename(columns={"chr": "Chromosome", "start": "Start", "end": "End"}))
for chip_type, bed_path in chip_beds.items():
    # Filter signal dataframe by chip type
    signal_df = full_df[full_df['chip_type'] == chip_type].copy()

    # Prepare PyRanges for just this chip type
    signal_ranges = pr.PyRanges(signal_df.rename(columns={"chr": "Chromosome", "start": "Start", "end": "End"}))

    # Load BED...
    snrna_bed = pd.read_csv(bed_path, sep="\t", header=None)
    snrna_bed.columns = ["Chromosome", "Start", "End", "Name", "Score", "Strand"]
    snrna_bed["Chromosome"] = snrna_bed["Chromosome"].astype(str).apply(lambda x: "chr" + x if not x.startswith("chr") else x)

    # Create PyRanges object
    snrna_ranges = pr.PyRanges(snrna_bed[["Chromosome", "Start", "End", "Name", "Strand"]])

    # Join
    overlap = signal_ranges.join(snrna_ranges)
    overlap_df = overlap.df
    overlap_df.to_csv(f"signal_mapped_to_snRNA_{chip_type}.csv", index=False)

### FOR TRANSCRIPTION FACTORS SNAP ###
import pandas as pd
import glob
import os

# Load metadata
metadata = pd.read_csv("/storage/research/dbmr_rubin_lab/projects/minor_splicing_anke/chip-seq_tf/encode_files_summary.csv")

# Build metadata dictionary using file_id as key
metadata_dict = {}
for _, row in metadata.iterrows():
    file_id = row["file_id"]
    metadata_dict[file_id] = {
        "Target": row["Target of assay"],
        "CellLine": row["Biosample term name"],
        "Condition": row["Biosample summary"],
        "BiosampleClassification": row["Biosample classification"],
        "LinkedAntibody": row["Linked antibody"]
    }

# Load signal matrix .tab files
tf_signal_matrix_dir = "/storage/research/dbmr_rubin_lab/projects/minor_splicing_anke/chip-seq_analysis_tf_bigwig"
tf_tab_files = glob.glob(os.path.join(tf_signal_matrix_dir, "*_snRNA_signal_matrix.tab"))

all_data = []

# Process each file
for tab_file in tf_tab_files:
    file_id = os.path.basename(tab_file).replace("_snRNA_signal_matrix.tab", "")

    if file_id not in metadata_dict:
        print(f"⚠️ Warning: {file_id} not found in metadata!")
        continue

    info = metadata_dict[file_id]

    df = pd.read_csv(tab_file, sep="\t", names=['chr', 'start', 'end', 'signal'])

    # Add metadata
    df['mark'] = info['Target']
    df['cell_line'] = info['CellLine']
    df['condition'] = info['Condition']
    df['biosample_classification'] = info['BiosampleClassification']
    df['linked_antibody'] = info['LinkedAntibody']
    df['file_id'] = file_id
    df['tab_file'] = os.path.basename(tab_file)
    df['chip_type'] = "tf"

    all_data.append(df)

# Combine and save
if all_data:
    full_df = pd.concat(all_data, ignore_index=True)
    print("✅ Successfully merged TF ChIP-seq signal matrices.")
    full_df.to_csv("./tf_chip_signal_summary.csv", index=False)
else:
    raise ValueError("❌ No matching metadata found — check file IDs.")


# === Load TF ChIP-seq merged signal matrix ===
full_df = pd.read_csv("/storage/homefs/tj23t050/snRNA_variant_predictions/scripts/1_chip-seq_analysis/tf_chip_signal_summary.csv")

# === Standardise coordinate column names ===
full_df = full_df.rename(columns={"chr": "Chromosome", "start": "Start", "end": "End"})

# === Filter out any non-numeric Start/End values (e.g. 'start' or malformed entries) ===
full_df = full_df[full_df["Start"].astype(str).str.isnumeric()]
full_df = full_df[full_df["End"].astype(str).str.isnumeric()]

# === Convert to correct data types ===
full_df["Chromosome"] = full_df["Chromosome"].astype(str)
full_df["Start"] = full_df["Start"].astype(int)
full_df["End"] = full_df["End"].astype(int)

# === Filter only TF data ===
tf_df = full_df[full_df["chip_type"] == "tf"].copy()

if tf_df.empty:
    print("⚠️ No TF ChIP-seq data available!")
else:
    print("🔎 Processing TF ChIP-seq overlaps...")

    # Convert to PyRanges
    tf_ranges = pr.PyRanges(tf_df[["Chromosome", "Start", "End"] + [c for c in tf_df.columns if c not in ["Chromosome", "Start", "End"]]])

    # Load BED file and fix chromosome naming if necessary
    tf_bed_path = "/storage/homefs/tj23t050/snRNA_variant_predictions/results/0_results/rfam_ensembl_infernal/final_snRNA_combined.TSSm70_m10.bed"
    tf_bed = pd.read_csv(tf_bed_path, sep="\t", header=None)
    tf_bed.columns = ["Chromosome", "Start", "End", "Name", "Score", "Strand"]
    tf_bed["Chromosome"] = tf_bed["Chromosome"].astype(str).apply(lambda x: "chr" + x if not x.startswith("chr") else x)

    # Create PyRanges from BED
    tf_ranges_bed = pr.PyRanges(tf_bed[["Chromosome", "Start", "End", "Name", "Strand"]])

    # Compute overlaps
    tf_overlap = tf_ranges.join(tf_ranges_bed)
    tf_overlap.df.to_csv("./signal_mapped_to_snRNA_tf.csv", index=False)
    print("✅ Saved: ./signal_mapped_to_snRNA_tf.csv")

# === Load TF metadata ===
metadata_path = "/storage/research/dbmr_rubin_lab/projects/minor_splicing_anke/chip-seq_tf/encode_files_summary.csv"
metadata = pd.read_csv(metadata_path)

# === Build metadata dictionary using file_id ===
metadata_dict = {}
for _, row in metadata.iterrows():
    file_id = row["file_id"]
    metadata_dict[file_id] = {
        "Target": row["Target of assay"],
        "CellLine": row["Biosample term name"],
        "Condition": row["Biosample summary"],
        "BiosampleClassification": row["Biosample classification"],
        "LinkedAntibody": row["Linked antibody"]
    }

# === Load TF signal matrix files ===
tf_signal_matrix_dir = "/storage/research/dbmr_rubin_lab/projects/minor_splicing_anke/chip-seq_analysis_tf_bigwig"
tf_tab_files = glob.glob(os.path.join(tf_signal_matrix_dir, "*_signal_matrix.tab"))

all_tf_data = []

# === Process each TF signal matrix ===
for tab_file in tf_tab_files:
    file_id = os.path.basename(tab_file).split("_")[0]

    if file_id not in metadata_dict:
        print(f"⚠️ Warning: {file_id} not found in metadata!")
        continue

    info = metadata_dict[file_id]
    df = pd.read_csv(tab_file, sep="\t")

    # Standardise column names
    df.columns = ['chr', 'start', 'end', 'signal']

    # Add metadata
    df['mark'] = info['Target']
    df['cell_line'] = info['CellLine']
    df['condition'] = info['Condition']
    df['biosample_classification'] = info['BiosampleClassification']
    df['linked_antibody'] = info['LinkedAntibody']
    df['file_id'] = file_id
    df['tab_file'] = os.path.basename(tab_file)
    df['chip_type'] = 'tf'

    all_tf_data.append(df)
# === Combine all TF data ===
if all_tf_data:
    tf_full_df = pd.concat(all_tf_data, ignore_index=True)
    tf_full_df.to_csv("./tf_chip_seq_data.csv", index=False)
    print("✅ Successfully saved: ./tf_chip_seq_data.csv")
else:
    raise ValueError("❌ No valid TF signal matrices loaded!")
