nextflow.enable.dsl=2

// --- Configuration & Paths ---
params.input_csv   = "$baseDir/data/nadph.csv"
params.scripts_dir = "$baseDir/scripts"
params.outdir      = "$baseDir/results"

// CRITICAL: Point strictly to your 'uv' virtual environment python
params.python      = "$baseDir/.venv/bin/python"

// Training Hyperparameters
params.epochs      = 10
params.batch_size  = 8
params.task        = "classification"
params.device      = "cuda" // Change to 'cpu' if debugging without GPU

log.info """
C E T S A - N A D P H   P I P E L I N E
=======================================
Python Env   : ${params.python}
Input CSV    : ${params.input_csv}
Scripts Dir  : ${params.scripts_dir}
Output Dir   : ${params.outdir}
Device       : ${params.device}
"""

// --- Process Definitions ---

process FIT_CURVES {
    tag "Fitting EC50"
    publishDir "${params.outdir}", mode: 'copy'

    input:
    path input_csv

    output:
    path "ec50_fits.csv", emit: fits

    script:
    """
    ${params.python} ${params.scripts_dir}/01_fit_itdr_curves.py \
        --input-csv ${input_csv} \
        --out-fits ec50_fits.csv
    """
}

process HIT_CALLING {
    tag "QC & Hit Calling"
    publishDir "${params.outdir}/hit_calling", mode: 'copy'

    input:
    path fits_csv

    output:
    path "cetsa_hits_ranked.csv", emit: hits
    path "*.png"

    script:
    """
    ${params.python} ${params.scripts_dir}/02_hit_calling_and_diagnostics.py \
        --fits-csv ${fits_csv} \
        --out-dir .
    """
}

process ANNOTATE_AND_FETCH {
    tag "MyGene & UniProt"
    publishDir "${params.outdir}", mode: 'copy'

    input:
    path fits_csv

    output:
    path "protein_annotations.csv", emit: annot
    path "protein_sequences.fasta", emit: fasta

    script:
    """
    # Running annotate.py to fetch sequences and gene names
    ${params.python} ${params.scripts_dir}/annotate.py \
        --fits-csv ${fits_csv} \
        --out-annot protein_annotations.csv \
        --out-fasta protein_sequences.fasta
    """
}

process SYSTEM_ANALYSIS {
    tag "PCA & Pathway"
    publishDir "${params.outdir}/system_analysis", mode: 'copy'

    input:
    path fits_csv
    path hits_csv
    path annot_csv

    output:
    path "*.csv"
    path "*.png"

    script:
    """
    ${params.python} ${params.scripts_dir}/03_system_level_analysis.py \
        --fits-csv ${fits_csv} \
        --hits-csv ${hits_csv} \
        --annot-csv ${annot_csv} \
        --out-dir .
    """
}

process TRAIN_MODEL {
    tag "ESM-2 Training"
    label 'gpu'
    publishDir "${params.outdir}", mode: 'copy'

    input:
    path fits_csv
    path fasta_file

    output:
    path "nadph_seq_supervised.csv", emit: supervised_data
    path "nadph_seq_head.pt", emit: checkpoint
    path "training_metrics.txt", optional: true

    script:
    """
    ${params.python} ${params.scripts_dir}/04_seq_build_and_train.py \
        --fits-csv ${fits_csv} \
        --fasta ${fasta_file} \
        --out-supervised nadph_seq_supervised.csv \
        --out-head nadph_seq_head.pt \
        --epochs ${params.epochs} \
        --task ${params.task} \
        --device ${params.device} \
        > training_metrics.txt
    """
}

process PREDICT {
    tag "Inference"
    label 'gpu'
    publishDir "${params.outdir}", mode: 'copy'

    input:
    path fasta_file
    path checkpoint

    output:
    path "predictions_nadph_seq.csv", emit: predictions

    script:
    """
    ${params.python} ${params.scripts_dir}/05_predict_nadph_effects.py \
        --fasta ${fasta_file} \
        --checkpoint ${checkpoint} \
        --out predictions_nadph_seq.csv \
        --task ${params.task} \
        --saliency \
        --ig \
        --device ${params.device}
    """
}

process VISUALIZE {
    tag "Plots"
    publishDir "${params.outdir}/plots", mode: 'copy'

    input:
    path predictions
    path supervised

    output:
    path "*.png"

    script:
    """
    ${params.python} ${params.scripts_dir}/06_model_predict_results.py \
        --pred-file ${predictions} \
        --truth-file ${supervised}
    """
}

// --- Workflow Logic ---

workflow {
    // 1. Define Input
    // checkIfExists stops the pipeline immediately if data is missing
    input_ch = Channel.fromPath(params.input_csv, checkIfExists: true)

    // 2. Curve Fitting
    FIT_CURVES(input_ch)

    // 3. Parallel Branching
    HIT_CALLING(FIT_CURVES.out.fits)
    ANNOTATE_AND_FETCH(FIT_CURVES.out.fits)

    // 4. System Analysis (Needs Annotation + Hits)
    SYSTEM_ANALYSIS(
        FIT_CURVES.out.fits,
        HIT_CALLING.out.hits,
        ANNOTATE_AND_FETCH.out.annot
    )

    // 5. Deep Learning Training (Needs Fits + Fasta)
    TRAIN_MODEL(
        FIT_CURVES.out.fits,
        ANNOTATE_AND_FETCH.out.fasta
    )

    // 6. Inference
    PREDICT(
        ANNOTATE_AND_FETCH.out.fasta,
        TRAIN_MODEL.out.checkpoint
    )

    // 7. Visualization
    VISUALIZE(
        PREDICT.out.predictions,
        TRAIN_MODEL.out.supervised_data
    )
}