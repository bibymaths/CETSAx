"""
Snakefile for the CETSA-NADPH analysis pipeline.
"""
# BSD 3-Clause License
# Copyright (c) 2025, Abhinav Mishra

from pathlib import Path

configfile: "config.yaml"

# Convenience
PY = config["python_bin"]
SCRIPTS = config["scripts_dir"]
RES = config["results_dir"]
SEQ = config["seq_model"]
PRED = config["predict"]
MAX_LEN = SEQ["max_len"]
MODEL_NAME = SEQ["model_name"]
REPR_LAYER = SEQ.get("repr_layer", 33)

BASE = Path(__file__).resolve().parent
SCRIPTS_ABS = (BASE / SCRIPTS).resolve()

# --- Target Rule (What we want to produce) ---
rule all:
    input:
        expand("results/plots/{plot}", plot=[
            "plot_1_confusion_matrix.png",
            "plot_2_roc_curves.png",
            "plot_3_confidence.png",
            "plot_4_saliency_map.png",
            "plot_5_ec50_correlation.png",
            "plot_6_deltamax_correlation.png",
            "plot_7_worst_misses.png",
            "plot_8_residue_importance.png",
            "plot_9_replicate_consistency.png",
            "plot_10_curve_reconstruction.png",
            "plot_11_bio_pathway_enrichment.png",
            "plot_12_bio_ec50_validation.png"
        ]),
        f"{RES}/system_analysis/mixture_cluster_labels.csv",
        f"{RES}/system_analysis/pca_scores.csv",
        f"{RES}/system_analysis/pathway_enrichment_overrepresentation.csv",
        f"{RES}/system_analysis/redox_axes_per_protein.csv",

        f"{RES}/network/network_modules.csv",
        f"{RES}/network/network_graph.gexf",

        f"{RES}/curve_ml/curve_clusters.csv",
        f"{RES}/curve_ml/curve_outliers.csv",

        f"{RES}/bayesian/bayesian_ec50_summaries.csv",

        f"{RES}/detailed_plots/global_goodness_of_fit.png",
        f"{RES}/detailed_plots/individual_curves",

        f"{RES}/model_performance_report.txt"


# --- 1. Fit Curves ---
rule fit_curves:
    input:
        csv=config["input_csv"]
    output:
        fits=f"{RES}/ec50_fits.csv"
    shell:
        r"""
        {PY} {SCRIPTS}/01_fit_itdr_curves.py \
            --input-csv {input.csv} \
            --out-fits {output.fits}
        """


# --- 2. Hit Calling ---
rule hit_calling:
    input:
        fits=rules.fit_curves.output.fits
    output:
        outdir=directory(f"{RES}/hit_calling"),
        hits_csv=f"{RES}/hit_calling/cetsa_hits_ranked.csv",
        diag1=f"{RES}/hit_calling/ec50_r1_vs_r2.png",
        diag2=f"{RES}/hit_calling/ec50_vs_delta_max.png",
        diag3=f"{RES}/hit_calling/ec50_vs_r2.png",
        diag4=f"{RES}/hit_calling/r2_vs_delta_max.png"
    shell:
        r"""
        mkdir -p {output.outdir}
        {PY} {SCRIPTS}/02_hit_calling_and_diagnostics.py \
            --fits-csv {input.fits} \
            --out-dir {output.outdir}
        """


# --- 3. Annotate & Fetch Sequences ---
rule annotate:
    input:
        fits=rules.fit_curves.output.fits,
        _wait_for_hits=rules.hit_calling.output.hits_csv
    output:
        annot=f"{RES}/protein_annotations.csv",
        fasta=f"{RES}/protein_sequences.fasta"
    shell:
        r"""
        {PY} cetsax/annotate.py \
            --fits-csv {input.fits} \
            --out-annot {output.annot} \
            --out-fasta {output.fasta}
        """


# --- 4. System Analysis ---
rule system_analysis:
    input:
        fits=rules.fit_curves.output.fits,
        hits=rules.hit_calling.output.hits_csv,
        annot=rules.annotate.output.annot
    output:
        outdir=directory(f"{RES}/system_analysis"),
        cluster_labels=f"{RES}/system_analysis/mixture_cluster_labels.csv",
        cluster_probs=f"{RES}/system_analysis/mixture_clusters_per_protein.csv",
        cluster_plot=f"{RES}/system_analysis/mixture_clusters_in_pca.png",
        pca_scores=f"{RES}/system_analysis/pca_scores.csv",
        pca_loadings=f"{RES}/system_analysis/pca_loadings.csv",
        fa_scores=f"{RES}/system_analysis/fa_scores.csv",
        pathway_enrich=f"{RES}/system_analysis/pathway_enrichment_overrepresentation.csv",
        redox_axes=f"{RES}/system_analysis/redox_axes_per_protein.csv",
        sensitivity=f"{RES}/system_analysis/sensitivity_scores.csv"
    shell:
        r"""
        mkdir -p {output.outdir}
        {PY} {SCRIPTS}/03_system_level_analysis.py \
            --fits-csv {input.fits} \
            --hits-csv {input.hits} \
            --annot-csv {input.annot} \
            --out-dir {output.outdir}
        """


# --- 5. Train ESM-2 Model (with caching + meta.json) ---
rule train_model:
    input:
        fits=rules.fit_curves.output.fits,
        fasta=rules.annotate.output.fasta,
        _wait_for_sys=rules.system_analysis.output.cluster_labels
    output:
        supervised=f"{RES}/nadph_seq_supervised.csv",
        head_ckpt=f"{RES}/nadph_seq_head.pt",
        meta=f"{RES}/nadph_seq_meta.json",
        token_cache= f"{RES}/cache/tokens_nadph_seq_supervised_{MAX_LEN}.pt",
        pooled_cache=f"{RES}/cache/pooled_tokens_nadph_seq_supervised_{MAX_LEN}_{MODEL_NAME}_L{REPR_LAYER}.pt",
        history=f"{RES}/nadph_seq_train_info.csv"
    params:
        model_name=SEQ["model_name"],
        max_len=SEQ["max_len"],
        num_classes=SEQ["num_classes"],
        task=SEQ["task"],
        device=SEQ["device"],
        epochs=SEQ["epochs"],
        lr=SEQ["lr"],
        batch_size=SEQ["batch_size"],
        esm_batch_size=SEQ["esm_batch_size"],
        head_batch_size=SEQ["head_batch_size"],
        cache_dir=SEQ["cache"]["dir"],
        token_cache="--use-token-cache" if SEQ["cache"].get("token_cache", True) else "--no-token-cache",
        pooled_cache="--use-pooled-cache" if SEQ["cache"].get("pooled_cache", True) else "--no-pooled-cache",
        reps_cache="--use-reps-cache" if SEQ["cache"].get("reps_cache", False) else "--no-use-reps-cache",
        fp16="--cache-fp16" if SEQ["cache"].get("fp16", True) else "--cache-fp32",
        patience_flag="--patience" if SEQ.get("patience", True) else "--no-patience"
    shell:
        r"""
        mkdir -p {RES}/cache

        {PY} {SCRIPTS}/04_seq_build_and_train.py \
          --fits-csv {input.fits} \
          --fasta {input.fasta} \
          --out-supervised {output.supervised} \
          --out-head {output.head_ckpt} \
          --out-meta {output.meta} \ 
          --out-info {output.history} \
          --cache-dir {params.cache_dir} \
          --model-name {params.model_name} \
          --max-len {params.max_len} \
          --task {params.task} \
          --num-classes {params.num_classes} \
          --device {params.device} \
          --epochs {params.epochs} \
          --lr {params.lr} \
          --batch-size {params.batch_size} \
          --esm-batch-size {params.esm_batch_size} \
          --head-batch-size {params.head_batch_size} \
          {params.token_cache} \
          {params.pooled_cache} \
          {params.reps_cache} \
          {params.fp16} \
          {params.patience_flag}
        """

# --- 6. Predict on Sequences (new predictor, reads meta.json) ---
rule predict:
    input:
        fasta=rules.annotate.output.fasta,
        head_ckpt=rules.train_model.output.head_ckpt,
        meta=rules.train_model.output.meta
    output:
        preds=f"{RES}/predictions_nadph_seq.csv"
    params:
        mode=PRED["mode"],
        task=SEQ["task"],
        device=SEQ["device"],
        batch_size=PRED["batch_size"],
        esm_batch_size=PRED["esm_batch_size"],
        saliency_flag="--saliency" if PRED.get("saliency", False) else "",
        ig_flag="--ig" if PRED.get("ig", False) else "",
        ig_steps=PRED.get("ig_steps", 50)
    shell:
        r"""
        {PY} {SCRIPTS}/05_predict_nadph_effects.py \
            --fasta {input.fasta} \
            --head {input.head_ckpt} \
            --meta {input.meta} \
            --mode {params.mode} \
            --task {params.task} \
            --device {params.device} \
            --batch-size {params.batch_size} \
            --esm-batch-size {params.esm_batch_size} \
            {params.saliency_flag} \
            {params.ig_flag} \
            --ig-steps {params.ig_steps} \
            --out {output.preds}
        """


# --- 7. Visualization ---
rule visualize:
    input:
        preds=rules.predict.output.preds,
        truth=rules.train_model.output.supervised,
        fits=rules.fit_curves.output.fits,
        annot=rules.annotate.output.annot,
        hist=rules.train_model.output.history
    output:
        expand(f"{RES}/plots/{{plot}}", plot=[
            "plot_1_confusion_matrix.png",
            "plot_2_roc_curves.png",
            "plot_3_confidence.png",
            "plot_4_saliency_map.png",
            "plot_5_ec50_correlation.png",
            "plot_6_deltamax_correlation.png",
            "plot_7_worst_misses.png",
            "plot_8_residue_importance.png",
            "plot_9_replicate_consistency.png",
            "plot_10_curve_reconstruction.png",
            "plot_11_bio_pathway_enrichment.png",
            "plot_12_bio_ec50_validation.png",
            "plot_13_training_loss.png",
            "plot_14_training_accuracy.png"
        ])
    params:
        outdir=f"{RES}/plots"
    shell:
        r"""
        mkdir -p {params.outdir}

        {PY} {SCRIPTS}/06_model_predict_results.py \
            --pred-file {input.preds} \
            --truth-file {input.truth} \
            --fit-file {input.fits} \
            --annot-file {input.annot} \
            --history-file {input.hist} \
            --out-dir {params.outdir}
        """


# --- 8. Network Analysis ---
rule network_analysis:
    input:
        csv=config["input_csv"],
        _wait_for_viz="results/plots/plot_1_confusion_matrix.png"
    output:
        outdir=directory(f"{RES}/network"),
        modules=f"{RES}/network/network_modules.csv",
        costab_hmap=f"{RES}/network/costab_matrix_heatmap.png",
        costab_csv=f"{RES}/network/costab_matrix.csv",
        graph_gexf=f"{RES}/network/network_graph.gexf"
    shell:
        r"""
        {PY} {SCRIPTS}/07_network_analysis.py \
            --input-csv {input.csv} \
            --out-dir {output.outdir}
        """


# --- 9. Curve ML ---
rule curve_ml:
    input:
        csv=config["input_csv"],
        _wait_for_net=rules.network_analysis.output.modules
    output:
        outdir=directory(f"{RES}/curve_ml"),
        clusters=f"{RES}/curve_ml/curve_clusters.csv",
        outliers=f"{RES}/curve_ml/curve_outliers.csv",
        pca_feats=f"{RES}/curve_ml/curve_pca_features.csv"
    shell:
        r"""
        {PY} {SCRIPTS}/08_curve_ml.py \
            --input-csv {input.csv} \
            --out-dir {output.outdir}
        """


# --- 10. Bayesian Validation ---
rule bayesian_fit:
    input:
        csv=config["input_csv"],
        hits=rules.hit_calling.output.hits_csv,
        _wait_for_ml=rules.curve_ml.output.clusters
    output:
        outdir=directory(f"{RES}/bayesian"),
        summary=f"{RES}/bayesian/bayesian_ec50_summaries.csv"
    shell:
        r"""
        {PY} {SCRIPTS}/09_bayesian_fit.py \
            --input-csv {input.csv} \
            --hits-csv {input.hits} \
            --out-dir {output.outdir}
        """


# --- 11. Detailed Plotting ---
rule detailed_plots:
    input:
        csv=config["input_csv"],
        fits=rules.fit_curves.output.fits,
        hits=rules.hit_calling.output.hits_csv,
        _wait_for_bayes=rules.bayesian_fit.output.summary
    output:
        outdir=directory(f"{RES}/detailed_plots"),
        gof=f"{RES}/detailed_plots/global_goodness_of_fit.png",
        curves_dir=directory(f"{RES}/detailed_plots/individual_curves")
    shell:
        r"""
        {PY} {SCRIPTS}/10_plot_curves.py \
            --input-csv {input.csv} \
            --fits-csv {input.fits} \
            --hits-csv {input.hits} \
            --out-dir {output.outdir}
        """


# --- 12. Model Performance Analysis ---
rule model_performance:
    input:
        supervised=f"{RES}/nadph_seq_supervised.csv",
        preds=f"{RES}/predictions_nadph_seq.csv"
    output:
        report=f"{RES}/model_performance_report.txt"
    shell:
        r"""
        {PY} {SCRIPTS}/11_model_performance.py > {output.report}
        """
