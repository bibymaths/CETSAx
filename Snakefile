configfile: "config.yaml"

# --- Target Rule ---
rule all:
    input:
        expand("results/plots/{plot}",plot=[
            "plot_1_confusion_matrix.png",
            "plot_2_roc_curves.png",
            "plot_3_confidence.png",
            "plot_4_saliency_map.png",
            "plot_5_ec50_correlation.png",
            "plot_6_deltamax_correlation.png",
            "plot_7_worst_misses.png",
            "plot_8_residue_importance.png"
        ]),
        f"{config['results_dir']}/system_analysis/mixture_cluster_labels.csv"

# --- 1. Fit Curves ---
rule fit_curves:
    input:
        csv=config["input_csv"]
    output:
        fits=f"{config['results_dir']}/ec50_fits.csv"
    shell:
        """
        {config[python_bin]} {config[scripts_dir]}/01_fit_itdr_curves.py \
            --input-csv {input.csv} \
            --out-fits {output.fits}
        """

# --- 2. Hit Calling ---
rule hit_calling:
    input:
        fits=rules.fit_curves.output.fits
    output:
        outdir=directory(f"{config['results_dir']}/hit_calling"),
        hits_csv=f"{config['results_dir']}/hit_calling/cetsa_hits_ranked.csv"
    shell:
        """
        mkdir -p {output.outdir}
        {config[python_bin]} {config[scripts_dir]}/02_hit_calling_and_diagnostics.py \
            --fits-csv {input.fits} \
            --out-dir {output.outdir}
        """

# --- 3. Annotate & Fetch Sequences ---
rule annotate:
    input:
        fits=rules.fit_curves.output.fits,
        # SERIALIZATION HACK: We add this input just to force
        # annotation to wait until hit_calling is done.
        _wait_for_hits=rules.hit_calling.output.hits_csv
    output:
        annot=f"{config['results_dir']}/protein_annotations.csv",
        fasta=f"{config['results_dir']}/protein_sequences.fasta"
    shell:
        """
        {config[python_bin]} cetsax/annotate.py \
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
        outdir=directory(f"{config['results_dir']}/system_analysis"),
        cluster_labels=f"{config['results_dir']}/system_analysis/mixture_cluster_labels.csv"
    shell:
        """
        mkdir -p {output.outdir}
        {config[python_bin]} {config[scripts_dir]}/03_system_level_analysis.py \
            --fits-csv {input.fits} \
            --hits-csv {input.hits} \
            --annot-csv {input.annot} \
            --out-dir {output.outdir}
        """

# --- 5. Train ESM-2 Model ---
rule train_model:
    input:
        fits=rules.fit_curves.output.fits,
        fasta=rules.annotate.output.fasta,
        # SERIALIZATION HACK: Force Training to wait for System Analysis
        _wait_for_sys=rules.system_analysis.output.cluster_labels
    output:
        supervised=f"{config['results_dir']}/nadph_seq_supervised.csv",
        checkpoint=f"{config['results_dir']}/nadph_seq_head.pt"
    params:
        epochs=config["epochs"],
        task=config["task"],
        device=config["device"]
    shell:
        """
        {config[python_bin]} {config[scripts_dir]}/04_seq_build_and_train.py \
            --fits-csv {input.fits} \
            --fasta {input.fasta} \
            --out-supervised {output.supervised} \
            --out-head {output.checkpoint} \
            --epochs {params.epochs} \
            --task {params.task} \
            --device {params.device}
        """

# --- 6. Predict on Sequences ---
rule predict:
    input:
        fasta=rules.annotate.output.fasta,
        checkpoint=rules.train_model.output.checkpoint
    output:
        preds=f"{config['results_dir']}/predictions_nadph_seq.csv"
    params:
        task=config["task"],
        device=config["device"]
    shell:
        """
        {config[python_bin]} {config[scripts_dir]}/05_predict_nadph_effects.py \
            --fasta {input.fasta} \
            --checkpoint {input.checkpoint} \
            --out {output.preds} \
            --task {params.task} \
            --saliency \
            --ig \
            --device {params.device}
        """

# --- 7. Visualization ---
rule visualize:
    input:
        preds=rules.predict.output.preds,
        truth=rules.train_model.output.supervised,
        fits=rules.fit_curves.output.fits,
        annot=rules.annotate.output.annot
    output:
        "results/plots/plot_1_confusion_matrix.png",
        "results/plots/plot_2_roc_curves.png",
        "results/plots/plot_3_confidence.png",
        "results/plots/plot_4_saliency_map.png",
        "results/plots/plot_5_ec50_correlation.png",
        "results/plots/plot_6_deltamax_correlation.png",
        "results/plots/plot_7_worst_misses.png",
        "results/plots/plot_8_residue_importance.png",
        "results/plots/plot_9_replicate_consistency.png",
        "results/plots/plot_10_curve_reconstruction.png",
        "results/plots/plot_11_bio_pathway_enrichment.png",
        "results/plots/plot_12_bio_ec50_validation.png"

    shell:
        """
        mkdir -p results/plots
        cd results/plots
        
        ../../{config[python_bin]} ../../{config[scripts_dir]}/06_model_predict_results.py \
            --pred-file ../../{input.preds} \
            --truth-file ../../{input.truth} \
            --fit-file ../../{input.fits} \
            --annot-file ../../{input.annot}
        """