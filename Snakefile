configfile: "config.yaml"

# --- Target Rule (What we want to produce) ---
rule all:
    input:
        # We want the final plots and the predictions
        expand("{res}/plots/{plot}", res=config["results_dir"], plot=[
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
        csv = config["input_csv"]
    output:
        fits = f"{config['results_dir']}/ec50_fits.csv"
    shell:
        """
        {config[python_bin]} {config[scripts_dir]}/01_fit_itdr_curves.py \
            --input-csv {input.csv} \
            --out-fits {output.fits}
        """

# --- 2. Hit Calling ---
rule hit_calling:
    input:
        fits = rules.fit_curves.output.fits
    output:
        # We define the directory as output to capture all plots/tables generated
        outdir = directory(f"{config['results_dir']}/hit_calling"),
        # We explicitly track the main CSV for downstream rules
        hits_csv = f"{config['results_dir']}/hit_calling/cetsa_hits_ranked.csv"
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
        fits = rules.fit_curves.output.fits
    output:
        annot = f"{config['results_dir']}/protein_annotations.csv",
        fasta = f"{config['results_dir']}/protein_sequences.fasta"
    shell:
        """
        # Updated path: pointing to cetsax/annotate.py instead of scripts/
        {config[python_bin]} cetsax/annotate.py \
            --fits-csv {input.fits} \
            --out-annot {output.annot} \
            --out-fasta {output.fasta}
        """

# --- 4. System Analysis ---
rule system_analysis:
    input:
        fits = rules.fit_curves.output.fits,
        hits = rules.hit_calling.output.hits_csv,
        annot = rules.annotate.output.annot
    output:
        outdir = directory(f"{config['results_dir']}/system_analysis"),
        # Tracking one file to verify completion
        cluster_labels = f"{config['results_dir']}/system_analysis/mixture_cluster_labels.csv"
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
        fits = rules.fit_curves.output.fits,
        fasta = rules.annotate.output.fasta
    output:
        supervised = f"{config['results_dir']}/nadph_seq_supervised.csv",
        checkpoint = f"{config['results_dir']}/nadph_seq_head.pt"
    params:
        epochs = config["epochs"],
        task = config["task"],
        device = config["device"]
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
        fasta = rules.annotate.output.fasta,
        checkpoint = rules.train_model.output.checkpoint
    output:
        preds = f"{config['results_dir']}/predictions_nadph_seq.csv"
    params:
        task = config["task"],
        device = config["device"]
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
        preds = rules.predict.output.preds,
        truth = rules.train_model.output.supervised
    output:
        # List all expected plots so Snakemake knows when it's done
        "results/plots/plot_1_confusion_matrix.png",
        "results/plots/plot_2_roc_curves.png",
        "results/plots/plot_3_confidence.png",
        "results/plots/plot_4_saliency_map.png",
        "results/plots/plot_5_ec50_correlation.png",
        "results/plots/plot_6_deltamax_correlation.png",
        "results/plots/plot_7_worst_misses.png",
        "results/plots/plot_8_residue_importance.png"
    shell:
        """
        mkdir -p results/plots
        # We change directory so the python script writes files directly into results/plots
        cd results/plots
        
        # We reference inputs using relative paths from the new directory
        {config[python_bin]} ../../{config[scripts_dir]}/06_model_predict_results.py \
            --pred-file ../../{input.preds} \
            --truth-file ../../{input.truth}
        """