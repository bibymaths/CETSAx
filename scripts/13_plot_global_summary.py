import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm


def generate_global_shap_summary(predictions_csv, supervised_csv):
    print("Loading datasets...")
    df_preds = pd.read_csv(predictions_csv)
    df_seqs = pd.read_csv(supervised_csv)

    # Quick lookup for ID -> Sequence
    seq_dict = dict(zip(df_seqs['id'], df_seqs['seq']))

    print("Aggregating residue scores across the entire dataset...")
    all_aas = []
    all_scores = []

    for _, row in tqdm(df_preds.iterrows(), total=len(df_preds)):
        prot_id = row['id']
        if prot_id not in seq_dict:
            continue

        full_seq = str(seq_dict[prot_id])
        scores_str = row.get("ig")

        if pd.isna(scores_str) or not isinstance(scores_str, str):
            continue

        # Parse scores
        scores = np.array([float(x) for x in scores_str.split(";")])

        # FIX: Strictly enforce equal lengths to prevent Pandas mismatch errors
        min_len = min(len(full_seq), len(scores))
        actual_seq = list(full_seq[:min_len])
        actual_scores = scores[:min_len]

        # Collect individual amino acids and their corresponding scores
        all_aas.extend(actual_seq)
        all_scores.extend(actual_scores)

    # Create a massive dataframe of every single residue evaluated by the model
    df_global = pd.DataFrame({
        "Amino_Acid": all_aas,
        "IG_Score": all_scores
    })

    # Filter out rare non-standard amino acids (like X, U, O) if they exist
    standard_aas = list("ACDEFGHIKLMNPQRSTVWY")
    df_global = df_global[df_global["Amino_Acid"].isin(standard_aas)]

    # Calculate median importance to sort the plots
    aa_medians = df_global.groupby("Amino_Acid")["IG_Score"].median().sort_values(ascending=False)
    sorted_aas = aa_medians.index.tolist()

    # ---------------------------------------------------------
    # PLOT 1: Global Mean/Median Importance (The Bar Plot)
    # ---------------------------------------------------------
    plt.figure(figsize=(12, 6))
    sns.barplot(
        data=df_global,
        x="Amino_Acid",
        y="IG_Score",
        order=sorted_aas,
        estimator=np.median,  # Using median as IG distributions have heavy right tails
        errorbar=('ci', 95),
        color="#3498db"
    )
    plt.title("Global Feature Importance (Median IG Score per Amino Acid)", fontsize=14, pad=15)
    plt.xlabel("Amino Acid Type", fontsize=12)
    plt.ylabel("Median Attribution Magnitude", fontsize=12)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig("results/global_importance_bar.png", dpi=300)
    plt.show()

    # ---------------------------------------------------------
    # PLOT 2: Global Distribution (The "Beeswarm" Equivalent)
    # ---------------------------------------------------------
    plt.figure(figsize=(14, 7))
    sns.violinplot(
        data=df_global,
        x="Amino_Acid",
        y="IG_Score",
        order=sorted_aas,
        palette="coolwarm",
        inner="quartile",
        linewidth=1
    )
    plt.title("Global Distribution of Attribution Scores by Amino Acid", fontsize=14, pad=15)
    plt.xlabel("Amino Acid Type (Sorted by Median Importance)", fontsize=12)
    plt.ylabel("Attribution Magnitude (IG Score)", fontsize=12)

    # Zoom in slightly to cut off extreme outliers so the bulk distribution is visible
    y_max = df_global["IG_Score"].quantile(0.995)
    plt.ylim(0, y_max)

    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig("results/global_importance_distribution.png", dpi=300)
    plt.show()

    return df_global


if __name__ == "__main__":
    preds_path = "results/predictions_nadph_seq.csv"
    supervised_path = "results/nadph_seq_supervised.csv"

    df_global_stats = generate_global_shap_summary(preds_path, supervised_path)
    print("\nPlots saved to the 'results/' folder!")