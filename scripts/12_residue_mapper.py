# SPDX-FileCopyrightText: 2024 Abhinav Mishra
# SPDX-License-Identifier: BSD-3-Clause
import numpy as np
import pandas as pd


def extract_top_drivers(predictions_csv, supervised_csv, top_n=10, score_col="ig"):
    """
    Maps attribution scores back to their specific amino acids using the supervised CSV.
    """
    print(f"Loading predictions from {predictions_csv}...")
    df_preds = pd.read_csv(predictions_csv)

    print(f"Loading sequences from {supervised_csv}...")
    df_seqs = pd.read_csv(supervised_csv)

    # Create a quick dictionary lookup for ID -> Sequence
    seq_dict = dict(zip(df_seqs['id'], df_seqs['seq'], strict=False))

    results = []

    for _, row in df_preds.iterrows():
        prot_id = row['id']

        if prot_id not in seq_dict:
            print(f"Warning: Sequence for {prot_id} not found in supervised CSV. Skipping.")
            continue

        full_seq = str(seq_dict[prot_id])
        scores_str = row.get(score_col)

        # Skip if no scores were generated (e.g., NaN or missing)
        if pd.isna(scores_str) or not isinstance(scores_str, str):
            continue

        # Convert the semicolon string to a numpy array of floats
        scores = np.array([float(x) for x in scores_str.split(";")])

        # Guardrail: Truncate the sequence string to match the length of the scores array
        # (Handles your model's max_len=1022 truncation perfectly)
        actual_seq = full_seq[:len(scores)]

        # Zip the amino acid, 1-based position index, and the score together
        residue_data = []
        for i, (aa, score) in enumerate(zip(actual_seq, scores, strict=False)):
            residue_data.append({
                "position": i + 1,  # Biologists use 1-based indexing
                "amino_acid": aa,
                "score": score
            })

        # Sort residues by score in descending order (highest importance first)
        residue_data.sort(key=lambda x: x["score"], reverse=True)

        # Grab the top N
        top_residues = residue_data[:top_n]

        # Format them into a readable string (e.g., "K142 (0.0123)")
        hotspot_strings = [f"{r['amino_acid']}{r['position']} ({r['score']:.4f})" for r in top_residues]

        # Append to our final results
        results.append({
            "id": prot_id,
            "predicted_class": row['pred_class'],
            "prob_strong": row.get('p_class1', np.nan),
            f"top_{top_n}_{score_col}_drivers": " | ".join(hotspot_strings)
        })

    # Convert to a clean DataFrame
    df_results = pd.DataFrame(results)
    return df_results


# --- Run it on your files ---
if __name__ == "__main__":
    # Point these to your actual files in the results directory
    preds_path = "results/predictions_nadph_seq.csv"
    supervised_path = "results/nadph_seq_supervised.csv"

    # Extract the top 10 Integrated Gradient (IG) hotspots
    df_hotspots = extract_top_drivers(
        predictions_csv=preds_path,
        supervised_csv=supervised_path,
        top_n=10,
        score_col="ig"
    )

    # Save the human-readable table
    out_file = "results/protein_hotspots.csv"
    df_hotspots.to_csv(out_file, index=False)
    print(f"\nSuccess! Top drivers mapped and saved to {out_file}")

    # Preview the exact protein you pasted above (A0AVT1)
    if not df_hotspots.empty:
        print("\nPreview of the first record:")
        for col, val in df_hotspots.iloc[0].items():
            print(f"{col}: {val}")
