"""
viz_predict.py   
---------------------
Visualization and Analysis of Model Predictions for Protein-Ligand Binding
"""
# BSD 3-Clause License
#
# Copyright (c) 2025, Abhinav Mishra
# All rights reserved.
# Email: mishraabhinav36@gmail.com
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of Abhinav Mishra nor the names of its contributors may
#    be used to endorse or promote products derived from this software without
#    specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize


def visualize_predictions(
        pred_file,
        truth_file
) -> None:
    """
    Generate a series of plots to visualize model predictions
    against ground truth labels and experimental data.
    Parameters
    ----------
    pred_file : str
        Path to CSV file containing model predictions.
    truth_file : str
        Path to CSV file containing ground truth labels and experimental data.
    -----------
    1. Confusion Matrix
    2. ROC Curves
    3. Probability Distributions (Box Plot)
    4. Saliency / Integrated Gradients Map
    5. Probability vs. EC50 Correlation
    6. Probability vs. Delta Max Correlation
    7. "The Worst Misses" (High Confidence Errors)
    8. Global Residue Importance (Aggregated IG)
    -----------
    """
    # 1. Load and Merge Data
    preds_df = pd.read_csv(pred_file)
    truth_df = pd.read_csv(truth_file)

    # Merge on 'id' to compare Truth (label_cls) vs Prediction (pred_class_idx)
    # We use an inner join to ensure we only plot proteins present in both files
    df = pd.merge(truth_df, preds_df, on='id', how='inner')

    print(f"Merged data: {len(df)} proteins found in both files.")

    # Set up plotting style
    sns.set(style="whitegrid")

    # -------------------------------------------------------
    # Plot 1: Confusion Matrix
    # -------------------------------------------------------
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(df['label_cls'], df['pred_class_idx'])

    # Create heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Weak', 'Medium', 'Strong'],
                yticklabels=['Weak', 'Medium', 'Strong'])

    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title('Confusion Matrix: Where is the model making mistakes?', fontsize=14)
    plt.tight_layout()
    plt.savefig('plot_1_confusion_matrix.png', dpi=300)
    # print("Saved plot_1_confusion_matrix.png")
    plt.close()

    # -------------------------------------------------------
    # Plot 2: ROC Curves (One per class)
    # -------------------------------------------------------
    # Binarize labels for multi-class ROC
    n_classes = 3
    y_true = label_binarize(df['label_cls'], classes=[0, 1, 2])
    y_score = df[['p_class0', 'p_class1', 'p_class2']].values
    class_names = ['Weak', 'Medium', 'Strong']

    plt.figure(figsize=(10, 7))
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'{class_names[i]} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2, alpha=0.5)
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves: How well does it separate classes?', fontsize=14)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig('plot_2_roc_curves.png', dpi=300)
    # print("Saved plot_2_roc_curves.png")
    plt.close()

    # -------------------------------------------------------
    # Plot 3: Probability Distributions (Box Plot)
    # -------------------------------------------------------
    # Melt for easier plotting with seaborn
    probs_cols = ['p_class0', 'p_class1', 'p_class2']
    plot_df = df.melt(id_vars=['hit_class'], value_vars=probs_cols,
                      var_name='Pred_Class_Prob', value_name='Probability')

    # Rename columns for readability
    plot_df['Pred_Class_Prob'] = plot_df['Pred_Class_Prob'].map({
        'p_class0': 'Weak', 'p_class1': 'Medium', 'p_class2': 'Strong'
    })

    plt.figure(figsize=(12, 6))
    sns.boxplot(x='hit_class', y='Probability', hue='Pred_Class_Prob',
                data=plot_df, order=['weak', 'medium', 'strong'], palette="Set2")

    plt.title('Model Confidence: Probabilities assigned to each class', fontsize=14)
    plt.xlabel('True Class (Ground Truth)', fontsize=12)
    plt.ylabel('Predicted Probability', fontsize=12)
    plt.legend(title="Probability For:")
    plt.tight_layout()
    plt.savefig('plot_3_confidence.png', dpi=300)
    # print("Saved plot_3_confidence.png")
    plt.close()

    # -------------------------------------------------------
    # Plot 4: Saliency / Integrated Gradients Map
    # -------------------------------------------------------
    # Find the Strong binder (True Class 2) that the model was MOST confident about
    # This is usually a good example to look at for "what did the model learn?"
    strong_hits = df[(df['label_cls'] == 2) & (df['pred_class_idx'] == 2)]

    if not strong_hits.empty:
        # Pick top 1 by probability
        best_hit = strong_hits.sort_values('p_class2', ascending=False).iloc[0]
        prot_id = best_hit['id']
        ig_str = best_hit.get('ig')  # Use 'ig' (Integrated Gradients) if available, else 'saliency'

        if isinstance(ig_str, str):
            # Parse the semicolon-separated string back into a list of floats
            scores = np.array([float(x) for x in ig_str.split(';')])

            plt.figure(figsize=(15, 5))

            # Plot the baseline signal
            plt.plot(scores, label='Importance Score', color='#2c3e50', linewidth=0.8)
            plt.fill_between(range(len(scores)), scores, alpha=0.3, color='#3498db')

            # Highlight the top 1% most important residues
            threshold = np.percentile(scores, 99)
            top_indices = np.where(scores > threshold)[0]
            plt.scatter(top_indices, scores[top_indices], color='red', s=15, label='Top 1% Residues', zorder=5)

            plt.title(f'Feature Importance Map for {prot_id} (Predicted: Strong)', fontsize=14)
            plt.xlabel('Residue Position (Sequence Index)', fontsize=12)
            plt.ylabel('Integrated Gradients Score', fontsize=12)
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig('plot_4_saliency_map.png', dpi=300)
            print(f"Saved plot_4_saliency_map.png (Example Protein: {prot_id})")
            plt.close()
        else:
            print("No IG/Saliency scores found in predictions file.")
    else:
        print("No correct 'Strong' predictions found to visualize.")

    # -------------------------------------------------------
    # Plot 5: Probability vs. EC50 (Biological Affinity)
    # -------------------------------------------------------
    # Ideally, 'Strong' probability should go UP as EC50 goes DOWN (higher affinity)

    plt.figure(figsize=(10, 6))

    # Filter out non-binders/failed fits for this plot (EC50 > 10 or NaNs)
    # Adjust '10' based on your specific data range
    plot_df = df[(df['EC50'] < 10) & (df['EC50'] > 0)].copy()

    scatter = sns.scatterplot(
        data=plot_df,
        x='EC50',
        y='p_class2',
        hue='hit_class',
        palette={'weak': 'gray', 'medium': 'orange', 'strong': 'green'},
        alpha=0.6
    )

    plt.xscale('log')  # EC50 is best viewed on log scale
    plt.axhline(0.5, color='red', linestyle='--', alpha=0.3, label='Decision Boundary')

    plt.title('Biological Validation: Does Model Confidence track with Affinity?', fontsize=14)
    plt.xlabel('Experimental EC50 (Log Scale)', fontsize=12)
    plt.ylabel('Predicted Probability of "Strong" (p_class2)', fontsize=12)
    plt.legend(title="True Class")
    plt.tight_layout()
    plt.savefig('plot_5_ec50_correlation.png', dpi=300)
    # print("Saved plot_5_ec50_correlation.png")
    plt.close()

    # -------------------------------------------------------
    # Plot 6: Probability vs. Delta Max (Thermal Shift)
    # -------------------------------------------------------
    # Higher thermal shift usually means tighter binding

    plt.figure(figsize=(12, 8))  # Increased size slightly for labels

    # 1. CHANGE: Update y to 'R2'
    sns.scatterplot(
        data=df,
        x='delta_max',
        y='R2',  # <--- Changed from 'p_class2'
        hue='hit_class',
        hue_order=['weak', 'medium', 'strong'],  # Optional: Ensures consistent color order
        palette={'weak': 'lightgray', 'medium': 'orange', 'strong': 'green'},
        alpha=0.6,
        s=60  # Increased dot size slightly
    )

    # 2. ADD: The "Clever" Labeling Logic
    # Filter for strong hits and sort by R2 to find the best ones
    top_hits = df[df['hit_class'] == 'strong'].sort_values('R2', ascending=False).head(10)

    for i, row in top_hits.iterrows():
        # Add text label
        plt.text(
            row['delta_max'] + 0.002,  # Slight offset X
            row['R2'] + 0.002,  # Slight offset Y
            row['id'],
            fontsize=9,
            fontweight='bold',
            color='darkgreen'
        )
        # Add a circle around the labeled point
        plt.scatter(row['delta_max'], row['R2'], s=150, facecolors='none', edgecolors='black', linewidth=0.8)

    # 3. CHANGE: Update Titles and Axes
    plt.title('Effect Size (Delta Max) vs Fit Quality (R2)', fontsize=15)
    plt.xlabel('Effect Size (Delta Max)', fontsize=12)
    plt.ylabel('Goodness of Fit (R2)', fontsize=12)  # <--- Changed label

    # Optional: Add threshold lines for context
    plt.axhline(0.8, color='gray', linestyle='--', alpha=0.3, label='Good Fit Threshold')

    plt.legend(title="Hit Classification", loc='lower right')
    plt.tight_layout()
    plt.savefig('plot_6_deltamax_correlation.png', dpi=300)
    # print("Saved plot_6_deltamax_correlation.png")
    plt.close()

    # -------------------------------------------------------
    # Plot 7: "The Worst Misses" (High Confidence Errors)
    # -------------------------------------------------------
    # Find proteins where Model was >90% sure but WRONG.

    # Wrong predictions
    errors = df[df['label_cls'] != df['pred_class_idx']].copy()

    # Calculate "Confidence in the Wrong Answer"
    # If model predicted class X, getting the prob of class X
    errors['wrong_conf'] = errors.apply(lambda x: x[f'p_class{int(x.pred_class_idx)}'], axis=1)

    # Get top 10 most confident errors
    top_misses = errors.sort_values('wrong_conf', ascending=False).head(10)

    if not top_misses.empty:
        plt.figure(figsize=(12, 6))
        sns.barplot(
            data=top_misses,
            x='id',
            y='wrong_conf',
            hue='pred_class',
            dodge=False
        )
        plt.axhline(0.9, color='r', linestyle='--', label='90% Confidence')
        plt.title('Top 10 "Worst Misses": High Confidence Failures', fontsize=14)
        plt.xlabel('Protein ID', fontsize=12)
        plt.ylabel('Confidence in Wrong Prediction', fontsize=12)
        plt.legend(title="Model Predicted:")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('plot_7_worst_misses.png', dpi=300)
        # print("Saved plot_7_worst_misses.png")
        plt.close()

    # -------------------------------------------------------
    # Plot 8: Global Residue Importance (Aggregated IG)
    # -------------------------------------------------------
    # Answers: "Which amino acids are universally important for Strong binders?"

    # Filter for Strong binders only
    strong_df = df[df['label_cls'] == 2].copy()

    all_residues = []
    all_scores = []

    print(f"Analyzing {len(strong_df)} strong binders for global importance...")

    for _, row in strong_df.iterrows():
        seq = row['seq']
        ig_str = row.get('ig')

        if pd.isna(ig_str) or not isinstance(ig_str, str):
            continue

        # Parse scores
        scores = np.array([float(x) for x in ig_str.split(';')])

        # Safety truncate to match lengths (in case of slight tokenization mismatches)
        min_len = min(len(seq), len(scores))

        all_residues.extend(list(seq[:min_len]))
        all_scores.extend(scores[:min_len])

    # Create Summary DataFrame
    if len(all_residues) > 0:
        res_df = pd.DataFrame({'Residue': all_residues, 'Importance': all_scores})

        # Calculate average importance per amino acid type
        summary = res_df.groupby('Residue')['Importance'].mean().sort_values(ascending=False)

        plt.figure(figsize=(12, 6))
        sns.barplot(x=summary.index, y=summary.values, palette="viridis", hue=summary.index, legend=False)

        # Add baseline (global average)
        plt.axhline(res_df['Importance'].mean(), color='red', linestyle='--', label='Global Avg')

        plt.title('Global Feature Importance: Which Amino Acids drive NADPH binding?', fontsize=14)
        plt.xlabel('Amino Acid Residue', fontsize=12)
        plt.ylabel('Avg Integrated Gradients Score', fontsize=12)
        plt.legend()
        plt.tight_layout()
        plt.savefig('plot_8_residue_importance.png', dpi=300)
        # print("Saved plot_8_residue_importance.png")
        plt.close()
    else:
        print("No IG scores found for Strong binders to generate Plot 8.")


def analyze_fitting_data(
        fits_file,
        pred_file
) -> None:
    """
    Analyze and visualize the quality of curve fitting data
    in relation to model predictions.
    Parameters
    ----------
    fits_file : str
        Path to CSV file containing curve fitting parameters.
    pred_file : str
        Path to CSV file containing model predictions.

    -----------
    1. Replicate Consistency Plot
    2. Curve Reconstruction for Top Predicted Targets
    -----------
    """
    # 1. Load Data
    fits = pd.read_csv(fits_file)
    preds = pd.read_csv(pred_file)

    # 2. Prepare Replicate Data (Pivot)
    fits['rep'] = fits['condition'].apply(lambda x: x.split('.')[-1])
    wide_fits = fits.pivot(index='id', columns='rep', values=['log10_EC50', 'R2', 'delta_max', 'Hill'])
    wide_fits.columns = [f"{col[0]}_{col[1]}" for col in wide_fits.columns]
    wide_fits = wide_fits.reset_index()

    # Merge with Predictions
    df = pd.merge(wide_fits, preds, on='id', how='inner')
    print(f"Analyzed {len(df)} proteins with curve data.")

    sns.set(style="whitegrid")

    # --- Plot 1: Replicate Consistency ---
    if 'log10_EC50_r1' in df.columns and 'log10_EC50_r2' in df.columns:
        plt.figure(figsize=(8, 8))
        sns.scatterplot(
            data=df, x='log10_EC50_r1', y='log10_EC50_r2',
            hue='pred_class', alpha=0.7, palette='viridis'
        )
        # Add diagonal
        lims = [
            min(df['log10_EC50_r1'].min(), df['log10_EC50_r2'].min()),
            max(df['log10_EC50_r1'].max(), df['log10_EC50_r2'].max())
        ]
        plt.plot(lims, lims, 'r--', alpha=0.5)
        plt.title('Replicate Consistency: Do "Strong" hits reproduce?')
        plt.savefig('plot_9_replicate_consistency.png', dpi=300)
        plt.close()

    # --- Plot 2: Curve Reconstruction (Top Hits) ---
    top_hits = df[df['pred_class_idx'] == 2].sort_values('p_class2', ascending=False).head(3)

    if not top_hits.empty:
        plt.figure(figsize=(10, 6))
        x_conc = np.logspace(-9, -3, 100)  # Simulated concentration range

        for idx, row in top_hits.iterrows():
            prot_id = row['id']
            prot_fits = fits[fits['id'] == prot_id]

            for _, fit_row in prot_fits.iterrows():
                # Reconstruct Sigmoid: Bottom + (Top-Bottom)/(1+(EC50/x)^Hill)
                y_vals = fit_row['E0'] + (fit_row['Emax'] - fit_row['E0']) / \
                         (1 + (fit_row['EC50'] / x_conc) ** fit_row['Hill'])
                plt.plot(np.log10(x_conc), y_vals, label=f"{prot_id} ({fit_row['rep']})")

        plt.title('Reconstructed ITDR Curves for Top Predicted Targets')
        plt.xlabel('Log10 Concentration')
        plt.ylabel('Response')
        plt.legend()
        plt.savefig('plot_10_curve_reconstruction.png', dpi=300)
        plt.close()


def generate_bio_insight(
        pred_file,
        truth_file,
        annot_file
) -> None:
    """
    Generate biological insight plots based on model predictions,
    experimental data, and pathway annotations.
    Parameters
    ----------
    pred_file : str
        Path to CSV file containing model predictions.
    truth_file : str
        Path to CSV file containing ground truth labels and experimental data.
    annot_file : str
        Path to CSV file containing protein annotations (e.g., pathways).
    -----------
    1. Pathway Enrichment in Predicted Strong Binders
    2. EC50 Validation Across Predicted Classes
    -----------
    """

    # 1. Load and Merge
    preds = pd.read_csv(pred_file)
    truth = pd.read_csv(truth_file)
    annot = pd.read_csv(annot_file)

    # Merge on 'id'
    df = pd.merge(truth, preds, on='id', how='inner')
    df = pd.merge(df, annot, on='id', how='left')

    # 2. Helper to clean pathway strings
    def get_terms(series):
        all_terms = []
        for item in series:
            if pd.isna(item): continue
            # Split "reactome:R-HSA-123:Name" -> "Name"
            terms = str(item).split(';')
            for t in terms:
                if ':' in t and not t.startswith('GO'):
                    all_terms.append(t.split(':')[-1])
                elif t.startswith('GO'):
                    parts = t.split(':')
                    if len(parts) > 2: all_terms.append(parts[-1])
                else:
                    all_terms.append(t)
        return all_terms

    # 3. Filter for predicted Strong binders
    strong_preds = df[df['pred_class_idx'] == 2]

    # --- Plot 1: Pathway Enrichment ---
    pathways = get_terms(strong_preds['pathway'])
    if pathways:
        counts = pd.DataFrame(Counter(pathways).most_common(10), columns=['Pathway', 'Count'])
        plt.figure(figsize=(10, 6))
        sns.barplot(data=counts, y='Pathway', x='Count', palette='viridis')
        plt.title('Top Pathways in Predicted Strong Binders')
        plt.tight_layout()
        plt.savefig('plot_11_bio_pathway_enrichment.png', dpi=300)
        plt.close()

    # --- Plot 2: EC50 Validation ---
    plt.figure(figsize=(8, 6))
    # Filter outliers for cleaner plot
    plot_df = df[df['EC50'] < 100]
    sns.boxplot(data=plot_df, x='pred_class', y='EC50', order=['weak', 'medium', 'strong'], palette='Set2')
    plt.yscale('log')
    plt.title('Do Predicted Strong Binders have lower EC50?')
    plt.ylabel('Experimental EC50 (Log Scale)')
    plt.savefig('plot_12_bio_ec50_validation.png', dpi=300)
    plt.close()
