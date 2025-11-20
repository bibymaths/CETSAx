import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize

def visualize_predictions(
        pred_file,
        truth_file
):
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
    plt.savefig('plot_1_confusion_matrix.png')
    print("Saved plot_1_confusion_matrix.png")
    # plt.show()

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
    plt.savefig('plot_2_roc_curves.png')
    print("Saved plot_2_roc_curves.png")
    # plt.show()

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
    plt.savefig('plot_3_confidence.png')
    print("Saved plot_3_confidence.png")
    # plt.show()

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
            plt.savefig('plot_4_saliency_map.png')
            print(f"Saved plot_4_saliency_map.png (Example Protein: {prot_id})")
            # plt.show()
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
    plt.savefig('plot_5_ec50_correlation.png')
    print("Saved plot_5_ec50_correlation.png")
    # plt.show()

    # -------------------------------------------------------
    # Plot 6: Probability vs. Delta Max (Thermal Shift)
    # -------------------------------------------------------
    # Higher thermal shift usually means tighter binding

    plt.figure(figsize=(10, 6))

    sns.scatterplot(
        data=df,
        x='delta_max',
        y='p_class2',
        hue='hit_class',
        palette={'weak': 'gray', 'medium': 'orange', 'strong': 'green'},
        alpha=0.6
    )

    plt.title('Thermal Shift Validation: Model Confidence vs Stabilization', fontsize=14)
    plt.xlabel('Delta Max (Thermal Stabilization)', fontsize=12)
    plt.ylabel('Predicted Probability of "Strong" (p_class2)', fontsize=12)
    plt.legend(title="True Class")
    plt.tight_layout()
    plt.savefig('plot_6_deltamax_correlation.png')
    print("Saved plot_6_deltamax_correlation.png")
    # plt.show()

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
        plt.savefig('plot_7_worst_misses.png')
        print("Saved plot_7_worst_misses.png")
        # plt.show()

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
        plt.savefig('plot_8_residue_importance.png')
        print("Saved plot_8_residue_importance.png")
        # plt.show()
    else:
        print("No IG scores found for Strong binders to generate Plot 8.")