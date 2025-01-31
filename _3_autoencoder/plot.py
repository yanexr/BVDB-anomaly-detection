import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde


def plot_losses_and_auc(train_losses, val_losses_normal, val_losses_anomaly, val_auc_scores, ax1ylim=None, ax2ylim=None):
        fig, ax1 = plt.subplots(figsize=(8, 5))

        ax1.set_yscale('log')
        ax1.set_ylabel('Loss (log scale)')
        ax1.set_xlabel('Epoch')
        ax1.plot(train_losses, color='tab:green', label='Train Loss')
        ax1.plot(val_losses_normal, color='tab:blue', label='Val Loss Normal')
        ax1.plot(val_losses_anomaly, color='tab:orange', label='Val Loss Anomaly')

        ax2 = ax1.twinx()
        ax2.set_ylabel('AUC')
        ax2.plot(val_auc_scores, color='tab:red', label='Val AUC')

        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, loc='upper left')
        if ax1ylim is not None:
            ax1.set_ylim(ax1ylim)
        if ax2ylim is not None:
            ax2.set_ylim(ax2ylim)

        fig.tight_layout()
        plt.show()

        best_auc_epoch = np.argmax(val_auc_scores)
        print(f'Best AUC: {val_auc_scores[best_auc_epoch]:.4f} at epoch {best_auc_epoch + 1}')


def plot_histogram(y_true, y_scores, mask=None, threshold=0):
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)

    if mask:
        lower, upper = mask
        mask = (y_scores >= lower) & (y_scores <= upper)
        y_true = y_true[mask]
        y_scores = y_scores[mask]
    scores_0 = y_scores[y_true == 0]
    scores_1 = y_scores[y_true == 1]
    range_min = min(np.concatenate([scores_0, scores_1]))
    range_max = max(np.concatenate([scores_0, scores_1]))
    
    plt.figure(figsize=(6,5))
    
    bins = np.logspace(np.log10(range_min), np.log10(range_max), 80)
    counts_0, bins_0, _ = plt.hist(scores_0, bins=bins, alpha=0.5, label='Normal')
    counts_1, bins_1, _ = plt.hist(scores_1, bins=bins, alpha=0.5, label='Anomaly')

    kde_0 = gaussian_kde(scores_0, bw_method=0.1)
    kde_1 = gaussian_kde(scores_1, bw_method=0.1)
    x_vals = np.logspace(np.log10(range_min), np.log10(range_max), 500)
    kde_0_vals = kde_0(x_vals) * max(counts_0) / max(kde_0(x_vals))
    kde_1_vals = kde_1(x_vals) * max(counts_1) / max(kde_1(x_vals))

    # plot the KDEs
    plt.plot(x_vals, kde_0_vals, color='tab:blue', linewidth=2)
    plt.plot(x_vals, kde_1_vals, color='tab:orange', linewidth=2)
    plt.axvline(x=threshold, color='black', linestyle='--', label='Threshold')
    plt.xlabel('Anomaly Score (Log Scale)', fontsize=14)
    plt.yticks([])
    plt.xscale('log')
    plt.minorticks_off()
    tick_positions = np.logspace(np.log10(range_min), np.log10(range_max), 3)
    plt.xticks(tick_positions, labels=[f'{pos:.2e}' for pos in tick_positions])
    plt.show()