import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_uncertainty_accuracy(
    y_true,
    prob,
    unc,
    window: int = 200,
    title: str = "Uncertainty vs Accuracy",
):
    """
    Sort samples by uncertainty (low -> high) and plot:
      - rolling accuracy (moving average)
      - cumulative accuracy

    Parameters
    ----------
    y_true : array-like (N,)
    prob   : array-like (N,) predicted probability for class 1
    unc    : array-like (N,) uncertainty score (higher = more uncertain)
    window : int rolling window size
    title  : str plot title
    """
    y_true = np.asarray(y_true).reshape(-1)
    prob = np.asarray(prob).reshape(-1)
    unc = np.asarray(unc).reshape(-1)

    df = pd.DataFrame({"y": y_true, "p": prob, "u": unc})
    df["correct"] = ((df["p"] >= 0.5).astype(int) == df["y"]).astype(int)

    df = df.sort_values("u").reset_index(drop=True)
    df["rolling_acc"] = df["correct"].rolling(window=window, min_periods=max(5, window // 10)).mean()
    df["cum_acc"] = df["correct"].expanding().mean()

    plt.figure(figsize=(10, 6))
    plt.plot(df["rolling_acc"].values, linewidth=2, label=f"Rolling acc (window={window})")
    plt.plot(df["cum_acc"].values, linewidth=2, linestyle="--", label="Cumulative acc")
    plt.xlabel("Samples (sorted: most certain → most uncertain)")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.show()


def plot_prob_vs_uncertainty(
    y_true,
    prob,
    unc,
    title="Predicted probability vs uncertainty",
    save_path=None,
):
    y_true = np.asarray(y_true).reshape(-1)
    prob = np.asarray(prob).reshape(-1)
    unc = np.asarray(unc).reshape(-1)

    correct = ((prob >= 0.5).astype(int) == y_true)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(unc[correct],  prob[correct],  alpha=0.3, s=15,
               c="tab:blue", label="Correct",   edgecolors="none")
    ax.scatter(unc[~correct], prob[~correct], alpha=0.5, s=15,
               c="tab:red",  label="Incorrect", edgecolors="none")
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.6)
    ax.set_xlabel("Uncertainty")
    ax.set_ylabel("Predicted P(Churn = 1)")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()

