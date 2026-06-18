import argparse
import datetime
import os
import random
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hw_noise_probit_limit import (
    calculate_cut,
    calculate_energy,
    get_graph_MAXCUT,
    probit_fitting_hardware_synchronous,
    read_file_MAXCUT,
)


@dataclass
class ConditionResult:
    name: str
    epsilon: float
    seeds: list
    energies_final: np.ndarray
    cuts_final: np.ndarray
    histories: np.ndarray  # shape: (trials, timesteps + 1)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate paper-quality energy plot: probit with mask vs without mask."
    )
    parser.add_argument("--file_path", type=str, required=True, help="Path to GSET file")
    parser.add_argument("--timesteps", type=int, default=1000, help="Annealing timesteps")
    parser.add_argument("--trials", type=int, default=30, help="Number of repeated trials")
    parser.add_argument("--sigma_start", type=float, default=5.0, help="Start sigma")
    parser.add_argument("--sigma_end", type=float, default=0.01, help="End sigma")
    parser.add_argument(
        "--schedule",
        type=str,
        default="down_counter",
        choices=["linear", "exponential", "down_counter"],
        help="Annealing schedule",
    )
    parser.add_argument(
        "--mask_epsilon",
        type=float,
        default=0.1,
        help="Epsilon for masked RPA (e.g. 0.1 means 10%% update ratio)",
    )
    parser.add_argument(
        "--no_mask_epsilon",
        type=float,
        default=1.0,
        help="Epsilon for no-mask baseline (1.0 = full update)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./paper_mask_vs_nomask",
        help="Directory for figures and CSV outputs",
    )
    parser.add_argument("--dpi", type=int, default=300, help="Figure DPI")
    return parser.parse_args()


def load_problem(file_path: str):
    first_line, _, _, fourth_line, lines = read_file_MAXCUT(file_path)
    n = int(first_line)
    best_known = int(fourth_line)
    g_matrix = get_graph_MAXCUT(n, lines)
    j_matrix = -g_matrix
    file_base = os.path.splitext(os.path.basename(file_path))[0]
    return j_matrix, g_matrix, best_known, file_base


def run_condition(
    condition_name: str,
    epsilon: float,
    seeds: list,
    j_matrix: np.ndarray,
    g_matrix: np.ndarray,
    timesteps: int,
    sigma_start: float,
    sigma_end: float,
    schedule: str,
) -> ConditionResult:
    histories = []
    energies_final = []
    cuts_final = []

    for idx, seed in enumerate(seeds, start=1):
        np.random.seed(seed)
        random.seed(seed)

        spins, energy_history = probit_fitting_hardware_synchronous(
            j_matrix,
            timesteps=timesteps,
            sigma_start=sigma_start,
            sigma_end=sigma_end,
            schedule=schedule,
            record_energy=True,
            epsilon=epsilon,
        )
        final_energy = float(calculate_energy(spins, j_matrix))
        final_cut = float(calculate_cut(spins, g_matrix))

        histories.append(np.asarray(energy_history, dtype=np.float64))
        energies_final.append(final_energy)
        cuts_final.append(final_cut)

        print(
            f"[{condition_name}] Trial {idx:02d}/{len(seeds)} | "
            f"seed={seed} | Energy={final_energy:.2f} | Cut={final_cut:.0f}"
        )

    return ConditionResult(
        name=condition_name,
        epsilon=epsilon,
        seeds=seeds,
        energies_final=np.asarray(energies_final, dtype=np.float64),
        cuts_final=np.asarray(cuts_final, dtype=np.float64),
        histories=np.asarray(histories, dtype=np.float64),
    )


def _mean_and_ci95(history_matrix: np.ndarray):
    mean = np.mean(history_matrix, axis=0)
    n = history_matrix.shape[0]
    if n > 1:
        std = np.std(history_matrix, axis=0, ddof=1)
    else:
        std = np.zeros_like(mean)
    sem = std / np.sqrt(max(n, 1))
    ci95 = 1.96 * sem
    return mean, ci95


def save_csv_outputs(output_dir: str, masked: ConditionResult, nomask: ConditionResult):
    t = np.arange(masked.histories.shape[1], dtype=int)
    masked_mean, masked_ci = _mean_and_ci95(masked.histories)
    nomask_mean, nomask_ci = _mean_and_ci95(nomask.histories)

    curve_df = pd.DataFrame(
        {
            "timestep": t,
            "masked_mean_energy": masked_mean,
            "masked_ci95": masked_ci,
            "nomask_mean_energy": nomask_mean,
            "nomask_ci95": nomask_ci,
        }
    )
    curve_df.to_csv(os.path.join(output_dir, "energy_curve_summary.csv"), index=False)

    trial_df = pd.DataFrame(
        {
            "trial": np.arange(1, len(masked.seeds) + 1),
            "seed": masked.seeds,
            "masked_energy": masked.energies_final,
            "masked_cut": masked.cuts_final,
            "nomask_energy": nomask.energies_final,
            "nomask_cut": nomask.cuts_final,
        }
    )
    trial_df.to_csv(os.path.join(output_dir, "trial_level_results.csv"), index=False)


def plot_paper_figure(
    output_dir: str,
    file_base: str,
    masked: ConditionResult,
    nomask: ConditionResult,
    best_known: int,
    dpi: int,
):
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "figure.dpi": dpi,
        }
    )

    masked_mean, masked_ci = _mean_and_ci95(masked.histories)
    nomask_mean, nomask_ci = _mean_and_ci95(nomask.histories)
    x = np.arange(masked.histories.shape[1])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)

    ax = axes[0]
    ax.plot(x, masked_mean, color="#1f77b4", linewidth=2.0, label=f"Masked (epsilon={masked.epsilon})")
    ax.fill_between(x, masked_mean - masked_ci, masked_mean + masked_ci, color="#1f77b4", alpha=0.18)

    ax.plot(x, nomask_mean, color="#d62728", linewidth=2.0, label=f"No Mask (epsilon={nomask.epsilon})")
    ax.fill_between(x, nomask_mean - nomask_ci, nomask_mean + nomask_ci, color="#d62728", alpha=0.18)

    ax.set_xlabel("Timestep")
    ax.set_ylabel("Energy")
    ax.set_title(f"Energy Trajectory ({file_base}, mean ± 95% CI)")
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend(loc="best", frameon=True)

    ax2 = axes[1]
    box = ax2.boxplot(
        [masked.energies_final, nomask.energies_final],
        tick_labels=[f"Masked\n(e={masked.epsilon})", f"No Mask\n(e={nomask.epsilon})"],
        patch_artist=True,
        widths=0.55,
        showmeans=True,
    )
    colors = ["#1f77b4", "#d62728"]
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.35)

    ax2.set_ylabel("Final Energy")
    ax2.set_title("Final Energy Distribution")
    ax2.grid(alpha=0.25, linestyle="--", axis="y")

    # Add compact text summary for paper caption drafting
    masked_mean_final = np.mean(masked.energies_final)
    nomask_mean_final = np.mean(nomask.energies_final)
    delta = masked_mean_final - nomask_mean_final
    txt = (
        f"Best-known cut: {best_known}\n"
        f"Mean Final E (Masked): {masked_mean_final:.2f}\n"
        f"Mean Final E (No Mask): {nomask_mean_final:.2f}\n"
        f"Delta (Masked - No Mask): {delta:.2f}"
    )
    ax2.text(
        0.02,
        0.03,
        txt,
        transform=ax2.transAxes,
        fontsize=9,
        va="bottom",
        ha="left",
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="gray"),
    )

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    png_path = os.path.join(output_dir, f"paper_energy_mask_vs_nomask_{file_base}_{timestamp}.png")
    pdf_path = os.path.join(output_dir, f"paper_energy_mask_vs_nomask_{file_base}_{timestamp}.pdf")
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return png_path, pdf_path


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    j_matrix, g_matrix, best_known, file_base = load_problem(args.file_path)
    seeds = [args.seed + i for i in range(args.trials)]

    print("=" * 80)
    print("Paper Figure: Probit Energy (Mask vs No-Mask)")
    print("=" * 80)
    print(f"Graph: {file_base}")
    print(f"Trials: {args.trials}")
    print(f"Timesteps: {args.timesteps}")
    print(f"Schedule: {args.schedule}")
    print(f"Sigma: {args.sigma_start} -> {args.sigma_end}")
    print(f"Masked epsilon: {args.mask_epsilon}")
    print(f"No-mask epsilon: {args.no_mask_epsilon}")
    print("=" * 80)

    masked = run_condition(
        condition_name="MASKED",
        epsilon=args.mask_epsilon,
        seeds=seeds,
        j_matrix=j_matrix,
        g_matrix=g_matrix,
        timesteps=args.timesteps,
        sigma_start=args.sigma_start,
        sigma_end=args.sigma_end,
        schedule=args.schedule,
    )
    nomask = run_condition(
        condition_name="NO_MASK",
        epsilon=args.no_mask_epsilon,
        seeds=seeds,
        j_matrix=j_matrix,
        g_matrix=g_matrix,
        timesteps=args.timesteps,
        sigma_start=args.sigma_start,
        sigma_end=args.sigma_end,
        schedule=args.schedule,
    )

    save_csv_outputs(args.output_dir, masked, nomask)
    png_path, pdf_path = plot_paper_figure(
        output_dir=args.output_dir,
        file_base=file_base,
        masked=masked,
        nomask=nomask,
        best_known=best_known,
        dpi=args.dpi,
    )

    print("\n=== Done ===")
    print(f"Figure (PNG): {png_path}")
    print(f"Figure (PDF): {pdf_path}")
    print(f"CSV summary: {os.path.join(args.output_dir, 'energy_curve_summary.csv')}")
    print(f"CSV trials : {os.path.join(args.output_dir, 'trial_level_results.csv')}")


if __name__ == "__main__":
    main()

