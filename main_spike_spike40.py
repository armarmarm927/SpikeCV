# -*- coding: utf-8 -*-
"""
Spike40 reconstruction with Hinge Loss ver.2 + 3D-TV + AdamW.

This script is adapted from the original main_spike.py while keeping the
core reconstruction algorithm unchanged:

    Hinge Loss ver.2 + 3D Total Variation + AdamW
    + Hooke-Jeeves hyperparameter search

Main changes for Spike40:
- Read spike streams directly from Spike40 .dat files.
- Use 41 spike frames centered at the key frame encoded in the filename.
- Use conversion_rate = 0.6 and threshold = 1.0.
- Remove PSF engineering.
- Remove AWGN.
- Remove image -> spike generation.
- Jointly estimate the intensity sequence and the initial residual voltage.
- Remove spike-count sweep.
- Remove TFW / TFI.
- Process all Spike40 samples in a single execution.
- Save reconstruction results and Hooke-Jeeves hyperparameters separately.

NOTE:
Hooke-Jeeves is still performed per test sample using the corresponding GT
center frame, because fixed hyperparameters are intentionally not introduced
at this stage.
"""

import argparse
import csv
import json
import os
import re
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
from skimage.metrics import structural_similarity as skimage_ssim
from tqdm import tqdm


# ============================================================================
# Default configuration
# ============================================================================

DEFAULT_DATASET_ROOT = Path(
    "/home/arima/work/SpikeCV/SpikeCV/spkData/datasets/"
    "Spike40/SpikeDataWithGT"
)

DEFAULT_OUTPUT_ROOT = Path("/home/arima/work/SpikeCV/outputs")

IMG_HEIGHT = 250
IMG_WIDTH = 400
EVAL_WINDOW_LENGTH = 41

CONVERSION_RATE = 0.6
THRESHOLD = 1.0

SEED_L_HAT = 42
SEED_V0_HAT = 43

# Optimization parameters
MARGIN = 0.0
X_WEIGHT = 1.0
Y_WEIGHT = 1.0

DEFAULT_LEARNING_RATE = 0.01
DEFAULT_LAMBDA_TV = 0.01
DEFAULT_T_WEIGHT = 1.0
DEFAULT_NUM_ITERATIONS = 10000

# Hooke-Jeeves parameters
DEFAULT_HJ_ALPHA = 0.5
DEFAULT_HJ_EPSILON = 0.01
DEFAULT_HJ_GAMMA = 0.5
DEFAULT_HJ_MAX_ITER = 12


# ============================================================================
# Spike40 loading
# ============================================================================

def raw_to_spike(dat_path: Path, height: int = IMG_HEIGHT, width: int = IMG_WIDTH):
    """
    Decode a Spike40 .dat file into a binary spike tensor of shape (T, H, W).

    The layout follows the Spk2ImgNet decoder:
    - 1 bit per pixel
    - LSB-first inside each byte
    - row-major pixel order
    - vertical flip after decoding
    """
    raw = np.fromfile(dat_path, dtype=np.uint8)

    pixels_per_frame = height * width
    if pixels_per_frame % 8 != 0:
        raise ValueError("height * width must be divisible by 8.")

    bytes_per_frame = pixels_per_frame // 8

    if raw.size % bytes_per_frame != 0:
        raise ValueError(
            f"{dat_path}: file size is not divisible by "
            f"{bytes_per_frame} bytes/frame."
        )

    num_frames = raw.size // bytes_per_frame
    raw = raw.reshape(num_frames, bytes_per_frame)

    bits = np.unpackbits(raw, axis=1, bitorder="little")
    spikes = bits.reshape(num_frames, height, width)

    # Match the official Spk2ImgNet raw_to_spike() behavior.
    spikes = np.flip(spikes, axis=1).copy()

    return torch.from_numpy(spikes.astype(np.float32))


def parse_key_id(dat_path: Path) -> int:
    """
    Extract the 1-based key frame id from a filename such as:
        200_part1_key_id151.dat
    """
    match = re.search(r"_key_id(\d+)", dat_path.stem)
    if match is None:
        raise ValueError(
            f"Could not parse key_id from filename: {dat_path.name}"
        )
    return int(match.group(1))


def get_eval_window(spikes: torch.Tensor, key_id: int, window_length: int = 41):
    """
    Extract an odd-length temporal window centered on the GT key frame.

    key_id is 1-based in the Spike40 filename.
    Example:
        key_id = 151
        center index (0-based) = 150
        41-frame window = spikes[130:171]
    """
    if window_length % 2 != 1:
        raise ValueError("window_length must be odd.")

    center_index = key_id - 1
    radius = window_length // 2
    start = center_index - radius
    end = center_index + radius + 1

    if start < 0 or end > spikes.shape[0]:
        raise ValueError(
            f"Requested window [{start}:{end}] is outside spike sequence "
            f"with T={spikes.shape[0]}."
        )

    window = spikes[start:end]

    if window.shape[0] != window_length:
        raise RuntimeError(
            f"Expected {window_length} frames, got {window.shape[0]}."
        )

    return window, center_index, start, end


def load_gt(gt_path: Path, device: torch.device):
    """
    Load the grayscale GT image and normalize only by 255.

    No per-image / per-video min-max normalization is applied.
    """
    gt = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
    if gt is None:
        raise FileNotFoundError(f"Could not load GT image: {gt_path}")

    if gt.shape != (IMG_HEIGHT, IMG_WIDTH):
        raise ValueError(
            f"Unexpected GT shape {gt.shape}; expected "
            f"({IMG_HEIGHT}, {IMG_WIDTH})."
        )

    gt = gt.astype(np.float32) / 255.0
    return torch.from_numpy(gt).to(device)


def load_spike40_sample(
    dat_path: Path,
    gt_dir: Path,
    device: torch.device,
):
    """
    Load one Spike40 sample.

    Returns:
        sigma_gt:        (41, 250, 400) binary spike tensor
        gt:              (250, 400) normalized GT image
        meta:            dictionary with indexing/statistics information
    """
    spikes_all = raw_to_spike(dat_path)
    key_id = parse_key_id(dat_path)

    sigma_gt, center_index, start, end = get_eval_window(
        spikes_all,
        key_id=key_id,
        window_length=EVAL_WINDOW_LENGTH,
    )

    gt_path = gt_dir / f"{dat_path.stem}.png"
    gt = load_gt(gt_path, device)

    total_spikes = int(sigma_gt.sum().item())
    spike_density = float(sigma_gt.mean().item())

    meta = {
        "sample": dat_path.stem,
        "dat_path": str(dat_path),
        "gt_path": str(gt_path),
        "total_frames_in_dat": int(spikes_all.shape[0]),
        "key_id_1based": int(key_id),
        "center_index_0based": int(center_index),
        "eval_start_index_0based": int(start),
        "eval_end_index_exclusive_0based": int(end),
        "eval_window_length": int(EVAL_WINDOW_LENGTH),
        "total_spikes_eval_window": total_spikes,
        "spike_density_eval_window": spike_density,
    }

    return sigma_gt.to(device), gt, meta


# ============================================================================
# Loss functions
# ============================================================================

def hinge_loss_ver2(
    intensity_seq: torch.Tensor,
    initial_voltage: torch.Tensor,
    sigma_gt: torch.Tensor,
    threshold: float,
    conversion_rate: float,
    epsilon: float,
):
    """
    Hinge-like spike consistency loss with unknown initial residual voltage.

    Spike-camera observation model:

        V(t) = V0 + sum_{tau<=t} conversion_rate * I(tau)
               - N(t) * threshold

    and the residual voltage must satisfy:

        0 <= V(t) < threshold

    Therefore:

        N(t) * threshold
            <= V0 + cumulative_charge(t)
            <= (N(t) + 1) * threshold

    The structure of the original hinge loss is preserved. Only the cumulative
    quantity is changed to match Spike40 and to include the unknown V0.

    epsilon is kept in the signature for compatibility with the original code.
    """
    del epsilon  # margin was unused in the original hinge_loss_ver2 as well.

    spike_accum = torch.cumsum(sigma_gt, dim=0)

    charge_accum = (
        initial_voltage.unsqueeze(0)
        + torch.cumsum(conversion_rate * intensity_seq, dim=0)
    )

    charge_sup = (spike_accum + 1.0) * threshold
    charge_inf = spike_accum * threshold

    over_loss = torch.clamp(charge_accum - charge_sup, min=0.0).sum()
    under_loss = torch.clamp(charge_inf - charge_accum, min=0.0).sum()

    return over_loss + under_loss


def total_variation_3d(
    x: torch.Tensor,
    x_weight: float,
    y_weight: float,
    t_weight: float,
    reduction: str = "sum",
):
    """Spatio-temporal total variation. Kept from the original algorithm."""
    if x.dim() != 3:
        raise ValueError("Input must be a 3D tensor.")

    T, H, W = x.shape
    eps = 1e-8

    dt = torch.cat(
        [
            x[1:] - x[:-1],
            torch.zeros(1, H, W, device=x.device, dtype=x.dtype),
        ],
        dim=0,
    )
    dy = torch.cat(
        [
            x[:, 1:] - x[:, :-1],
            torch.zeros(T, 1, W, device=x.device, dtype=x.dtype),
        ],
        dim=1,
    )
    dx = torch.cat(
        [
            x[:, :, 1:] - x[:, :, :-1],
            torch.zeros(T, H, 1, device=x.device, dtype=x.dtype),
        ],
        dim=2,
    )

    tv = torch.sqrt(
        eps
        + t_weight * dt**2
        + y_weight * dy**2
        + x_weight * dx**2
    )

    if reduction == "sum":
        return tv.sum()
    if reduction == "mean":
        return tv.mean()

    raise ValueError(f"Unsupported reduction: {reduction}")


# ============================================================================
# Evaluation
# ============================================================================

def psnr(gt: torch.Tensor, pred: torch.Tensor):
    """PSNR for a single [0, 1] grayscale image."""
    mse = torch.mean((gt - pred) ** 2)

    if mse.item() == 0.0:
        return float("inf")

    value = 20.0 * torch.log10(
        torch.tensor(1.0, device=gt.device) / torch.sqrt(mse)
    )
    return float(value.item())


def ssim_image(gt: torch.Tensor, pred: torch.Tensor, data_range: float = 1.0):
    """SSIM for a single grayscale image."""
    gt_np = gt.detach().cpu().numpy()
    pred_np = pred.detach().cpu().numpy()

    return float(
        skimage_ssim(
            gt_np,
            pred_np,
            data_range=data_range,
        )
    )


# ============================================================================
# Reconstruction optimization
# ============================================================================

def run_optimization_ver2(
    seed_L_hat: int,
    seed_V0_hat: int,
    num_frames: int,
    img_height: int,
    img_width: int,
    learning_rate: float,
    num_iterations: int,
    x_weight: float,
    y_weight: float,
    t_weight: float,
    lambda_tv: float,
    threshold: float,
    conversion_rate: float,
    sigma_gt: torch.Tensor,
    margin: float,
    device: torch.device,
    show_progress: bool = True,
):
    """
    Main reconstruction loop.

    Core reconstruction algorithm is preserved:
        Hinge Loss ver.2 + 3D-TV + AdamW.

    Optimized variables:
        L_hat  : intensity sequence, shape (T, H, W)
        V0_hat : initial residual voltage, shape (H, W)
    """
    gen_L_hat = torch.Generator().manual_seed(seed_L_hat)
    gen_V0_hat = torch.Generator().manual_seed(seed_V0_hat)

    # Keep the original L_hat initialization behavior.
    L_hat = torch.rand(
        num_frames,
        img_height,
        img_width,
        generator=gen_L_hat,
    )
    L_hat /= L_hat.mean()
    L_hat = (
        L_hat.to(device)
        .detach()
        .clone()
        .requires_grad_(True)
    )

    # Residual voltage before the first frame in the 41-frame window.
    V0_hat = torch.rand(
        img_height,
        img_width,
        generator=gen_V0_hat,
    )
    V0_hat = (
        (V0_hat * threshold)
        .to(device)
        .detach()
        .clone()
        .requires_grad_(True)
    )

    optimizer = torch.optim.AdamW(
        [L_hat, V0_hat],
        lr=learning_rate,
    )

    loss_history = []
    loss_spike_history = []
    loss_tv_history = []

    best_loss = float("inf")
    best_L_hat = None
    best_V0_hat = None

    iterator = range(1, num_iterations + 1)
    if show_progress:
        iterator = tqdm(
            iterator,
            desc="AdamW",
            leave=False,
        )

    log_interval = max(1, num_iterations // 10)

    for iteration in iterator:
        optimizer.zero_grad()

        # No PSF: estimated scene intensity itself is the incident intensity.
        l_event = hinge_loss_ver2(
            intensity_seq=L_hat,
            initial_voltage=V0_hat,
            sigma_gt=sigma_gt,
            threshold=threshold,
            conversion_rate=conversion_rate,
            epsilon=margin,
        )

        l_tv = total_variation_3d(
            L_hat,
            x_weight=x_weight,
            y_weight=y_weight,
            t_weight=t_weight,
            reduction="sum",
        )

        loss = l_event + lambda_tv * l_tv

        loss.backward()
        optimizer.step()

        # Projection / constraints
        with torch.no_grad():
            L_hat.clamp_(min=0.0, max=1.0)
            V0_hat.clamp_(
                min=0.0,
                max=threshold - 1e-6,
            )

        loss_val = float(loss.item())

        loss_history.append(loss_val)
        loss_spike_history.append(float(l_event.item()))
        loss_tv_history.append(float(l_tv.item()))

        # Preserve the original "best loss" selection behavior.
        if loss_val < best_loss:
            best_loss = loss_val
            best_L_hat = L_hat.detach().clone()
            best_V0_hat = V0_hat.detach().clone()

        if show_progress and iteration % log_interval == 0:
            tqdm.write(
                f"  Iter {iteration}: "
                f"Loss={loss_val:.6f} "
                f"(Ev={l_event.item():.6f}, TV={l_tv.item():.6f})"
            )

    return (
        best_L_hat,
        best_V0_hat,
        loss_history,
        loss_spike_history,
        loss_tv_history,
        best_loss,
    )


# ============================================================================
# Hooke-Jeeves hyperparameter optimization
# ============================================================================

def make_objective_function(
    sigma_gt: torch.Tensor,
    gt: torch.Tensor,
    device: torch.device,
    num_iterations: int,
    show_inner_progress: bool,
):
    """
    Build the per-sample Hooke-Jeeves objective.

    x:
        x[0] = learning rate
        x[1] = lambda_tv
        x[2] = t_weight

    The reconstruction itself does not use GT.
    GT is used only to evaluate the center frame for Hooke-Jeeves.
    """
    center_local_index = EVAL_WINDOW_LENGTH // 2

    def objective(x):
        learning_rate = float(x[0])
        lambda_tv = float(x[1])
        t_weight = float(x[2])

        # Reject invalid parameters safely.
        if learning_rate <= 0 or lambda_tv < 0 or t_weight < 0:
            return float("inf"), float("-inf")

        (
            best_L_hat,
            _best_V0_hat,
            _loss_history,
            _loss_sp,
            _loss_tv,
            _best_loss,
        ) = run_optimization_ver2(
            seed_L_hat=SEED_L_HAT,
            seed_V0_hat=SEED_V0_HAT,
            num_frames=EVAL_WINDOW_LENGTH,
            img_height=IMG_HEIGHT,
            img_width=IMG_WIDTH,
            learning_rate=learning_rate,
            num_iterations=num_iterations,
            x_weight=X_WEIGHT,
            y_weight=Y_WEIGHT,
            t_weight=t_weight,
            lambda_tv=lambda_tv,
            threshold=THRESHOLD,
            conversion_rate=CONVERSION_RATE,
            sigma_gt=sigma_gt,
            margin=MARGIN,
            device=device,
            show_progress=show_inner_progress,
        )

        pred_center = best_L_hat[center_local_index]

        score_psnr = psnr(gt, pred_center)
        score_ssim = ssim_image(gt, pred_center)

        # Minimize negative PSNR, matching the original code.
        return -score_psnr, score_ssim

    return objective


def hooke_jeeves_opt(
    f,
    x0,
    alpha,
    epsilon,
    gamma=0.5,
    max_iter=12,
):
    """
    Hooke-Jeeves-style coordinate search.

    The search behavior follows the original script:
    each parameter is perturbed by +/- alpha * x[i].
    """
    x = np.array(x0, dtype=float)

    y, z = f(x)
    n = len(x)
    iteration = 1

    while alpha > epsilon and iteration <= max_iter:
        improved = False

        x_best = x.copy()
        y_best = y
        z_best = z

        for i in range(n):
            for sign in (-1, 1):
                x_phi = x.copy()
                x_phi[i] += sign * alpha * x[i]

                y_phi, z_phi = f(x_phi)

                if y_phi < y_best:
                    x_best = x_phi
                    y_best = y_phi
                    z_best = z_phi
                    improved = True

        x = x_best
        y = y_best
        z = z_best

        if not improved:
            alpha *= gamma
            print(
                f"  Step size reduced: alpha={alpha:.6f}, "
                f"Current PSNR={-y:.4f}, "
                f"SSIM={z:.6f}"
            )
        else:
            print(
                "  Current best: "
                f"LR={x[0]:.8f}, "
                f"lambda_tv={x[1]:.8f}, "
                f"t_weight={x[2]:.8f}, "
                f"PSNR={-y:.4f}, "
                f"SSIM={z:.6f}"
            )

        iteration += 1

    return x, -y, z


# ============================================================================
# Saving
# ============================================================================

def save_center_png(tensor_2d: torch.Tensor, path: Path):
    """Save a [0,1] grayscale tensor as uint8 PNG."""
    arr = (
        tensor_2d.detach()
        .cpu()
        .clamp(0.0, 1.0)
        .numpy()
    )
    arr = np.round(arr * 255.0).astype(np.uint8)

    path.parent.mkdir(parents=True, exist_ok=True)

    ok = cv2.imwrite(str(path), arr)
    if not ok:
        raise IOError(f"Failed to save image: {path}")


def save_loss_history(
    path: Path,
    loss_history,
    loss_spike_history,
    loss_tv_history,
):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "iteration",
                "total_loss",
                "event_loss",
                "tv_loss",
            ]
        )

        for i, (loss, event, tv) in enumerate(
            zip(
                loss_history,
                loss_spike_history,
                loss_tv_history,
            ),
            start=1,
        ):
            writer.writerow([i, loss, event, tv])


def create_output_dirs(output_root: Path):
    now = datetime.now()
    run_dir = (
        output_root
        / now.strftime("%Y%m%d")
        / now.strftime("%Y%m%d_%H%M%S")
    )

    dirs = {
        "run": run_dir,
        "reconstructions": run_dir / "reconstructions",
        "initial_voltage": run_dir / "initial_voltage",
        "hyperparameters": run_dir / "hyperparameters",
        "loss_history": run_dir / "loss_history",
    }

    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)

    return dirs


# ============================================================================
# One-sample pipeline
# ============================================================================

def process_sample(
    dat_path: Path,
    gt_dir: Path,
    dirs,
    device: torch.device,
    num_iterations: int,
    initial_params,
    hj_alpha: float,
    hj_epsilon: float,
    hj_gamma: float,
    hj_max_iter: int,
    show_inner_progress: bool,
):
    sample_name = dat_path.stem

    print()
    print("=" * 80)
    print(f"Sample: {sample_name}")
    print("=" * 80)

    sigma_gt, gt, meta = load_spike40_sample(
        dat_path=dat_path,
        gt_dir=gt_dir,
        device=device,
    )

    print(
        f"  frames in .dat : {meta['total_frames_in_dat']}\n"
        f"  eval window    : "
        f"[{meta['eval_start_index_0based']}:"
        f"{meta['eval_end_index_exclusive_0based']}]\n"
        f"  total spikes   : {meta['total_spikes_eval_window']}\n"
        f"  spike density  : "
        f"{meta['spike_density_eval_window']:.8f}"
    )

    # ----------------------------------------------------------------------
    # Hooke-Jeeves
    # ----------------------------------------------------------------------
    objective = make_objective_function(
        sigma_gt=sigma_gt,
        gt=gt,
        device=device,
        num_iterations=num_iterations,
        show_inner_progress=show_inner_progress,
    )

    print("Starting Hooke-Jeeves optimization...")

    best_params, hj_psnr, hj_ssim = hooke_jeeves_opt(
        f=objective,
        x0=initial_params,
        alpha=hj_alpha,
        epsilon=hj_epsilon,
        gamma=hj_gamma,
        max_iter=hj_max_iter,
    )

    learning_rate = float(best_params[0])
    lambda_tv = float(best_params[1])
    t_weight = float(best_params[2])

    print(
        "Optimal hyperparameters:\n"
        f"  learning_rate = {learning_rate}\n"
        f"  lambda_tv     = {lambda_tv}\n"
        f"  t_weight      = {t_weight}"
    )

    hyperparam_record = {
        **meta,
        "learning_rate": learning_rate,
        "lambda_tv": lambda_tv,
        "t_weight": t_weight,
        "x_weight": X_WEIGHT,
        "y_weight": Y_WEIGHT,
        "num_iterations": int(num_iterations),
        "conversion_rate": CONVERSION_RATE,
        "threshold": THRESHOLD,
        "margin": MARGIN,
        "seed_L_hat": SEED_L_HAT,
        "seed_V0_hat": SEED_V0_HAT,
        "hooke_jeeves_alpha_initial": hj_alpha,
        "hooke_jeeves_epsilon": hj_epsilon,
        "hooke_jeeves_gamma": hj_gamma,
        "hooke_jeeves_max_iter": hj_max_iter,
        "hooke_jeeves_best_psnr": float(hj_psnr),
        "hooke_jeeves_best_ssim": float(hj_ssim),
    }

    hyperparam_path = (
        dirs["hyperparameters"]
        / f"{sample_name}.json"
    )

    with hyperparam_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            hyperparam_record,
            f,
            indent=2,
            ensure_ascii=False,
        )

    # ----------------------------------------------------------------------
    # Final reconstruction with the selected per-sample parameters
    # ----------------------------------------------------------------------
    print("Running final reconstruction...")

    (
        best_L_hat,
        best_V0_hat,
        loss_history,
        loss_spike_history,
        loss_tv_history,
        best_loss,
    ) = run_optimization_ver2(
        seed_L_hat=SEED_L_HAT,
        seed_V0_hat=SEED_V0_HAT,
        num_frames=EVAL_WINDOW_LENGTH,
        img_height=IMG_HEIGHT,
        img_width=IMG_WIDTH,
        learning_rate=learning_rate,
        num_iterations=num_iterations,
        x_weight=X_WEIGHT,
        y_weight=Y_WEIGHT,
        t_weight=t_weight,
        lambda_tv=lambda_tv,
        threshold=THRESHOLD,
        conversion_rate=CONVERSION_RATE,
        sigma_gt=sigma_gt,
        margin=MARGIN,
        device=device,
        show_progress=True,
    )

    center_local_index = EVAL_WINDOW_LENGTH // 2
    pred_center = best_L_hat[center_local_index]

    final_psnr = psnr(gt, pred_center)
    final_ssim = ssim_image(gt, pred_center)

    print(
        f"Final result: PSNR={final_psnr:.4f}, "
        f"SSIM={final_ssim:.6f}, "
        f"best_loss={best_loss:.6f}"
    )

    # Save the full 41-frame reconstruction.
    torch.save(
        best_L_hat.detach().cpu(),
        dirs["reconstructions"] / f"{sample_name}.pt",
    )

    # Save the center reconstruction as PNG for direct comparison with GT.
    save_center_png(
        pred_center,
        dirs["reconstructions"]
        / f"{sample_name}_center.png",
    )

    # Save the estimated initial residual voltage map separately.
    torch.save(
        best_V0_hat.detach().cpu(),
        dirs["initial_voltage"] / f"{sample_name}.pt",
    )

    save_loss_history(
        dirs["loss_history"] / f"{sample_name}.csv",
        loss_history,
        loss_spike_history,
        loss_tv_history,
    )

    return {
        **meta,
        "learning_rate": learning_rate,
        "lambda_tv": lambda_tv,
        "t_weight": t_weight,
        "best_loss": float(best_loss),
        "psnr": float(final_psnr),
        "ssim": float(final_ssim),
        "reconstruction_pt": str(
            dirs["reconstructions"] / f"{sample_name}.pt"
        ),
        "center_png": str(
            dirs["reconstructions"]
            / f"{sample_name}_center.png"
        ),
        "initial_voltage_pt": str(
            dirs["initial_voltage"] / f"{sample_name}.pt"
        ),
        "hyperparameters_json": str(hyperparam_path),
    }


# ============================================================================
# Main
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run Hinge-Loss reconstruction on all Spike40 test samples."
        )
    )

    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="SpikeDataWithGT directory containing input/ and gt/.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Root directory for results.",
    )
    parser.add_argument(
        "--num-iterations",
        type=int,
        default=DEFAULT_NUM_ITERATIONS,
        help="AdamW iterations per reconstruction.",
    )
    parser.add_argument(
        "--initial-lr",
        type=float,
        default=DEFAULT_LEARNING_RATE,
    )
    parser.add_argument(
        "--initial-lambda-tv",
        type=float,
        default=DEFAULT_LAMBDA_TV,
    )
    parser.add_argument(
        "--initial-t-weight",
        type=float,
        default=DEFAULT_T_WEIGHT,
    )
    parser.add_argument(
        "--hj-alpha",
        type=float,
        default=DEFAULT_HJ_ALPHA,
    )
    parser.add_argument(
        "--hj-epsilon",
        type=float,
        default=DEFAULT_HJ_EPSILON,
    )
    parser.add_argument(
        "--hj-gamma",
        type=float,
        default=DEFAULT_HJ_GAMMA,
    )
    parser.add_argument(
        "--hj-max-iter",
        type=int,
        default=DEFAULT_HJ_MAX_ITER,
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help=(
            "Optional limit for debugging. "
            "Default: process every .dat file."
        ),
    )
    parser.add_argument(
        "--show-inner-progress",
        action="store_true",
        help=(
            "Show AdamW progress bars inside Hooke-Jeeves trials. "
            "Off by default to reduce console output."
        ),
    )

    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Device: {device}")

    input_dir = args.dataset_root / "input"
    gt_dir = args.dataset_root / "gt"

    if not input_dir.exists():
        raise FileNotFoundError(
            f"Input directory not found: {input_dir}"
        )

    if not gt_dir.exists():
        raise FileNotFoundError(
            f"GT directory not found: {gt_dir}"
        )

    dat_files = sorted(input_dir.glob("*.dat"))

    if len(dat_files) == 0:
        raise FileNotFoundError(
            f"No .dat files found in: {input_dir}"
        )

    if args.max_samples is not None:
        dat_files = dat_files[: args.max_samples]

    print(f"Spike40 samples to process: {len(dat_files)}")

    dirs = create_output_dirs(args.output_root)

    run_config = {
        "dataset_root": str(args.dataset_root),
        "input_dir": str(input_dir),
        "gt_dir": str(gt_dir),
        "output_dir": str(dirs["run"]),
        "device": str(device),
        "num_samples": len(dat_files),
        "img_height": IMG_HEIGHT,
        "img_width": IMG_WIDTH,
        "eval_window_length": EVAL_WINDOW_LENGTH,
        "conversion_rate": CONVERSION_RATE,
        "threshold": THRESHOLD,
        "num_iterations": int(args.num_iterations),
        "initial_params": {
            "learning_rate": args.initial_lr,
            "lambda_tv": args.initial_lambda_tv,
            "t_weight": args.initial_t_weight,
        },
        "hooke_jeeves": {
            "alpha": args.hj_alpha,
            "epsilon": args.hj_epsilon,
            "gamma": args.hj_gamma,
            "max_iter": args.hj_max_iter,
        },
    }

    with (dirs["run"] / "run_config.json").open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            run_config,
            f,
            indent=2,
            ensure_ascii=False,
        )

    initial_params = [
        args.initial_lr,
        args.initial_lambda_tv,
        args.initial_t_weight,
    ]

    rows = []
    failed_rows = []

    for sample_index, dat_path in enumerate(dat_files, start=1):
        print(
            f"\n[{sample_index}/{len(dat_files)}] "
            f"{dat_path.name}"
        )

        try:
            row = process_sample(
                dat_path=dat_path,
                gt_dir=gt_dir,
                dirs=dirs,
                device=device,
                num_iterations=args.num_iterations,
                initial_params=initial_params,
                hj_alpha=args.hj_alpha,
                hj_epsilon=args.hj_epsilon,
                hj_gamma=args.hj_gamma,
                hj_max_iter=args.hj_max_iter,
                show_inner_progress=args.show_inner_progress,
            )
            rows.append(row)

        except Exception as exc:
            print(
                f"ERROR: failed to process "
                f"{dat_path.name}: {exc}"
            )

            failed_rows.append(
                {
                    "sample": dat_path.stem,
                    "dat_path": str(dat_path),
                    "error": repr(exc),
                }
            )

            # Continue so that one bad sample does not stop all 40 samples.
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # ----------------------------------------------------------------------
    # Save metrics
    # ----------------------------------------------------------------------
    metrics_path = dirs["run"] / "metrics.csv"

    if rows:
        fieldnames = list(rows[0].keys())

        with metrics_path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as f:
            writer = csv.DictWriter(
                f,
                fieldnames=fieldnames,
            )
            writer.writeheader()
            writer.writerows(rows)

        mean_psnr = float(
            np.mean([row["psnr"] for row in rows])
        )
        std_psnr = float(
            np.std([row["psnr"] for row in rows])
        )
        mean_ssim = float(
            np.mean([row["ssim"] for row in rows])
        )
        std_ssim = float(
            np.std([row["ssim"] for row in rows])
        )

        summary = {
            "processed_samples": len(rows),
            "failed_samples": len(failed_rows),
            "mean_psnr": mean_psnr,
            "std_psnr": std_psnr,
            "mean_ssim": mean_ssim,
            "std_ssim": std_ssim,
        }

        with (dirs["run"] / "summary.json").open(
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                summary,
                f,
                indent=2,
                ensure_ascii=False,
            )

        print()
        print("=" * 80)
        print("Spike40 summary")
        print("=" * 80)
        print(f"Processed samples: {len(rows)}")
        print(f"Failed samples   : {len(failed_rows)}")
        print(
            f"PSNR: {mean_psnr:.4f} ± {std_psnr:.4f}"
        )
        print(
            f"SSIM: {mean_ssim:.6f} ± {std_ssim:.6f}"
        )
        print(f"Results: {dirs['run']}")

    if failed_rows:
        failed_path = dirs["run"] / "failed_samples.csv"

        with failed_path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "sample",
                    "dat_path",
                    "error",
                ],
            )
            writer.writeheader()
            writer.writerows(failed_rows)


if __name__ == "__main__":
    main()
