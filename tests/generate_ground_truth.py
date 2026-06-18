import os

import numpy as np
import torch

from csrc.dmsc import compute_dmsc
from test_dmsc import generate_noisy_landscape


def save_ground_truth():
    seed = 1780627675
    torch.manual_seed(seed)
    np.random.seed(seed)

    H, W = 32, 32
    img = generate_noisy_landscape(H, W, with_loop=True)

    os.makedirs("tests/data", exist_ok=True)
    torch.save(img, "tests/data/landscape_32x32.pt")

    configs = [("st", "cpu", 1), ("mt", "cpu", 8), ("gpu", "mps" if torch.backends.mps.is_available() else "cuda", 8)]

    for mode, device_str, threads in configs:
        torch.set_num_threads(threads)
        device = torch.device(device_str)
        img_dev = img.to(device)

        for is_dual in [False, True]:
            dual_str = "dual" if is_dual else "primal"

            # Raw
            ms_raw = compute_dmsc(img_dev, -1.0, return_gradient=True, is_dual=is_dual, block_size=32)

            # Simplified
            ms_flt = compute_dmsc(img_dev, 0.15, return_gradient=True, is_dual=is_dual, block_size=32)

            # Extract attributes to dict to save
            def ms_to_dict(ms):
                return {
                    "shape": ms.shape.cpu(),
                    "max_pts": ms.max_pts.cpu(),
                    "min_pts": ms.min_pts.cpu(),
                    "sad_pts": ms.sad_pts.cpu(),
                    "e_max": ms.e_max.cpu(),
                    "e_min": ms.e_min.cpu(),
                    "p_max": ms.p_max.cpu(),
                    "p_min": ms.p_min.cpu(),
                    "ppairs_max": ms.ppairs_max.cpu() if ms.ppairs_max is not None else None,
                    "ppairs_min": ms.ppairs_min.cpu() if ms.ppairs_min is not None else None,
                    "peaks": ms.peaks.cpu(),
                    "basins": ms.basins.cpu(),
                    "ridges": ms.ridges.cpu(),
                    "ridge_offsets": ms.ridge_offsets.cpu(),
                    "valleys": ms.valleys.cpu(),
                    "valley_offsets": ms.valley_offsets.cpu(),
                }

            torch.save(ms_to_dict(ms_raw), f"tests/data/ms_{mode}_{dual_str}_raw.pt")
            torch.save(ms_to_dict(ms_flt), f"tests/data/ms_{mode}_{dual_str}_flt.pt")

    print("Ground truth generation complete.")


if __name__ == "__main__":
    save_ground_truth()
