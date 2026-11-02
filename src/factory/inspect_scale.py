"""run with
source /n/hekstra_lab/people/aldama/software/dials-v3-16-1/dials_env.sh
dials.python ../factory/src/factory/inspect_scale.py"""

#!/usr/bin/env dials.python

import sys

import matplotlib.pyplot as plt

# matplotlib.use("Agg")           # ← use a headless backend
import wandb
from dials.array_family import flex


def main(refl_file):
    refl = flex.reflection_table.from_file(refl_file)
    if "inverse_scale_factor" not in refl:
        raise RuntimeError("No inverse_scale_factor column found in " + refl_file)

    # Extract the scale factors
    scales = refl["inverse_scale_factor"]

    scales_list = list(scales)
    scales = [1 / x for x in scales_list]
    scales_list = scales

    mean_scale = sum(scales) / len(scales)
    variance_scale = sum((s - mean_scale) ** 2 for s in scales) / len(scales)
    max_scale = max(scales)
    min_scale = min(scales)
    print("Mean of the 1/scale factors:", mean_scale)
    print("Max of the 1/scale factors:", max_scale)
    print("Min of the 1/scale factors:", min_scale)
    print("Variance of the 1/scale factors:", variance_scale)

    # Convert to a regular Python list for plotting

    # Plot histogram
    plt.figure(figsize=(8, 5))
    plt.hist(scales_list, bins=100, edgecolor="black", alpha=0.6, label="Data")
    plt.xlabel("1/Inverse Scale Factor")
    plt.ylabel("Number of Observations")
    plt.title("Histogram of Per‐Reflection 1/Scale Factors")
    plt.tight_layout()

    # Overlay Gamma distribution
    import numpy as np
    from scipy.stats import gamma

    x = np.linspace(min(scales_list), max(scales_list), 1000)
    gamma_shape = 6.68
    gamma_scale = 0.155
    y = gamma.pdf(x, a=gamma_shape, scale=gamma_scale)
    # Scale the gamma PDF to match the histogram
    y_scaled = (
        y * len(scales_list) * (max(scales_list) - min(scales_list)) / 100
    )  # 100 bins
    plt.plot(x, y_scaled, "r-", lw=2, label=f"Gamma({gamma_shape}, {gamma_scale})")
    plt.legend()

    # Show or save
    # out_png = "scale_histogram.png"
    # plt.savefig(out_png, dpi=300)  # ← write to disk
    # print(f"Histogram written to {out_png}")

    # Log to wandb
    wandb.init(project="inspection", name="inspect_scale_histogram", reinit=True)
    wandb.log({"scale_histogram": wandb.Image(plt.gcf())})
    wandb.finish()
    # If you’d rather save to disk, uncomment:
    # plt.savefig("scale_histogram.png", dpi=300)


if __name__ == "__main__":
    main("/n/holylabs/LABS/hekstra_lab/Users/fgiehr/creat_dials_unmerged/scaled.refl")
