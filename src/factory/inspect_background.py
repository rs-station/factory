#!/usr/bin/env dials.python
"""
inspect_background.py

Extracts per-reflection background summaries from a DIALS reflection table
and plots a histogram of the background.sum.value column.

run with
    source /n/hekstra_lab/people/aldama/software/dials-v3-16-1/dials_env.sh
    dials.python ../factory/src/factory/inspect_background.py
"""

import sys
import os
from dials.array_family import flex
import matplotlib.pyplot as plt
import wandb
import statistics
import numpy as np
from scipy.stats import gamma


def main(refl_file):
    # Load the reflection table
    rt = flex.reflection_table.from_file(refl_file)
    # Check required columns
    if "background.sum.value" not in rt or "background.sum.variance" not in rt:
        raise KeyError(
            "Required background columns not found in %s. "
            "Available columns: %s" % (refl_file, sorted(rt.keys()))
        )
    # Extract data
    bg_sum = list(rt["background.sum.mean"])
    bg_var = list(rt["background.sum.variance"])

    # Print basic summary
    print(f"Reflections: {len(bg_sum)}")
    print(f"Min background.sum.value: {min(bg_sum):.3f}")
    print(f"Max background.sum.value: {max(bg_sum):.3f}")
    print(f"Mean background.sum.value: {sum(bg_sum)/len(bg_sum):.3f}")
    print(f"Min background.sum.variance: {min(bg_var):.3e}")
    print(f"Max background.sum.variance: {max(bg_var):.3e}")
    print(f"Mean background.sum.variance: {sum(bg_var)/len(bg_var):.3e}")
    variance = statistics.variance(bg_sum)
    print(f"Variance of background.sum.value: {variance:.3f}")

    plt.figure(figsize=(8, 5))
    plt.hist(bg_sum, bins=100, edgecolor='black', alpha=0.6, label='background.sum.value')
    plt.xlabel('Background Sum Value')
    plt.ylabel('Count')
    plt.title('Histogram of background.sum.value')

    # Add Gamma distribution histogram
    alpha = 1.9802
    scale = 75.4010
    gamma_samples = gamma.rvs(a=alpha, scale=scale, size=len(bg_sum))
    plt.hist(gamma_samples, bins=100, edgecolor='red', alpha=0.4, label='Gamma(alpha=1.98, scale=75.4)')
    plt.legend()
    plt.tight_layout()

    wandb.init(project="inspection", name="inspect_background_histogram", reinit=True)
    wandb.log({"background_histogram": wandb.Image(plt.gcf())})
    wandb.finish()

if __name__ == "__main__":
    refl_file = "/n/holylabs/LABS/hekstra_lab/Users/fgiehr/creat_dials_unmerged/scaled.refl"
    main(refl_file)
