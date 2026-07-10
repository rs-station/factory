# factory

**Infer X-ray structure factors directly from diffraction images.**

`factory` takes raw, indexed diffraction data and predicts **structure factor amplitudes** that allow to reconstruct the molecule's electron density. It can replace the traditional multi-step *integrate → scale → merge* pipeline (e.g. DIALS) with a single deep probabilistic model that learns these steps jointly and returns calibrated uncertainty estimates.

## Installation

```bash
git clone https://github.com/rs-station/factory.git
cd factory
pip install -e ".[test,docs]"
```

Requires Python 3.9+ and a [Weights & Biases](https://wandb.ai) account for logging (`wandb login`).

## Usage

```bash
# 1. Copy the template config and point it at your data
cp configs/config_example.yaml configs/my_run.yaml

# 2. Train — writes checkpoints and an .mtz of structure factors to --save_directory
python src/factory/run_factory.py --config configs/config_example.yaml --save_directory ./results
```

The config controls data paths, distribution choices, hyperparameters, and encoder settings. The resulting amplitudes are written into an `.mtz` file and can be fed straight into standard crystallography tools (e.g. Phenix) for refinement and map generation.

## Details

The input is a set of **shoeboxes** (small 3D arrays of photon counts around each reflection) plus per-reflection metadata. Photon counts are modeled with Poisson statistics, and the model separates each reflection into four interpretable pieces:

$$\lambda = \Sigma \cdot |F|^2 \cdot p + b$$

a per-reflection **scale** `Σ`, the **structure factor** `|F|`, the spot **profile** `p`, and the **background** `b`. Neural-network encoders infer `Σ`, `p`, and `b` (shared across reflections), while the amplitudes `F` are optimized individually per reflection. Training maximizes the ELBO.

![Graphical model of factory](docs/images/architecture.svg)

## License

MIT — see [LICENSE](./LICENSE).