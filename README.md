# DFM-LC

DFM means deterministic forecast model: an autoregressive machine-learning model that predicts future atmospheric states from an initial ERA5 state.

This repository contains the code for comparing two DFM training objectives:

- **DFM-LC**: DFM trained with a latent-space constraint.
- **DFM-MC**: DFM trained with the model-space constraint baseline.

The example notebook runs both models from the same ERA5 initial condition and compares 15-day forecasts of `T500`. In the animation, DFM-LC preserves more small-scale details than DFM-MC and looks closer to the real atmospheric structure.

![DFM-LC vs DFM-MC T500 15-day forecast](results/dfm_lc_mc_t500_15day.gif)

## Files

```text
configs/DFM-warm-start.yaml         # step1 one-step warm-start training
configs/DFM-LC.yaml                 # DFM-LC training
configs/DFM-MC.yaml                 # DFM-MC training
train.py                            # training entry point
dataset/                            # ERA5 sequence loader and normalization stats
model/                              # training wrapper and latent constraint adapter
dfm_networks/                       # forecast model structure
notebooks/dfm_lc_mc_case.ipynb      # live 15-day forecast comparison
results/dfm_lc_mc_t500_15day.gif    # rendered T500 comparison animation
```

Model checkpoints are not included in this GitHub repository. They will be uploaded before the formal paper publication.

## Training

First train the warm-start model. This is a one-step forecast model: it learns most of the atmospheric dynamics and can produce detailed short-range forecasts, but it is not sufficient for stable long-range autoregressive forecasts.

```bash
python train.py \
  --cfg configs/DFM-warm-start.yaml \
  --outdir output \
  --desc DFM-warm-start \
  --world_size 1 \
  --per_cpus 1
```

Then expose the selected warm-start checkpoint with the filename expected by the LC and MC configs:

```bash
mkdir -p checkpoints
ln -s ../output/DFM-warm-start/world_size1-DFM-warm-start/checkpoint_best.pth \
  checkpoints/DFM-warm-start.pth
```

Train DFM-LC:

```bash
python train.py \
  --cfg configs/DFM-LC.yaml \
  --outdir output \
  --desc dfm_lc \
  --world_size 1 \
  --per_cpus 1
```

Train DFM-MC:

```bash
python train.py \
  --cfg configs/DFM-MC.yaml \
  --outdir output \
  --desc dfm_mc \
  --world_size 1 \
  --per_cpus 1
```

Slurm examples:

```bash
CFG=configs/DFM-warm-start.yaml DESC=DFM-warm-start bash scripts/slurm_train.sh
CFG=configs/DFM-LC.yaml DESC=dfm_lc bash scripts/slurm_train.sh
CFG=configs/DFM-MC.yaml DESC=dfm_mc bash scripts/slurm_train.sh
```

## Notebook

Open:

```text
notebooks/dfm_lc_mc_case.ipynb
```

The notebook loads one ERA5 initial condition, runs DFM-LC and DFM-MC for 60 six-hour steps, saves the 15-day forecast arrays, and renders the `T500` comparison gif.

Required local files:

```text
data/dfm_lc_mc_case.npz
checkpoints/dfm_lc.pth
checkpoints/dfm_mc.pth
```

Runtime outputs:

```text
results/dfm_lc_mc_15day_forecasts.npz
results/dfm_lc_mc_t500_15day.gif
```
