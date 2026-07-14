# Data

`dfm_lc_mc_case.npz` is the one-case artifact for the comparison notebook. It contains:

```text
input          shape (1, 69, 128, 256), physical-unit initial state
era5_initial   shape (1, 69, 128, 256), alias for the saved ERA5 initial state
mean           shape (69,), channel normalization mean
std            shape (69,), channel normalization std
channel_names  shape (69,)
lead_steps     scalar rollout length, set to 60 for the 15-day paper example
```

This file is intended for the Zenodo/full artifact. It does not contain DFM-LC or DFM-MC forecasts; the notebook computes those by loading the model checkpoints and running the forecast model.
