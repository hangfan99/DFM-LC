import os

import numpy as np
import torch
import torch.nn as nn


try:
    from networks.AutoEncoder import AutoEncoder
except ImportError as exc:
    raise ImportError(
        "DFM-LC requires the AutoEncoder implementation from LDA_1.41. "
        "Clone https://github.com/hangfan99/LDA_1.41 and add its root to PYTHONPATH."
    ) from exc


class ForecastAutoEncoder(nn.Module):
    """Adapter from DFM forecast fields to the LDA_1.41 autoencoder space."""

    def __init__(self, model_version="34_4", stats_dir=None):
        super().__init__()
        self.ae = AutoEncoder(model_version=model_version)

        stats_dir = stats_dir or os.environ.get("DFM_AE_STATS_DIR")
        if stats_dir is None:
            raise ValueError(
                "Set DFM_AE_STATS_DIR to the LDA_1.41 preprocessing_params directory "
                "before using AE_loss=True."
            )

        self.register_buffer(
            "fw_mean_layer",
            torch.from_numpy(np.load(os.path.join(stats_dir, "fw_mean_layer.npy"))).float().reshape(1, -1, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "fw_std_layer",
            torch.from_numpy(np.load(os.path.join(stats_dir, "fw_std_layer.npy"))).float().reshape(1, -1, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "mean_layer",
            torch.from_numpy(np.load(os.path.join(stats_dir, "mean_layer.npy"))).float().reshape(1, -1, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "std_layer",
            torch.from_numpy(np.load(os.path.join(stats_dir, "std_layer.npy"))).float().reshape(1, -1, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "ae_index",
            torch.tensor([3, 0, 1, 2] + [i + 4 for i in range(65)], dtype=torch.long),
            persistent=False,
        )

    def fw2ae(self, x, inverse=False):
        if not inverse:
            x = x * self.fw_std_layer + self.fw_mean_layer
            x = x[:, self.ae_index]
            x = (x - self.mean_layer) / self.std_layer
            return x

        inverse_index = torch.argsort(self.ae_index)
        x = x * self.std_layer + self.mean_layer
        x = x[:, inverse_index]
        x = (x - self.fw_mean_layer) / self.fw_std_layer
        return x

    def encode(self, x):
        return self.ae.encode(self.fw2ae(x))

    def decode(self, z):
        return self.fw2ae(self.ae.decode(z), inverse=True)

    def forward(self, x):
        posterior = self.encode(x)
        z = posterior.mode()
        return self.decode(z), posterior
