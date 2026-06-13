from torch.utils.data import DataLoader, Dataset
import torch
import numpy as np


class WebsterDataset(Dataset):
    """
    Возвращает:
      X: dict с геометрией (S, l, x_nodes) в torch.float32
      Y: в зависимости от output_mode:
         - "out"     : (Nf,)   — комплексная/вещественная АЧХ на выходе
         - "heatmap" : (Nf, M-1) — комплексная/вещественная АЧХ вдоль трубы
         - "both"    : tuple(Y_out, Y_heatmap)

    Параметры:
      output_mode: "out" | "heatmap" | "both"
      y_rep: "complex" | "abs" | "db" | "realimag"
        complex  -> complex64 тензор (если torch поддерживает downstream)
        abs      -> |T|
        db       -> 20*log10(|T|+eps)
        realimag -> 2 канала: (..., 2) = [Re, Im]
    """

    def __init__(
        self,
        n_samples: int,
        M: int = 32,
        L: float = 0.17,
        Fs: float = 10_000.0,
        N: int = 512,

        vloss=None,
        wall_param=None,

        output_mode: str = "out",
        y_rep: str = "db",
        return_geometry: bool = True,
        seed: int = 123,
        S_min: float = 0.2,
        S_max: float = 8.0,
        smooth_sigma: float = 2.0,
        cache: bool = False,

        tline_fn=None,
    ):
        super().__init__()
        assert output_mode in ("out", "heatmap", "both")
        assert y_rep in ("complex", "abs", "db", "realimag")

        self.tline_fn = tline_fn

        self.n_samples = int(n_samples)
        self.M = int(M)
        self.L = float(L)
        self.Fs = float(Fs)
        self.N = int(N)

        self.vloss = vloss
        self.wall_param = wall_param

        self.output_mode = output_mode
        self.y_rep = y_rep
        self.return_geometry = return_geometry
        self.seed = int(seed)
        self.S_min = float(S_min)
        self.S_max = float(S_max)
        self.smooth_sigma = float(smooth_sigma)
        self.cache = bool(cache)
        self._cache = {} if cache else None

        # частотная сетка фиксированная для всех
        self.freq_hz = np.linspace(0.0, self.Fs / 2.0, self.N + 1, dtype=float)
        self.omega = 2.0 * np.pi * self.freq_hz

        # x_nodes одинаковые (если L фиксирован), а S — разные
        self.x_nodes = np.linspace(0.0, self.L, self.M, dtype=float)
        self.l = np.diff(self.x_nodes)
        self.x_seg = np.cumsum(self.l)  # стыки после сегментов (M-1)

    def __len__(self):
        return self.n_samples

    def _make_target(self, T: np.ndarray) -> torch.Tensor:
        """
        T: complex numpy array
        """
        eps = 1e-12
        if self.y_rep == "complex":
            return torch.from_numpy(T.astype(np.complex64))
        if self.y_rep == "abs":
            return torch.from_numpy(np.abs(T).astype(np.float32))
        if self.y_rep == "db":
            y = 20.0 * np.log10(np.abs(T) + eps)
            return torch.from_numpy(y.astype(np.float32))
        # realimag
        y = np.stack([T.real, T.imag], axis=-1).astype(np.float32)
        return torch.from_numpy(y)

    def __getitem__(self, idx: int):
        if self.cache and idx in self._cache:
            return self._cache[idx]

        rng = np.random.default_rng(self.seed + int(idx))

        # геометрия: фиксируем x_nodes/l, рандомим S
        _, S, _ = random_tube_geometry(
            M=self.M,
            L=self.L,
            S_min=self.S_min,
            S_max=self.S_max,
            smooth_sigma=self.smooth_sigma,
            rng=rng,
        )

        # расчёт
        T_out, T_x = self.tline_fn(
            S0=S,
            l=self.l,
            freq_grid=self.omega,
            vloss=self.vloss,
            wall_param=self.wall_param,
        )

        # входы
        X = {
            "S": torch.from_numpy(S.astype(np.float32)),              # (M,)
            "l": torch.from_numpy(self.l.astype(np.float32)),         # (M-1,)
            "x_nodes": torch.from_numpy(self.x_nodes.astype(np.float32)),  # (M,)
        } if self.return_geometry else torch.from_numpy(S.astype(np.float32))

        # выходы
        if self.output_mode == "out":
            Y = self._make_target(T_out)         # (Nf,) или (Nf,2)
        elif self.output_mode == "heatmap":
            Y = self._make_target(T_x)           # (Nf,M-1) или (Nf,M-1,2)
        else:  # both
            Y = (self._make_target(T_out), self._make_target(T_x))

        out = (X, Y)

        if self.cache:
            self._cache[idx] = out
        return out
