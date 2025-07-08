import os

# Default directory for efficiency visualization outputs
DEFAULT_EFF_VIZ_DIR = "eff_viz"


def _resolve_eff_viz_path(save_path: str | None, filename: str) -> str:
    """Return absolute path for saving efficiency visualizations.

    Parameters
    ----------
    save_path : str or None
        Requested output path. If ``None`` or just a filename without a
        directory component, the file will be stored inside
        ``DEFAULT_EFF_VIZ_DIR``.
    filename : str
        Default file name to use when ``save_path`` is ``None``.

    Returns
    -------
    str
        Path pointing to the desired output location.
    """

    if save_path is None:
        save_path = filename
    # Only prepend the default directory when no directory information is
    # provided by the caller. ``os.path.dirname`` returns an empty string in
    # that case. This keeps absolute or explicitly relative directories
    # untouched.
    if not os.path.isabs(save_path) and os.path.dirname(save_path) == "":
        save_path = os.path.join(DEFAULT_EFF_VIZ_DIR, save_path)
    return save_path

try:  # optional heavy deps are loaded lazily
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover - handled at runtime
    np = None  # type: ignore

try:
    import matplotlib.pyplot as plt  # type: ignore
except Exception:  # pragma: no cover - handled at runtime
    plt = None  # type: ignore

try:
    from sklearn.manifold import TSNE  # type: ignore
    from sklearn.decomposition import PCA  # type: ignore
except Exception:  # pragma: no cover - handled at runtime
    TSNE = PCA = None  # type: ignore

try:
    import torch  # type: ignore
except Exception:  # pragma: no cover - handled at runtime
    torch = None  # type: ignore

try:
    import umap  # type: ignore
except Exception:  # pragma: no cover - handled at runtime
    umap = None  # type: ignore

try:
    import ruptures as rpt  # type: ignore
except Exception:  # pragma: no cover - handled at runtime
    rpt = None  # type: ignore


def _ensure_deps(extra=None):
    """Raise ``ImportError`` if core dependencies are missing."""
    missing = []
    if np is None:
        missing.append("numpy")
    if plt is None:
        missing.append("matplotlib")
    if TSNE is None:
        missing.append("scikit-learn")
    if torch is None:
        missing.append("torch")
    if extra:
        for name, mod in extra.items():
            if mod is None:
                missing.append(name)
    if missing:
        raise ImportError(
            "Missing required packages: "
            + ", ".join(missing)
            + ". Install them with 'pip install -r requirements-demo.txt'"
        )


def _collect_latents(model, loader, n_samples):
    """Return original and replay latent vectors."""
    _ensure_deps()
    device = next(model.parameters()).device
    orig_latents = []
    seen = 0
    for batch, _ in loader:
        batch = batch.to(device).float()
        with torch.no_grad():
            enc = model.embedding(batch)
            enc, _, _, _ = model.encoder(enc)
            pooled = enc.mean(dim=1)
            if hasattr(model, "fc_mu"):
                lat = model.fc_mu(pooled)
            elif hasattr(model, "fc_latent"):
                lat = model.fc_latent(pooled)
            else:
                raise AttributeError(
                    "Model does not expose a latent projection via `fc_mu` or `fc_latent`"
                )
        orig_latents.append(lat.cpu())
        seen += len(batch)
        if seen >= n_samples:
            break
    if not orig_latents:
        raise ValueError("loader did not yield any samples")
    orig_latents = torch.cat(orig_latents, dim=0)[:n_samples].numpy()

    if not model.z_bank:
        raise ValueError("z_bank is empty; train the model before calling")
    replay_latents = torch.stack([entry[1] for entry in model.z_bank]).cpu().numpy()
    replay_latents = replay_latents[-n_samples:]

    return orig_latents, replay_latents


def _scatter_projection(orig_latents, replay_latents, reduced, title, save_path):
    _ensure_deps()
    count_orig = orig_latents.shape[0]
    plt.figure()
    plt.scatter(
        reduced[:count_orig, 0], reduced[:count_orig, 1], s=10, label="Original"
    )
    plt.scatter(
        reduced[count_orig:, 0],
        reduced[count_orig:, 1],
        s=10,
        label="Replay",
        alpha=0.7,
    )
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path)
    plt.close()


def plot_z_bank_tsne(model, loader, n_samples=500, save_path="z_bank_tsne.png"):
    """Visualize latent vectors stored in ``z_bank`` with t-SNE."""
    _ensure_deps()
    orig_latents, replay_latents = _collect_latents(model, loader, n_samples)
    combined = np.concatenate([orig_latents, replay_latents], axis=0)
    reduced = TSNE(n_components=2, random_state=0).fit_transform(combined)
    _scatter_projection(
        orig_latents, replay_latents, reduced, "t-SNE of Latent Vectors", save_path
    )


def plot_z_bank_pca(model, loader, n_samples=500, save_path="z_bank_pca.png"):
    """Visualize latent vectors stored in ``z_bank`` with PCA."""
    _ensure_deps()
    orig_latents, replay_latents = _collect_latents(model, loader, n_samples)
    combined = np.concatenate([orig_latents, replay_latents], axis=0)
    reduced = PCA(n_components=2).fit_transform(combined)
    _scatter_projection(
        orig_latents, replay_latents, reduced, "PCA of Latent Vectors", save_path
    )


def plot_z_bank_umap(model, loader, n_samples=500, save_path="z_bank_umap.png"):
    """Visualize latent vectors stored in ``z_bank`` with UMAP."""
    _ensure_deps({"umap-learn": umap})
    orig_latents, replay_latents = _collect_latents(model, loader, n_samples)
    combined = np.concatenate([orig_latents, replay_latents], axis=0)
    reducer = umap.UMAP(n_components=2, random_state=0)
    reduced = reducer.fit_transform(combined)
    _scatter_projection(
        orig_latents, replay_latents, reduced, "UMAP of Latent Vectors", save_path
    )


def _collect_recon(autoencoder, dataset, n_samples):
    """Return original windows and reconstructions."""
    _ensure_deps()
    loader = torch.utils.data.DataLoader(dataset, batch_size=1)
    orig = []
    recon = []
    autoencoder.eval()
    with torch.no_grad():
        for z, x in loader:
            out = autoencoder(z)
            # ``ZBankAutoencoder`` returns only the reconstruction while
            # ``BasicWindowAutoencoder`` returns ``(recon, latents)``.
            r = out[0] if isinstance(out, (tuple, list)) else out
            orig.append(x.squeeze(0))
            recon.append(r.squeeze(0))
            if len(orig) >= n_samples:
                break
    orig = torch.stack(orig).numpy().reshape(len(orig), -1)
    recon = torch.stack(recon).numpy().reshape(len(recon), -1)
    return orig, recon


def plot_reconstruction_tsne(autoencoder, dataset, n_samples=500, save_path="recon_tsne.png"):
    """Visualize autoencoder reconstructions with t-SNE."""
    _ensure_deps()
    orig, recon = _collect_recon(autoencoder, dataset, n_samples)
    combined = np.concatenate([orig, recon], axis=0)
    reduced = TSNE(n_components=2, random_state=0).fit_transform(combined)
    _scatter_projection(orig, recon, reduced, "t-SNE of Reconstructions", save_path)


def plot_reconstruction_pca(autoencoder, dataset, n_samples=500, save_path="recon_pca.png"):
    """Visualize autoencoder reconstructions with PCA."""
    _ensure_deps()
    orig, recon = _collect_recon(autoencoder, dataset, n_samples)
    combined = np.concatenate([orig, recon], axis=0)
    reduced = PCA(n_components=2).fit_transform(combined)
    _scatter_projection(orig, recon, reduced, "PCA of Reconstructions", save_path)


def visualize_cpd_detection(
    series,
    penalty=None,
    min_size=30,
    save_path="cpd_detection.png",
    *,
    zoom_range=None,
    top_k=None,
    zoom_margin=50,
    extra_zoom_ranges=None,
):
    """Plot change-point locations predicted by ``ruptures``.

    Parameters
    ----------
    series : np.ndarray
        Sequence with shape ``(time, features)`` or ``(time,)``.
    penalty : float, optional
        Penalty passed to ``rpt.Pelt.predict``. If ``None`` a heuristic based
        on sequence length and variance is used.
    min_size : int, optional
        Minimum distance between change points, defaults to ``30``.
    save_path : str, optional
        Path to save the visualization.
    zoom_range : tuple(int, int), optional
        When set, only ``series[start:end]`` is plotted while the x-axis keeps
        the original indices.
    top_k : int, optional
        Create ``top_k`` additional zoomed views of the most significant change
        points. Saved with ``_top{i}`` suffixes next to ``save_path``.
    zoom_margin : int, optional
        Half-window size around a selected change point for the zoomed views,
        defaulting to ``50``.
    extra_zoom_ranges : list of tuple(int, int), optional
        Additional fixed ranges to visualize. Each range is saved next to
        ``save_path`` with a ``_range{i}`` suffix.
    """
    _ensure_deps({"ruptures": rpt})

    series = np.asarray(series)
    orig_series = series
    if zoom_range is not None:
        start, end = zoom_range
        start = max(start, 0)
        end = min(end, len(series))
        series = series[start:end]
        offset = start
    else:
        offset = 0

    if series.ndim == 1:
        data = series.reshape(-1, 1)
        plot_target = series
    else:
        data = series.reshape(series.shape[0], -1)
        plot_target = series[:, 0]

    if penalty is None:
        penalty = np.log(len(data)) * np.var(data)

    algo = rpt.Pelt(model="l2", min_size=min_size).fit(data)
    result = algo.predict(pen=penalty)

    plt.figure()
    x_vals = np.arange(offset, offset + len(plot_target))
    plt.plot(x_vals, plot_target, label="series")
    for cp in result[:-1]:
        plt.axvline(cp + offset, color="r", linestyle="--", alpha=0.8)
    plt.xlabel("Time")
    plt.title("Change Point Detection")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path)
    plt.close()

    if top_k:
        metrics = []
        for cp in result[:-1]:
            left = max(cp - zoom_margin, 0)
            right = min(cp + zoom_margin, len(data))
            before = data[left:cp]
            after = data[cp:right]
            var_change = abs(np.var(after) - np.var(before))
            metrics.append((var_change, cp))

        metrics.sort(reverse=True)
        base, ext = os.path.splitext(save_path)
        top_cps = [cp for _, cp in metrics[:top_k]]
        for i, cp in enumerate(top_cps, 1):
            global_cp = cp + offset
            start = max(global_cp - zoom_margin, 0)
            end = min(global_cp + zoom_margin, len(orig_series))
            zoom_path = f"{base}_top{i}{ext}"
            visualize_cpd_detection(
                orig_series,
                penalty=penalty,
                min_size=min_size,
                save_path=zoom_path,
                zoom_range=(start, end),
                top_k=None,
                zoom_margin=zoom_margin,
            )

    if extra_zoom_ranges:
        base, ext = os.path.splitext(save_path)
        for i, (start, end) in enumerate(extra_zoom_ranges, 1):
            zoom_path = f"{base}_range{i}{ext}"
            visualize_cpd_detection(
                orig_series,
                penalty=penalty,
                min_size=min_size,
                save_path=zoom_path,
                zoom_range=(start, end),
                top_k=None,
                zoom_margin=zoom_margin,
            )


def plot_replay_vs_series(
    model, series, *, start=0, end=4000, save_path="replay_vs_series.png", ordered=False
):
    """Compare replay-generated samples with the original series.

    Parameters
    ----------
    model : AnomalyTransformerAE
        Model containing a populated ``z_bank``.
    series : array-like
        1D sequence used during training.
    start : int, optional
        Starting index of the slice to plot.
    end : int, optional
        End index of the slice to plot.
    save_path : str, optional
        Location where the figure will be saved.
    ordered : bool, optional
        When ``True`` use stored latents in chronological order instead of
        random sampling.
    """
    _ensure_deps()
    if not model.z_bank:
        raise ValueError("z_bank is empty; train the model before calling")

    series = np.asarray(series).squeeze()
    start = max(0, start)
    end = min(len(series), end)

    n_samples = end - start
    if ordered:
        replay = model.generate_replay_sequence(deterministic=True)
        if replay is not None:
            replay = replay[:n_samples]
    else:
        replay = model.generate_replay_samples(n_samples)
    if replay is None:
        raise ValueError("Not enough entries in z_bank for replay")
    replay = replay.detach().cpu().numpy()[:, :, 0]

    win_size = replay.shape[1]
    available = replay.shape[0]
    max_len = min(n_samples, available + win_size - 1)
    recon = np.zeros(max_len)
    counts = np.zeros(max_len)
    for i in range(available):
        idx_start = i
        idx_end = i + win_size
        if idx_start >= max_len:
            break
        win = replay[i]
        if idx_end > max_len:
            win = win[: max_len - idx_start]
            idx_end = max_len
        recon[idx_start:idx_end] += win
        counts[idx_start:idx_end] += 1
    counts[counts == 0] = 1
    recon /= counts

    actual = series[start : start + max_len]
    x = np.arange(start, start + len(actual))
    plt.figure()
    plt.plot(x, actual, label="Actual")
    plt.plot(x[: len(recon)], recon, label="Replay", alpha=0.7)
    plt.xlabel("Time")
    plt.title("Replay vs Actual")
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path)
    plt.close()


def plot_autoencoder_vs_series(
    autoencoder,
    dataset,
    series,
    *,
    start=0,
    end=None,
    save_path="ae_vs_series.png",
):
    """Compare autoencoder reconstruction with the original series.

    Parameters
    ----------
    autoencoder : ZBankAutoencoder
        Trained autoencoder used to reconstruct ``dataset`` windows.
    dataset : ZBankDataset
        Dataset of ``(z, x)`` pairs the autoencoder was trained on.
    series : array-like
        Original sequence used to build ``dataset``.
    start : int, optional
        Starting index of the slice to plot.
    end : int, optional
        End index of the slice to plot. Defaults to ``len(series)``.
    save_path : str, optional
        Location where the figure will be saved.
    """

    _ensure_deps()

    series = np.asarray(series).squeeze()
    if end is None:
        end = len(series)
    start = max(0, start)
    end = min(len(series), end)

    win_size = dataset[0][1].shape[0]
    max_len = end - start
    recon = np.zeros(max_len)
    counts = np.zeros(max_len)

    autoencoder.eval()
    with torch.no_grad():
        for i in range(start, min(end - win_size + 1, len(dataset))):
            z, _ = dataset[i]
            # ``BasicWindowAutoencoder`` returns a tuple ``(recon, latents)`` by
            # default. Extract the reconstruction before squeezing.
            r = autoencoder(z.unsqueeze(0))[0].squeeze(0).cpu().numpy()[:, 0]
            idx_start = i - start
            idx_end = idx_start + win_size
            if idx_end > max_len:
                r = r[: max_len - idx_start]
                idx_end = max_len
            recon[idx_start:idx_end] += r
            counts[idx_start:idx_end] += 1

    counts[counts == 0] = 1
    recon /= counts

    actual = series[start:end]
    x = np.arange(start, start + len(actual))
    plt.figure()
    plt.plot(x, actual, label="Actual")
    plt.plot(x[: len(recon)], recon, label="Reconstruction", alpha=0.7)
    plt.xlabel("Time")
    plt.title("Autoencoder Reconstruction")
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path)
    plt.close()

# New functions for raw data visualization

DEFAULT_RAW_VIZ_DIR = "dataset_dist_proof"


def plot_feature_distribution_by_segment(
    data, segments, feature=0, save_path=os.path.join(DEFAULT_RAW_VIZ_DIR, "feature_dist.png")
):
    """Boxplot showing how a raw feature's distribution changes across segments.

    Parameters
    ----------
    data : array-like of shape (time, features) or (time,)
        Raw sequence to analyze.
    segments : list of tuple(int, int)
        Each ``(start, end)`` pair defines a slice ``data[start:end]``.
    feature : int, optional
        Index of the feature to visualize when ``data`` is 2D.
    save_path : str, optional
        Location where the figure will be saved.
    """
    _ensure_deps()
    data = np.asarray(data)
    if data.ndim == 1:
        data = data[:, None]
    seg_data = []
    valid_labels = []
    for start, end in segments:
        start = max(0, start)
        end = min(len(data), end)
        if start >= end:
            continue
        seg_data.append(data[start:end, feature])
        valid_labels.append(f"{start}-{end}")
    if not seg_data:
        raise ValueError("No valid segments provided")
    plt.figure()
    plt.boxplot(seg_data, labels=valid_labels, showfliers=False)
    plt.xlabel("Segment")
    plt.ylabel(f"Feature {feature}")
    plt.title("Feature Distribution by Segment")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path)
    plt.close()


def plot_rolling_stats(
    data,
    feature=None,
    window=50,
    save_path=os.path.join(DEFAULT_RAW_VIZ_DIR, "rolling_stats.png"),
):
    """Plot rolling mean, std, and min-max range of a raw feature."""
    _ensure_deps()
    data = np.asarray(data)
    if data.ndim == 1:
        series = data.astype(float)
    else:
        series = data[:, feature].astype(float)
    means = []
    stds = []
    mins = []
    maxs = []
    for i in range(len(series)):
        start = max(0, i - window + 1)
        win = series[start : i + 1]
        means.append(win.mean())
        stds.append(win.std())
        mins.append(win.min())
        maxs.append(win.max())
    x = np.arange(len(series))
    plt.figure()
    plt.plot(x, means, label="mean")
    plt.fill_between(x, mins, maxs, alpha=0.2, label="min-max")
    plt.plot(x, stds, label="std")
    plt.xlabel("Time")
    plt.title("Rolling Statistics")
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path)
    plt.close()


def plot_projection_by_segment(
    data,
    segments,
    *,
    feature=None,
    method="tsne",
    save_path=None,
):
    """Visualize raw data segments with t-SNE or PCA.

    Parameters
    ----------
    data : array-like of shape (time, features) or (time,)
        Raw sequence to analyze.
    segments : list of tuple(int, int)
        Each ``(start, end)`` pair defines a slice ``data[start:end]``.
    feature : int or None, optional
        Index of the feature to visualize when ``data`` is 2D. ``None`` uses all
        features and is the default when the input has multiple dimensions.
    method : {"tsne", "pca"}, optional
        Dimensionality reduction technique to apply.
    save_path : str, optional
        Figure location. Defaults to ``dataset_dist_proof/<method>_segments.png``.
    """
    _ensure_deps()
    if save_path is None:
        save_path = os.path.join(DEFAULT_RAW_VIZ_DIR, f"{method}_segments.png")

    data = np.asarray(data)
    if data.ndim == 1:
        data = data[:, None]

    points = []
    labels = []
    for idx, (start, end) in enumerate(segments):
        start = max(0, start)
        end = min(len(data), end)
        if start >= end:
            continue
        if feature is None:
            seg = data[start:end]
        else:
            seg = data[start:end, feature]
        seg = seg.reshape(len(seg), -1)
        points.append(seg)
        labels.append(np.full(len(seg), idx))
    if not points:
        raise ValueError("No valid segments provided")

    combined = np.concatenate(points, axis=0)
    label_arr = np.concatenate(labels)

    if method == "tsne":
        reducer = TSNE(n_components=2, random_state=0)
    elif method == "pca":
        reducer = PCA(n_components=2)
    else:
        raise ValueError("method must be 'tsne' or 'pca'")

    reduced = reducer.fit_transform(combined)

    plt.figure()
    sc = plt.scatter(reduced[:, 0], reduced[:, 1], c=label_arr, cmap="tab10", s=10)
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.title(f"{method.upper()} by Segment")
    plt.colorbar(sc, label="Segment")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path)
    plt.close()


def plot_memory_usage_curve(steps, continual_mem, batch_mem, save_path="memory_usage.png"):
    """Plot memory usage for continual vs batch learning over time.

    When ``save_path`` does not specify a directory, the figure is saved under
    ``DEFAULT_EFF_VIZ_DIR``.
    """
    _ensure_deps()
    steps = np.asarray(steps)
    continual_mem = np.asarray(continual_mem)
    batch_mem = np.asarray(batch_mem)
    if steps.ndim != 1 or continual_mem.shape != steps.shape or batch_mem.shape != steps.shape:
        raise ValueError("inputs must be 1D arrays of the same length")
    plt.figure()
    plt.plot(steps, continual_mem, label="Continual")
    plt.plot(steps, batch_mem, label="Batch")
    plt.xlabel("Training Step")
    plt.ylabel("Memory Usage")
    plt.title("Memory Usage over Training")
    plt.legend()
    plt.tight_layout()
    save_path = _resolve_eff_viz_path(save_path, "memory_usage.png")
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path)
    plt.close()


def plot_parameter_update_efficiency(param_counts, performance, *, labels=None, save_path="param_efficiency.png"):
    """Plot model performance as a function of updated parameters.

    When ``save_path`` lacks directory information, the figure is written under
    ``DEFAULT_EFF_VIZ_DIR``.
    """
    _ensure_deps()
    param_counts = np.asarray(param_counts)
    performance = np.asarray(performance)
    if param_counts.ndim != 1 or performance.shape != param_counts.shape:
        raise ValueError("param_counts and performance must be 1D arrays of the same length")
    plt.figure()
    plt.scatter(param_counts, performance)
    if labels is not None:
        for x, y, text in zip(param_counts, performance, labels):
            plt.text(x, y, str(text))
    plt.xlabel("Updated Parameters")
    plt.ylabel("Model Performance")
    plt.title("Parameter Update Efficiency")
    plt.tight_layout()
    save_path = _resolve_eff_viz_path(save_path, "param_efficiency.png")
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path)
    plt.close()


def plot_latency_vs_model_size(model_sizes, latencies, *, labels=None, save_path="latency_vs_size.png"):
    """Plot inference latency as a function of model size.

    Figures are saved inside ``DEFAULT_EFF_VIZ_DIR`` when no directory is
    specified for ``save_path``.
    """
    _ensure_deps()
    model_sizes = np.asarray(model_sizes)
    latencies = np.asarray(latencies)
    if model_sizes.ndim != 1 or latencies.shape != model_sizes.shape:
        raise ValueError("model_sizes and latencies must be 1D arrays of the same length")
    plt.figure()
    plt.scatter(model_sizes, latencies)
    if labels is not None:
        for x, y, text in zip(model_sizes, latencies, labels):
            plt.text(x, y, str(text))
    plt.xlabel("Model Size")
    plt.ylabel("Inference Latency")
    plt.title("Latency vs Model Size")
    plt.tight_layout()
    save_path = _resolve_eff_viz_path(save_path, "latency_vs_size.png")
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path)
    plt.close()


def plot_vector_projection(vectors, *, method="tsne", title=None, save_path="projection.png"):
    """Project a sequence of vectors with t-SNE or PCA and save a scatter plot."""

    _ensure_deps()
    arr = np.asarray(vectors)
    arr = arr.reshape(-1, arr.shape[-1])
    if method == "tsne":
        reducer = TSNE(n_components=2, random_state=0)
    elif method == "pca":
        reducer = PCA(n_components=2)
    else:
        raise ValueError("method must be 'tsne' or 'pca'")
    reduced = reducer.fit_transform(arr)

    if title is None:
        title = f"{method.upper()} Projection"

    plt.figure()
    plt.scatter(reduced[:, 0], reduced[:, 1], s=10)
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.title(title)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path)
    plt.close()

