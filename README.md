# Anomaly-Transformer (ICLR 2022 Spotlight)
Anomaly Transformer: Time Series Anomaly Detection with Association Discrepancy

Unsupervised detection of anomaly points in time series is a challenging problem, which requires the model to learn informative representation and derive a distinguishable criterion. In this paper, we propose the Anomaly Transformer in these three folds:

- An inherent distinguishable criterion as **Association Discrepancy** for detection.
- A new **Anomaly-Attention** mechanism to compute the association discrepancy.
- A **minimax strategy** to amplify the normal-abnormal distinguishability of the association discrepancy.

<p align="center">
<img src=".\pics\structure.png" height = "350" alt="" align=center />
</p>

## Get Started

1. Install Python 3.6 and the required packages:

   ```bash
   pip install -r requirements-demo.txt
   ```

   This pulls in PyTorch (>=1.4.0) and other optional dependencies.
   (Thanks Élise for the contribution in solving the environment. See this [issue](https://github.com/thuml/Anomaly-Transformer/issues/11) for details.)
2. Download data. You can obtain four benchmarks from [Google Cloud](https://drive.google.com/drive/folders/1gisthCoE-RrKJ0j3KPV7xiibhHWT9qRm?usp=sharing). **All the datasets are well pre-processed**. For the SWaT dataset, you can apply for it by following its official tutorial.
3. Train and evaluate. We provide the experiment scripts of all benchmarks under the folder `./scripts`. You can reproduce the experiment results as follows:
```bash
bash ./scripts/SMD.sh
bash ./scripts/MSL.sh
bash ./scripts/SMAP.sh
bash ./scripts/PSM.sh
```

After training completes, you can evaluate the saved checkpoint with
```bash
python main.py --mode test [your args]
```
Use the same arguments as during training&mdash;especially the `--model_tag`
option&mdash;so that the correct model is loaded for testing.

Especially, we use the adjustment operation proposed by [Xu et al, 2018](https://arxiv.org/pdf/1802.03903.pdf) for model evaluation. If you have questions about this, please see this [issue](https://github.com/thuml/Anomaly-Transformer/issues/14) or email us.

## Main Result

We compare our model with 15 baselines, including THOC, InterFusion, etc. **Generally,  Anomaly-Transformer achieves SOTA.**

<p align="center">
<img src=".\pics\result.png" height = "450" alt="" align=center />
</p>

## Continual Experiment

`incremental_experiment.py` now trains a single model on the full dataset. When
CPD (change-point detection) signals drift, previously generated normal samples
are replayed together with the new data to update the model. This dynamic
approach leverages the VAE branch to mitigate concept drift.

### Required arguments

- `--dataset`: name of the dataset to use.
- `--data_path`: path to the dataset directory.
- `--win_size`: sliding window size (default `100`).
- `--input_c`: number of input channels.
- `--output_c`: number of output channels.
- `--batch_size`: training batch size (default `256`).
- `--num_epochs`: training epochs (default `10`).
- `--lr`: learning rate for the Adam optimizer (default `1e-4`).
- `--k`: weighting factor for the association discrepancy losses (default `3`).
- `--anomaly_ratio`: anomaly ratio in training set (default `1.0`).
- `--model_save_path`: directory for checkpoints and results. By default a
  timestamped folder is created under `outputs/<dataset>/`.
- `--model_type`: `transformer` or `transformer_ae` (default
  `transformer_ae`).
- `--cpd_penalty`: penalty used by `ruptures` for change point detection
  (default `20`). A larger value results in fewer detected drifts.
- `--replay_horizon`: keep latent vectors for at most this many training
  steps when using the VAE model (default `None`).
- `--store_mu`: store `(mu, logvar)` pairs instead of sampled `z` for replay.
- `--freeze_after`: freeze the encoder after this many updates (default `None`).
- `--ema_decay`: apply EMA to encoder weights with this decay (default `None`).
- `--decoder_type`: choose decoder architecture: `mlp`, `rnn`, or `attention`.
- `--min_cpd_gap`: minimum separation between detected change points (default
  `30`).
- `--cpd_log_interval`: evaluate and print metrics only after this many CPD
  updates (default `20`).
- `--replay_plot`: optional path for saving a figure comparing replayed samples
  with the training data. A success message with the absolute location is
  printed after saving.
- `--cpd_top_k`: number of zoomed views for CPD visualization (default `3`).
- `--cpd_extra_ranges`: comma-separated `start:end` pairs for fixed CPD zoom
  windows (default `0:4000`).

After training, the script prints the number of updates triggered by CPD events.
Install the `ruptures` package (e.g., via `pip install ruptures`) so that these
change-point detection updates can occur. The `--cpd_penalty` argument controls
the sensitivity of this detection.

### Example

```bash
python incremental_experiment.py \
    --dataset SMD --data_path dataset/SMD \
    --input_c 38 --output_c 38
```

Use `--cpd_penalty` to tune how aggressively change points are detected. Larger
values, such as `--cpd_penalty 40`, will trigger fewer updates.

Training and evaluation artifacts are saved under `--model_save_path`.
When left at its default value this directory is automatically created as
`outputs/<dataset>/<timestamp>` so that results from different runs remain
organized.
Two figures, `f1_score.png` and `roc_auc.png`, visualize F1 score and ROC AUC
across the number of CPD-triggered updates. Starting with this version the
metrics are evaluated **whenever CPD causes a model update**, so each point
corresponds to a detected drift event rather than an epoch boundary.
F1 Score와 ROC AUC가 CPD 업데이트가 발생할 때마다 기록되어
`f1_score.png`와 `roc_auc.png` 파일로 저장됩니다.

## Visualization Utilities

The new module `utils/analysis_tools.py` provides helper functions for
qualitatively inspecting continual learning behavior.

- `plot_z_bank_tsne(model, loader, n_samples=500, save_path="z_bank_tsne.png")`
  projects latent vectors from the model's `z_bank` and from the provided
  dataset loader using **t-SNE**.
- `plot_z_bank_pca(model, loader, n_samples=500, save_path="z_bank_pca.png")`
  performs the same comparison with **PCA**, and `plot_z_bank_umap` relies on
  **UMAP** if available. Each helper saves a scatter plot contrasting original
  and replayed vectors.
- `visualize_cpd_detection(series, penalty=None, min_size=30, save_path="cpd_detection.png")`
  draws change points detected by `ruptures`. When `penalty` is ``None`` a
  heuristic based on series length is used, and `min_size` enforces a minimum
  gap between change points so the plot remains readable. The helper also
  accepts `zoom_range` to focus on a specific slice of the sequence, a
  `top_k` option that automatically creates additional zoomed-in figures around
  the most significant change points, and `extra_zoom_ranges` for arbitrary
  fixed-range views (e.g. `0:4000`).
- `plot_projection_by_segment(data, segments, method="tsne", feature=None)`
  visualizes raw data windows with **t-SNE** or **PCA**. Using `feature=None`
  (the default) plots all features and colors each point by its time segment so
  distribution shifts become apparent.
- `visualize_cpd_detection(series, penalty=20, save_path="cpd_detection.png")`
  draws change points detected by `ruptures` on top of a sequence so that you
  can confirm whether CPD corresponds to actual distribution shifts.
- A quick demo script `scripts/visualize_cpd_demo.py` generates a toy series and
  saves `cpd_demo.png`, `tsne_demo.png`, and `pca_demo.png` so you can verify
  that these utilities work without preparing a real dataset. Install the
  dependencies (`numpy`, `matplotlib`, `scikit-learn`, `ruptures`, and `torch`)
  listed in `requirements-demo.txt` with
  ```bash
  pip install -r requirements-demo.txt
  ```
  and then run the demo script from the repository root:
  ```bash
  python -m scripts.visualize_cpd_demo
  ```
  The script checks for these dependencies and exits with a message if any are
  missing.

- `scripts/zbank_autoencoder_demo.py` illustrates how to train a lightweight
  autoencoder purely on the latent vectors stored in `z_bank`. After training it
  generates `recon_tsne.png` and `recon_pca.png` visualizing how well the
  reconstructions match the original windows. Run the demo with

  ```bash
  python -m scripts.zbank_autoencoder_demo
  ```
  You can supply `--load_model path/to/checkpoint.pth` to build the
  `z_bank` from pretrained weights and `--ae_epochs` to control how long the
  lightweight autoencoder trains.

- `scripts/raw_autoencoder_demo.py` trains an encoder+decoder autoencoder
  directly on windowed time series without relying on a `z_bank`. Usage:

  ```bash
  python -m scripts.raw_autoencoder_demo
  ```
  Adjust `--epochs`, `--latent_dim`, and other arguments to explore how well a
  simple AE reconstructs the data. The demo now saves `latents.npy`,
  `hidden.npy`, and `recon_errors.npy` for inspection. It also plots
  `latent_tsne.png`, `latent_pca.png`, `hidden_tsne.png`, and `hidden_pca.png`
  showing the encoder and decoder representations. The window with the largest
  reconstruction error is stored in `worst_window.npy` and its index is printed
  to the console.
  simple AE reconstructs the data.

- `scripts/visualize_dataset_distribution.py` contrasts the training and test
  splits of the benchmark datasets (SMD, SMAP, MSL, PSM) using
  `plot_projection_by_segment`. Provide the dataset name and path and it will
  save a scatter plot like `smd_tsne_segments.png`. Example usage:

  ```bash
  python -m scripts.visualize_dataset_distribution --dataset SMD --data_path dataset/SMD
  ```


Directories in the provided `save_path` are created automatically, so you can
use paths such as `outputs/z_bank_tsne.png` without pre-creating the folder.

When using the AE-based model (`--model_type transformer_ae`), these
visualizations are generated automatically at the end of training and saved
alongside the metric plots.


## Citation
If you find this repo useful, please cite our paper.

```
@inproceedings{
xu2022anomaly,
title={Anomaly Transformer: Time Series Anomaly Detection with Association Discrepancy},
author={Jiehui Xu and Haixu Wu and Jianmin Wang and Mingsheng Long},
booktitle={International Conference on Learning Representations},
year={2022},
url={https://openreview.net/forum?id=LzQQ89U1qm_}
}
```

## Contact
If you have any question, please contact wuhx23@mails.tsinghua.edu.cn.
