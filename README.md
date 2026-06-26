# Geometry-Aware Wafer Map Anomaly Detection

This repository is adapted from PatchCore for wafer map anomaly detection and localization. The current codebase focuses on **detection and localization only**. The archived anomaly classification path, Frequency and Channel Attention (FCA), and randomly initialized CBAM execution are not part of the current main experiment path.

The main research path is:

1. original PatchCore baseline;
2. foreground-aware wafer preprocessing;
3. polar geometry-aware patch metadata;
4. geometry-balanced memory-bank sampling;
5. neighboring-bin nearest-neighbor matching;
6. optional mask-pretrained CBAM and embedding adapter;
7. controlled localization evaluation with paired synthetic images and masks.

The repository still keeps the original PatchCore implementation as a baseline. The wafer-specific contribution is implemented through the geometry-aware PatchCore path and the synthetic-mask-assisted localization evaluation path.

---

## Environment

Install the requirements and expose the source folder:

```bash
pip install -r requirements.txt
export PYTHONPATH=src
```

On Windows PowerShell:

```powershell
$env:PYTHONPATH='src'
```

Use `bin/run_patchcore_compat.py` for current wafer experiments. This compatibility launcher maps the public CLI flag `--anomaly_scorer_num_nn` to the internal model argument `anomaly_score_num_nn`, so the nearest-neighbor setting is not silently ignored by older model classes.

---

## Expected WM811K layout

The WM811K data should be prepared into an MVTec-like folder layout:

```text
wm811k/
  prepare_dataset_train_ratio10p/
    train/
      good/
        *.png
    test/
      good/
        *.png
      Center/
        *.png
      Donut/
        *.png
      Edge-Loc/
        *.png
      Edge-Ring/
        *.png
      Loc/
        *.png
      Near-full/
        *.png
      Random/
        *.png
      Scratch/
        *.png
    ground_truth/
      Center/
        *_mask.png
      Donut/
        *_mask.png
      ...
```

When using the command-line interface, the dataset root is passed as `/path/to/wm811k`, and the prepared split folder is passed through `-d prepare_dataset_train_ratio10p`.

---

## Baseline PatchCore run

```bash
PYTHONPATH=src python bin/run_patchcore_compat.py \
  --gpu 0 --seed 0 --save_patchcore_model --save_image_scores \
  --log_group wm811k_patchcore_baseline --log_project wm811k results \
  patch_core \
  -b wideresnet50 -le layer2 -le layer3 \
  --pretrain_embed_dimension 1024 \
  --target_embed_dimension 1024 \
  --anomaly_scorer_num_nn 1 \
  --patchsize 3 \
  sampler -p 0.01 --seed 0 seeded_random \
  dataset \
  --resize 128 --imagesize 128 \
  --subdatasets prepare_dataset_train_ratio10p \
  wm811k /path/to/wm811k
```

The command saves image-level scores when `--save_image_scores` is enabled. By default, the score file is written under the run folder as `image_scores_<dataset>.csv`.

---

## Geometry-aware PatchCore run

The geometry-aware path adds wafer foreground estimation, polar patch bins, geometry-balanced memory-bank sampling, and neighboring-bin nearest-neighbor search.

```bash
PYTHONPATH=src python bin/run_patchcore_compat.py \
  --gpu 0 --seed 0 --save_patchcore_model --save_image_scores \
  --log_group wm811k_geometry_patchcore --log_project wm811k results \
  patch_core \
  -b wideresnet50 -le layer2 -le layer3 \
  --pretrain_embed_dimension 1024 \
  --target_embed_dimension 1024 \
  --anomaly_scorer_num_nn 1 \
  --patchsize 3 \
  --use_geometry \
  --radial_bins 4 \
  --angular_bins 8 \
  --min_wafer_coverage 0.5 \
  --geometry_radial_neighbors 1 \
  --geometry_angular_neighbors 1 \
  sampler -p 0.1 --seed 0 geometry_coreset \
  dataset \
  --resize 128 --imagesize 128 \
  --transform_mode resize_pad \
  --subdatasets prepare_dataset_train_ratio10p \
  wm811k /path/to/wm811k
```

For geometry-aware PatchCore, use `geometry_coreset`. The geometry path requires this sampler because each selected memory-bank patch must keep its polar-bin metadata.

---

## Optional mask-pretrained modules

The geometry-aware path can optionally load a mask-pretrained CBAM checkpoint and an embedding adapter:

```bash
PYTHONPATH=src python bin/run_patchcore_compat.py \
  --gpu 0 --seed 0 --save_patchcore_model --save_image_scores \
  --log_group wm811k_geometry_cbam_adapter --log_project wm811k results \
  patch_core \
  -b wideresnet50 -le layer2 -le layer3 \
  --pretrain_embed_dimension 1024 \
  --target_embed_dimension 1024 \
  --anomaly_scorer_num_nn 1 \
  --patchsize 3 \
  --use_geometry \
  --cbam_checkpoint checkpoints/mask_cbam.pth \
  --embedding_adapter_path checkpoints/embedding_adapter.pth \
  sampler -p 0.1 --seed 0 geometry_coreset \
  dataset \
  --resize 128 --imagesize 128 \
  --transform_mode resize_pad \
  --subdatasets prepare_dataset_train_ratio10p \
  wm811k /path/to/wm811k
```

Randomly initialized CBAM and FCA are archived in the main runner. Use `--cbam_checkpoint` together with `--use_geometry` if CBAM is needed.

---

## Synthetic-mask localization evaluation

Pixel-level labels are often unavailable for real wafer maps. This repository therefore supports a controlled localization evaluation set built from paired synthetic images and masks.

```text
synthetic_root/
  images/
    train/
      good/
    val/
      Center/
      Donut/
      Edge-Loc/
      ...
  masks/
    train/
      good/
    val/
      Center/
      Donut/
      Edge-Loc/
      ...
```

Image and mask stems must match. A mask may also use the `_mask` suffix. Use the synthetic mask set only as a controlled localization benchmark. It should not be described as real wafer pixel-level ground truth.

Example:

```bash
PYTHONPATH=src python bin/run_patchcore_compat.py \
  --gpu 0 --seed 0 --save_image_scores \
  --log_group wm811k_geometry_synth_pixel_eval --log_project wm811k results \
  patch_core \
  -b wideresnet50 -le layer2 -le layer3 \
  --pretrain_embed_dimension 1024 \
  --target_embed_dimension 1024 \
  --anomaly_scorer_num_nn 1 \
  --patchsize 3 \
  --use_geometry \
  sampler -p 0.1 --seed 0 geometry_coreset \
  dataset \
  --resize 128 --imagesize 128 \
  --transform_mode resize_pad \
  --synthetic_pixel_eval_path /path/to/synthetic_root \
  --synthetic_pixel_eval_subdataset all \
  --subdatasets prepare_dataset_train_ratio10p \
  wm811k /path/to/wm811k
```

---

## Filtering and preprocessing

The WM811K loader applies a constrained mean filter by default. It can be controlled with:

```bash
--filter_window_size 3 --filter_threshold 1.25
```

To disable the filter:

```bash
--no-apply_filter
```

The default wafer transform mode is `resize_pad`, which preserves wafer content and avoids center-crop loss. Other available modes are `resize_only` and `resize_crop`.

---

## Outputs

Each run stores the main metrics in the run folder. Depending on the flags, the run may also save:

```text
results/
  <project>/
    <group>_<run_id>/
      results.csv
      image_scores_<dataset>.csv
      per_category_metrics_<dataset>.csv
      models/
        <dataset>/
          patchcore_params.pkl
          nnscorer_search_index.faiss
          geometry_memory.pkl
```

For geometry-aware runs, `geometry_memory.pkl` stores the sampled memory-bank features and their polar-bin metadata.

---

## Recommended ablations

For a paper experiment, evaluate the following components under fixed data splits and at least three random seeds:

1. original PatchCore;
2. foreground-aware wafer masking;
3. geometry-balanced coreset sampling;
4. neighboring-bin nearest-neighbor matching;
5. mask-pretrained CBAM;
6. embedding adapter;
7. complete geometry-aware method.

Report image-level AUROC on real WM811K labels. Report pixel-level AUROC only on real masks when real masks are available. If the masks are generated or synthetic, report them as controlled synthetic-mask localization results.

---

## Archived components

The following components are kept only for historical reference and are not part of the current main method:

- anomaly classification after detection;
- Frequency and Channel Attention through `--use_fca`;
- randomly initialized CBAM through `--use_cbam`.

The current main runner will reject archived paths that are no longer supported.

---

## Citation

This repository is adapted from PatchCore. If you use the original PatchCore implementation, please cite:

```bibtex
@misc{roth2021total,
  title={Towards Total Recall in Industrial Anomaly Detection},
  author={Karsten Roth and Latha Pemula and Joaquin Zepeda and Bernhard Schölkopf and Thomas Brox and Peter Gehler},
  year={2021},
  eprint={2106.08265},
  archivePrefix={arXiv},
  primaryClass={cs.CV}
}
```

## License

This project is licensed under the Apache-2.0 License.
