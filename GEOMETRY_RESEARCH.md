# Geometry-Aware Wafer PatchCore

This repository keeps the original PatchCore path as an ablation baseline and
adds a detection/localization-only wafer path.

## Synthetic data layout

Synthetic data must use parallel `images/train`, `masks/train`, `images/val`,
and `masks/val` trees. Image and mask relative directories must match. Mask
stems may equal the image stem or end in `_mask`.

## Pretraining

Set `PYTHONPATH=src`, then run `bin/train_mask_cbam.py` with the synthetic root,
checkpoint path, backbone, and selected layers. Run
`bin/train_embedding_adapter.py` with the synthetic root, real normal root,
subdataset, CBAM checkpoint, and output checkpoint.

Both commands support CPU and CUDA. The CBAM checkpoint is selected by
synthetic validation AUPRO. The embedding adapter is trained from real normal
patches and synthetic masks, then frozen before memory-bank construction.

## Geometry-aware run

Use the normal `run_patchcore.py` command structure with `--use_geometry`,
`--cbam_checkpoint`, `--embedding_adapter_path`, the polar-bin options, and
`sampler -p 0.1 --seed 0 geometry_coreset`.

The geometry memory bank stores normal patches only. Classification, FCA, and
randomly initialized CBAM execution are disabled in the main runner.

## Required ablations

Evaluate the original core, foreground geometry, geometry-balanced sampling,
neighboring-bin matching, mask-pretrained CBAM, the embedding adapter, and the
complete method with fixed data splits and at least three seeds.
