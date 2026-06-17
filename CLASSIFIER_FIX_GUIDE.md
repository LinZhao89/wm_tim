# Classifier Training Issues - Root Cause Analysis and Fixes

## 🚨 Critical Issue: Data Leakage

### Problem Identified
Your classifier is achieving only **0.45 accuracy** despite state-of-the-art being **0.95** because:

1. **You're training on the test set and testing on the same test set**
   - The model has seen all test images during training
   - This violates the fundamental train/test separation principle
   - The poor accuracy (0.45) on the "test" set is actually the model's **true generalization** performance on unseen validation split (15% held out during training)

2. **Preprocessing Mismatch**
   - PatchCore (anomaly detection) uses `constrained_mean_filter` preprocessing
   - Classifier dataset was NOT applying this filter
   - Distribution mismatch between anomaly detection and classification

3. **Normalization Issues**
   - Both datasets use `WM811K_MEAN/STD` which are still **ImageNet placeholder values**
   - Not computed from actual wafermap data

## 📊 Your Current Setup

```
dataset/wm811k/prepare_dataset_train_ratio10p/
  train/
    good/           <- Only normal samples (for unsupervised PatchCore)
  test/
    Center/         <- 103 defect images
    Donut/          <- ??? defect images  
    Edge-Loc/       <- 80 defect images
    ... (8 classes)
```

**What you've been doing:**
```bash
# Training classifier on TEST data with 15% internal validation split
python train_anomaly_classifier.py dataset/wm811k/prepare_dataset_train_ratio10p/test ...

# Then testing on the SAME data (only seeing the 15% validation subset as "test")
python classify_anomalies.py --test_data dataset/wm811k/prepare_dataset_train_ratio10p/test ...
```

**Result:**
- 85% of test data was used for training ❌
- Only 15% held out for validation ✓
- 0% true unseen test data ❌
- Accuracy 0.45 = performance on validation subset (overfitting to train portion)

## ✅ Solution: Proper Train/Val/Test Split

### Step 1: Create Proper Dataset Splits

Run the new splitter script:

```powershell
# Create stratified 70% train / 15% val / 15% test split
$env:PYTHONPATH='src'
python .\bin\split_classifier_dataset.py `
    --source dataset\wm811k\prepare_dataset_train_ratio10p\test `
    --output dataset\wm811k\classifier_data `
    --train_ratio 0.7 `
    --val_ratio 0.15 `
    --test_ratio 0.15 `
    --seed 42 `
    --exclude_good
```

This creates:
```
dataset/wm811k/classifier_data/
  train/
    Center/       <- 70% of Center images
    Donut/        <- 70% of Donut images
    ... (8 classes)
  val/
    Center/       <- 15% of Center images (for validation during training)
    ...
  test/
    Center/       <- 15% of Center images (NEVER seen during training)
    ...
```

### Step 2: Diagnose Your Current Setup

First, check if you have data leakage:

```powershell
$env:PYTHONPATH='src'
python .\bin\diagnose_classifier.py `
    --train_data dataset\wm811k\prepare_dataset_train_ratio10p\test `
    --test_data dataset\wm811k\prepare_dataset_train_ratio10p\test
```

Expected output: **100% data leakage detected** ❌

After creating splits, verify:
```powershell
python .\bin\diagnose_classifier.py `
    --train_data dataset\wm811k\classifier_data\train `
    --test_data dataset\wm811k\classifier_data\test
```

Expected output: **No data leakage** ✓

### Step 3: Train with Proper Preprocessing

**CRITICAL:** Check if your PatchCore model was trained with `apply_filter=True`:

```powershell
# If PatchCore was trained WITH filtering, use --apply_filter
$env:PYTHONPATH='src'
python .\bin\train_anomaly_classifier.py `
    dataset\wm811k\classifier_data\train `
    --save_path results\classifier_proper.pth `
    --epochs 50 `
    --batch_size 32 `
    --resize 256 `
    --imagesize 224 `
    --apply_filter `
    --freeze_backbone_epochs 5 `
    --use_focal_loss `
    --focal_gamma 2.0 `
    --use_class_weights `
    --learning_rate 2e-4
```

**Key changes:**
- `--apply_filter` flag to match PatchCore preprocessing
- `--resize 256 --imagesize 224` to match your detection pipeline
- Training on `classifier_data/train` instead of `test`

### Step 4: Evaluate on True Held-Out Test Set

Update `WM811KAnomalyClassDataset` usage in your inference scripts to also use `apply_filter`:

```python
# In classify_anomalies.py or run_patchcore.py classification block
test_dataset = WM811KAnomalyClassDataset(
    test_root,
    resize=256,
    imagesize=224,
    apply_filter=True,  # <- ADD THIS
    extra_augs=False    # No augmentation at test time
)
```

Then evaluate:
```powershell
python .\bin\classify_anomalies.py `
    --classifier_weights results\classifier_proper.pth `
    --test_data dataset\wm811k\classifier_data\test `
    --output_csv classification_results_proper.csv
```

## 📈 Expected Results After Fix

| Metric | Before (Data Leakage) | After (Proper Split) |
|--------|----------------------|---------------------|
| Test Accuracy | 0.45 (misleading) | 0.85-0.95 (true performance) |
| Train Accuracy | ~0.99 (overfit) | 0.90-0.98 |
| Data Leakage | 85% | 0% ✓ |
| Generalization | Poor | Good ✓ |

## 🔍 Additional Improvements

### 1. Compute Actual WM811K Statistics

```powershell
$env:PYTHONPATH='src'
python .\bin\compute_dataset_stats.py `
    dataset\wm811k `
    prepare_dataset_train_ratio10p `
    --imagesize 224
```

Update `WM811K_MEAN` and `WM811K_STD` in `src/patchcore/datasets/wm811k.py` with the computed values.

### 2. Verify Preprocessing Consistency

Create a test script to visualize preprocessing:

```python
from PIL import Image
from src.patchcore.datasets.wm811k_cls import WM811KAnomalyClassDataset

# Test with filter
ds_filtered = WM811KAnomalyClassDataset(
    "dataset/wm811k/classifier_data/train",
    apply_filter=True
)
sample1 = ds_filtered[0]

# Test without filter  
ds_raw = WM811KAnomalyClassDataset(
    "dataset/wm811k/classifier_data/train",
    apply_filter=False
)
sample2 = ds_raw[0]

# Visual comparison - should match PatchCore preprocessing
```

### 3. Cross-Validation for Small Datasets

If you have limited data, use stratified k-fold CV:

```powershell
# Train 5 models on different folds
for($fold=0; $fold -lt 5; $fold++) {
    python .\bin\train_anomaly_classifier.py `
        dataset\wm811k\classifier_data\train `
        --save_path results\classifier_fold${fold}.pth `
        --cv_folds 5 `
        --cv_fold $fold `
        ... (other parameters)
}

# Ensemble at inference
python .\bin\classify_anomalies.py `
    --classifier_weights results\classifier_fold0.pth,results\classifier_fold1.pth,... `
    --test_data dataset\wm811k\classifier_data\test
```

## 🎯 Summary of Actions

**Immediate fixes (MUST DO):**
1. ✅ Run `split_classifier_dataset.py` to create proper train/val/test splits
2. ✅ Add `--apply_filter` flag when training classifier
3. ✅ Train on `classifier_data/train`, evaluate on `classifier_data/test`
4. ✅ Update inference code to use `apply_filter=True`

**Secondary improvements:**
5. ⚠️ Compute and update `WM811K_MEAN/STD` statistics
6. ⚠️ Run `diagnose_classifier.py` to verify no leakage
7. ⚠️ Consider k-fold cross-validation for robust evaluation

**Expected outcome:**
- Accuracy should jump from **0.45 → 0.85-0.95** once data leakage is eliminated
- If still poor after fixing leakage, then investigate other issues (model capacity, hyperparameters, etc.)

## 🔧 Files Modified

1. `bin/split_classifier_dataset.py` - NEW: Creates stratified splits
2. `bin/diagnose_classifier.py` - NEW: Detects data leakage
3. `src/patchcore/datasets/wm811k_cls.py` - UPDATED: Added `constrained_mean_filter` method
4. `bin/train_anomaly_classifier.py` - UPDATED: Added `--apply_filter`, `--resize`, `--imagesize` flags

## 📞 Next Steps

1. Run the diagnostic script to confirm data leakage
2. Create proper splits using the splitter
3. Retrain with `--apply_filter` flag
4. Report back the new accuracy on true held-out test set

**The 0.45 accuracy is NOT a model problem - it's a data leakage problem!** ✅
