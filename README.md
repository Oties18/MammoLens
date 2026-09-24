# MammoLens

Benign versus malignant classification of mammograms with a fine-tuned EfficientNet-B0, evaluated separately in non-dense and dense breasts and inspected with Grad-CAM. The work is an undergraduate project prototype, not a medical device, and nothing here supports clinical use.

**Author:** [NAME] ([STUDENT NUMBER]), [MODULE], [INSTITUTION]

## The short version

The first run reached a test AUC of 0.7541 and appeared to show that the model performs worse in dense breasts (AUC 0.8188 against 0.6950, p = 0.0116). An audit then found that the data pipeline had silently discarded 246 usable full mammograms and 106 patients because of a gap in the dataset's metadata. Recovering them enlarged the dataset from 2857 to 3085 images, and the retrained model was **worse**, not better: test AUC 0.6627.

Two further experiments traced that loss to its cause. Running the full training schedule instead of stopping early changed nothing (0.6694), so the model had not been undertrained. Adding the recovered images to the original partitions, rather than re-splitting the data, recovered the original performance in full: test AUC 0.7527, within 0.0014 of the first run, and a paired difference of -0.0041 on the 400 images both runs held out. **The deficit came from re-splitting the data, not from the recovered images.** The recovered data are safe to use.

The density difference is the part that did not survive. Four estimates of the gap exist (0.1238, 0.0750, 0.0673 and 0.0945), and the two that reach significance are the two measured on largely the same 400 test images. It should be reported as unreplicated across independent samples.

## Dataset

CBIS-DDSM, the Curated Breast Imaging Subset of the Digital Database for Screening Mammography: a curated set of digitised screen-film mammograms with verified pathology, published by Lee et al. (2017, *Scientific Data* 4, 170177, doi:10.1038/sdata.2017.177) and derived from DDSM (Heath et al., 2001). This project uses the JPEG conversion distributed with `dicom_info.csv` and the four case description files, held here as `archive.zip` and extracted into `data/`.

Every image in the dataset contains an annotated abnormality, so the task is separating benign from malignant findings, not detecting cancer among normal screening examinations. The dataset is not redistributed in this repository.

## Layout

```
MammoLens/
├── configs/config.yaml          hyperparameters and paths, the single source of truth
├── data/                        csv/ and jpeg/ from the archive, plus the working split manifests
├── src/                         config, data, model, train, evaluate, gradcam
├── notebooks/                   one notebook per run (see below)
├── initial_run/                 v1 artefacts, frozen and verified
├── recovered_data_run/          v2 artefacts
├── recovered_data_full_schedule/  v3 artefacts
└── original_split_with_recovered_images/  v4 artefacts
```

Each run folder holds `results/`, `checkpoints/`, `figures/` and, where relevant, `manifests/`. The folders were renamed from their original `*_v1`, `*_v3` and bare `results/` forms on 23 September 2026; `initial_run/manifests/SNAPSHOT_COMPLETE.json` records the old names alongside the md5 of every file it protects.

## The four runs

| Run | Notebook | Artefacts | What it is |
|---|---|---|---|
| v1 | `MammoLens.ipynb` | `initial_run/` | The main analysis: linkage, patient-grouped split, training, evaluation, Grad-CAM. Ran June 2026. |
| v2 | `MammoLens_recovered_data_rebuild.ipynb` | `recovered_data_run/` | Recovers the 246 discarded mammograms, rebuilds the manifest, re-splits, retrains, re-evaluates, and compares itself with v1. |
| v3 | `MammoLens_recovered_data_full_schedule.ipynb` | `recovered_data_full_schedule/` | Control for undertraining: the v2 data and split with early stopping disabled, so all 12 epochs run. |
| v4 | `MammoLens_original_split_with_recovered_images.ipynb` | `original_split_with_recovered_images/` | Separates the two remaining explanations by adding the recovered images to the v1 partitions instead of re-splitting. **The primary result.** |

`notebooks/MammoLens.backup-before-reformat.ipynb` is the v1 notebook as it stood before its prose was rewritten. It is kept only as a record.

## Which run is the primary result, and why

**v4 (`MammoLens_original_split_with_recovered_images.ipynb`) is the primary result.** It is the only run trained on the complete dataset, 3085 images from 1565 patients, that also performs at the level of the original run: test AUC 0.7527 with sensitivity 0.6369 and specificity 0.7027 on 438 images, and the best calibration of any run (expected calibration error 0.1654 for non-dense and 0.2182 for dense images). It also has the best-documented training run, with a full per-epoch log.

v1 remains the corroborating result at 0.7541 on 400 images, and the two agree on identical data: on the 400 test images they share, the paired AUC difference is -0.0041 with a 95% interval of [-0.0421, +0.0343]. Because v4's test set is built from v1's, those two are not independent samples, and that should be stated wherever both are quoted.

v2 and v3 should be reported as what they are: the record of a rebuild that went wrong, and the two controls that found out why.

The density finding should be reported as unreplicated. Four estimates of the non-dense minus dense AUC gap exist, all from the same code: 0.1238 (p = 0.0116) in v1, 0.0750 (p = 0.1518) in v2, 0.0673 (p = 0.1915) in v3 and 0.0945 (p = 0.0398) in v4. The two that reach significance at image level are the two whose test sets overlap by 400 images, and v4's own patient-level test does not agree (p = 0.0736). On this evidence the gap is at the edge of what a few hundred images per stratum can detect.

## Reproducing a run

The notebooks are written for Google Colab with the project on Google Drive, because training needs a GPU and each epoch reads about 2200 images.

1. Upload the project to Drive at `MyDrive/MammoLens` and open the notebook for the run you want in Colab.
2. Set the runtime to GPU (Runtime, Change runtime type, GPU).
3. Run the first cell, which mounts Drive and installs `requirements.txt`, then run the rest in order.
4. Expect roughly 7 minutes per epoch on a T4, so about 85 minutes for a 12-epoch run, plus 10 minutes for scoring and the bootstrap analyses.

Hyperparameters live in `configs/config.yaml` and are identical across runs: AdamW, learning rate 0.0003, weight decay 0.0001, cosine annealing over 12 epochs, batch size 32, dropout 0.3, mixed precision, unweighted cross-entropy, no class weighting, random horizontal flip at p = 0.5 with rotation within 10 degrees, 224 by 224 inputs and seed 42. Each experiment notebook redirects the configured checkpoint and results paths into its own run folder, so `configs/config.yaml` can keep the plain defaults.

Training can also be run outside a notebook:

```bash
python -m src.train --config configs/config.yaml
```

A run tag guards against silent reuse: `src.train.train(cfg, run_tag=...)` refuses to resume a checkpoint written by a different run, which is what stops a rebuild from quietly reporting an earlier model as its own. Passing `log_csv=` appends one row per epoch, and the same rows are stored inside the checkpoint so a resumed run rewrites the log without gaps.

## Caveats worth carrying into the report

Each result rests on a single training run with a single seed, and the test sets hold 400 to 438 images. Bootstrap intervals for AUC span roughly 0.10, so small differences between runs cannot be resolved. Mammograms of several thousand pixels per side are reduced to 224 by 224 without preserving aspect ratio, which is likely to erase microcalcifications. The models are poorly calibrated, with expected calibration errors between 0.18 and 0.29 by density group in the later runs, so their probabilities should not be read as risk estimates. The v4 test set overlaps v1's by construction, so those two runs are not independent.
