# MammoLens

**Author:** Ibrahim Tamilore Esther

I fine-tuned an EfficientNet-B0 to tell benign from malignant findings on whole mammograms, evaluated it separately in non-dense and dense breasts, and used Grad-CAM to look at what it attends to. This is an undergraduate project prototype. It is not a medical device and nothing here should be used to make decisions about anyone's care.

## What happened, in short

My first run reached a test AUC of 0.7541 and looked like it had found something: the model performed worse in dense breasts, AUC 0.8188 against 0.6950, with p = 0.0116. Then I audited my own data pipeline and found it had quietly thrown away 246 usable full mammograms and 106 patients because of a gap in the dataset's metadata. Recovering them took my dataset from 2857 to 3085 images, and the retrained model came out **worse**, not better, at a test AUC of 0.6627. That was not what I expected, and most of this project is the work of finding out why.

Two further experiments traced it. Running the full training schedule instead of stopping early changed almost nothing (0.6694), so the model had not been undertrained. Adding the recovered images to my original partitions, rather than splitting the data again, recovered the original performance in full: test AUC 0.7527, within 0.0014 of my first run, and a paired difference of -0.0041 on the 400 images both runs held out. **So the loss came from re-splitting the data, not from the images I recovered.** The recovered data are safe to use, and I use them.

The density finding is the part that did not survive. I have four estimates of the gap (0.1238, 0.0750, 0.0673 and 0.0945), and the two that reach significance are the two measured on largely the same 400 test images. I report it as unreplicated across independent samples rather than as a result.

## The data

CBIS-DDSM, the Curated Breast Imaging Subset of the Digital Database for Screening Mammography: a curated set of digitised screen-film mammograms with verified pathology, published by Lee et al. (2017, *Scientific Data* 4, 170177, doi:10.1038/sdata.2017.177) and derived from DDSM (Heath et al., 2001). I use the JPEG conversion that ships with `dicom_info.csv` and the four case description files, held here as `archive.zip` and extracted into `data/`.

One thing worth being clear about, because it limits every claim I make: every image in this dataset already contains an annotated abnormality. So the task I am actually studying is separating benign findings from malignant ones, not finding cancer among the mostly normal examinations a screening programme sees. I do not redistribute the dataset in this repository.

## Layout

```
MammoLens/
├── configs/config.yaml          hyperparameters and paths, my single source of truth
├── data/                        csv/ and jpeg/ from the archive, plus the working split manifests
├── src/                         config, data, model, train, evaluate, gradcam
├── notebooks/                   one notebook per run (see below)
├── initial_run/                 v1 artefacts, frozen and verified
├── recovered_data_run/          v2 artefacts
├── recovered_data_full_schedule/  v3 artefacts
└── original_split_with_recovered_images/  v4 artefacts
```

Each run folder holds `results/`, `checkpoints/`, `figures/` and, where it applies, `manifests/`. I renamed these folders from their original `*_v1`, `*_v3` and bare `results/` forms on 23 September 2026, and `initial_run/manifests/SNAPSHOT_COMPLETE.json` records the old names along with the md5 of every file it protects.

## My four runs

| Run | Notebook | Artefacts | What it is |
|---|---|---|---|
| v1 | `MammoLens.ipynb` | `initial_run/` | My main analysis: linkage, patient-grouped split, training, evaluation, Grad-CAM. Ran June 2026. |
| v2 | `MammoLens_recovered_data_rebuild.ipynb` | `recovered_data_run/` | Recovers the 246 discarded mammograms, rebuilds the manifest, re-splits, retrains, re-evaluates and compares itself with v1. |
| v3 | `MammoLens_recovered_data_full_schedule.ipynb` | `recovered_data_full_schedule/` | My control for undertraining: the v2 data and split with early stopping disabled, so all 12 epochs run. |
| v4 | `MammoLens_original_split_with_recovered_images.ipynb` | `original_split_with_recovered_images/` | Separates the two remaining explanations by adding the recovered images to my v1 partitions instead of re-splitting. **My primary result.** |

`notebooks/MammoLens.backup-before-reformat.ipynb` is my v1 notebook as it stood before I rewrote its prose. I keep it only as a record.

## Which run I report, and why

**v4 (`MammoLens_original_split_with_recovered_images.ipynb`) is my primary result.** It is the only run trained on the complete dataset, 3085 images from 1565 patients, that also performs at the level of my original run: test AUC 0.7527 with sensitivity 0.6369 and specificity 0.7027 on 438 images, and the best calibration of any run I did, with an expected calibration error of 0.1654 for non-dense and 0.2182 for dense images. It also has the best-documented training run, because by then I was logging every epoch to a CSV.

v1 is the corroborating result at 0.7541 on 400 images, and the two agree on identical data: on the 400 test images they share, the paired AUC difference is -0.0041 with a 95% interval of [-0.0421, +0.0343]. Because my v4 test set is built out of v1's, those two are not independent samples, and I say so wherever I quote both.

v2 and v3 I report as what they are: the record of a rebuild that went wrong, and the two controls that found out why. I have not hidden them, because the mistake and the diagnosis are the most useful thing I learned in this project.

On the density gap, my four estimates from the same code are 0.1238 (p = 0.0116) in v1, 0.0750 (p = 0.1518) in v2, 0.0673 (p = 0.1915) in v3 and 0.0945 (p = 0.0398) in v4. The two that reach significance at image level are the two whose test sets overlap by 400 images, and v4's own patient-level test does not agree (p = 0.0736). On that evidence the gap sits at the edge of what a few hundred images per stratum can detect, and I do not claim it as a finding.

## Reproducing a run

The notebooks are written for Google Colab with the project on Google Drive, because training needs a GPU and every epoch reads about 2200 images.

1. Upload the project to Drive at `MyDrive/MammoLens` and open the notebook for the run you want in Colab.
2. Set the runtime to GPU (Runtime, Change runtime type, GPU).
3. Run the first cell, which mounts Drive and installs `requirements.txt`, then run the rest in order.
4. Expect roughly 7 minutes per epoch on a T4, so about 85 minutes for a 12-epoch run, plus another 10 minutes for the scoring and the bootstrap analyses.

Every hyperparameter lives in `configs/config.yaml` and they are identical across all four runs: AdamW, learning rate 0.0003, weight decay 0.0001, cosine annealing over 12 epochs, batch size 32, dropout 0.3, mixed precision, unweighted cross-entropy, no class weighting, a random horizontal flip at p = 0.5 with rotation within 10 degrees, 224 by 224 inputs and seed 42. Holding them fixed is what makes the runs comparable. Each experiment notebook redirects the configured checkpoint and results paths into its own run folder, so `configs/config.yaml` can keep the plain defaults.

Training also runs outside a notebook:

```bash
python -m src.train --config configs/config.yaml
```

Two things in `src/train.py` are worth knowing about, because I added both after being caught out. `train(cfg, run_tag=...)` refuses to resume a checkpoint written by a different run, which is what stops a rebuild from quietly reporting an earlier model as its own; that guard is the reason my v2 numbers are v2's. And `log_csv=` appends one row per epoch, with the same rows stored inside the checkpoint, so a run resumed after a dropped Colab session rewrites its log without a gap.

The figures, tables and numbers I use in the report are generated by a separate throwaway script that writes outside this repository, so nothing in here changes when I rebuild them.

## Caveats I carry into the report

Every result rests on a single training run with a single seed, and my test sets hold between 400 and 438 images. The bootstrap intervals for AUC span roughly 0.10, so small differences between runs cannot be resolved and I do not try to read them. Mammograms of several thousand pixels per side are reduced to 224 by 224 without preserving aspect ratio, which is likely to erase microcalcifications, and that matters because calcification cases make up part of the data. The models are poorly calibrated, with expected calibration errors between 0.18 and 0.29 by density group in the later runs, so their probabilities should not be read as risk estimates. And my v4 test set overlaps v1's by construction, so those two runs are not independent.
