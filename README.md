![DeepMORSE](fig/fig.jpg)

# About DeepMORSE

Deep <u>mo</u>dality-sha<u>r</u>ed <u>s</u>elf-<u>e</u>xpressive model (DeepMORSE) is a simple but principled approach for image clustering assisted with textual information. It discovers cross-modal structures via a modality-shared self-expressive model and simultaneously learns structured representations that conform to a union of modality-specific subspaces.

# How to Run DeepMORSE?

**Step 1:** Download the extracted CLIP image and text features from [here](https://drive.google.com/drive/folders/1_qqetrZHSEjAsM8Xp7fQUUgdjLVExRnl?usp=drive_link) and put them under `./data/datasets`.

**Step 2:** Train DeepMORSE by running:

```bash
python main.py --data CIFAR-10 --n_clusters 10 --epo 50
python main.py --data CIFAR-20 --n_clusters 20 --epo 50
python main.py --data ImageNet-Dogs --n_clusters 15 --epo 50
python main.py --data DTD --n_clusters 47 --epo 100
python main.py --data UCF101 --n_clusters 101 --epo 100
python main.py --data ImageNet --n_clusters 1000 --epo 500 --hidden_dim 8192 --z_dim 1024 --bs 8192 --gamma 600
```

## Evaluation protocol

The default evaluates only the predefined final training epoch. Hyperparameters,
epoch counts and training seeds must be fixed before evaluating test labels.
`--validate_every` can enable diagnostic intermediate scores; these scores must
not be used to select the reporting checkpoint.

For each checkpoint, spectral clustering produces 10 K-means partitions using
`--eval_seed 0`. Each K-means fit uses 10 restarts selected by its unlabeled inertia.
ACC, NMI and ARI are computed from the same 10 partitions and averaged; no metric
is maximized using labels. `--eval_repeats` controls the number of repeated fits.
The evaluator uses a local RNG, so evaluating does not consume training RNG state.

Each run saves all partitions and their paired metric triples in
`evaluation_epoch_*.npz`, a final summary in `evaluation_final.json`, and the
predefined final checkpoint in `checkpoints/final.pt`. Standard deviations in
these summaries are population standard deviations (`ddof=0`) across K-means fits.
For a multi-training-seed paper table, first average the K-means fits within each
training seed, then report the mean and standard deviation across those seed means.
Do not treat repeated K-means fits as independent network-training runs.

Earlier code printed independent label-based maxima of ACC, NMI and ARI. Those
values can correspond to different partitions. This correction does not
retroactively recompute historical paper tables; those results require
reevaluation under the stated protocol.
