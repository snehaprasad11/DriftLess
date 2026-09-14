# notebooks/

Exploration and offline training — the "build-time" rhythm.

This is where the heavy, rare work happens (it does **not** run in the live
services):

- **Phase 1** — look at the raw signal, confirm drift is real and person-specific.
- **Phase 2** — cut trials into fixed-length windows, normalise, split *by person*.
- **Phase 3** — train the shared base model (run this one on **Google Colab's
  free GPU**), then record the honest "before calibration" error number.

Keep notebooks numbered so the pipeline reads top to bottom, e.g.
`01_explore.ipynb`, `02_windowing.ipynb`, `03_train_base.ipynb`.
