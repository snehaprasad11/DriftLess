# data/

Raw EOG datasets live here. **This folder is gitignored** — the recordings are
large and EOG is biometric (a person can be re-identified from it), so it must
never be committed. Only this README is tracked.

## What to download (Phase 1)

Start with **Dataset 1 only: zero-centred horizontal + vertical bipolar EOG**
from the UM Malta EyeCon collection.

**Why this one first:** it's the cleanest — already zero-centred, only two
channels (horizontal + vertical) — so you learn the pipeline without head-pose
complications. Expand to Datasets 2–3 only after the full pipeline works on this.

## First step

Read the dataset's **"Data Description" file fully** before writing any code —
you cannot model a signal you have never looked at. Then plot a few trials
(Phase 1.2) and find the two saccade "steps", the blink "spike", and the slow
baseline drift with your own eyes.

## Expected structure (fill in once downloaded)

```
data/
├── README.md          # this file (tracked)
└── dataset1/          # (gitignored) the zero-centred H+V bipolar EOG recordings
    └── ...
```
