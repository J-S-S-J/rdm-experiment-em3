# Random Dot Motion Task (DDM-ready)

A fully functional PsychoPy implementation of the Random Dot Kinematogram (RDK) task, designed for fitting a **Drift Diffusion Model (DDM)**.

---

## Directory Structure

```
rdm_experiment/
├── src/
│   ├── main.py            ← Entry point; run this file
│   ├── stimulus.py        ← RandomDotMotion class (Britten et al. 1992 RDK)
│   ├── trial.py           ← Single-trial logic + trial-list builder
│   ├── utils.py           ← Config loader, CSV logger, UI helpers
│   └── analyze_ddm.py     ← Post-hoc data check + HDDM-ready export
├── raw/
│   └── config.json        ← ALL experiment parameters (edit here!)
└── results/               ← Created automatically; one CSV per participant
    └── logs/              ← PsychoPy log files
```

---

## Quick Start

### 1. Install dependencies

```bash
# Option A – PsychoPy standalone (recommended for timing accuracy)
# Download from https://www.psychopy.org/download.html

# Option B – pip install into a virtual environment
pip install psychopy numpy
```

### 2. Run the experiment

```bash
cd rdm_experiment
python src/main.py
```

A dialog will appear asking for **Participant ID**, session number, age, and handedness.

### 3. Find your data

Results are saved to:
```
results/<ParticipantID>_ses<N>_<YYYYMMDD_HHMMSS>.csv
```

---

## CSV Output Columns

| Column | Description |
|---|---|
| `participant_id` | ID entered at dialog |
| `session` | Session number |
| `trial_number` | Sequential trial index |
| `block_number` | Block index (0 = practice) |
| `is_practice` | 1 for practice trials, 0 for main |
| `coherence` | Fraction of coherent dots (0–1) |
| `direction` | Correct direction (`left` / `right`) |
| `response` | Participant's response (`left` / `right` / `none`) |
| `accuracy` | 1 = correct, 0 = error, NaN = no response |
| `reaction_time` | Time from stimulus onset to keypress (seconds) |
| `stimulus_onset_time` | Absolute time of stimulus onset since experiment start |
| `response_time` | Absolute time of keypress since experiment start |
| `trial_duration` | Duration from stimulus onset to trial end |

---

## Modifying Parameters

All parameters live in **`raw/config.json`** — no code changes needed.

### Key parameters to adjust

```json
"stimulus": {
    "coherence_levels": [0.05, 0.1, 0.2, 0.4],  ← your coherence conditions
    "n_dots": 200,                                 ← dot density
    "dot_speed_deg_per_sec": 5.0,                 ← motion speed
    "dot_lifetime_frames": 3,                     ← limited lifetime (noise control)
    "aperture_radius_deg": 5.0                    ← stimulus window size
},

"design": {
    "n_trials_per_condition": 10,   ← trials per coherence × direction cell
    "n_practice_trials": 8,
    "n_blocks": 4
},

"timing": {
    "stimulus_max_duration": 2.0,   ← response deadline (seconds)
    "fixation_duration": 0.5,
    "fixation_jitter": 0.25         ← ± jitter on fixation duration
}
```

### Response keys

```json
"keys": {
    "left_response":  "f",    ← change to any key
    "right_response": "j"
}
```

---

## Fitting a Drift Diffusion Model

### Step 1: Check and export data

```bash
python src/analyze_ddm.py results/<ParticipantID>_ses1_<timestamp>.csv
```

This prints an accuracy × RT summary and writes two files:
- `*_summary.csv`  — condition-level means
- `*_ddm_ready.csv` — HDDM-compatible trial file

### Step 2: Fit with HDDM

```python
import hddm

data = hddm.load_csv('results/<participant>_ddm_ready.csv')

# Drift rate varies with coherence (the key DDM prediction)
m = hddm.HDDM(data, depends_on={'v': 'coherence'})
m.find_starting_values()
m.sample(2000, burn=500)
m.print_stats()
```

Install HDDM:
```bash
conda create -n ddm python=3.9
conda activate ddm
pip install hddm
```

### Step 2 (alternative): Fit with PyDDM

```python
from pyddm import Model, Fittable
from pyddm.models import DriftCoherence, NoiseConstant, BoundConstant

model = Model(drift=DriftCoherence(driftcoh=Fittable(minval=0, maxval=20)),
              noise=NoiseConstant(noise=1),
              bound=BoundConstant(B=Fittable(minval=0.1, maxval=1.5)),
              T_dur=2.0)
# See https://pyddm.readthedocs.io for full fitting tutorial
```

Install PyDDM:
```bash
pip install pyddm
```

---

## Timing Precision Notes

- The window uses `waitBlanking=True` for frame-locked timing.
- Response timestamps use PsychoPy's `event.getKeys(timeStamped=clock)` for sub-millisecond accuracy relative to stimulus onset.
- A 60-frame timing check runs at startup; a warning is logged if frame intervals are variable (CV > 5%).
- Fixation and ITI durations are jittered to decorrelate neural/behavioural events.
- For maximum timing precision, use the **PsychoPy standalone app** and close other applications.

---

## References

- Britten, K. H., Shadlen, M. N., Newsome, W. T., & Movshon, J. A. (1992). The analysis of visual motion: a comparison of neuronal and psychophysical performance. *Journal of Neuroscience*, 12(12), 4745–4765.
- Ratcliff, R., & McKoon, G. (2008). The diffusion decision model: theory and data for two-choice decision tasks. *Psychological Review*, 115(2), 350–378.
- Wiecki, T. V., Sofer, I., & Frank, M. J. (2013). HDDM: Hierarchical Bayesian estimation of the drift-diffusion model in Python. *Frontiers in Neuroinformatics*, 7, 14.
