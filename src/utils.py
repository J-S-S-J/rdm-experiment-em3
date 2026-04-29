"""
utils.py — Utility Functions
==============================
Timing helpers, CSV data logger, config loader, and GUI dialog.

Usage
-----
    from utils import load_config, DataLogger, show_instructions, get_participant_info
"""

import os
import csv
import json
import datetime
import numpy as np
from psychopy import visual, core, event, gui, logging as psy_logging


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

def load_config(config_path):
    """
    Load experiment parameters from a JSON config file.

    Parameters
    ----------
    config_path : str  – absolute or relative path to the JSON file

    Returns
    -------
    dict
    """
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    return config


# ---------------------------------------------------------------------------
# Participant info dialog
# ---------------------------------------------------------------------------

def get_participant_info():
    """
    Show a GUI dialog to collect participant ID and session number.
    Blocks until the user submits or cancels.

    Returns
    -------
    dict with keys 'participant_id' and 'session'
    Raises SystemExit if the user cancels.
    """
    dlg_data = {
        'Participant ID': '',
        'Session': '1',
        'Task Type': ['control', 'binaural'],
        'Age': '',
        'Handedness': ['right', 'left', 'ambidextrous'],
    }
    dlg = gui.DlgFromDict(
        dictionary=dlg_data,
        title='Random Dot Motion Task',
        order=['Participant ID', 'Session', 'Task Type', 'Age', 'Handedness'],
    )
    if not dlg.OK:
        core.quit()

    participant_id = str(dlg_data['Participant ID']).strip()
    if not participant_id:
        participant_id = 'ANON'

    return {
        'participant_id': participant_id,
        'session': str(dlg_data['Session']),
        'task_type': str(dlg_data['Task Type']).lower(),
        'age': str(dlg_data['Age']),
        'handedness': str(dlg_data['Handedness']),
    }


# ---------------------------------------------------------------------------
# Results directory & file naming
# ---------------------------------------------------------------------------

def make_results_path(results_dir, participant_id, session='1'):
    """
    Build a timestamped CSV filename and ensure the results directory exists.

    Returns
    -------
    str – full path to the output CSV
    """
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{participant_id}_ses{session}_{timestamp}.csv"
    return os.path.join(results_dir, filename)


# ---------------------------------------------------------------------------
# CSV data logger
# ---------------------------------------------------------------------------

# Canonical column order for DDM analysis
CSV_COLUMNS = [
    'participant_id',
    'session',
    'task_type',
    'trial_number',
    'block_number',
    'is_practice',
    'coherence',
    'direction',
    'trial_start_trigger',
    'stimulus_trigger',
    'response_trigger',
    'feedback_trigger',
    'response',
    'accuracy',
    'reaction_time',
    'stimulus_onset_time',
    'response_time',
    'trial_duration',
]


class DataLogger:
    """
    Writes trial-by-trial data to a CSV file row-by-row
    (safe against crashes – data is flushed after every trial).

    Parameters
    ----------
    filepath : str     – output CSV path
    participant_info : dict  – keys: participant_id, session, etc.
    """

    def __init__(self, filepath, participant_info):
        self.filepath = filepath
        self.participant_info = participant_info
        self._file = open(filepath, 'w', newline='', encoding='utf-8')
        self._writer = csv.DictWriter(self._file, fieldnames=CSV_COLUMNS,
                                      extrasaction='ignore')
        self._writer.writeheader()
        self._file.flush()

    def log_trial(self, trial_result, block_number=0, is_practice=False):
        """
        Append one trial row to the CSV.

        Parameters
        ----------
        trial_result : dict   – returned by trial.run_trial()
        block_number : int
        is_practice  : bool
        """
        row = {
            'participant_id':    self.participant_info['participant_id'],
            'session':           self.participant_info.get('session', '1'),
            'task_type':         self.participant_info.get('task_type', 'control'),
            'block_number':      block_number,
            'is_practice':       int(is_practice),
            **trial_result,
        }
        self._writer.writerow(row)
        self._file.flush()   # write to disk immediately

    def close(self):
        """Close the CSV file."""
        if not self._file.closed:
            self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# ---------------------------------------------------------------------------
# Screen / instruction helpers
# ---------------------------------------------------------------------------

def show_message(win, text, keys=None, wait_secs=None, text_height=0.6):
    """
    Display a text message and wait for a keypress or timeout.

    Parameters
    ----------
    win : psychopy.visual.Window
    text : str
    keys : list of str  – accepted keys (None = any key)
    wait_secs : float   – optional timeout in seconds
    """
    stim = visual.TextStim(
        win,
        text=text,
        height=text_height,
        units='deg',
        wrapWidth=25,
        color=[-1, -1, -1],
        colorSpace='rgb',
        alignText='center',
    )
    event.clearEvents(eventType='keyboard')
    wait_clock = core.Clock()
    while True:
        stim.draw()
        win.flip()
        if wait_secs is not None and wait_clock.getTime() >= wait_secs:
            break
        pressed = event.getKeys(keyList=keys)
        if pressed:
            if 'escape' in pressed:
                win.close()
                core.quit()
            break


def build_instruction_text(template, config):
    """
    Fill in key names in instruction template strings.

    Parameters
    ----------
    template : str  – text with {left_key}, {right_key} etc. placeholders
    config : dict   – full experiment config

    Returns
    -------
    str
    """
    return template.format(
        left_key=config['keys']['left_response'].upper(),
        right_key=config['keys']['right_response'].upper(),
        continue_key=config['keys']['continue'].upper(),
        quit_key=config['keys']['quit'].upper(),
        n=config['design']['n_practice_trials'],
    )


# ---------------------------------------------------------------------------
# Fixation cross
# ---------------------------------------------------------------------------

def make_fixation(win, config):
    """
    Create a fixation cross ShapeStim.

    Returns
    -------
    psychopy.visual.ShapeStim
    """
    fix_cfg = config['fixation']
    size    = float(fix_cfg['size_deg'])
    color   = fix_cfg['color']

    # Draw as two overlapping rectangles (horizontal + vertical bars)
    fix = visual.ShapeStim(
        win,
        vertices=[
            (-size, 0.05), (size, 0.05),
            (size, -0.05), (-size, -0.05),
        ],
        fillColor=color,
        lineColor=color,
        colorSpace='rgb',
        units='deg',
        closeShape=True,
    )
    fix2 = visual.ShapeStim(
        win,
        vertices=[
            (-0.05, size), (0.05, size),
            (0.05, -size), (-0.05, -size),
        ],
        fillColor=color,
        lineColor=color,
        colorSpace='rgb',
        units='deg',
        closeShape=True,
    )
    # Combine into a single drawable object via a helper class
    return FixationCross(fix, fix2)


class FixationCross:
    """Thin wrapper so we can call .draw() on both bars at once."""
    def __init__(self, bar1, bar2):
        self._bars = [bar1, bar2]

    def draw(self):
        for b in self._bars:
            b.draw()


# ---------------------------------------------------------------------------
# Timing precision check
# ---------------------------------------------------------------------------

def check_timing(win, n_frames=120):
    """
    Measure actual frame duration and warn if it is irregular.
    Call after window creation.

    Returns
    -------
    float  – estimated frames-per-second
    """
    intervals = []
    t0 = win.flip()
    for _ in range(n_frames):
        t1 = win.flip()
        intervals.append(t1 - t0)
        t0 = t1

    intervals = np.array(intervals)
    fps = 1.0 / np.mean(intervals)
    cv  = np.std(intervals) / np.mean(intervals)   # coefficient of variation

    if cv > 0.05:
        psy_logging.warning(
            f"Frame timing variability is high (CV={cv:.3f}). "
            "Consider closing other applications."
        )

    return fps
