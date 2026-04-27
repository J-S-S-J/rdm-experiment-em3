"""
main.py — Random Dot Motion Task (DDM-ready)
=============================================
Entry point for the experiment. Run this file directly:

    python src/main.py

Or from the project root:

    python -m src.main

Directory layout expected
-------------------------
    rdm_experiment/
    ├── src/
    │   ├── main.py          ← this file
    │   ├── stimulus.py      ← RDM stimulus class
    │   ├── trial.py         ← single-trial logic & trial-list builder
    │   └── utils.py         ← config loader, logger, helpers
    ├── raw/
    │   └── config.json      ← all experiment parameters
    └── results/             ← created automatically; one CSV per participant

Dependencies
------------
    pip install psychopy numpy

References
----------
    Britten et al. (1992). The analysis of visual motion: a comparison of
        neuronal and psychophysical performance. J Neurosci.
    Ratcliff & McKoon (2008). The diffusion decision model: theory and data
        for two-choice decision tasks. Psychol Rev.
"""

import os
import sys
import math

# ---------------------------------------------------------------------------
# Path setup – allows running from any working directory
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# ---------------------------------------------------------------------------
# PsychoPy must be imported before other psychopy submodules
# ---------------------------------------------------------------------------
from psychopy import visual, core, event, monitors, sound, logging as psy_logging

# Local modules
from stimulus import RandomDotMotion
from trial    import run_trial, build_trial_list, split_into_blocks
from utils    import (
    load_config, get_participant_info, make_results_path,
    DataLogger, show_message, build_instruction_text,
    make_fixation, check_timing,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
CONFIG_PATH  = os.path.join(_ROOT, 'raw', 'config.json')
RESULTS_DIR  = os.path.join(_ROOT, 'results')
LOG_DIR      = os.path.join(_ROOT, 'results', 'logs')
SOUNDS_DIR   = os.path.join(_ROOT, 'raw', 'sounds')

PRE_EXPOSURE_DURATION_SECS = 20 * 60
PRE_EXPOSURE_SOUND_FILES = {
    'control': 'Pink-noise.wav',
    'binaural': 'BB-400-420.wav',
}


def _open_trigger_port(trigger_cfg):
    """Open an optional serial port for hardware triggers."""
    if not trigger_cfg.get('enabled'):
        return None

    port_name = trigger_cfg.get('serial_port')
    if not port_name:
        psy_logging.warning('Triggering is enabled but no serial_port was set.')
        return None

    try:
        import serial
    except ImportError:
        psy_logging.warning('pyserial is not installed; trigger output disabled.')
        return None

    baudrate = int(trigger_cfg.get('baudrate', 115200))
    try:
        return serial.Serial(port_name, baudrate=baudrate)
    except Exception as exc:
        psy_logging.warning(f'Could not open trigger port {port_name}: {exc}')
        return None


def _get_pre_exposure_setup(task_type):
    """Resolve pre-exposure file and display behavior for a task type."""
    task_type = str(task_type).strip().lower()
    if task_type not in PRE_EXPOSURE_SOUND_FILES:
        raise ValueError(
            f"Unknown task type '{task_type}'. Use one of: "
            f"{list(PRE_EXPOSURE_SOUND_FILES.keys())}"
        )

    filename = PRE_EXPOSURE_SOUND_FILES[task_type]
    file_path = os.path.join(SOUNDS_DIR, filename)
    if not os.path.isfile(file_path):
        raise FileNotFoundError(
            f"Missing pre-exposure sound file: {file_path}. "
            f"Please add it to {SOUNDS_DIR}."
        )

    if task_type == 'control':
        return file_path, True, 'CONTROL (pink noise + fixation cross)'
    return file_path, False, 'BINAURAL'


def _run_pre_exposure_block(win, fixation, config, task_type):
    """Run 20-minute pre-exposure audio phase before the RDM task."""
    audio_path, show_fixation, phase_label = _get_pre_exposure_setup(task_type)
    continue_key = config['keys']['continue']
    quit_key = config['keys']['quit']

    intro_lines = [
        f"{phase_label} PRE-PHASE",
        '',
        "Duration: 20 minutes",
    ]
    if show_fixation:
        intro_lines.append("Keep your eyes on the fixation cross.")
    intro_lines.extend([
        '',
        f"Press {continue_key.upper()} to start.",
    ])
    show_message(win, "\n".join(intro_lines), keys=[continue_key])

    pre_sound = sound.Sound(value=audio_path)
    clip_duration = float(pre_sound.getDuration())
    if not math.isfinite(clip_duration) or clip_duration <= 0.0:
        raise ValueError(f'Invalid audio duration for file: {audio_path}')

    n_loops = int(math.ceil(PRE_EXPOSURE_DURATION_SECS / clip_duration))
    phase_clock = core.Clock()
    event.clearEvents(eventType='keyboard')

    for _ in range(n_loops):
        remaining = PRE_EXPOSURE_DURATION_SECS - phase_clock.getTime()
        if remaining <= 0:
            break

        segment_duration = min(clip_duration, remaining)
        pre_sound.play()
        segment_clock = core.Clock()

        while segment_clock.getTime() < segment_duration:
            if event.getKeys(keyList=[quit_key]):
                pre_sound.stop()
                _graceful_exit(win)

            if show_fixation:
                fixation.draw()
            win.flip()

        pre_sound.stop()

    show_message(
        win,
        f"Pre-phase complete.\n\nPress {continue_key.upper()} to continue to the main experiment.",
        keys=[continue_key],
    )


# ===========================================================================
# Main experiment
# ===========================================================================

def _build_monitor(exp_cfg):
    """Create a monitor object with enough metadata for deg unit conversions."""
    mon_name = exp_cfg.get('monitor_name', 'rdm_monitor')
    mon_width_cm = float(exp_cfg.get('monitor_width_cm', 53.0))
    viewing_distance_cm = float(exp_cfg.get('viewing_distance_cm', 60.0))

    mon = monitors.Monitor(name=mon_name, width=mon_width_cm,
                           distance=viewing_distance_cm)
    mon.setSizePix((int(exp_cfg['screen_width']), int(exp_cfg['screen_height'])))
    return mon

def run_experiment():
    # ------------------------------------------------------------------
    # 1. Load config
    # ------------------------------------------------------------------
    config = load_config(CONFIG_PATH)
    exp_cfg = config['experiment']
    trigger_cfg = config.get('triggers', {})

    # ------------------------------------------------------------------
    # 2. Participant info dialog
    # ------------------------------------------------------------------
    participant_info = get_participant_info()
    pid = participant_info['participant_id']

    # ------------------------------------------------------------------
    # 3. Create results / log directories
    # ------------------------------------------------------------------
    os.makedirs(LOG_DIR, exist_ok=True)
    csv_path = make_results_path(RESULTS_DIR, pid, participant_info['session'])

    # PsychoPy log file
    psy_logging.setDefaultClock(core.Clock())
    log_path = os.path.join(
        LOG_DIR,
        os.path.splitext(os.path.basename(csv_path))[0] + '.log'
    )
    psy_logging.LogFile(log_path, level=psy_logging.WARNING)

    # ------------------------------------------------------------------
    # 4. Open window
    # ------------------------------------------------------------------
    monitor = _build_monitor(exp_cfg)
    win = visual.Window(
        size=(exp_cfg['screen_width'], exp_cfg['screen_height']),
        fullscr=exp_cfg['fullscreen'],
        color=exp_cfg['background_color'],
        colorSpace='rgb',
        units=exp_cfg['units'],
        monitor=monitor,
        allowGUI=False,
        waitBlanking=True,   # crucial for timing accuracy
    )
    win.mouseVisible = False

    # Timing check (warns if frame rate is unstable)
    measured_fps = check_timing(win, n_frames=60)
    psy_logging.info(f"Measured frame rate: {measured_fps:.1f} Hz")

    # ------------------------------------------------------------------
    # 5. Create shared stimulus objects
    # ------------------------------------------------------------------
    rdm_stim = RandomDotMotion(win, config['stimulus'])
    fixation = make_fixation(win, config)
    trigger_port = _open_trigger_port(trigger_cfg)

    # Global experiment clock (starts here; all onset times relative to this)
    global_clock = core.Clock()

    # ------------------------------------------------------------------
    # 6. Open data logger
    # ------------------------------------------------------------------
    with DataLogger(csv_path, participant_info) as logger:

        # --------------------------------------------------------------
        # 7. Instructions
        # --------------------------------------------------------------
        instr_text = build_instruction_text(
            config['text']['instructions'], config
        )
        show_message(win, instr_text, keys=[config['keys']['continue']])

        # --------------------------------------------------------------
        # 7b. 20-minute pre-exposure phase (task-type specific)
        # --------------------------------------------------------------
        _run_pre_exposure_block(
            win=win,
            fixation=fixation,
            config=config,
            task_type=participant_info['task_type'],
        )

        # --------------------------------------------------------------
        # 8. Practice block
        # --------------------------------------------------------------
        practice_text = build_instruction_text(
            config['text']['practice_start'], config
        )
        show_message(win, practice_text, keys=[config['keys']['continue']])

        practice_trials = build_trial_list(config, practice=True)
        n_correct_practice = 0

        for t_def in practice_trials:
            result = run_trial(
                win, rdm_stim, fixation, t_def, config, global_clock,
                show_feedback=config['design']['show_feedback_in_practice'],
                trigger_port=trigger_port,
                trigger_cfg=trigger_cfg,
            )
            # Adjust trial number to clearly mark as practice
            result['trial_number'] = t_def['trial_number']
            logger.log_trial(result, block_number=0, is_practice=True)

            if result['accuracy'] == 1:
                n_correct_practice += 1

            # Allow quit during practice
            if event.getKeys(keyList=[config['keys']['quit']]):
                _graceful_exit(win)

        # Practice summary
        pct_correct = round(
            100 * n_correct_practice / max(len(practice_trials), 1)
        )
        practice_end_text = config['text']['practice_end'].format(
            score=pct_correct
        )
        show_message(win, practice_end_text, keys=[config['keys']['continue']])

        # --------------------------------------------------------------
        # 9. Main experiment blocks
        # --------------------------------------------------------------
        main_trials = build_trial_list(config, practice=False)
        n_blocks    = int(config['design']['n_blocks'])
        blocks      = split_into_blocks(main_trials, n_blocks)

        # Re-number trials continuously across blocks
        trial_counter = 1
        for block_idx, block_trials in enumerate(blocks, start=1):

            for t_def in block_trials:
                t_def['trial_number'] = trial_counter
                trial_counter += 1

                result = run_trial(
                    win, rdm_stim, fixation, t_def, config, global_clock,
                    show_feedback=config['design']['show_feedback_in_main'],
                    trigger_port=trigger_port,
                    trigger_cfg=trigger_cfg,
                )
                logger.log_trial(result, block_number=block_idx,
                                 is_practice=False)

                # Allow quit at any time
                if event.getKeys(keyList=[config['keys']['quit']]):
                    _graceful_exit(win)

            # Block break (skip after the last block)
            if block_idx < n_blocks:
                break_text = config['text']['block_break'].format(
                    current=block_idx,
                    total=n_blocks,
                )
                show_message(win, break_text,
                             keys=[config['keys']['continue']])

        # --------------------------------------------------------------
        # 10. End screen
        # --------------------------------------------------------------
        show_message(win, config['text']['end'],
                     keys=[config['keys']['continue']])

    # ------------------------------------------------------------------
    # 11. Cleanup
    # ------------------------------------------------------------------
    if trigger_port is not None:
        trigger_port.close()
    win.close()
    print(f"\nExperiment complete.")
    print(f"Data saved to: {csv_path}")
    core.quit()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _graceful_exit(win):
    """Close window and exit cleanly on Escape press."""
    win.close()
    print("Experiment aborted by user.")
    core.quit()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    run_experiment()
