"""
trial.py — Trial Logic for RDM Task
=====================================
Handles single-trial execution: fixation → stimulus → response → feedback.
Returns a structured result dict suitable for the CSV logger.

Usage
-----
    from trial import run_trial
    result = run_trial(win, rdm_stim, fixation, trial_def, config, clock)
"""

import numpy as np
from psychopy import visual, core, event


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_trial(win, rdm_stim, fixation_stim, trial_def, config, global_clock,
              show_feedback=False, trigger_port=None, trigger_cfg=None):
    """
    Execute one RDM trial.

    Parameters
    ----------
    win : psychopy.visual.Window
    rdm_stim : stimulus.RandomDotMotion
    fixation_stim : psychopy.visual.ShapeStim  (crosshair)
    trial_def : dict  with keys:
        'coherence'  – float, fraction of coherent dots
        'direction'  – str, 'left' or 'right'
        'trial_number' – int
    config : dict  (full experiment config)
    global_clock : psychopy.core.Clock  (started at experiment onset)
    show_feedback : bool
    trigger_port : object or None
        Optional serial-like object with a ``write`` method.
    trigger_cfg : dict or None
        Optional trigger mapping and encoding configuration.

    Returns
    -------
    dict with all columns required for the DDM CSV output.
    """
    timing    = config['timing']
    keys_cfg  = config['keys']
    left_key  = keys_cfg['left_response']
    right_key = keys_cfg['right_response']
    trigger_cfg = trigger_cfg or config.get('triggers', {})

    trial_start_code = int(trigger_cfg.get('trial_start', 20))
    stimulus_code = _build_stimulus_trigger_code(
        trial_def['coherence'], trial_def['direction'], trigger_cfg
    )
    left_response_code = int(trigger_cfg.get('left_response', 1))
    right_response_code = int(trigger_cfg.get('right_response', 2))
    correct_feedback_code = int(trigger_cfg.get('correct_feedback', 10))
    incorrect_feedback_code = int(trigger_cfg.get('incorrect_feedback', 11))
    timeout_feedback_code = int(trigger_cfg.get('timeout_feedback', 12))

    response_keys = [left_key, right_key, keys_cfg['quit']]

    # ---------- Configure stimulus ----------
    rdm_stim.set_coherence(trial_def['coherence'])
    rdm_stim.set_direction(trial_def['direction'])
    rdm_stim.reset()

    # ---------- Fixation period ----------
    fix_dur = _jittered_duration(timing['fixation_duration'],
                                 timing['fixation_jitter'])
    _draw_fixation(
        win, fixation_stim, fix_dur,
        trigger_port=trigger_port, trigger_code=trial_start_code,
    )

    # ---------- Stimulus + response window ----------
    max_dur = timing['stimulus_max_duration']
    stimulus_onset_holder = [np.nan]

    event.clearEvents(eventType='keyboard')
    trial_clock = core.Clock()
    stimulus_trigger_sent = False

    response        = None
    reaction_time   = None
    response_time   = None   # absolute time since experiment start

    while trial_clock.getTime() < max_dur:
        if not stimulus_trigger_sent:
            win.callOnFlip(_send_trigger, trigger_port, stimulus_code)
            win.callOnFlip(_store_clock_time, global_clock, stimulus_onset_holder)
            win.callOnFlip(trial_clock.reset)
            stimulus_trigger_sent = True
        rdm_stim.draw()
        fixation_stim.draw()
        win.flip()

        # Check for response
        keys = event.getKeys(keyList=response_keys, timeStamped=trial_clock)
        if keys:
            key_name, rt = keys[0]
            if key_name == keys_cfg['quit']:
                _quit_experiment(win)
            response      = 'left'  if key_name == left_key else 'right'
            reaction_time = rt
            stimulus_onset = stimulus_onset_holder[0]
            response_time = stimulus_onset + rt
            if response == 'left':
                _send_trigger(trigger_port, left_response_code)
            else:
                _send_trigger(trigger_port, right_response_code)
            break

    # If no response within window, mark as missing
    if response is None:
        reaction_time = np.nan
        response_time = np.nan
        stimulus_onset = stimulus_onset_holder[0]
    else:
        stimulus_onset = stimulus_onset_holder[0]

    trial_end = global_clock.getTime()

    # ---------- Accuracy ----------
    if response is None:
        accuracy = np.nan
    else:
        accuracy = int(response == trial_def['direction'])

    # ---------- Optional feedback ----------
    if show_feedback and response is not None:
        feedback_code = correct_feedback_code if accuracy == 1 else incorrect_feedback_code
        _show_feedback(
            win, accuracy, timing['feedback_duration'],
            trigger_port=trigger_port, trigger_code=feedback_code,
        )
    elif show_feedback and response is None:
        _show_text_feedback(
            win, "Too slow!", timing['feedback_duration'],
            trigger_port=trigger_port, trigger_code=timeout_feedback_code,
        )

    # ---------- Inter-trial interval ----------
    iti = _jittered_duration(timing['iti_duration'], timing['iti_jitter'])
    _blank_screen(win, iti)

    # ---------- Build result record ----------
    result = {
        'trial_number':        trial_def['trial_number'],
        'coherence':           trial_def['coherence'],
        'direction':           trial_def['direction'],
        'trial_start_trigger': trial_start_code,
        'stimulus_trigger':    stimulus_code,
        'response_trigger':    left_response_code if response == 'left' else right_response_code if response == 'right' else np.nan,
        'feedback_trigger':    correct_feedback_code if accuracy == 1 else incorrect_feedback_code if accuracy == 0 else timeout_feedback_code if response is None and show_feedback else np.nan,
        'response':            response if response is not None else 'none',
        'accuracy':            accuracy,
        'reaction_time':       round(reaction_time, 6) if not _is_nan(reaction_time) else np.nan,
        'stimulus_onset_time': round(stimulus_onset, 6),
        'response_time':       round(response_time, 6) if not _is_nan(response_time) else np.nan,
        'trial_duration':      round(trial_end - stimulus_onset, 6),
    }
    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _draw_fixation(win, fixation_stim, duration, trigger_port=None,
                   trigger_code=None):
    """Draw fixation cross for `duration` seconds."""
    fixation_clock = core.Clock()
    first_frame = True
    while fixation_clock.getTime() < duration:
        if first_frame and trigger_code is not None:
            win.callOnFlip(_send_trigger, trigger_port, trigger_code)
            first_frame = False
        fixation_stim.draw()
        win.flip()


def _blank_screen(win, duration):
    """Show blank screen for `duration` seconds."""
    blank_clock = core.Clock()
    while blank_clock.getTime() < duration:
        win.flip()


def _show_feedback(win, accuracy, duration, trigger_port=None,
                   trigger_code=None):
    """Show green tick or red cross feedback."""
    color = [0.0, 0.8, 0.0] if accuracy else [0.9, 0.0, 0.0]
    symbol = '✓' if accuracy else '✗'
    fb = visual.TextStim(win, text=symbol, color=color,
                         height=1.5, units='deg', bold=True)
    fb_clock = core.Clock()
    first_frame = True
    while fb_clock.getTime() < duration:
        if first_frame and trigger_code is not None:
            win.callOnFlip(_send_trigger, trigger_port, trigger_code)
            first_frame = False
        fb.draw()
        win.flip()


def _show_text_feedback(win, text, duration, trigger_port=None,
                        trigger_code=None):
    """Show a text string as feedback."""
    fb = visual.TextStim(win, text=text, color=[0.9, 0.5, 0.0],
                         height=0.7, units='deg')
    fb_clock = core.Clock()
    first_frame = True
    while fb_clock.getTime() < duration:
        if first_frame and trigger_code is not None:
            win.callOnFlip(_send_trigger, trigger_port, trigger_code)
            first_frame = False
        fb.draw()
        win.flip()


def _jittered_duration(base, jitter):
    """Return base ± Uniform(0, jitter/2)."""
    return base + np.random.uniform(-jitter / 2, jitter / 2)


def _build_stimulus_trigger_code(coherence, direction, trigger_cfg):
    """Encode condition-specific stimulus onset trigger."""
    base_code = int(trigger_cfg.get('stimulus_base', 100))
    coherence_scale = float(trigger_cfg.get('coherence_scale', 100))
    direction_codes = trigger_cfg.get('direction_codes', {})
    direction_code = int(direction_codes.get(direction, 0))
    coherence_code = int(round(float(coherence) * coherence_scale))
    return base_code + coherence_code + direction_code


def _store_clock_time(clock, holder):
    """Store the current time from a clock into a single-item list."""
    holder[0] = clock.getTime()


def _send_trigger(port, code):
    """Write a single-byte trigger code to an optional hardware port."""
    if port is None or code is None:
        return

    code = int(code)
    if code < 0 or code > 255:
        raise ValueError(f'Trigger code must fit in one byte: {code}')

    port.write(code.to_bytes(1, 'big'))


def _quit_experiment(win):
    """Clean exit on Escape key."""
    win.close()
    core.quit()


def _is_nan(x):
    """Safe NaN check for scalars."""
    try:
        return np.isnan(x)
    except (TypeError, ValueError):
        return False


# ---------------------------------------------------------------------------
# Trial list generation
# ---------------------------------------------------------------------------

def build_trial_list(config, practice=False):
    """
    Create a fully-specified list of trial dicts.

    Parameters
    ----------
    config : dict  (full experiment config)
    practice : bool  – if True, builds practice trial list

    Returns
    -------
    list of dicts, each containing 'coherence', 'direction', 'trial_number'
    """
    design = config['design']
    stim   = config['stimulus']

    if practice:
        coherences  = [float(design['practice_coherence'])]
        n_per_cond  = design['n_practice_trials'] // 2  # half each direction
        if n_per_cond < 1:
            n_per_cond = 1
    else:
        coherences = [float(c) for c in stim['coherence_levels']]
        n_per_cond = int(design['n_trials_per_condition'])

    directions = stim['directions']

    trials = []
    for coh in coherences:
        for direction in directions:
            for _ in range(n_per_cond):
                trials.append({'coherence': coh, 'direction': direction})

    if design['randomize_trials']:
        np.random.shuffle(trials)

    # Assign sequential trial numbers
    for i, t in enumerate(trials):
        t['trial_number'] = i + 1

    return trials


def split_into_blocks(trials, n_blocks):
    """
    Divide a flat trial list into `n_blocks` roughly equal blocks.
    Returns a list of lists.
    """
    total = len(trials)
    block_size = int(np.ceil(total / n_blocks))
    blocks = []
    for i in range(0, total, block_size):
        blocks.append(trials[i: i + block_size])
    return blocks
