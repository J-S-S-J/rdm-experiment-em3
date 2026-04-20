"""
stimulus.py — Random Dot Motion Stimulus
=========================================
Implements a random dot kinematogram (RDK) where each dot keeps a motion
vector within a trial, inspired by classical RDK implementations.

At trial reset, each dot is sampled as either signal or noise:
- signal dots move in the trial direction
- noise dots move in random directions

Dots are respawned at the aperture edge opposite to the signal direction
when they leave the aperture, which maintains stable global motion flow.
"""

import numpy as np
from psychopy import visual


class RandomDotMotion:
    """
    Random Dot Motion (RDM) stimulus.

    Parameters
    ----------
    win : psychopy.visual.Window
        The PsychoPy window to draw into.
    config : dict
        The 'stimulus' section of the experiment config JSON.
    """

    # Direction angles in mathematical coordinates (degrees)
    _DIRECTION_ANGLES = {
        'right': 0.0,
        'up': 90.0,
        'left': 180.0,
        'down': 270.0,
    }

    def __init__(self, win, config):
        self.win = win
        self.cfg = config

        # ---------- core parameters ----------
        self.n_dots      = int(config['n_dots'])
        self.aperture_r  = float(config['aperture_radius_deg'])
        self.dot_speed   = float(config['dot_speed_deg_per_sec'])   # °/s
        self.lifetime    = int(config['dot_lifetime_frames'])        # frames
        self.coherence   = 0.0
        self.direction   = 'right'

        # Convert speed from °/s → °/frame using the monitor frame rate
        # We store as °/frame and multiply by frame delta in draw() for
        # sub-frame precision, but default to the nominal frame rate here.
        try:
            self._fps = win.getActualFrameRate(nIdentical=10, nMaxFrames=100,
                                               nWarmUpFrames=10, threshold=1)
            if self._fps is None or self._fps < 20:
                self._fps = 60.0
        except Exception:
            self._fps = 60.0

        self._speed_per_frame = self.dot_speed / self._fps

        # ---------- dot arrays ----------
        # Positions in Cartesian degrees, centred on (0, 0)
        self._pos = np.zeros((self.n_dots, 2), dtype=float)
        # Per-dot motion angle (radians) and per-frame displacement
        self._theta = np.zeros(self.n_dots, dtype=float)
        self._dxy = np.zeros((self.n_dots, 2), dtype=float)
        # Age counter for optional finite lifetime replacement
        self._age = np.zeros(self.n_dots, dtype=int)

        # ---------- PsychoPy dot object ----------
        self._dot_stim = visual.ElementArrayStim(
            win,
            nElements=self.n_dots,
            elementTex=None,
            elementMask='circle',
            sizes=float(config['dot_size_deg']),
            colors=config['dot_color'],
            colorSpace='rgb',
            units='deg',
            xys=self._pos,
        )

        # Keep dots inside a virtual aperture in software for speed.

        # Initialise all dots after the visual stim exists.
        self.reset()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_coherence(self, coherence: float):
        """Set the fraction of signal (coherently moving) dots, 0–1."""
        self.coherence = float(np.clip(coherence, 0.0, 1.0))

    def set_direction(self, direction: str):
        """Set the signal direction ('left', 'right', 'up', 'down')."""
        if direction not in self._DIRECTION_ANGLES:
            raise ValueError(f"Unknown direction '{direction}'. "
                             f"Choose from {list(self._DIRECTION_ANGLES)}")
        self.direction = direction

    def reset(self):
        """Reinitialise all dot positions and ages (call between trials)."""
        idx = np.arange(self.n_dots)
        self._age = np.random.randint(0, self.lifetime, size=self.n_dots)
        self._randomise_positions(idx)
        self._assign_motion_vectors(idx)
        self._update_stim_xys()

    def draw(self):
        """
        Advance dot positions by one frame and draw.
        Call once per display frame inside your stimulus presentation loop.
        """
        self._step()
        self._update_stim_xys()
        self._dot_stim.draw()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _step(self):
        """Advance all dots by one frame."""
        # Increment ages; finite-lifetime dots are replaced.
        self._age += 1
        dead = self._age >= self.lifetime

        # Reborn dots get new position and a fresh motion assignment.
        if dead.any():
            dead_idx = np.where(dead)[0]
            self._randomise_positions(dead_idx)
            self._assign_motion_vectors(dead_idx)
            self._age[dead] = 0

        # Advance all dots using their fixed per-dot motion vectors.
        self._pos += self._dxy

        # Dots leaving the aperture are reinserted at the opposite edge.
        outside = self._outside_aperture()
        if outside.any():
            out_idx = np.where(outside)[0]
            self._respawn_at_edge(out_idx)
            self._assign_motion_vectors(out_idx)
            self._age[out_idx] = 0

    def _outside_aperture(self) -> np.ndarray:
        """Return boolean mask of dots outside the circular aperture."""
        dist_sq = self._pos[:, 0] ** 2 + self._pos[:, 1] ** 2
        return dist_sq > self.aperture_r ** 2

    def _randomise_positions(self, indices):
        """
        Place dots at indices randomly in aperture using area-uniform sampling.
        """
        n = len(indices)
        if n == 0:
            return

        # r ~ sqrt(U) gives area-uniform disk samples (reduces center clustering).
        r = np.sqrt(np.random.uniform(0, self.aperture_r ** 2, size=n))
        theta = np.random.uniform(0, 2 * np.pi, size=n)
        self._pos[indices, 0] = r * np.cos(theta)
        self._pos[indices, 1] = r * np.sin(theta)

    def _assign_motion_vectors(self, indices):
        """Assign coherent or random motion vectors for selected dots."""
        n = len(indices)
        if n == 0:
            return

        signal_theta = np.deg2rad(self._DIRECTION_ANGLES[self.direction])
        is_signal = np.random.uniform(0.0, 1.0, size=n) < self.coherence

        theta = np.random.uniform(0.0, 2 * np.pi, size=n)
        theta[is_signal] = signal_theta

        self._theta[indices] = theta
        self._dxy[indices, 0] = np.cos(theta) * self._speed_per_frame
        self._dxy[indices, 1] = np.sin(theta) * self._speed_per_frame

    def _respawn_at_edge(self, indices):
        """
        Reinsert dots at aperture edge opposite to signal flow.

        A jitter of +/-90 degrees around the opposite direction avoids
        visible streaking from strict single-angle reinsertion.
        """
        n = len(indices)
        if n == 0:
            return

        signal_deg = self._DIRECTION_ANGLES[self.direction]
        opposite_deg = (signal_deg + 180.0) % 360.0
        jitter = np.random.uniform(-90.0, 90.0, size=n)
        spawn_theta = np.deg2rad((opposite_deg + jitter) % 360.0)

        edge_r = self.aperture_r
        self._pos[indices, 0] = np.cos(spawn_theta) * edge_r
        self._pos[indices, 1] = np.sin(spawn_theta) * edge_r

    def _update_stim_xys(self):
        """Push current positions into the PsychoPy ElementArrayStim."""
        self._dot_stim.setXYs(self._pos)
