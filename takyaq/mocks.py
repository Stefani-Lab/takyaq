#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mocks
=====

  This module provide a mock camera and piezo stage. It is useful for testing and
  developing without equipment.

Copyright (C) 2025 Andrés Zelcer and others

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Affero General Public License as
    published by the Free Software Foundation, either version 3 of the
    License, or (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Affero General Public License for more details.

    You should have received a copy of the GNU Affero General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
import numpy as _np
import time as _time
import logging as _lgn
from .base_classes import BaseCamera, BasePiezo


_lgn.basicConfig()
_lgr = _lgn.getLogger(__name__)
_lgr.setLevel(_lgn.DEBUG)


def gaussian2D(grid, amplitude, x0, y0, sigma, offset):
    """2D gaussian."""

    x, y = grid
    x0 = float(x0)
    y0 = float(y0)
    a = 1.0 / (2 * sigma**2)
    G = offset + amplitude * _np.exp(-(a * ((x - x0) ** 2) + a * ((y - y0) ** 2)))
    return G


class MockCamera(BaseCamera):
    """Mock camera for testing and development.

    Simulates a drift that:
        - Shifts X position in a sinusoidal way with period 4.
        - Shifts Y position in a sinusoidal way with period 4e.
        - Shifts Z position in a triangular way with period 4pi.

    Moreover, it adds random noise to the position of the marks and a
    background noise.
    """

    max_x = 1200
    max_y = 900
    centers = (
        (100, 100),
        (100, 300),
        (100, 500),
        (100, 700),
        (300, 100),
        (300, 300),
        (300, 500),
        (300, 700),
        (500, 100),
        (500, 300),
        (500, 500),
        (500, 700),
        (800, 800),
    )
    sigma = 200.  # FWHM of the signals, in nm
    _X_PERIOD = 4
    _Y_PERIOD = _np.e*4
    _Z_PERIOD = _np.pi*4
    _shifts = _np.zeros((3,), dtype=_np.float64)
    grid = _np.array(_np.meshgrid(_np.arange(max_x), _np.arange(max_y), indexing="ij"))
    _exposure = 0.050
    _gain = 1.
    # f = True  # Camera fail first call flag

    def __init__(self, nmpp_x, nmpp_y, nmpp_z, sigma, z_ang: float, noise_level=3,
                 drift_amplitude=3):
        """Init Mock camera.

        Parameters
        ----------
        nmpp_x : float
            nanometers per pixel in X direction.
        nmpp_y : float
            nanometers per pixel in X direction.
        nmpp_z : float
            nanometers per pixel in Z direction (see comment).
        sigma : float
            FWHM of the signals, in nm.
        z_ang: float (radians)
            angle between z spot displacement and positive x direction
        noise_level : float, optional
            Random shifts of XYZ positions in nm. The default is 3.
        drift_amplitude : float, optional
            Amplitude of the periodic shift in nm. The default is 3.
        """
        self._nl = noise_level
        self._drift = drift_amplitude
        self._nmpp_x = nmpp_x
        self._nmpp_y = nmpp_y
        self._nmpp_z = nmpp_z
        self._z_ang = z_ang
        self._rot_vec = _np.array((_np.cos(self._z_ang), _np.sin(self._z_ang),))
        self.sigma = sigma

    def get_image(self):
        """Return a faked image."""
        # if not self.f:  # Camera fail mocking
        #     if _np.random.random_sample() > 0.9:  # falla una de cada 10
        #         raise ValueError("error en camara")
        # self.f = False
        rv = _np.zeros((self.max_x, self.max_y,), dtype=_np.float64)
        t = _time.monotonic()
        # limit gaussian creation to +-4 sigma from center for speed
        slice_size = int(self.sigma / self._nmpp_x * 4)  # in pixels
        for x0, y0 in self.centers[:-1]:
            x0 += self._shifts[0] / self._nmpp_x
            y0 += self._shifts[1] / self._nmpp_y
            x0 += (
                (_np.random.random_sample() - 0.5) * self._nl
                + _np.sin(t / self._X_PERIOD * 2 * _np.pi) * self._drift
            ) / self._nmpp_x
            y0 += (
                (_np.random.random_sample() - 0.5) * self._nl
                + _np.sin(t / self._Y_PERIOD * 2 * _np.pi) * self._drift
            ) / self._nmpp_y
            cx = int(x0)
            cy = int(y0)
            slicex = slice(max(cx - slice_size, 0), min(cx + slice_size, self.max_x))
            slicey = slice(max(cy - slice_size, 0), min(cy + slice_size, self.max_y))
            rv[slicex, slicey] += gaussian2D(
                self.grid[:, slicex, slicey], 100, x0, y0, self.sigma / self._nmpp_x, 0
            )

        # Z mocking: triangular wave
        r = self._shifts[2] + self._drift * (2 * abs(2 * (
            t / self._Z_PERIOD - _np.floor(t / self._Z_PERIOD + 0.5)))-1)
        r *= self._rot_vec / self._nmpp_z
        r += (_np.random.random_sample((2,)) - 0.5) * self._nl
        r += _np.array(self.centers[-1])

        cx = int(r[0])
        cy = int(r[1])
        slicex = slice(max(cx - slice_size, 0), min(cx + slice_size, self.max_x))
        slicey = slice(max(cy - slice_size, 0), min(cy + slice_size, self.max_y))
        rv[slicex, slicey] += gaussian2D(  # use X coordinate nmpp, since it maps OK
            self.grid[:, slicex, slicey], 100, r[0], r[1], self.sigma / self._nmpp_x, 0
        )
        rv *= self._gain
        rv += _np.random.poisson(2, (self.max_x, self.max_y))
        rv *= (self._exposure / 0.05)  # 50ms default
        _time.sleep(max(t + self._exposure - _time.monotonic(), 0))
        return rv.astype(_np.uint16)

    def shift(self, dx: float, dy: float, dz: float):
        """Shift origin of coordinates.

        Simulates a stage movement.
        """
        self._shifts += _np.array(
            (
                dx,
                dy,
                dz,
            )
        )

    def set_exposure(self, new_exp: float):
        """Set camera exposure."""
        _lgr.info("New exposure: %s", new_exp)
        self._exposure = float(new_exp)

    def set_gain(self, new_gain: float):
        """Set camera gain."""
        _lgr.info("New gain: %s", new_gain)
        self._gain = float(new_gain)


class MockPiezo(BasePiezo):
    """Mock piezoelectric motor.

    It can shift the zero of the mock camera, to test stabilization strategies.
    """

    _pos: _np.ndarray = _np.zeros((3,))

    def __init__(self, camera=None):
        self._camera = camera

    def get_position(self):
        return tuple(self._pos)

    def set_position_xy(self, x: float, y: float):
        npos = _np.array((x, y, self._pos[2],), dtype=float)
        if self._camera:
            self._camera.shift(
                *(npos - self._pos)
            )
        self._pos = npos
        return

    def set_position_z(self, z: float):
        npos = _np.array((self._pos[0], self._pos[1], z,), dtype=float)
        if self._camera:
            self._camera.shift(
                *(npos - self._pos)
            )
        self._pos = npos
        return

    def get_limits(self):
        return ((-10, 10),) * 3
