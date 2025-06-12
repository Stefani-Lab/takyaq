# -*- coding: utf-8 -*-
"""
The module implement objects that react after a fiduciary localization event


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
import logging as _lgn
from typing import Optional as _Optional, Union as _Union, Tuple as _Tuple
from takyaq.base_classes import BaseController as _BaseController


_lgn.basicConfig()
_lgr = _lgn.getLogger(__name__)
_lgr.setLevel(_lgn.DEBUG)


class PIController(_BaseController):
    """PI Controller."""

    _Kp = _np.ones((3,))
    _Ki = _np.ones((3,))
    _deriv = _np.zeros((3,))
    _last_e = _np.zeros((3,))
    _cum = _np.zeros((3,))
    next_val = 0
    _last_times = _np.zeros((3,))

    def __init__(self, Kp: _Union[float, _Tuple[float]] = 1.,
                 Ki: _Union[float, _Tuple[float]] = 1.,
                 ):
        self.set_Kp(Kp)
        self.set_Ki(Ki)

    def set_Kp(self, Kp: _Union[float, _Tuple[float]]):
        self._Kp[:] = _np.array(Kp)

    def set_Ki(self, Ki: _Union[float, _Tuple[float]]):
        self._Ki[:] = _np.array(Ki)

    def reset_xy(self, n_xy_rois: int):
        """Initialize all necesary internal structures for XY."""
        self._cum[0:2] = 0.
        self._last_times[0:2] = 0.

    def reset_z(self):
        """Initialize all necesary internal structures for Z."""
        self._cum[2] = 0.
        self._last_times[2] = 0.

    def response(self, t: float, xy_shifts: _Optional[_np.ndarray], z_shift: float):
        """Process a mesaurement of the displacements.

        Any parameter can be NAN, so we have to take it into account.

        If xy_shifts has not been measured, a None will be received.

        Must return a 3-item tuple representing the response in x, y and z
        """
        if xy_shifts is None:
            x_shift = y_shift = 0.0
        else:
            x_shift, y_shift = _np.nanmean(xy_shifts, axis=0)
        if x_shift is _np.nan:
            _lgr.warning("x shift is NAN")
            x_shift = 0.0
        if y_shift is _np.nan:
            _lgr.warning("y shift is NAN")
            y_shift = 0.0

        error = _np.array((x_shift, y_shift, z_shift))
        self._last_times[_np.where(self._last_times <= 0.)] = t
        delta_t = t - self._last_times
        delta_t[_np.where(delta_t > 1)] = 1.  # protect against suspended processes
        self._cum += error * delta_t
        rv = error * self._Kp + self._Ki * self._cum
        self._last_times[:] = t
        return -rv


class RejectPIControllerSD:
    """PI Controller that rejects outliers."""

    _Kp = _np.ones((3,))
    _Ki = _np.ones((3,))
    _deriv = _np.zeros((3,))
    _last_e = _np.zeros((3,))
    _cum = _np.zeros((3,))
    _threshold = 2.0
    next_val = 0
    _last_times = _np.zeros((3,))

    def __init__(self, Kp: _Union[float, _Tuple[float]] = 1.,
                 Ki: _Union[float, _Tuple[float]] = 1.,
                 threshold: float = 2.0,
                 ):
        self.set_Kp(Kp)
        self.set_Ki(Ki)
        self._threshold = threshold  # TODO: seteable

    def set_Kp(self, Kp: _Union[float, _Tuple[float]]):
        self._Kp[:] = _np.array(Kp)

    def set_Ki(self, Ki: _Union[float, _Tuple[float]]):
        self._Ki[:] = _np.array(Ki)

    def reset_xy(self, n_xy_rois: int):
        """Initialize all necesary internal structures for XY."""
        self._cum[0:2] = 0.
        self._last_times[0:2] = 0.

    def reset_z(self):
        """Initialize all necesary internal structures for Z."""
        self._cum[2] = 0.
        self._last_times[2] = 0.

    def response(self, t: float, xy_shifts: _Optional[_np.ndarray], z_shift: float):
        """Process a mesaurement of the displacements.

        Any parameter can be NAN, so we have to take it into account.

        If xy_shifts has not been measured, a None will be received.

        Must return a 3-item tuple representing the response in x, y and z
        """
        if xy_shifts is None:
            x_shift = y_shift = 0.0
        elif len(xy_shifts) < 4:
            x_shift, y_shift = _np.nanmean(xy_shifts, axis=0)
        else:
            outliers = _np.zeros((len(xy_shifts),), dtype=bool)
            valid_idx = ~outliers
            shifts = _np.nanmean(xy_shifts[valid_idx], axis=0)
            desvs = _np.nanstd(xy_shifts[valid_idx], axis=0)
            outliers = ((_np.abs(xy_shifts - shifts) / desvs) >
                        self._threshold).sum(axis=1, dtype=bool)
            valid_idx = ~outliers
            n_valids = valid_idx.sum()  # actual
            last_n_valids = len(xy_shifts)
            while n_valids >= 3 and n_valids != last_n_valids:
                last_n_valids = n_valids
                shifts = _np.nanmean(xy_shifts[valid_idx], axis=0)
                desvs = _np.nanstd(xy_shifts[valid_idx], axis=0)
                outliers = (
                    (_np.abs(xy_shifts - shifts) / desvs) > self._threshold
                    ).sum(axis=1, dtype=bool)
                valid_idx = ~outliers
                n_valids = valid_idx.sum()
            if n_valids < 3:
                _lgr.warning("There might be invalid positions")
            x_shift, y_shift = shifts

        if x_shift is _np.nan:
            _lgr.warning("x shift is NAN")
            x_shift = 0.0
        if y_shift is _np.nan:
            _lgr.warning("y shift is NAN")
            y_shift = 0.0

        error = _np.array((x_shift, y_shift, z_shift))
        self._last_times[_np.where(self._last_times <= 0.)] = t
        delta_t = t - self._last_times
        delta_t[_np.where(delta_t > 1)] = 1.  # protect against suspended processes
        self._cum += error * delta_t
        rv = error * self._Kp + self._Ki * self._cum
        self._last_times[:] = t
        return -rv


class RejectPIControllerMAD:
    """PI Controller that rejects outliers."""

    _Kp = _np.ones((3,))
    _Ki = _np.ones((3,))
    _deriv = _np.zeros((3,))
    _last_e = _np.zeros((3,))
    _cum = _np.zeros((3,))
    _threshold = 1.5
    next_val = 0
    _last_times = _np.zeros((3,))

    def __init__(self, Kp: _Union[float, _Tuple[float]] = 1.,
                 Ki: _Union[float, _Tuple[float]] = 1.,
                 threshold: float = 1.5,
                 ):
        self.set_Kp(Kp)
        self.set_Ki(Ki)
        self._threshold = threshold  # TODO: seteable

    def set_Kp(self, Kp: _Union[float, _Tuple[float]]):
        self._Kp[:] = _np.array(Kp)

    def set_Ki(self, Ki: _Union[float, _Tuple[float]]):
        self._Ki[:] = _np.array(Ki)

    def reset_xy(self, n_xy_rois: int):
        """Initialize all necesary internal structures for XY."""
        self._cum[0:2] = 0.
        self._last_times[0:2] = 0.

    def reset_z(self):
        """Initialize all necesary internal structures for Z."""
        self._cum[2] = 0.
        self._last_times[2] = 0.

    def response(self, t: float, xy_shifts: _Optional[_np.ndarray], z_shift: float):
        """Process a mesaurement of the displacements.

        Any parameter can be NAN, so we have to take it into account.

        If xy_shifts has not been measured, a None will be received.

        Must return a 3-item tuple representing the response in x, y and z
        """
        if xy_shifts is None:
            x_shift = y_shift = 0.0
        else:
            shifts = _np.nanmean(xy_shifts, axis=0)
            if len(xy_shifts) >= 4:
                adifs = _np.abs(xy_shifts - shifts)
                MAD = _np.median(adifs, axis=0)
                aMi = 0.6745 * adifs / MAD
                outliers = (aMi > self._threshold).sum(axis=1, dtype=bool)
                valid_idx = ~outliers
                shifts = _np.nanmean(xy_shifts[valid_idx], axis=0)
                if valid_idx.sum() < 3:
                    _lgr.warning("There might be invalid positions")
            x_shift, y_shift = shifts

        if x_shift is _np.nan:
            _lgr.warning("x shift is NAN")
            x_shift = 0.0
        if y_shift is _np.nan:
            _lgr.warning("y shift is NAN")
            y_shift = 0.0

        error = _np.array((x_shift, y_shift, z_shift))
        self._last_times[_np.where(self._last_times <= 0.)] = t
        delta_t = t - self._last_times
        delta_t[_np.where(delta_t > 1)] = 1.  # protect against suspended processes
        self._cum += error * delta_t
        rv = error * self._Kp + self._Ki * self._cum
        self._last_times[:] = t
        return -rv
