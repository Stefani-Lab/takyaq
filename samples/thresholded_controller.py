"""
Sample thresholded controller.

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
from __future__ import annotations
import numpy as np
from takyaq.base_classes import BaseController


class ThresholdPIController(BaseController):
    """Thresholded Proportional controller."""

    def __init__(self,
                 threshold_distance_nm: float,
                 Kp_above: float,
                 max_shift: float):
        """Init internal data."""
        self._threshold = threshold_distance_nm
        self._max_disp = max_shift
        self._Kp_plusultra = Kp_above
        self._Kp = np.ones((3,))  # Kp for X, Y and Z

    def set_Kp(self, Kp: float | tuple[float]):
        """Save new Kp."""
        self._Kp[:] = np.array(Kp)

    def set_Ki(self, *args):
        """Ki is unsupported by this controller, ignore."""
        pass

    def reset_xy(self, n_xy_rois: int):
        """Initialize all necessary internal structures for XY.

        Nothing to do in this case
        """
        pass

    def reset_z(self):
        """Initialize all necessary internal structures for Z.

        Nothing to do in this case
        """
        pass

    def response(self, t: float, xy_shifts: np.ndarray | None, z_shift: float):
        """Process a mesaurement of the displacements.

        Any parameter can be NAN, so we have to take it into account.
        If xy_shifts has not been measured, a None will be received.
        Must return a 3-item tuple representing the response in x, y and z
        """
        if xy_shifts is None:
            x_shift = y_shift = 0.0
        else:
            x_shift, y_shift = np.nanmean(xy_shifts, axis=0)
        if x_shift is np.nan:
            print("x shift is NAN")
            x_shift = 0.0
        if y_shift is np.nan:
            print("y shift is NAN")
            y_shift = 0.0
        error = np.array((x_shift, y_shift, z_shift))

        Kp_to_use = self._Kp.copy()
        Kp_to_use[np.abs(error) >= self._threshold] = self._Kp_plusultra

        rv = error * Kp_to_use
        rv = np.where(rv < self._max_disp, rv, np.sign(rv) * self._max_disp)
        return -rv
