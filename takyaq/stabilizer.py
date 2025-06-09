# -*- coding: utf-8 -*-
"""
Takyaq Stabilizer core

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
import scipy as _sp
import threading as _th
import logging as _lgn
import time as _time
import os as _os
from typing import Optional as _Optional, List as _List, Tuple as _Tuple
from concurrent.futures import ProcessPoolExecutor as _PPE
import warnings as _warnings
from typing import Union as _Union
from .info_types import (ROI, PointInfo, CameraInfo, StabilizationType,
                         report_callback_type, init_callback_type,
                         end_callback_type)
from . import base_classes as _bc

_lgn.basicConfig()
_lgr = _lgn.getLogger(__name__)
_lgr.setLevel(_lgn.DEBUG)


def _gaussian2D(grid, amplitude, x0, y0, sigma, offset, ravel=True):
    """Generate a 2D gaussian.

    The amplitude is not normalized, as the function result is meant to be used
    just for fitting centers.

    Parameters
    ----------
    grid: numpy.ndarray
        X, Y coordinates grid to generate the gaussian over
    amplitude: float
        Non-normalized amplitude
    x0, y0: float
        position of gaussian center
    sigma: float
        FWHM
    offset:
        uniform background value
    ravel: bool, default True
        If True, returns the raveled values, otherwise return a 2D array
    """
    x, y = grid
    x0 = float(x0)
    y0 = float(y0)
    a = 1.0 / (2 * sigma**2)
    G = offset + amplitude * _np.exp(
        -(a * ((x - x0) ** 2) + a * ((y - y0) ** 2))
    )
    if ravel:
        G = G.ravel()
    return G


def _gaussian_fit(
    data: _np.ndarray, x_max: float, y_max: float, sigma: float
) -> _Tuple[float, float, float]:
    """Fit a gaussian to an image.

    All data is in PIXEL units.

    Parameters
    ----------
    data : numpy.ndarray
        image as a 2D array
    x_max : float
        initial estimator of X position of the maximum
    y_max : float
        initial estimator of Y position of the maximum
    sigma : float
        initial estimator of spread of the gaussian

    Returns
    -------
    x : float
        X position of the maximum
    y : float
        y position of the maximum
    sigma : float
        FWHM

    If case of an error, returns numpy.nan for every value

    Raises
    ------
        Should not raise
    """
    try:
        xdata = _np.meshgrid(
            _np.arange(data.shape[0]), _np.arange(data.shape[1]), indexing="ij"
        )
        v_min = data.min()
        v_max = data.max()
        args = (v_max - v_min, x_max, y_max, sigma, v_min)
        popt, pcov = _sp.optimize.curve_fit(
            _gaussian2D, xdata, data.ravel(), p0=args
        )
    except Exception as e:
        _lgr.warning("Error fiting: %s, %s", e, type(e))
        return _np.nan, _np.nan, _np.nan
    return popt[1:4]


def _check_max_displacement(disp: float, max_value: float, axis_name: str) -> float:
    if abs(disp) > max_value:
        _lgr.warning("%s movement of %.2f exceeded max displacement",
                     axis_name, disp)
        return 0.0
    return disp


class Stabilizer(_th.Thread):
    """Wraps a stabilization thread.

    As usual, functions names beggining with an underscore do not belong to the
    public interface.

    Functions that *do* belong to the public interface communicate with the
    running thread relying on the GIL (like setting a value) or in events.
    """

    _report_cb = _List[_Optional[report_callback_type]]
    _init_cb = _List[_Optional[init_callback_type]]
    _end_cb = _List[_Optional[end_callback_type]]

    # Status flags
    _xy_tracking: bool = False
    _z_tracking: bool = False
    _xy_stabilization: bool = False
    _z_stabilization: bool = False

    # ROIS from the user
    _xy_rois: _np.ndarray = None  # [ [min, max]_x, [min, max]_y] * n_rois
    _z_roi = None  # min/max x, min/max y

    _last_image: _np.ndarray = _np.empty((50, 50))
    _pos = _np.zeros((3,))  # current position in nm
    _period = 0.150  # minumum loop time in seconds
    _reference_shift = _np.zeros((3,))
    _z_shift: _np.float64 = _np.nan
    _xy_shifts: _np.ndarray = _np.full((1, 2, ), _np.nan)
    _running_thread: _th.Thread = None

    def __init__(
        self,
        camera: _bc.BaseCamera,
        piezo: _bc.BasePiezo,
        camera_info: CameraInfo,
        corrector: _bc.BaseController,
        max_displacement: _Union[float, _Tuple[float]] = 200.,
        *args,
        **kwargs,
    ):
        """Init stabilization thread.

        Parameters
        ----------
        camera:
            Camera. Must implement a method called `get_image`, that returns
            a 2d numpy.ndarray representing the image
        piezo:
            Piezo controller. Must implement a method called `set_position` that
            accepts x, y and z positions
        camera_info: info_types.CameraInfo
            Holds information about camera and (x,y) and z marks relation
        corrector:
            object that provides a response
        """
        super().__init__(*args, **kwargs)

        # check if camera and piezo are OK
        if not callable(getattr(camera, "get_image", None)):
            raise ValueError(
                "The camera object does not expose a 'get_image' method"
            )
        self._camera = camera
        self._nmpp_xy = camera_info.nm_ppx_xy
        self._nmpp_z = camera_info.nm_ppx_z
        self._rot_vec = _np.array(
            (
                _np.cos(camera_info.angle),
                _np.sin(camera_info.angle),
            )
        )

        if not callable(getattr(piezo, "set_position_xy", None)):
            raise ValueError(
                "The piezo object does not expose a 'set_position_xy' method"
            )
        if not callable(getattr(piezo, "set_position_z", None)):
            raise ValueError(
                "The piezo object does not expose a 'set_position_z' method"
            )
        if not callable(getattr(piezo, "get_position", None)):
            raise ValueError(
                "The piezo object does not expose a 'get_position' method"
            )
        self._piezo = piezo
        self._stop_event = _th.Event()
        self._stop_event.set()

        # Clearing these events requests an update
        self._xy_track_event = _th.Event()
        self._xy_track_event.set()
        self._z_track_event = _th.Event()
        self._z_track_event.set()
        # self._xy_lock_event = _th.Event()
        # self._xy_lock_event.set()
        # self._z_lock_event = _th.Event()
        # self._z_lock_event.set()
        self._calibrate_event = _th.Event()  # Unset by default
        self._move_event = _th.Event()
        self._moveto_pos = _np.zeros((3,))

        self._rsp = corrector
        self._report_cb = []
        self._init_cb = []
        self._end_cb = []
        self._last_params = {}

        # Avoid users from shooting themselves in the foot
        self._old_run = self.run
        self.run = self._donotcall
        self._old_start = self.start
        self.start = self._donotcall
        self._max_displacement = _np.zeros((3,))
        self.set_max_displacement(max_displacement)
        self._initial_z_position = None

    def __enter__(self):
        self.start_loop()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_loop()
        return False

    def set_max_displacement(self, threshold: _Union[float, _Tuple[float]]):
        """Set the threshold above which no movement is performed."""
        self._max_displacement[:] = _np.array(threshold)

    def shift_reference(self, dx: float, dy: float, dz: float):
        """Shift the reference setpoint."""
        self._reference_shift = _np.array((dx, dy, dz,))

    def set_log_level(self, loglevel: int):
        """Set log level for module."""
        if loglevel < 0:
            _lgr.warning("Invalid log level asked: %s", loglevel)
        else:
            _lgr.setLevel(loglevel)

    def add_callbacks(self, report_cb: _Optional[report_callback_type],
                      init_cb: _Optional[init_callback_type],
                      end_cb: _Optional[end_callback_type],
                      ):
        """Add callbacks functions.

        report_cb: takyaq.info_types.report_callback_type | None
            Function to be called to report a new stabilization cycle. This
            function is called from an internal thread.
        init_cb: takyaq.info_types.init_callback_type | None
            Function to be called before starting the stabilization.
        end_cb: takyaq.info_types.end_callback_type | None
            Function to be called after stopping the stabilization.

        Any parameter can be None. In this case that CB is not called
        """
        self._report_cb.append(report_cb)
        self._init_cb.append(init_cb)
        self._end_cb.append(end_cb)

    def set_min_period(self, period: float):
        """Set minimum period between position adjustments.

        Parameters
        ----------
        period: float
            Minimum period, in seconds.

        The period is not precise, and might be longer than selected if the
        time needed to locate the stage real position takes longer.

        The thread always sleeps for at least 1 ms, in order to let other
        threads run (see main loop).

        Raises
        ------
        ValuError if requested period is negative
        """
        if period < 0:
            raise ValueError(f"Period can not be negative ({period})")
        self._period = period
        _lgr.debug("New period set: %s", period)

    def set_xy_rois(self, rois: _List[ROI]) -> bool:
        """Set ROIs for xy stabilization.

        Can not be used while XY tracking is active.

        Parameters
        ----------
        rois: list[info_types.ROI]
            list of XY rois (in pixels)

        Return
        ------
        True if successful, False otherwise
        """
        if self._xy_tracking:
            _lgr.warning("Trying to change xy ROIs while tracking is active")
            return False
        self._xy_rois = _np.array(
            # TODO: protect against negative numbers max(0, _.min_x), min(_.max_x, self.img.shape[1])
            [[[_.min_x, _.max_x], [_.min_y, _.max_y]] for _ in rois],
            dtype=_np.uint16,  # int type to use as indexes
        )
        return True

    def set_z_roi(self, roi: ROI) -> bool:
        """Set ROI for z stabilization.

        Can not be used while Z tracking is active.

        Parameter
        ---------
        roi: ROI
            Z roi

        Return
        ------
        True if successful, False otherwise
        """
        # TODO: protect against negative numbers max(0, _.min_x), min(_.max_x, self.img.shape[1])
        self._z_roi = _np.array(
            [[roi.min_x, roi.max_x], [roi.min_y, roi.max_y]], dtype=_np.uint32
        )
        return True

    def restore_z_lock(self, x: float, y: float, roi: ROI) -> bool:
        """Restore a saved Z lock position and ROI.

        Parameters
        ----------
        x, y: float
            center of mass in the selected roi
        roi: ROI
            Z roi
        """
        self.set_z_roi(roi)
        self._initial_z_position = _np.array((x, y,))

    def get_z_lock(self) -> _Tuple[float, float, ROI]:
        """Get Z lock position and ROI.

        Returns
        ------
           Tuple[float, float, ROI]: (x coordinate of center, y coordinate of
           center, RO)I

        Raises
        ------
            ValueError if lock not set
        """
        if self._z_roi is None or self._initial_z_position is None:
            raise ValueError("z lock not set")
        roi = ROI(*self._z_roi.flat)
        return (*self._initial_z_position, roi)

    def get_current_displacement(self) -> _Tuple[float, float, float]:
        """Returns last XYZ displacement from initial tracking.

        Values are numpy.nan if tracking was never enabled for that axis.

        Should NOT be called from any direct or indirect report handler.
        """
        if self._stop_event.is_set():
            raise RuntimeError("Stabilization thread is not running")
        # This guard is here to implement a better handling in future versions
        if _th.current_thread() == self._running_thread:
            raise RuntimeError("Can not be called from callbacks")
        xy = _np.nanmean(self._xy_shifts, axis=0)
        z = self._z_shift
        return tuple(_np.array([xy[0], xy[1], z]))

    def enable_xy_tracking(self) -> bool:
        """Enable tracking of XY fiduciaries."""
        if self._xy_rois is None:
            _lgr.warning("Trying to enable xy tracking without ROIs")
            return False

        self._xy_track_event.clear()
        self._xy_track_event.wait()
        return self._xy_tracking

    def disable_xy_tracking(self) -> bool:
        """Disable tracking of XY fiduciaries."""
        if self._xy_stabilization:
            _lgr.warning("Trying to disable xy tracking while feedback active")
            return False
        self._xy_tracking = False
        return True

    def set_xy_tracking(self, enabled: bool) -> bool:
        """Set XY tracking ON or OFF."""
        if enabled:
            return self.enable_xy_tracking()
        return self.disable_xy_tracking()

    @property
    def is_xy_tracking_enabled(self) -> bool:
        """Whether tracking is enabled."""
        return self._xy_tracking

    def _report_end_stabilization(self, st_type: StabilizationType,
                                  idx_end: int = None):
        """Call the stabilization en callbacks up to idx.

        Helper function.
        """
        for cb in self._end_cb[: idx_end]:
            try:
                if cb:
                    cb(st_type)
            except Exception as e:
                _lgr.warning("Error %s reporting stabilization end: %s", type(e), e)

    def enable_xy_stabilization(self) -> bool:
        """Enable stabilization on XY plane."""
        if not self._xy_tracking:
            _lgr.warning("Trying to enable xy stabilization without tracking")
            return False
        must_unroll = False
        for idx, cb in enumerate(self._init_cb):
            try:
                if cb:
                    if not cb(StabilizationType.XY_stabilization):
                        must_unroll = True
                        break
            except Exception as e:
                _lgr.warning("Error %s reporting stabilization end: %s", type(e), e)
                must_unroll = True
        if must_unroll:
            self._report_end_stabilization(StabilizationType.XY_stabilization,
                                           idx + 1)
            return False
        self._rsp.reset_xy(len(self._xy_rois))
        self._xy_stabilization = True
        return True

    def disable_xy_stabilization(self) -> bool:
        """Disable stabilization on XY plane."""
        if not self._xy_stabilization:
            # _lgr.warning("Trying to disable xy feedback but is not active")
            return False
        self._xy_stabilization = False
        self._report_end_stabilization(StabilizationType.XY_stabilization)
        return True

    def set_xy_stabilization(self, enabled: bool) -> bool:
        """Set XY stabilization ON or OFF."""
        if enabled:
            return self.enable_xy_stabilization()
        return self.disable_xy_stabilization()

    @property
    def is_xy_stabilization_enabled(self) -> bool:
        """Whether stabilization is enabled."""
        return self._xy_stabilization

    def enable_z_tracking(self) -> bool:
        """Enable tracking of Z position."""
        if self._z_roi is None:
            _lgr.warning("Trying to enable z tracking without ROI")
            return False
        self._z_track_event.clear()
        if self.is_alive():
            self._z_track_event.wait()
        return True

    def disable_z_tracking(self) -> bool:
        """Disable tracking of Z position."""
        if self._z_stabilization:
            _lgr.warning("Trying to disable z tracking while feedback active")
            return False
        self._z_tracking = False
        return True

    def set_z_tracking(self, enabled: bool) -> bool:
        """Set tracking of Z position ON or OFF."""
        if enabled:
            return self.enable_z_tracking()
        return self.disable_z_tracking()

    @property
    def is_z_tracking_enabled(self) -> bool:
        """Whether tracking is enabled."""
        return self._z_tracking


    def enable_z_stabilization(self) -> bool:
        """Enable stabilization of Z position."""
        if not self._z_tracking:
            _lgr.warning("Trying to enable z stabilization without tracking")
            return False
        must_unroll = False
        for idx, cb in enumerate(self._init_cb):
            try:
                if cb:
                    if not cb(StabilizationType.Z_stabilization):
                        must_unroll = True
                        break
            except Exception as e:
                _lgr.warning("Error %s reporting stabilization end: %s", type(e), e)
                must_unroll = True
        if must_unroll:
            _lgr.warning("UNROLLING Z")
            self._report_end_stabilization(StabilizationType.Z_stabilization,
                                           idx + 1)
            return False
        self._rsp.reset_z()
        self._z_stabilization = True
        return True

    def disable_z_stabilization(self) -> bool:
        """Disable stabilization of Z position."""
        if not self._z_stabilization:
            # _lgr.warning("Trying to disable z feedback but is not active")
            return False
        self._report_end_stabilization(StabilizationType.Z_stabilization)
        self._z_stabilization = False
        return True

    @property
    def is_z_stabilization_enabled(self) -> bool:
        """Whether stabilization is enabled."""
        return self._z_stabilization

    def calibrate(self, direction: str) -> bool:
        """Perform calibration of pixel size."""
        if direction not in ["x", "y", "z"]:
            _lgr.warning("Invalid calibration direction: %s", direction)
            return False

        # no `match` yet (we support python 3.7), so do a dirty trick
        self._calib_idx = {"x": 0, "y": 1, "z": 2}[direction]
        self._calibrate_event.set()
        return True

    def set_z_stabilization(self, enabled: bool) -> bool:
        """Set stabilization of Z position ON or OFF."""
        if enabled:
            return self.enable_z_stabilization()
        return self.disable_z_stabilization()

    def move(self, x: float, y: float, z: float) -> bool:
        """Start tracking and stabilization loop."""
        if self._stop_event.is_set():
            _lgr.warning("Trying to move without a running loop")
            return False
        if self._z_stabilization or self._xy_stabilization:
            _lgr.warning("Trying to move while stabilization is active")
            return False
        self._moveto_pos[:] = x, y, z
        self._move_event.set()
        return True

    def start_loop(self) -> bool:
        """Start tracking and stabilization loop."""
        if not self._stop_event.is_set():
            _lgr.warning("Trying to start already running loop")
            return False
        self._executor = _PPE()
        # prime pool for responsiveness (a _must_ on windows).
        nproc = _os.cpu_count()
        params = [
            [_np.eye(3)] * nproc,
            [1.0] * nproc,
            [1.0] * nproc,
            [1.0] * nproc,
        ]
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore")
            _ = tuple(self._executor.map(_gaussian_fit, *params))
        self._stop_event.clear()
        self.run = self._old_run
        self._old_start()
        self.run = self._donotcall
        return True

    def stop_loop(self):
        """Stop tracking and stabilization loop and release resources.

        Must be called from another thread to avoid deadlocks.
        """
        if self._stop_event.is_set():
            _lgr.warning("Trying to stop already finished loop")
            return False
        self._stop_event.set()
        self.join()
        self._executor.shutdown()
        _lgr.debug("Loop ended")
        return True

    def _donotcall(self, *args, **kwargs):
        """Advice against running forbidden functions."""
        raise ValueError("Do not call this function directly")

    def _locate_xy_centers(self, image: _np.ndarray) -> _np.ndarray:
        """Locate centers in XY ROIS.

        Returns values in pixels

        Parameter
        ---------
        image: numpy.ndarray
            2D array with the image to process

        Return
        ------
            numpy.ndarray of shape (NROIS, 2) with x,y center in nm
        """
        trimmeds = [
            image[roi[0, 0]: roi[0, 1], roi[1, 0]: roi[1, 1]]
            for roi in self._xy_rois
        ]
        x = self._last_params["x"]
        y = self._last_params["y"]
        s = self._last_params["s"]
        locs = _np.array(
            tuple(self._executor.map(_gaussian_fit, trimmeds, x, y, s))
        )
        self._last_params["x"] = locs[:, 0]
        nanloc = _np.isnan(locs[:, 0])  # if x is nan, y also is nan
        self._last_params["x"][nanloc] = x[nanloc]
        self._last_params["y"] = locs[:, 1]
        self._last_params["y"][nanloc] = y[nanloc]
        self._last_params["s"] = locs[:, 2]
        self._last_params["s"][nanloc] = s[nanloc]
        rv = locs[:, :2] + self._xy_rois[:, :, 0]
        rv *= self._nmpp_xy
        return rv

    def _initialize_last_params(self):
        """Initialize fitting parameters.

        All values are *in pixels* inside each ROI.

        TODO: Protect against errors (image must exist, ROIS must fit into
        image, etc.)
        """
        trimmeds = [
            self._last_image[roi[0, 0]: roi[0, 1], roi[1, 0]: roi[1, 1]]
            for roi in self._xy_rois
        ]
        pos_max = [
            _np.unravel_index(_np.argmax(data), data.shape) for data in trimmeds
        ]
        sigmas = [data.shape[0] / 3 for data in trimmeds]
        self._last_params = {
            "x": _np.array([p[0] for p in pos_max], dtype=float),
            "y": _np.array([p[1] for p in pos_max], dtype=float),
            "s": _np.array(sigmas, dtype=float),
        }

    def _locate_z_center(self, image: _np.ndarray) -> _np.ndarray:
        """Locate the center of the reflection used to infer Z position."""
        if self._z_roi is None:
            _lgr.error("Trying to locate z position without a ROI")
            return _np.full((2,), _np.nan)
        roi = image[slice(*self._z_roi[0]), slice(*self._z_roi[1])]
        roi = _np.array(roi)
        # roi[roi < 25] = 0
        return _np.array(_sp.ndimage.center_of_mass(roi))

    def _move_relative_xy(self, dx: float, dy: float):
        """Perform a relative movement in xy."""
        self._pos[0] += dx
        self._pos[1] += dy
        try:
            self._piezo.set_position_xy(*self._pos[:2])
        except Exception as e:
            _lgr.error("Error %s (%s) moving the stage", type(e), e)

    def _move_relative_z(self, dz: float):
        """Perform a relative movement in Z."""
        self._pos[2] += dz
        try:
            self._piezo.set_position_z(self._pos[2])
        except Exception as e:
            _lgr.error("Error %s (%s) moving the stage", type(e), e)

    def _report(
        self,
        t: float,
        image: _np.ndarray,
        xy_shifts: _Union[_np.ndarray, None],
        z_shift: float,
    ):
        """Send data to provided callback."""
        if xy_shifts is None:
            xy_shifts = _np.empty((0,))
        rv = PointInfo(t, image, z_shift, xy_shifts)
        for cb in self._report_cb:
            try:
                if cb:
                    cb(rv)
            except Exception as e:
                _lgr.warning(
                    "Exception reporting to callback: %s(%s)", type(e), e
                )

    def _calibrate_xy(
        self, length: float, initial_xy_positions: _np.ndarray, points: int = 20
    ):
        """Calibrate nm per pixel in XY plane.

        Runs it own small loop.
        moves around current position
        TODO: Inform about XY coupling (camera rotation?)
        WARNING: Z is handled separately

        Parameters
        ----------
        length: float
            calibration displacement in nm
        initial_xy_positions: numpy.ndarray
            initial positions of the fiduciary marks
        points: int, default 20
            number of calibration points
        """
        c_idx = self._calib_idx  # 0 for X, 1 for Y, Z is complicated
        shifts, step = _np.linspace(
            -length / 2.0, length / 2.0, points, retstep=True
        )
        response = _np.empty_like(shifts)
        if self._xy_rois is None:
            _lgr.warning("Trying to calibrate xy without ROIs")
            return False
        if not self._xy_tracking:
            _lgr.warning("Trying to calibrate xy without tracking")
            return False
        oldpos = _np.copy(self._pos)
        rel_vec = _np.zeros((2,))
        try:
            rel_vec[c_idx] = -length / 2.0
            self._move_relative_xy(*rel_vec)
            rel_vec[c_idx] = step
            image = self._camera.get_image()
            self._initialize_last_params()  # we made a LARGE shift
            for idx, s in enumerate(shifts):
                image = self._camera.get_image()
                xy_shifts = self._locate_xy_centers(image)
                self._report(
                    _time.time(), image, xy_shifts - initial_xy_positions, 0
                )
                x = _np.nanmean(xy_shifts[:, c_idx])
                response[idx] = x / self._nmpp_xy
                self._move_relative_xy(*rel_vec)
                _time.sleep(0.10)
            # TODO: better reporting
            for x, y in zip(shifts, response):
                print(f"{x}, {y}")
            vec, _ = _np.linalg.lstsq(
                _np.vstack([shifts, _np.ones(points)]).T, response, rcond=None
            )[0]
            print("slope = ", 1 / vec)
        except Exception as e:
            _lgr.warning("Exception calibrating x: %s(%s)", type(e), e)
        self._pos[:] = oldpos
        self._piezo.set_position_xy(*self._pos[:2])
        return True

    def _calibrate_z(
        self, length: float, initial_xy_positions: _np.ndarray, points: int = 20
    ):
        """Calibrate nm per pixel when Z shifts.

        Runs its own loop.

        Parameters
        ----------
        length: float
            calibration displacement in nm
        initial_xy_positions: numpy.ndarray
            initial positions of the fiduciary marks
        points: int, default 20
            number of calibration points
        """
        shifts, step = _np.linspace(
            -length / 2.0, length / 2.0, points, retstep=True
        )
        response = _np.empty(
            (
                points,
                2,
            )
        )
        if self._z_roi is None:
            _lgr.warning("Trying to calibrate z without ROI")
            return False
        if not self._z_tracking:
        # if not self._z_track_event.is_set():
            _lgr.warning("Trying to calibrate z without tracking")
            return False
        oldpos = _np.copy(self._pos)
        rel_mov = 0.0
        try:
            rel_mov = -length / 2.0
            self._move_relative_z(rel_mov)
            rel_mov = step
            image = self._camera.get_image()
            for idx, s in enumerate(shifts):
                image = self._camera.get_image()
                # roi = image[slice(*self._z_roi[0]), slice(*self._z_roi[1])]
                # c = _np.array(_sp.ndimage.center_of_mass(roi))
                z_position = self._locate_z_center(image)
                c = z_position - self._initial_z_position
                xy_data = (
                    None
                    if self._xy_rois is None
                    else self._locate_xy_centers(image) - initial_xy_positions
                )
                self._report(_time.time(), image, xy_data, _np.nan)
                response[idx] = c
                self._move_relative_z(rel_mov)
                _time.sleep(0.10)
            # TODO: better reporting
            print("z, x, y")
            for z, xy in zip(shifts, response):
                print(f"{z}, {xy[0]}, {xy[1]}")
            vec, _ = _np.linalg.lstsq(
                _np.vstack([shifts, _np.ones(points)]).T, response, rcond=None
            )[0]
            rot_angle = _np.arctan2(vec[1], vec[0])
            print("Angle(rad) = ", rot_angle)
            rot_vec = _np.array((_np.cos(rot_angle), _np.sin(rot_angle)))
            disp = _np.sum(_np.array(response) * rot_vec, axis=1)
            px_per_nm, _ = _np.linalg.lstsq(
                _np.vstack([shifts, _np.ones(points)]).T, disp, rcond=None
            )[0]
            print("Z nanometer per pixel= ", 1. / px_per_nm)
        except Exception as e:
            _lgr.warning("Exception calibrating z: %s(%s)", type(e), e)
        self._pos[:] = oldpos
        self._piezo.set_position_z(self._pos[2])
        return True

    def run(self):
        """Run main stabilization loop."""
        if callable(getattr(self._piezo, 'init', None)):
            self._piezo.init()
        self._running_thread = _th.current_thread()
        initial_xy_positions = None
        self._initial_z_position = None
        self._pos[:] = self._piezo.get_position()
        lt = _time.monotonic()  # or 0
        while not self._stop_event.is_set():
            z_shift = 0.0
            xy_shifts = None
            # Check external events
            if self._calibrate_event.is_set():
                _lgr.debug("Calibration event received")
                try:
                    self._pos[:] = self._piezo.get_position()
                    if self._calib_idx >= 0 and self._calib_idx < 2:
                        if self._xy_tracking:
                            self._calibrate_xy(100.0, initial_xy_positions)
                        else:
                            _lgr.warning("can not calibrate XY without tracking")
                    elif self._calib_idx == 2:
                        if self._z_tracking:
                            self._calibrate_z(100.0, initial_xy_positions)
                        else:
                            _lgr.warning("can not calibrate Z without tracking")
                    else:
                        _lgr.warning("Invalid calibration direction detected")
                except Exception as e:
                    _lgr.error("An exception (%s) happened during calibration: %s",
                               type(e), e)
                self._calibrate_event.clear()
            if self._move_event.is_set():
                self._piezo.set_position_xy(*self._moveto_pos[:2])
                self._piezo.set_position_z(self._moveto_pos[2])
                self._pos[:] = self._moveto_pos
                self._move_event.clear()
            # Tracking and stabilization starts here
            nt = _time.monotonic()
            delay = self._period - (nt - lt)
            _time.sleep(max(delay, 0.001))  # be nice to other threads
            lt = nt
            try:
                image = self._camera.get_image()
                t = _time.time()
                self._last_image = image
            except Exception as e:
                _lgr.error("Could not acquire image: %s (%s)", type(e), e)
                image = _np.diag(_np.full(max(*self._last_image.shape), 255))
                t = _time.time()
                self._report(
                    t,
                    image,
                    _np.full_like(initial_xy_positions, _np.nan),
                    _np.nan,
                )
                _time.sleep(self._period)
                continue
            # Process start tracking commands
            if not self._xy_track_event.is_set():
                _lgr.info("Setting xy initial positions")
                self._pos[:2] = self._piezo.get_position()[:2]
                self._initialize_last_params()
                initial_xy_positions = self._locate_xy_centers(image)
                self._xy_track_event.set()
                self._xy_tracking = True
            if not self._z_track_event.is_set():
                self._pos[2] = self._piezo.get_position()[2]
                _lgr.info("Setting z initial positions")
                self._initial_z_position = self._locate_z_center(image)
                self._z_track_event.set()
                self._z_tracking = True
            # Do actual work
            if self._z_tracking:
                z_position = self._locate_z_center(image)
                z_disp = z_position - self._initial_z_position
                # ang is measured counterclockwise from the X axis. We rotate *clockwise*
                z_shift = _np.sum(z_disp * self._rot_vec) * self._nmpp_z
                self._z_shift = z_shift
                # TODO: handle Z reference shift
            if self._xy_tracking:
                xy_positions = self._locate_xy_centers(image)
                xy_shifts = xy_positions - initial_xy_positions
                self._xy_shifts = xy_shifts
            self._report(t, image, xy_shifts, z_shift)
            if xy_shifts is not None:
                xy_shifts = xy_shifts + self._reference_shift[0:2]
            if self._z_stabilization or self._xy_stabilization:
                if z_shift is _np.nan:
                    _lgr.warning("z shift is NAN")
                    z_shift = 0.0
                try:
                    x_resp, y_resp, z_resp = self._rsp.response(
                        t, xy_shifts, z_shift
                    )
                except Exception as e:
                    _lgr.warning("Error getting correction: %s, %s", e, type(e))
                    x_resp = y_resp = z_resp = 0.0
                if not self._z_stabilization:
                    z_resp = 0.0
                else:
                    z_resp = _check_max_displacement(z_resp, self._max_displacement[2], 'Z')
                if not self._xy_stabilization:
                    x_resp = y_resp = 0.0
                else:
                    x_resp = _check_max_displacement(x_resp, self._max_displacement[0], 'X')
                    y_resp = _check_max_displacement(y_resp, self._max_displacement[1], 'Y')
                if self._z_stabilization:
                    self._move_relative_z(z_resp)
                if self._xy_stabilization:
                    self._move_relative_xy(x_resp, y_resp)
        _lgr.debug("Ending loop.")
