"""
Sample PyQT frontend for Takyaq.


Use:
    - Create XY ROIS and move and size them to encompass the fiducial marks
    - Create a Z ROI and move and size them to encompass the beam reflection
    - Start tracking of XY and Z rois. While tracking is active, erasing or
    changing the ROIs positions has no effect.
    - Start correction of XY and Z positions.


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
import datetime as _datetime
import pathlib as _pathlib
from typing import Optional as _Optional, BinaryIO as _BinaryIO, Tuple as _Tuple
from configparser import ConfigParser as _ConfigParser
import warnings
from PyQt5.QtCore import pyqtSignal, pyqtSlot, QObject, Qt
from PyQt5.QtWidgets import (
    QGroupBox,
    QFrame,
    QGridLayout,
    QLabel,
    QPushButton,
    QCheckBox,
    QHBoxLayout,
    QVBoxLayout,
    QLineEdit,
    QDoubleSpinBox,
)
from PyQt5.QtGui import QDoubleValidator
from .qt_utils import create_spin as _create_spin, GroupedCheckBoxes
import pyqtgraph as _pg

import logging as _lgn

from ..stabilizer import Stabilizer, PointInfo, ROI, CameraInfo

import takyaq.base_classes as _bc
from .pattern_GUI import PatternWindow

_lgr = _lgn.getLogger(__name__)
_lgr.setLevel(_lgn.DEBUG)

# default configuration filename
_CONFIG_FILENAME = 'takyaq.ini'
_Z_LOCK_FILENAME = "z_lock.cfg"

_DEFAULT_CONFIG = {
        'display_points': 400,
        'save_buffer_length': 500,
        'period': 0.05,
        'output_base_dir': _pathlib.Path.home() / "takyaq_data",
        'XY ROIS': {
            'size': 60,
        },
        'Z ROI': {
            'size': 100,
        }
    }

_NPY_Z_DTYPE = _np.dtype([
    ('t', _np.float64),
    ('z', _np.float64),
    ])


def _has_method(obj, method_name: str) -> bool:
    """Check if an object implements a method, logging."""
    if not hasattr(obj, method_name):
        _lgr.info("Method %s not present")
        return False
    if not callable(getattr(obj, method_name)):
        _lgr.error("Attribute %s is not a valid method", method_name)
        return False
    return True


class QReader(QObject):
    """Helper class to send data from stabilizar to Qt GUI.

    Implements a method that receives a PointInfo object. This is how the
    stabilizer thread reports the last data.

    In this case (a PyQT application), we emit a signal to the GUI.
    """

    new_data = pyqtSignal(float, _np.ndarray, float, _np.ndarray)

    def cb(self, data: PointInfo):
        """Report data."""
        self.new_data.emit(data.time, data.image, data.z_shift, data.xy_shifts)


def save_config(config_data: dict, filename: str = _CONFIG_FILENAME):
    """Save data to file.

    TODO: remove default params
    """
    config = _ConfigParser()
    config["General"] = {
        'display_points': config_data.get('display_points', 200),
        'save_buffer_length': config_data.get('save_buffer_length', 200),
        'output_base_dir': config_data.get('output_base_dir', _DEFAULT_CONFIG['output_base_dir']),
        'period': config_data.get('period', 0.100),
    }
    XY_dict = config_data.get('XY ROIS', {})
    config['XY ROIS'] = {
        'size': XY_dict.get('size', 60),
        }
    # TODO: guardar posicion ROI Z
    Z_dict = config_data.get('Z ROI', {})
    config['Z ROI'] = {
        'size': Z_dict.get('size', 100),
        }
    with open(filename, "wt") as configfile:
        config.write(configfile)


def load_config(filename: str = _CONFIG_FILENAME):
    """Load config."""
    config = _ConfigParser()
    rv = dict(_DEFAULT_CONFIG)
    if not config.read(filename):
        _lgr.info("No config file: using defaults")
        return rv
    if 'General' in config:
        gnrl = config['General']
        for k in ('display_points', 'save_buffer_length',):
            rv[k] = gnrl.getint(k, fallback=rv[k])
        for k in ('period',):
            rv[k] = gnrl.getfloat(k, fallback=rv[k])
        rv['output_base_dir'] = gnrl.get('output_base_dir', fallback=rv['output_base_dir'])
    if 'XY ROIS' in config:
        xy_cfg = config['XY ROIS']
        for k in ('size',):
            rv['XY ROIS'][k] = xy_cfg.getint(k)
    if 'Z ROI' in config:
        z_cfg = config['Z ROI']
        for k in ('size',):
            rv['Z ROI'][k] = z_cfg.getint(k)
    return rv


def load_camera_info(filename: str = _CONFIG_FILENAME) -> CameraInfo:
    """Return camera info from config file."""
    config = _ConfigParser()
    if not config.read(filename):
        raise FileNotFoundError("Camera info configuration file not found.")
    if 'Camera' not in config:
        raise KeyError("Camera info not found in configuration file.")
    nm_ppx_xy = config.getfloat('Camera', 'nm_ppx_xy')
    nm_ppx_z = config.getfloat('Camera', 'nm_ppx_z')
    angle = config.getfloat('Camera', 'angle')
    return CameraInfo(nm_ppx_xy, nm_ppx_z, angle)


def save_z_lock(x: float, y: float, roi: ROI):
    """Save z lock data to file."""
    config = _ConfigParser()
    config["Z lock"] = {
        'x': x,
        'y': y,
    }
    config['ROI'] = {
        'min_x': roi.min_x,
        'max_x': roi.max_x,
        'min_y': roi.min_y,
        'max_y': roi.max_y,
        }
    with open(_Z_LOCK_FILENAME, "wt") as configfile:
        config.write(configfile)
        _lgr.info("Z lock data saved to %s", _Z_LOCK_FILENAME)


def load_z_lock() -> _Tuple[float, float, ROI]:
    """Load z lock data from file."""
    config = _ConfigParser()
    if not config.read(_Z_LOCK_FILENAME):
        _lgr.warning("Z lock file not found")
        raise FileNotFoundError
    z_data = config["Z lock"]
    x = z_data.getfloat('x')
    y = z_data.getfloat('y')
    roi = ROI(*[config['ROI'].getfloat(k) for k in ('min_x', 'max_x', 'min_y', 'max_y')])
    return x, y, roi


class ConfigWindow(QFrame):
    def __init__(self, parent, controller: _bc.BaseController, *args, **kwargs):
        # TODO: accept a loaded config parameter
        super().__init__(*args, **kwargs)
        self._controller = controller
        self.setWindowFlag(Qt.WindowCloseButtonHint, False)
        self._init_gui(parent)
        self._PID_changed(0.0)

    def _init_gui(self, parent):
        self.setWindowTitle("Takyaq configuration")
        layout = QVBoxLayout()

        PI_gb = QGroupBox("PI")
        PI_layout = QGridLayout()
        PI_gb.setLayout(PI_layout)
        self._KP_sp: list[QDoubleSpinBox] = []
        self._KI_sp: list[QDoubleSpinBox] = []
        # self._KD_sp: list[QDoubleSpinBox] = []
        PI_layout.addWidget(QLabel('Kp'), 1, 0)
        PI_layout.addWidget(QLabel('Ki'), 2, 0)
        # PI_layout.addWidget(QLabel('Kd'), 3, 0)
        for idx, coord in enumerate(['x', 'y', 'z']):
            PI_layout.addWidget(QLabel(coord), 0, 1+idx)
            kpsp = _create_spin(.75, 3, 0.005)
            kpsp.valueChanged.connect(self._PID_changed)
            self._KP_sp.append(kpsp)
            kisp = _create_spin(0, 3, 0.005)
            kisp.valueChanged.connect(self._PID_changed)
            self._KI_sp.append(kisp)
            PI_layout.addWidget(kpsp, 1, 1+idx)
            PI_layout.addWidget(kisp, 2, 1+idx)
        PI_gb.setFlat(True)
        layout.addWidget(PI_gb)

        calibration_gb = QGroupBox("Calibration")
        calibration_layout = QHBoxLayout()
        calibration_gb.setLayout(calibration_layout)
        self.calibrateXButton = QPushButton('X')
        self.calibrateXButton.clicked.connect(parent._calibrate_x)
        self.calibrateYButton = QPushButton('Y')
        self.calibrateYButton.clicked.connect(parent._calibrate_y)
        self.calibrateZButton = QPushButton('Z')
        self.calibrateZButton.clicked.connect(parent._calibrate_z)
        calibration_layout.addWidget(self.calibrateXButton)
        calibration_layout.addWidget(self.calibrateYButton)
        calibration_layout.addWidget(self.calibrateZButton)
        calibration_gb.setFlat(True)
        layout.addWidget(calibration_gb)

        movement_gb = QGroupBox("Movement")
        mv_layout = QGridLayout()
        movement_gb.setLayout(mv_layout)
        self._mv_sp: list[QDoubleSpinBox] = []
        limits = ((0, 20),) * 3
        if hasattr(parent._piezo, 'get_limits'):
            try:
                limits = parent._piezo.get_limits()
            except Exception as e:
                _lgr.warning("Error %s (%s) getting limits from piezo",
                             type(e), e)
        else:
            _lgr.info("No piezo limits reported, using defaults")
        for idx, (coord, v0, extents) in enumerate(zip(['x', 'y', 'z'],
                                              parent._piezo.get_position(),
                                              limits)):
            mv_layout.addWidget(QLabel(coord + ' / µm'), 0, idx)
            pos_spin = _create_spin(v0, 3, 0.01, extents[0], extents[1])
            mv_layout.addWidget(pos_spin, 1, idx)
            self._mv_sp.append(pos_spin)
        self._mv_btn = QPushButton('Move')
        mv_layout.addWidget(self._mv_btn, 0, 3, 2, 1)
        self._mv_btn.clicked.connect(lambda: parent.goto_position(
            *[sb.value() for sb in self._mv_sp])
            )
        movement_gb.setFlat(True)
        layout.addWidget(movement_gb)

        self.setLayout(layout)

    @pyqtSlot(float)
    def _PID_changed(self, newvalue: float):
        Kp = [sp.value() for sp in self._KP_sp]
        Ki = [sp.value() for sp in self._KI_sp]
        # Kd = [sp.value() for sp in self._KD_sp]
        self._controller.set_Kp(Kp)
        self._controller.set_Ki(Ki)
        # self._controller.set_Kd(Kd)


class Frontend(QFrame):
    """PyQt Frontend for Takyaq.

    Implemented as a QFrame so it can be easily integrated within a larger app.
    """
    # For displaying
    _t_data = _np.full((0,), _np.nan)
    _z_data = _np.full((0,), _np.nan)
    _xy_data = _np.full((0, 0, 2), _np.nan)
    _graph_pos = 0

    # For saving
    # z and xy are completely decoupled. As we roll data for graphic
    # and we do not need huge amounts of memory, we decouple completely
    # data saving and graphs
    _save_pos = 0  # shared for xy and z so data alignment is easier
    _save_data: bool = False
    _SAVE_PERIOD = 0.05
    _xy_fd: _BinaryIO = None
    _z_fd: _BinaryIO = None
    _t_save_data = _np.full((0,), _np.nan)
    _z_save_data = _np.full((0,), _np.nan)
    _xy_save_data = _np.full((0, 0, 2), _np.nan)

    _x_plots = []
    _y_plots = []
    _roilist = []
    _z_ROI = None
    lastimage: _np.ndarray = None
    _z_tracking_enabled: bool = False
    _xy_tracking_enabled: bool = False
    _z_locking_enabled: bool = False
    _xy_locking_enabled: bool = False

    _camera_info: CameraInfo

    def __init__(self, camera: _bc.BaseCamera, piezo: _bc.BasePiezo,
                 controller: _bc.BaseController, camera_info: _Optional[CameraInfo],
                 stabilizer: Stabilizer,
                 *args, **kwargs):
        """Init Frontend."""
        super().__init__(*args, **kwargs)
        self._load_config()
        self._camera = camera
        self._piezo = piezo
        self._controller = controller
        self._camera_info = camera_info if camera_info else load_camera_info()
        self.setup_gui()
        # Callback object
        self._cbojt = QReader()
        self._cbojt.new_data.connect(self.get_data)
        self.reset_data_buffers()
        self.reset_xy_data_buffers(len(self._roilist))
        self.reset_z_data_buffers()
        self._stabilizer = stabilizer
        self._stabilizer.add_callbacks(self._cbojt.cb, None, None)
        self._t0 = _time.time()
        self._set_delay(True)
        self._config_window = ConfigWindow(self, controller)
        self._config_window.hide()
        self._pattern_window = PatternWindow(self, stabilizer)
        self._pattern_window.hide()

    def _load_config(self):
        self._config = load_config()
        self._MAX_POINTS = self._config['display_points']
        self._SAVE_PERIOD = self._config['save_buffer_length']
        self._XY_ROI_SIZE = self._config['XY ROIS']['size']
        self._Z_ROI_SIZE = self._config['Z ROI']['size']
        self._period = self._config['period']

    @pyqtSlot(bool)
    def clear_data(self, *args):
        """Clears all data buffers."""
        self.reset_data_buffers()

    def reset_data_buffers(self):
        """Reset data buffers unrelated to localization.

        Also resets base timer
        """
        # self._I_data = _np.full((self._MAX_POINTS,), _np.nan)
        self._t_data = _np.full((self._MAX_POINTS,), _np.nan)
        self._t_save_data = _np.full((self._SAVE_PERIOD,), _np.nan)
        self._graph_pos = 0
        self._save_pos = 0
        self._t0 = _time.time()

    def reset_xy_data_buffers(self, roi_len: int):
        """Reset data buffers related to XY localization."""
        self._xy_data = _np.full((self._MAX_POINTS, roi_len, 2), _np.nan)  # sample #, roi
        self._xy_save_data = _np.full((self._SAVE_PERIOD, roi_len, 2), _np.nan)

    def reset_z_data_buffers(self):
        """Reset data buffers related to Z localization."""
        self._z_data = _np.full((self._MAX_POINTS,), _np.nan)
        self._z_save_data = _np.full((self._SAVE_PERIOD,), _np.nan)

    def reset_graphs(self, roi_len: int):
        """Reset graphs contents and adjust to number of XY ROIs."""
        neplots = len(self._x_plots)
        if roi_len > neplots:
            self._x_plots.extend(
                [
                    self.xyzGraph.xPlot.plot(
                        pen="w", alpha=0.3, auto=False, connect="finite"
                    )
                    for _ in range(roi_len - neplots)
                ]
            )
            self._y_plots.extend(
                [
                    self.xyzGraph.yPlot.plot(
                        pen="w", alpha=0.3, auto=False, connect="finite"
                    )
                    for _ in range(roi_len - neplots)
                ]
            )
        elif roi_len < neplots:
            for i in range(neplots - roi_len):
                self.xyzGraph.xPlot.removeItem(self._x_plots.pop())
                self.xyzGraph.yPlot.removeItem(self._y_plots.pop())
        for p in self._x_plots:
            p.clear()
            p.setAlpha(0.3, auto=False)
        for p in self._y_plots:
            p.clear()
            p.setAlpha(0.3, auto=False)
        self.xmeanCurve.setZValue(roi_len)
        self.ymeanCurve.setZValue(roi_len)

    @pyqtSlot(bool)
    def _add_xy_ROI(self, checked: bool):
        """Add a new XY ROI."""
        if self.lastimage is None:
            _lgr.warning("No image to set ROI")
            return
        w, h = self.lastimage.shape[0:2]
        ROIpos = (w / 2 - self._XY_ROI_SIZE / 2, h / 2 - self._XY_ROI_SIZE / 2)
        ROIsize = (self._XY_ROI_SIZE, self._XY_ROI_SIZE)
        roi = _pg.ROI(ROIpos, ROIsize, rotatable=False)
        roi.addScaleHandle((1, 0), (0, 1), lockAspect=True)
        self.image_pi.addItem(roi)
        self._roilist.append(roi)
        self.delete_roiButton.setEnabled(True)

    @pyqtSlot(bool)
    def _remove_xy_ROI(self, checked: bool):
        """Remove last XY ROI."""
        if not self._roilist:
            _lgr.warning("No ROI to delete")
            return
        roi = self._roilist.pop()
        self.image_pi.removeItem(roi)
        del roi
        if not self._roilist:
            self.delete_roiButton.setEnabled(False)

    @pyqtSlot(bool)
    def _add_z_ROI(self, checked: bool):
        """Create the Z ROI."""
        if self._z_ROI:
            _lgr.warning("A Z ROI already exists")
            return
        w, h = self.lastimage.shape[0:2]
        ROIpos = (w / 2 - self._Z_ROI_SIZE / 2, h / 2 - self._Z_ROI_SIZE / 2)
        ROIsize = (self._Z_ROI_SIZE, self._Z_ROI_SIZE)
        roi = _pg.ROI(ROIpos, ROIsize, pen={"color": "red", "width": 2}, rotatable=False)
        roi.addScaleHandle((1, 0), (0, 1), lockAspect=True)
        self.image_pi.addItem(roi)
        self._z_ROI = roi
        self.zROIButton.setEnabled(False)

    @pyqtSlot(int)
    def _send_z_rois_and_track(self, state: int):
        """Send Z roi data to the stabilizer and start tracking the Z position."""
        if state == Qt.CheckState.Unchecked:
            if self._z_locking_enabled:
                _lgr.warning("Z locking enabled: can not disable tracking")
                self.trackZBox.setCheckState(Qt.CheckState.Checked)
                return
            self._stabilizer.set_z_tracking(False)
            self._z_tracking_enabled = False
        else:
            if self._z_tracking_enabled:
                return
            if not self._z_ROI:
                _lgr.warning("We need a Z ROI to init tracking")
                self.trackZBox.setCheckState(Qt.CheckState.Unchecked)
                return
            self._stabilizer.set_z_roi(ROI.from_pyqtgraph(self._z_ROI))
            self.reset_z_data_buffers()
            if not self._xy_tracking_enabled:
                self.reset_data_buffers()
            self._stabilizer.set_z_tracking(True)
            self._z_tracking_enabled = True

    @pyqtSlot(int)
    def _change_z_lock(self, state: int):
        """Start stabilization of Z position."""
        if state == Qt.CheckState.Unchecked:
            self._stabilizer.set_z_stabilization(False)
            self._z_locking_enabled = False
        else:
            if not self._z_tracking_enabled:
                _lgr.warning("We need Z tracking to init locking")
                self.lockZBox.setCheckState(Qt.CheckState.Unchecked)
                return
            if not self._stabilizer.enable_z_stabilization():
                _lgr.warning("Could not start z stabilization")
                self.lockZBox.setCheckState(Qt.CheckState.Unchecked)
                return
            self._z_locking_enabled = True

    @pyqtSlot(int)
    def _send_xy_rois_and_track(self, state: int):
        """Send XY roi data to the stabilizer and start tracking the XY position."""
        if state == Qt.CheckState.Unchecked:
            if self._xy_locking_enabled:
                _lgr.warning("XY locking enabled: can not disable tracking")
                self.trackXYBox.setCheckState(Qt.CheckState.Checked)
                return
            self._stabilizer.set_xy_tracking(False)
            self._xy_tracking_enabled = False
        else:
            if not self._roilist:
                _lgr.warning("We need XY ROIs to init tracking")
                self.trackXYBox.setCheckState(Qt.CheckState.Unchecked)
                return
            if self._xy_tracking_enabled:
                _lgr.warning("XY tracking was already enabled")
                return
            self._npy_xy_dtype = _np.dtype([
                ('t', _np.float64),
                ('xy', _np.float64, (len(self._roilist), 2)),
                ])
            self.reset_graphs(len(self._roilist))
            self.reset_xy_data_buffers(len(self._roilist))
            if not self._z_tracking_enabled:
                self.reset_data_buffers()
            self._stabilizer.set_xy_rois([ROI.from_pyqtgraph(roi) for roi in self._roilist])
            self._stabilizer.set_xy_tracking(True)
            self._xy_tracking_enabled = True

    @pyqtSlot(int)
    def _change_xy_lock(self, state: int):
        """Start stabilization of XY position."""
        if state == Qt.CheckState.Unchecked:
            self._stabilizer.set_xy_stabilization(False)
            self._xy_locking_enabled = False
        else:
            if not self._xy_tracking_enabled:
                _lgr.warning("We need XY tracking to init locking")
                self.lockXYBox.setCheckState(Qt.CheckState.Unchecked)
                return
            if not self._stabilizer.enable_xy_stabilization():
                _lgr.warning("Could not start xy stabilization")
                self.lockXYBox.setCheckState(Qt.CheckState.Unchecked)
                return
            self._xy_locking_enabled = True

    @pyqtSlot(bool)
    def _set_delay(self, checked: bool):
        delay = float(self.delay_le.text())
        self._period = delay
        self._config['period'] = delay
        self._stabilizer.set_min_period(delay)

    @pyqtSlot(bool)
    def _manage_exposure_set(self, checked: bool):
        try:
            self._camera.set_exposure(float(self._exposure_sb.value()))
        except Exception as e:
            _lgr.warning("Error %s (%s) setting exposure", type(e), e)

    @pyqtSlot(bool)
    def _manage_gain_set(self, checked: bool):
        try:
            self._camera.set_gain(float(self._gain_sb.value()))
        except Exception as e:
            _lgr.warning("Error %s (%s) setting gain", type(e), e)

    @pyqtSlot(bool)
    def _calibrate_x(self, clicked: bool):
        self._stabilizer.calibrate('x')

    @pyqtSlot(bool)
    def _calibrate_y(self, clicked: bool):
        self._stabilizer.calibrate('y')

    @pyqtSlot(bool)
    def _calibrate_z(self, clicked: bool):
        self._stabilizer.calibrate('z')

    @pyqtSlot(int)
    def _change_save(self, check_status: int):
        if check_status == Qt.CheckState.Unchecked:
            if self._save_data:  # in case file open failed
                self._save_and_reset()
                self._xy_fd.close()
                self._xy_fd = None
                self._z_fd.close()
                self._z_fd = None
                self._save_data = False
        else:
            base_dir = _pathlib.Path.home() / "takyaq_data"
            base_dir = self._config['output_base_dir']
            base_dir.mkdir(parents=True, exist_ok=True)
            date_str = _datetime.datetime.now().isoformat(
                timespec='seconds').replace('-', '').replace(':', '-')
            xy_filename = base_dir / ('xy_data' + date_str + '.npy')
            z_filename = base_dir / ('z_data' + date_str + '.npy')
            try:
                self._xy_fd = open(xy_filename, 'wb')
                self._z_fd = open(z_filename, 'wb')
            except Exception as e:
                if self._xy_fd:
                    self._xy_fd.close()
                    self._xy_fd = None
                _lgr.error("Can not open output files: %s (%s)", type(e), e)
                self.export_chkbx.setChecked(False)
            self._save_data = True
            self._save_pos = 0

    def _save_and_reset(self):
        if self._xy_tracking_enabled:
            save_data = _np.array(
                list(zip(self._t_save_data[:self._save_pos],
                         self._xy_save_data[:self._save_pos])),
                dtype=self._npy_xy_dtype)
            _np.save(self._xy_fd, save_data)
        if self._z_tracking_enabled:
            save_data = _np.array(
                list(zip(self._t_save_data[:self._save_pos],
                         self._z_save_data[:self._save_pos],
                         )),
                dtype=_NPY_Z_DTYPE)
            _np.save(self._z_fd, save_data)
        self._save_pos = 0

    @pyqtSlot(bool)
    def _save_z_lock(self, clicked: bool):
        """Save Z lock position to file."""
        try:
            x, y, roi = self._stabilizer.get_z_lock()
            save_z_lock(x, y, roi)
        except ValueError:
            _lgr.warning("Can not save: Z has not been locked")

    @pyqtSlot(bool)
    def _load_z_lock(self, clicked: bool):
        """Load Z lock position to file."""
        try:
            x, y, roi = load_z_lock()
        except FileNotFoundError:
            _lgr.warning("Can not load: Z locked file not found")
            return
        self._stabilizer.restore_z_lock(x, y, roi)
        if not self._z_ROI:
            self._add_z_ROI(False)
        ROIsize = (roi.max_x - roi.min_x, roi.max_y - roi.min_y)
        self._z_ROI.setSize(ROIsize, update=False)
        self._z_ROI.setPos(roi.min_x, roi.min_y)

    @pyqtSlot(float, _np.ndarray, float, _np.ndarray)
    def get_data(self, t: float, img: _np.ndarray, z: float, xy_shifts: _np.ndarray):
        """Receive data from the stabilizer and graph it."""
        if self._save_pos >= self._SAVE_PERIOD and self._save_data:
            self._save_and_reset()

        # Graphics
        # Roll data
        if self._graph_pos >= self._MAX_POINTS:
            self._t_data[0:-1] = self._t_data[1:]
            # self._I_data[0:-1] = self._I_data[1:]
            if self._z_tracking_enabled:
                self._z_data[0:-1] = self._z_data[1:]
            if self._xy_tracking_enabled and xy_shifts.shape[0]:
                self._xy_data[0:-1] = self._xy_data[1:]
            self._graph_pos -= 1

        # manage image data
        self.img.setImage(img, autoLevels=self.lastimage is None)
        self.lastimage = img
        # self._I_data[self._graph_pos] = _np.average(img)
        self._t_save_data[self._save_pos] = self._t_data[self._graph_pos] = t
        t_data = self._t_data[: self._graph_pos + 1] - self._t0
        # self.avgIntCurve.setData(t_data, self._I_data[: self._graph_pos + 1])

        # manage tracking data
        if self._z_tracking_enabled:
            self._z_save_data[self._save_pos] = self._z_data[self._graph_pos] = z
            # update reports
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.zstd_value.setText(f"{_np.nanstd(self._z_data[:self._graph_pos]):.2f}")
            # update Graphs
            z_data = self._z_data[: self._graph_pos + 1]
            self.zCurve.setData(t_data, z_data)
            try:  # It is possible to have all NANs data
                hist, bin_edges = _np.histogram(z_data, bins=30,
                                                range=(_np.nanmin(z_data),
                                                       _np.nanmax(z_data)))
                self.zHistogram.setOpts(
                    x=_np.mean((bin_edges[:-1], bin_edges[1:],), axis=0),
                    height=hist,
                    width=bin_edges[1]-bin_edges[0]
                    )
            except Exception as e:
                _lgr.warning("Exception %s plotting z: %s", type(e), e)

        if self._xy_tracking_enabled and xy_shifts.shape[0]:
            self._xy_save_data[self._save_pos] = self._xy_data[self._graph_pos] = xy_shifts
            t_data = _np.copy(t_data)  # pyqtgraph does not keep a cpoy

            x_data = self._xy_data[: self._graph_pos + 1, :, 0]
            y_data = self._xy_data[: self._graph_pos + 1, :, 1]
            # update reports
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                x_mean = _np.nanmean(x_data, axis=1)
                y_mean = _np.nanmean(y_data, axis=1)
                self.xstd_value.setText(
                    f"{_np.nanstd(x_mean):.2f} - {_np.nanmean(_np.nanstd(x_data, axis=0)):.2f}"
                )
                self.ystd_value.setText(
                    f"{_np.nanstd(y_mean):.2f} - {_np.nanmean(_np.nanstd(y_data, axis=0)):.2f}"
                )
            # update Graphs
            for i, p in enumerate(self._x_plots):
                p.setData(t_data, x_data[:, i])
            self.xmeanCurve.setData(t_data, x_mean)
            for i, p in enumerate(self._y_plots):
                p.setData(t_data, y_data[:, i])
            self.ymeanCurve.setData(t_data, y_mean)
            self.xyDataItem.setData(x_mean, y_mean)
        self._graph_pos += 1
        if self._save_data:
            self._save_pos += 1

    def setup_gui(self):
        """Create and lay out all GUI objects."""
        # GUI layout
        main_layout = QVBoxLayout()
        self.setLayout(main_layout)

        # image widget layout
        imageWidget = _pg.GraphicsLayoutWidget()
        imageWidget.setMinimumHeight(250)
        imageWidget.setMinimumWidth(350)

        # setup axis
        self.xaxis = _pg.AxisItem(orientation="bottom", maxTickLength=5)
        self.xaxis.showLabel(show=True)
        self.xaxis.setLabel("x", units="µm")

        self.yaxis = _pg.AxisItem(orientation="left", maxTickLength=5)
        self.yaxis.showLabel(show=True)
        self.yaxis.setLabel("y", units="µm")
        self.xaxis.setScale(scale=self._camera_info.nm_ppx_xy / 1000)
        self.yaxis.setScale(scale=self._camera_info.nm_ppx_xy / 1000)

        self.image_pi = imageWidget.addPlot(
            axisItems={"bottom": self.xaxis, "left": self.yaxis}
        )
        self.image_pi.setAspectLocked(True)
        self.img = _pg.ImageItem()
        imageWidget.translate(-0.5, -0.5)
        self.image_pi.addItem(self.img)
        self.image_pi.setAspectLocked(True)
        imageWidget.setAspectLocked(True)

        self.hist = _pg.HistogramLUTItem(image=self.img)
        self.hist.gradient.loadPreset("viridis")
        imageWidget.addItem(self.hist, row=0, col=1)

        # parameters widget
        self.paramWidget = QGroupBox("Tracking and feedback")

        # ROI buttons
        self.xyROIButton = QPushButton("xy ROI")
        self.xyROIButton.clicked.connect(self._add_xy_ROI)
        self.zROIButton = QPushButton("z ROI")
        self.zROIButton.setEnabled(True)
        self.zROIButton.clicked.connect(self._add_z_ROI)
        self.delete_roiButton = QPushButton("Delete last xy ROI")
        self.delete_roiButton.clicked.connect(self._remove_xy_ROI)
        self.delete_roiButton.setEnabled(False)
        self.toggle_options_button = QPushButton("Show Options Window")
        self.toggle_options_button.clicked.connect(self._toggle_options_window)
        self.toggle_options_button.setEnabled(True)
        self.toggle_options_button.setCheckable(True)
        self.toggle_pattern_button = QPushButton("Show pattern Window")
        self.toggle_pattern_button.clicked.connect(self._toggle_pattern_window)
        self.toggle_pattern_button.setEnabled(True)
        self.toggle_pattern_button.setCheckable(True)

        # Tracking control
        trackgb = QGroupBox("Track")
        trackLayout = QHBoxLayout()
        self.trackAllBox = QCheckBox("All")
        self.trackXYBox = QCheckBox("xy")
        self.trackZBox = QCheckBox("z")
        trackLayout.addWidget(self.trackAllBox)
        trackLayout.addWidget(self.trackXYBox)
        trackLayout.addWidget(self.trackZBox)
        trackgb.setLayout(trackLayout)
        trackgb.updateGeometry()
        trackgb.adjustSize()
        trackgb.setMinimumSize(trackgb.sizeHint())
        self.trackManager = GroupedCheckBoxes(
            self.trackAllBox,
            self.trackXYBox,
            self.trackZBox,
        )
        trackgb.setFlat(True)
        self.trackZBox.stateChanged.connect(self._send_z_rois_and_track)
        self.trackXYBox.stateChanged.connect(self._send_xy_rois_and_track)

        # Correction controls
        lockgb = QGroupBox("Lock")
        lockLayout = QHBoxLayout()
        lockgb.setLayout(lockLayout)
        self.lockAllBox = QCheckBox("All")
        self.lockXYBox = QCheckBox("xy")
        self.lockZBox = QCheckBox("z")
        lockLayout.addWidget(self.lockAllBox)
        lockLayout.addWidget(self.lockXYBox)
        lockLayout.addWidget(self.lockZBox)
        lockgb.setMinimumSize(lockgb.sizeHint())
        self.lockManager = GroupedCheckBoxes(
            self.lockAllBox,
            self.lockXYBox,
            self.lockZBox,
        )
        lockgb.setFlat(True)
        self.lockZBox.stateChanged.connect(self._change_z_lock)
        self.lockXYBox.stateChanged.connect(self._change_xy_lock)

        # Data saving
        datagb = QGroupBox("Data")
        data_layout = QHBoxLayout()
        datagb.setLayout(data_layout)
        self.export_chkbx = QCheckBox("Save")
        self.export_chkbx.stateChanged.connect(self._change_save)
        self.clearDataButton = QPushButton("Clear")
        self.saveZButton = QPushButton("Save Z")
        self.loadZButton = QPushButton("Load Z")
        data_layout.addWidget(self.export_chkbx)
        data_layout.addWidget(self.clearDataButton)
        data_layout.addWidget(self.saveZButton)
        data_layout.addWidget(self.loadZButton)
        self.clearDataButton.clicked.connect(self.clear_data)
        self.saveZButton.clicked.connect(self._save_z_lock)
        self.loadZButton.clicked.connect(self._load_z_lock)
        datagb.setFlat(True)

        numeric_gb = QGroupBox("External settings")
        numeric_layout = QGridLayout()
        numeric_gb.setLayout(numeric_layout)
        self.delay_le = QLineEdit(str(self._period))
        self.delay_le.setValidator(QDoubleValidator(1E-3, 1., 3))
        self.set_delay_button = QPushButton('Set Period')
        self.set_delay_button.clicked.connect(self._set_delay)
        numeric_layout.addWidget(QLabel("Period / s",
                                        alignment=Qt.AlignRight + Qt.AlignVCenter), 0, 0)
        numeric_layout.addWidget(self.delay_le, 0, 1)
        numeric_layout.addWidget(self.set_delay_button, 0, 2)
        if _has_method(self._camera, 'set_exposure'):
            row = numeric_layout.rowCount()
            self._exposure_sb = _create_spin(0.05, 3, 0.01, 0.01, 1.)
            self._set_exposure_button = QPushButton('Set exposure')
            numeric_layout.addWidget(QLabel("Exposure / s",
                                            alignment=Qt.AlignRight + Qt.AlignVCenter), row, 0)
            numeric_layout.addWidget(self._exposure_sb, row, 1)
            numeric_layout.addWidget(self._set_exposure_button, row, 2)
            self._set_exposure_button.clicked.connect(self._manage_exposure_set)

        if _has_method(self._camera, 'set_gain'):
            row = numeric_layout.rowCount()
            self._gain_sb = _create_spin(1, 2, 0.01, 0.01, 10.)
            self._set_gain_button = QPushButton('Set gain')
            numeric_layout.addWidget(QLabel("Gain",
                                            alignment=Qt.AlignRight + Qt.AlignVCenter), row, 0)
            numeric_layout.addWidget(self._gain_sb, row, 1)
            numeric_layout.addWidget(self._set_gain_button, row, 2)
            self._set_gain_button.clicked.connect(self._manage_gain_set)

        numeric_gb.setFlat(True)
        numeric_gb.setMinimumSize(numeric_gb.minimumSizeHint())

        param_layout = QVBoxLayout()
        self.paramWidget.setLayout(param_layout)

        param_layout.addWidget(self.xyROIButton)
        param_layout.addWidget(self.zROIButton)
        param_layout.addWidget(self.delete_roiButton)
        param_layout.addWidget(self.toggle_options_button)
        param_layout.addWidget(self.toggle_pattern_button)

        param_layout.addWidget(trackgb)
        param_layout.addWidget(lockgb)
        param_layout.addWidget(datagb)

        param_layout.addStretch()

        param_layout.addWidget(numeric_gb)
        self.paramWidget.setMinimumSize(self.paramWidget.sizeHint())

        # stats widget
        self.statWidget = QGroupBox("Live statistics")

        self.xstd_value = QLabel("00.00-00.00")
        self.ystd_value = QLabel("-")
        self.zstd_value = QLabel("-")

        stat_subgrid = QGridLayout()
        self.statWidget.setLayout(stat_subgrid)
        stat_subgrid.addWidget(QLabel("\u03C3X/nm"), 0, 0)
        stat_subgrid.addWidget(QLabel("\u03C3Y/nm"), 1, 0)
        stat_subgrid.addWidget(QLabel("\u03C3Z/nm"), 2, 0)
        stat_subgrid.addWidget(self.xstd_value, 0, 1)
        stat_subgrid.addWidget(self.ystd_value, 1, 1)
        stat_subgrid.addWidget(self.zstd_value, 2, 1)

        self.statWidget.setMinimumSize(self.statWidget.sizeHint())
        self.xstd_value.setText("-")

        # drift (and signal intensity) graphs
        self.xyzGraph = _pg.GraphicsLayoutWidget()
        self.xyzGraph.setAntialiasing(True)

        # TODO: Wrap boilerplate into a function
        self.xyzGraph.xPlot = self.xyzGraph.addPlot(row=0, col=0)
        self.xyzGraph.xPlot.setLabels(bottom=("Time", "s"), left=("X shift", "nm"))
        self.xyzGraph.xPlot.showGrid(x=True, y=True)
        self.xmeanCurve = self.xyzGraph.xPlot.plot(pen="r", width=140)

        self.xyzGraph.yPlot = self.xyzGraph.addPlot(row=1, col=0)
        self.xyzGraph.yPlot.setLabels(bottom=("Time", "s"), left=("Y shift", "nm"))
        self.xyzGraph.yPlot.showGrid(x=True, y=True)
        self.ymeanCurve = self.xyzGraph.yPlot.plot(pen="r", width=140)

        self.xyzGraph.zPlot = self.xyzGraph.addPlot(row=2, col=0)
        self.xyzGraph.zPlot.setLabels(bottom=("Time", "s"), left=("Z shift", "nm"))
        self.xyzGraph.zPlot.showGrid(x=True, y=True)
        self.zCurve = self.xyzGraph.zPlot.plot(pen="y")

        # self.xyzGraph.avgIntPlot = self.xyzGraph.addPlot(row=3, col=0)
        # self.xyzGraph.avgIntPlot.setLabels(
        #     bottom=("Time", "s"), left=("Av. intensity", "Counts")
        # )
        # self.xyzGraph.avgIntPlot.showGrid(x=True, y=True)
        # self.avgIntCurve = self.xyzGraph.avgIntPlot.plot(pen="g")

        # xy drift graph (2D point plot)
        self.xyPoint = _pg.GraphicsLayoutWidget()
        self.xyPoint.resize(400, 400)
        self.xyPoint.setAntialiasing(False)

        self.xyplotItem = self.xyPoint.addPlot()
        self.xyplotItem.showGrid(x=True, y=True)
        self.xyplotItem.setLabels(
            bottom=("X position", "nm"), left=("Y position", "nm")
        )

        self.xyDataItem = self.xyplotItem.plot(
            [], pen=None, symbolBrush=(255, 0, 0), symbolSize=5, symbolPen=None
        )

        # z drift graph (1D histogram)
        x = _np.arange(-20, 20)
        y = _np.zeros(len(x))

        self.zHistogram = _pg.BarGraphItem(x=x, height=y, width=0.5, brush="#008a19")
        self.zPlot = self.xyPoint.addPlot()
        self.zPlot.addItem(self.zHistogram)

        # Lay everything in place
        top_layout = QHBoxLayout()
        bottom_layout = QHBoxLayout()
        top_layout.addWidget(imageWidget, stretch=2)
        top_layout.addWidget(self.paramWidget, stretch=0)
        top_layout.addWidget(self.statWidget, stretch=0)
        bottom_layout.addWidget(self.xyzGraph)
        bottom_layout.addWidget(self.xyPoint)
        main_layout.addLayout(top_layout)
        main_layout.addLayout(bottom_layout)

    def goto_position(self, x, y, z):
        """Move to a defined position."""
        self._stabilizer.move(x * 1E3, y * 1E3, z * 1E3)

    @pyqtSlot(bool)
    def _toggle_options_window(self, checked: bool):
        if checked:
            self.toggle_options_button.setText("Hide options window")
            self._config_window.show()
        else:
            self.toggle_options_button.setText("Show options window")
            self._config_window.hide()

    @pyqtSlot(bool)
    def _toggle_pattern_window(self, checked: bool):
        if checked:
            self.toggle_pattern_button.setText("Hide pattern window")
            self._pattern_window.show()
        else:
            self.toggle_pattern_button.setText("Show pattern window")
            self._pattern_window.hide()

    def closeEvent(self, *args, **kwargs):
        """Shut down stabilizer on exit."""
        _lgr.debug("Closing stabilization window")
        if self._save_data:
            self._change_save(Qt.CheckState.Unchecked)
        self._config_window.close()
        self._pattern_window.close()
        super().closeEvent(*args, **kwargs)
