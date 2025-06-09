#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  9 11:08:00 2025

@author: azelcer
"""

# from pylablib.devices import Thorlabs, Andor  # import the device libraries
# import numpy as np  # import numpy for saving

# # connect to the devices
# with Thorlabs.KinesisMotor("27000000") as stage, Andor.AndorSDK2Camera() as cam:
#    # change some camera parameters
#    cam.set_exposure(50E-3)
#    cam.set_roi(0, 128, 0, 128, hbin=2, vbin=2)
#    # start the stepping loop
#    images = []
#    for _ in range(10):
#       stage.move_by(10000)  # initiate a move
#       stage.wait_move()  # wait until it's done
#       img = cam.snap()  # grab a single frame
#       images.append(img)

# from pylablib.devices import uc480, NI

# # Set the DLL path
# dll_path = r"C:\Program Files\Thorlabs\Scientific Imaging\ThorCam"
# os.environ["PATH"] += os.pathsep + dll_path

from __future__ import annotations
import logging as _lgn
from ..base_classes import BaseCamera as _bc, BasePiezo  as _bp
from pylablib.devices.interface.camera import ICamera as _ICamera
from pylablib.devices.Thorlabs.kinesis import BasicKinesisDevice as _Kinesis

_lgr = _lgn.getLogger(__name__)
_lgr.setLevel(_lgn.DEBUG)


class PyLabLibCameraWrapper(_bc):
    """Generic wrapper for pylablib cameras."""

    def __init__(self, camera: _ICamera):
        """Keep a reference to the camera and start acquisition."""
        self._cam = camera
        self._cam.start_acquisition()

    def get_image(self):
        """Grab and return a single image."""
        return self._cam.snap()
        # might need rotation / flipping / mirror in numpy (rot90, ...)

    def set_exposure(self, new_exp: float):
        """Set camera exposure."""
        _lgr.info("New exposure: %s", new_exp)
        self._cam.set_exposure(new_exp)

    def close(self):
        """Shutdown everything."""
        self._cam.stop_acquisition()
        self._cam.close()

    def __enter__(self):
        """Nothing to do.

        Could start acquisition loop here.
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Cleanup."""
        self.close()
        return False


class PyLabLibKinesisPiezoControllerWrapper(_bp):
    """Generic wrapper for pylablib Kinesis piezo Controllers.

    Works on open-loop mode
    """

    def __init__(self, stage_ctrl: _Kinesis, nm_per_volt: float):
        """Keep a reference to the controller.

        The controller must have 3 channels, attached o X, Y and Z respectively.
        """
        self._ctrl = stage_ctrl
        self._NM_PER_VOLT = nm_per_volt
        self._channels = self._ctrl.get_all_channels()
        if len(self._channels) != 3:
            raise ValueError("Device does not have 3 channels")

    def get_position(self) -> tuple[float, float, float]:
        """Return (x, y, z) position of the piezo in nanometers."""
        return tuple(self._ctrl.get_output_voltage(channel=c) * self._NM_PER_VOLT
                     for c in self._channels)

    def set_position_xy(self, x: float, y: float):
        """Move to xy position specified in nanometers."""
        [self._ctrl.set_output_voltage(pos / self._NM_PER_VOLT, channel=c) for
         (pos, c) in zip((x, y,), self._channels[:2])]

    def set_position_z(self, z: float):
        """Move to z position, specified in nanometers."""
        self._ctrl.set_output_voltage(z / self._NM_PER_VOLT, channel=self._channels[2])

    def close(self):
        """Shutdown everything."""
        self._ctrl.close()

    def __enter__(self):
        """Nothing to do."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Cleanup."""
        self.close()
        return False
