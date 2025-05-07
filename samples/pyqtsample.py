# -*- coding: utf-8 -*-
"""
Sample PyQT frontend for Takyaq.

Uses a mocked camera and piezo motor: replace those parts with real interfaces
to have a fully functional stabilization program.

Use:
    - Set the parameters:
         - nm per pixel XY
         - nm per pixel Z
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
import logging as _lgn

from takyaq.frontends.PyQt_frontend import Frontend
from takyaq.stabilizer import Stabilizer
from PyQt5.QtWidgets import QApplication

from takyaq.controllers import PIController as PIController
from takyaq.mocks import MockCamera, MockPiezo
from takyaq.info_types import CameraInfo


_lgr = _lgn.getLogger(__name__)
_lgr.setLevel(_lgn.DEBUG)


# Constants for camera mock
_CAMERA_XY_NMPPX = 23.5
_CAMERA_Z_NMPPX = 10
_CAMERA_Z_ROTATION = 3.1415 / 4


if __name__ == "__main__":
    camera_info = CameraInfo(
        _CAMERA_XY_NMPPX,
        _CAMERA_Z_NMPPX,
        _CAMERA_Z_ROTATION,
    )
    # Mock camera, replace with a real one
    camera = MockCamera(
        _CAMERA_XY_NMPPX,
        _CAMERA_XY_NMPPX,
        _CAMERA_Z_NMPPX,
        _CAMERA_XY_NMPPX * 7,  # en pixeles
        _CAMERA_Z_ROTATION,
        1,  # Center position noise in pixels
        40,
    )
    # Mock piezo motor, replace with your own
    piezo = MockPiezo(camera)
    controller = PIController()

    if not QApplication.instance():
        app = QApplication([])
    else:
        app = QApplication.instance()
    app.setStyle('Windows')
    with Stabilizer(camera, piezo, camera_info, controller) as stb:
        gui = Frontend(camera, piezo, controller, camera_info, stb)
        gui.setWindowTitle("Takyaq with PyQt frontend")
        gui.show()
        gui.raise_()
        gui.activateWindow()

        app.exec_()
        app.quit()
