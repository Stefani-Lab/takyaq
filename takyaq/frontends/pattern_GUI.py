#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pattern generation window.

TODO: Use QTableWidget for position definition.

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
import json as _json
from itertools import repeat as _repeat, product as _product
from typing import List as _List, Tuple as _Tuple
from PyQt5.QtCore import pyqtSlot, QTimer, Qt
from PyQt5.QtGui import QDoubleValidator
from PyQt5.QtWidgets import (
    QGroupBox,
    QFrame,
    QLabel,
    QPushButton,
    QFileDialog,
    QHBoxLayout,
    QVBoxLayout,
    QLineEdit,
    QTextEdit,
    QCheckBox,
)
import numpy as np
import pyqtgraph as _pg
_lgr = _lgn.getLogger(__name__)
_lgr.setLevel(_lgn.DEBUG)


def text2list(txt: str) -> np.ndarray:
    r"""Interpret a text as an Nx3 array.

    Expects empty spaces (' ', '\t') or semicolons as spacers. We avoid commas as
    spacers as it is the decimal separator in many languages.
    """
    x = []
    y = []
    t = []
    for line in txt.split('\n'):
        originalline = line
        line = line.replace(';', ' ').strip()
        if not line:
            continue
        _ = line.split()
        if len(_) != 3:
            raise ValueError(f'Can not interpret "{originalline}" as a 3-tuple')
        for orig, dest in zip(_, (x, y, t)):
            value = float(orig)  # raises the proper exception
            dest.append(value)
    return np.array((x, y, t))


def list2txt(positions: _List[_Tuple[float, float, float]]) -> str:
    """Render an Nx3 array as text."""
    return '\n'.join([' '.join([str(c) for c in p]) for p in positions])


def _create_square_array(n_points: int, L: float, period: float):
    """Create a square grid vertexes with the same residence time.

    Note that n_points vertexes give n_points-1 spaces.
    """
    rng = np.linspace(0, 1, n_points)
    rv = [(x, y, t) for (y, x), t in zip(_product(rng, rng), _repeat(period))]
    return {"L": L, "positions": rv}


class PatternWindow(QFrame):
    """Window for defining a shift pattern."""

    def __init__(self, parent, stabilizer, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if parent:
            self.setWindowFlag(Qt.WindowCloseButtonHint, False)
        self._stabilizer = stabilizer
        self._init_gui()
        self._timer = QTimer()
        self._timer.timeout.connect(self.click)
        self.setWindowTitle('Patterns')

    def _init_gui(self):
        self.setWindowTitle("Patterns")
        layout = QHBoxLayout()

        definition_gb = QGroupBox("Pattern definition")
        definition_layout = QVBoxLayout()
        definition_gb.setLayout(definition_layout)
        self.points_te = QTextEdit()
        self.points_te.setToolTip(
            """Enter a list of vertexes for the pattern, one vertex per line.
Each vertex is specified by 3 numbers separated by spaces: x shift, y shift and residence time.
Shifts are specified in units of the the 'L' parameter, and time is in seconds."""
        )
        L_layout = QHBoxLayout()
        L_layout.addWidget(QLabel("L / nm"))
        self._length_le = QLineEdit("10.0")
        self._length_le.setValidator(QDoubleValidator(0, 200., 2))
        L_layout.addWidget(self._length_le)
        xtra_gb = QGroupBox("Extra shift / nm")
        xtra_shift_layout = QHBoxLayout()
        xtra_gb.setLayout(xtra_shift_layout)
        self._use_extra_chkbx = QCheckBox("Use")
        self._use_extra_chkbx.stateChanged.connect(self._handle_use_toggle)
        xtra_shift_layout.addWidget(self._use_extra_chkbx)
        self.xtra_x_le = QLineEdit("0.0")
        self.xtra_x_le.setValidator(QDoubleValidator(-200., 200., 1))
        self.xtra_y_le = QLineEdit("0.0")
        self.xtra_y_le.setValidator(QDoubleValidator(-200., 200., 1))
        xtra_shift_layout.addWidget(QLabel("x"))
        xtra_shift_layout.addWidget(self.xtra_x_le)
        xtra_shift_layout.addWidget(QLabel("y"))
        xtra_shift_layout.addWidget(self.xtra_y_le)
        self.parseButton = QPushButton('Process pattern')
        self.parseButton.clicked.connect(self._interpret)
        self.loadButton = QPushButton('Load pattern')
        self.loadButton.clicked.connect(self.load_dialog)
        self.saveButton = QPushButton('Save pattern')
        self.saveButton.clicked.connect(self.save_dialog)
        load_save_layout = QHBoxLayout()
        load_save_layout.addWidget(self.loadButton)
        load_save_layout.addWidget(self.saveButton)
        self.startButton = QPushButton('Start')
        self.startButton.clicked.connect(self._start)
        self.stopButton = QPushButton('Stop')
        self.stopButton.setEnabled(False)
        self.stopButton.clicked.connect(self._finish_pattern)
        start_stop_layout = QHBoxLayout()
        start_stop_layout.addWidget(self.startButton)
        start_stop_layout.addWidget(self.stopButton)

        definition_layout.addWidget(self.points_te)
        definition_layout.addLayout(L_layout)
        definition_layout.addWidget(xtra_gb)
        definition_layout.addWidget(self.parseButton)
        definition_layout.addLayout(load_save_layout)
        definition_layout.addLayout(start_stop_layout)
        definition_gb.setFlat(True)
        self.xyPoint = _pg.GraphicsLayoutWidget()
        self.xyPoint.resize(400, 400)
        self.xyPoint.setAntialiasing(False)

        self.xyplotItem = self.xyPoint.addPlot()
        self.xyplotItem.showGrid(x=True, y=True)
        self.xyplotItem.setLabels(
            bottom=("X shift", "nm"), left=("Y shift", "nm")
        )

        self.xyDataItem = self.xyplotItem.plot(
            [], symbolBrush=(255, 0, 0), symbolSize=10, symbolPen=None
        )
        layout.addWidget(definition_gb)
        layout.addWidget(self.xyPoint)
        self.setLayout(layout)

    def _goto_rest_reference(self):
        """Goes back to rest reference, unless a pattern is being executed.

        Rest point is 0,0 if extra shift is not enabled, or whatever the user selected
        if enabled.
        """
        if not self._timer.isActive():
            if self._use_extra_chkbx.isChecked():
                self._stabilizer.shift_reference(*self._read_xtras(), 0.)
            else:
                self._stabilizer.shift_reference(0., 0., 0.)

    @pyqtSlot(int)
    def _handle_use_toggle(self, state: int):
        """Handle 'Use extra shift' checkbox toggle."""
        self.xtra_x_le.setEnabled(state == 0)
        self.xtra_y_le.setEnabled(state == 0)
        self._goto_rest_reference()

    def _read_xtras(self) -> _Tuple[float, float]:
        """Read user entered extra shift position."""
        return float(self.xtra_x_le.text()), float(self.xtra_y_le.text())

    def _start(self):
        """Start moving."""
        points = self._interpret()
        if not points:
            _lgr.info("Wrong format for positions list")
            return
        self._points = np.array(points['positions'])
        if not len(self._points):
            _lgr.warning("Empty position list")
            return
        shift_x, shift_y = (0., 0.)
        if self._use_extra_chkbx.isChecked():
            try:
                shift_x, shift_y = self._read_xtras()
            except Exception as e:
                _lgr.warning("Invalid extra shift: %s (%s)", type(e), e)
        self._points[:, 0:2] *= points['L']
        self._points[:, 0] += shift_x
        self._points[:, 1] += shift_y
        self._timer.setInterval(0)
        self._current_step = 0
        self.startButton.setEnabled(False)
        self.stopButton.setEnabled(True)
        self._timer.start()

    def _finish_pattern(self):
        """Stop timer, restore buttons and reset position."""
        self._timer.stop()
        self.startButton.setEnabled(True)
        self.stopButton.setEnabled(False)
        self._goto_rest_reference()

    @pyqtSlot()
    def click(self):
        """Handle each point in the pattern.

        Move to next step or finish.
        """
        if self._current_step >= len(self._points):
            self._finish_pattern()
            _lgr.info("Pattern finished")
            return
        self._stabilizer.shift_reference(*self._points[self._current_step][0:2], 0.)
        self._timer.setInterval(int(self._points[self._current_step, 2] * 1000))
        self._current_step += 1

    def _interpret(self) -> dict:
        """Produce a pattern dict from user provided data."""
        try:
            x, y, t = text2list(self.points_te.toPlainText())
            L = float(self._length_le.text())
            rv = {"L": L, 'positions': [v for v in zip(x, y, t)]}
            self.xyDataItem.setData(np.array(x) * L, np.array(y) * L)
            return rv
        except Exception as e:
            _lgr.warning("Error %s parsing text: %s", type(e), e, )
            raise

    @pyqtSlot(bool)
    def load_dialog(self, clicked: bool):
        """Load file dialog."""
        filename = QFileDialog.getOpenFileName(
            self, "Select pattern", "", "json files (*.json);;all files (*.*)")
        if filename[0]:
            try:
                self._do_load(filename[0])
                self._interpret()
            except Exception as e:
                _lgr.warning("Error %s (%s) opening and interpreting file %s",
                             type(e), e, filename[0])

    @pyqtSlot(bool)
    def save_dialog(self, clicked: bool):
        """Save file dialog."""
        filename = QFileDialog.getSaveFileName(
            self, "Save pattern", "pattern.json", "json files (*.json);;all files (*.*)")
        if filename[0]:
            self._do_save(filename[0])

    def _do_load(self, filename: str):
        """Load and check pattern definition."""
        with open(filename, "rt") as fd:
            data = _json.load(fd)
        # Everything raises the expected exceptions, so no handling
        L = data['L']
        pos = data['positions']
        for p in pos:
            if len(p) != 3:
                _lgr.error("Invalid data length in file: %s", p)
                return
        self.points_te.setText(list2txt(pos))
        self._length_le.setText(str(L))

    def _do_save(self, filename: str):
        """Save pattern definition."""
        data = self._interpret()
        with open(filename, "wt") as fd:
            data = _json.dump(data, fd)


if __name__ == '__main__':
    from PyQt5.QtWidgets import QApplication

    class DummyStabilizer:
        def shift_reference(self, dx: float, dy: float, dz: float):
            print("Set shift reference to", dx, dy)

    if not QApplication.instance():
        app = QApplication([])
    else:
        app = QApplication.instance()

    gui = PatternWindow(None, DummyStabilizer())
    gui.show()
    gui.raise_()
    gui.activateWindow()
    app.exec_()
