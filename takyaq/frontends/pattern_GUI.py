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
)
import numpy as np
import pyqtgraph as _pg
_lgr = _lgn.getLogger(__name__)
_lgr.setLevel(_lgn.DEBUG)


def text2list(txt: str) -> np.ndarray:
    """Transforma un texto en un array de Nx3."""
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
    """Transforma un array de Nx3 en un texto."""
    return '\n'.join([' '.join([str(c) for c in p]) for p in positions])


class PatternWindow(QFrame):
    """Window for defining a shift pattern."""

    def __init__(self, parent, stabilizer, *args, **kwargs):
        super().__init__(*args, **kwargs)
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
        L_layout = QHBoxLayout()
        L_layout.addWidget(QLabel("L / nm"))
        self._length_le = QLineEdit("10.0")
        self._length_le.setValidator(QDoubleValidator(0, 200., 2))
        L_layout.addWidget(self._length_le)
        self.parseButton = QPushButton('Process pattern')
        self.parseButton.clicked.connect(self._interpret)
        self.loadButton = QPushButton('Load pattern')
        self.loadButton.clicked.connect(self.load_dialog)
        self.saveButton = QPushButton('Save pattern')
        self.saveButton.clicked.connect(self.save_dialog)
        self.startButton = QPushButton('Start')
        self.startButton.clicked.connect(self._start)
        self.stopButton = QPushButton('Stop')
        self.stopButton.setEnabled(False)
        self.stopButton.clicked.connect(self._finish_pattern)
        definition_layout.addWidget(self.points_te)
        definition_layout.addLayout(L_layout)
        definition_layout.addWidget(self.parseButton)
        definition_layout.addWidget(self.loadButton)
        definition_layout.addWidget(self.saveButton)
        definition_layout.addWidget(self.startButton)
        definition_layout.addWidget(self.stopButton)
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

    def _start(self):
        points = self._interpret()
        if not points:
            _lgr.info("Wrong format for positions list")
            return

        self._points = np.array(points['positions'])
        self._points[:, 0:2] *= points['L']
        self._timer.setInterval(0)
        self._current_step = 0
        self.startButton.setEnabled(False)
        self.stopButton.setEnabled(True)
        self._timer.start()

    def _finish_pattern(self):
        """Stop timer and restore buttons."""
        self._timer.stop()
        self.startButton.setEnabled(True)
        self.stopButton.setEnabled(False)

    @pyqtSlot()
    def click(self):
        """Move to next step or end."""
        if self._current_step >= len(self._points):
            self._finish_pattern()
            _lgr.info("Pattern finished")
            return
        self._stabilizer.shift_reference(*self._points[self._current_step][0:2], 0.)
        self._timer.setInterval(int(self._points[self._current_step, 2] * 1000))
        self._current_step += 1

    def _interpret(self):
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

    gui = PatternWindow(DummyStabilizer())
    gui.show()
    gui.raise_()
    gui.activateWindow()
    app.exec_()
