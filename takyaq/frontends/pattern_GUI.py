#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 12 12:21:58 2025

@author: azelcer
"""
import logging as _lgn
from PyQt5.QtCore import pyqtSignal, pyqtSlot, QObject, Qt, QTimer
from PyQt5.QtGui import QDoubleValidator
from PyQt5.QtWidgets import (
    QGroupBox,
    QFrame,
    QLabel,
    QPushButton,
    QCheckBox,
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


# def list2txt(positions: list[tuple[float, float, float]]) -> str:
#     """Transforma un array de Nx3 en un texto."""
#     returrn '\n'.join([str()])
#     for pos in data:
#         originalline = line
#         line = line.replace(';', ' ').strip()
#         if not line:
#             continue
#         _ = line.split()
#         if len(_) != 3:
#             raise ValueError(f'Can not interpret "{originalline}" as a 3-tuple')
#         for orig, dest in zip(_, (x, y, t)):
#             value = float(orig)  # raises the proper exception
#             dest.append(value)
#     return np.array((x, y, t))

class PatternWindow(QFrame):
    """Window for defining a shift pattern."""

    def __init__(self, parent, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.setWindowFlag(Qt.WindowCloseButtonHint, False)
        self._init_gui(parent)
        self._timer = QTimer()
        self._timer.timeout.connect(self.click)

    def _init_gui(self, parent):
        self.setWindowTitle("Patterns")
        layout = QHBoxLayout()

        definition_gb = QGroupBox("Pattern definition")
        definition_layout = QVBoxLayout()
        definition_gb.setLayout(definition_layout)
        self.points_te = QTextEdit()
        L_layout = QHBoxLayout()
        L_layout.addWidget(QLabel("L / nm"))
        self._length_le = QLineEdit("10.0")
        L_layout.addWidget(self._length_le)
        self.parseButton = QPushButton('Process pattern')
        self.parseButton.clicked.connect(self._interpret)
        self._length_le.setValidator(QDoubleValidator(0, 100., 2))
        self.startButton = QPushButton('Start')
        self.startButton.clicked.connect(self._start)
        self.stopButton = QPushButton('Stop')
        self.stopButton.setEnabled(False)
        self.stopButton.clicked.connect(self._finish_pattern)
        definition_layout.addWidget(self.points_te)
        definition_layout.addLayout(L_layout)
        definition_layout.addWidget(self.parseButton)
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
        puntos = self._interpret()
        if not puntos:
            _lgr.info("Wrong format for positions list")
            return

        self._points = np.array(puntos['positions'])
        self._points[:, 0:2] *= puntos['L']
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
            print("Listo")
            return
        print("mover a ", self._points[self._current_step][0:2])
        self._timer.setInterval(int(self._points[self._current_step, 2] * 1000))
        self._current_step += 1

    def _interpret(self):
        try:
            x, y, t = text2list(self.points_te.toPlainText())
            print(x, y, t)
            L = float(self._length_le.text())
            rv = {"L": L, 'positions': [v for v in zip(x, y, t)]}
            print(rv)
            self.xyDataItem.setData(np.array(x) * L, np.array(y) * L)
            return rv

        except Exception as e:
            print("error en el texto: ", e, type(e))


if __name__ == '__main__':
    from PyQt5.QtWidgets import QApplication

    if not QApplication.instance():
        app = QApplication([])
    else:
        app = QApplication.instance()

    gui = PatternWindow(None)
    gui.setWindowTitle('E.A.E.A.P.P')
    gui.show()
    gui.raise_()
    gui.activateWindow()
    app.exec_()
