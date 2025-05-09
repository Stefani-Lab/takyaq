"""
Utility classes for PyQt.

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
from PyQt5.QtWidgets import (QCheckBox, QDoubleSpinBox)


class GroupedCheckBoxes:
    """Manages grouped CheckBoxes states.

    This is a helper class to ease GUI implementation. It implements an 'All'
    checkbox that controls and stays sinchronized with others. It should be used
    carefully to avoid state changes loops.
    """

    def __init__(self, all_checkbox: QCheckBox, *other_checkboxes):
        """Init class.

        Parameters
        ----------
        all_checkbox : QCheckBox
            The checkbox that checks/unchecks all others.
        *other_checkboxes : Iterable[QCheckBox]
            All other checkboxes.
        """
        self.acb = all_checkbox
        self.others = other_checkboxes
        for _ in other_checkboxes:
            _.stateChanged.connect(self.on_state)
        all_checkbox.clicked.connect(self.on_click)

    def on_state(self, state: int):
        """Handle single items change."""
        self.acb.setChecked(all([_.isChecked() for _ in self.others]))

    def on_click(self, is_checked: bool):
        """Handle 'All' checkbox click."""
        for _ in self.others:
            _.setChecked(is_checked)


def create_spin(value: float, decimals: int, step: float,
                minimum: float = 0., maximum: float = 1.):
    """Create spin with properties.

    Parameters
    ----------
    value: float
        Inital value
    decimals: int
        Number of decimals to show
    step: float
        Change on up/down click
    minimum: float = 0.
        Minimum allowed value.
    maximum: float = 1.
        Maximum allowed value.
    """
    rv = QDoubleSpinBox()
    rv.setValue(value)
    rv.setDecimals(decimals)
    rv.setMinimum(minimum)
    rv.setMaximum(maximum)
    rv.setSingleStep(step)
    return rv
