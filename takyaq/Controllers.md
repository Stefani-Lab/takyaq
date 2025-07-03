Let’s assume that we want to build a proportional controller that uses a user-selectable `Kp` if the shift is below a defined threshold, and a different (fixed) `Kp` if the shift above the threshold.

First we import the Controller abstract base class to ensure that our implementation matches the required interface, and numpy to create our internal data arrays. We import annotations to use modern python annotations while keeping compatibility with Python 3.7:
```python  
from __future__ import annotations
import numpy as np
from takyaq.base_classes import BaseController
```  
Now, let’s define our class and its constructor. Takyaq does impose requirements on this function, so it may have any signature we need. We will let the user provide three parameters:

- The distance threshold at which the controller will switch the behaviour  
- The Kp to use when the shift is above the threshold  
- The maximum allowable shift to return, so we don’t jerk the piezo stage (the stabilization system already has a protection system, but it is illustrative and extra safe)

In the contructor, we just save this parameters as properties for later use. We also initialize the instance property `_Kp` which will hold the proportionally constant values for each axis:
```python  
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
```


The GUI calls methods for setting `Kp` and `Ki`. These methods will receive either a single value, if the same constant must be used for all axes, or a sequence of lenght 3 with values for `X`, `Y` and `Z` corrections respectively. Passing a sequence of length different to three will raise an error.

We just save the received `Kp` values into the `_Kp` property. This controller doesn't use `Ki`, so let's ignore it.
```python
    def set_Kp(self, Kp: float | tuple[float]):
        """Save new Kp."""
        self._Kp[:] = np.array(Kp)

    def set_Ki(self, *args):
        """Ki is unsupported by this controller, ignore."""
        pass
```

Controllers must provide methods for resetting their internal state when stabilization is engaged. They must expose one method to be used when `XY` stabilization starts and one to be used when `Z` stabilization Starts. The method for `Z` doesn't take parameters, but the method for `XY` receives the number of ROIs as its single parameter.
In this case, we don't keep a special internal state, so we don't do anything.
```python
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
```

At last, we reach the core of the controller: the response function. This function receives 3 parameters:
 - `t`, the time tag of the shifts, as standard POSIX time (i.e. as returned by python's `time.time()`).
 - `xy_shifts`, an optional `numpy.ndarray` of shape `(number_of_ROIs, 2)`. For each ROI, it holds the measured value of the X and Y shifts. Any of this values might be `np.nan`, in which case it must be ignored. This may happen, for example, if the fitting procedure fails for a fiducial marker. If `XY` stabilization isn't engaged, this parameter is `None`.
 - `z_shift`, a float with the measured `Z` shift. If stabilization on this axis isn't engaged, its value will be 0.

We first calculate the average of all measured `XY` shifts, checking and managing special cases.
For ease of calculation we save the shifts on a `numpy,ndarray` called `error`. The variable name is derived from PID nomenclature.
```python
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
```

We then calculate the `Kp` we should use. We make a copy of the array holding the standard `Kp`s that should be used for shifts below the threshold, and set the values for the axis whose shifts are above the threshold to the special `Kp` value.
```python
        Kp_to_use = self._Kp.copy()
        Kp_to_use[np.abs(error) >= self._threshold] = self._Kp_plusultra
```

Finally, we calculate the correction to be performed and clip values to the user selected maximum displacement. Notice the sign change of the return value, as we must return the desired relative movement to be performed.
```python
        rv = np.clip(error * Kp_to_use, -self._max_disp, self._max_disp)
        return -rv
```


Here's the full code of the controller. It can be found in `samples/thresholded_controller.py`. This kind of controllers may be useful when testing patterns that require large (>10 nm) movements, as Kp values typically used for stabilization are small and thus a large number of steps are required for these shifts.

```python  
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
            _lgr.warning("x shift is NAN")
            x_shift = 0.0
        if y_shift is np.nan:
            _lgr.warning("y shift is NAN")
            y_shift = 0.0
        error = np.array((x_shift, y_shift, z_shift))

        Kp_to_use = self._Kp.copy()
        Kp_to_use[np.abs(error) >= self._threshold] = self._Kp_plusultra

        rv = np.clip(error * Kp_to_use, -self._max_disp, self._max_disp)
        return -rv
```
