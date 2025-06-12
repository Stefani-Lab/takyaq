# Takyaq

Takyaq is a python module that uses piezoelectric stages to achieve within sub nm 3D stabilization of samples. It is is designed to be used in superresolution microscopy, altough it can be useful for many other applications, including photolithography.
The module performance is shown in the [article] *Open-source Sub-Nanometer Stabilization System for Super-resolution Fluorescence Microscopy*.


## Features

 - Pure pyhthon implementation.
 - GUI agnostic.
 - Customizable and adaptable to any piezo stage and camera.
 - Tested and explained.


## What is included

 - A stabilizer module.
 - A fully functional PyQt frontend that is more than an example.
 - Mock camera and piezo modules, so you can develop, test and try without real equipment.
 - Different stabilization strategies so you can choose your own withouth rolling your own.
 
## How to install

Takyaq is not yet available in PyPi. Meanwhile installation and development is managed using [Poetry].
Clone the repository, either using `ssh`:
```sh
git clone git@github.com:Stefani-Lab/takyaq.git
```
or `https`:
```sh
git clone https://github.com/Stefani-Lab/takyaq.git
```

Install required dependencies using:
```sh
cd takyaq
poetry install
```

If you want to use the provided GUI, muy must also install some extra dependencies (PyQT and pyqtgraph):
```sh
poetry install --with qt
```

Takyaq includes a mock Piezo and Camera module for testing and development.
 

Takyaq uses a number of open source projects to work properly:

- [NumPy] - For efficient scientific computing
- [SciPy] - For data fitting


## How to use

### Module

Interfaces needed:
 - A camera object, that exposes a function named `get_image` and returns a single color image (a [NumPy] 2D array)
 - A piezo object, that must expose three functions
   - One called `get_position` that returns a 3-element collection (list, array, tuple, etc) of x, y, and z positions *in nanometers*.
   - One called `set_position_xy` that takes 2 arguments: the x and y positions where the stage should move *in nanometers*.
   - One called `set_position_z` that takes 1 arguments: the z position where the stage should move *in nanometers*.

Note that the stabilization module internally uses nanometers for all distance measurements. On the contrary, the provided GUI uses µm on the "Move to" textboxes.

The provided GUI will use some optional extra methods if implemented. The piezo object can optionally expose a method called `get_limits`, that takes no parameters and return a tuple of three pairs of floats, each pair representing the minimum and maximum values for X, Y and Z axes. The camera object can optionally expose methods called `set_exposure` and `set_gain`. Both methods take a single float parameter. For `set_exposure`, this value is the desired exposure time in seconds. For `set_gain`, the parameter is a value between 0 and 10. For each of these methods implemented in the object, the corresponding control will be available to set the parameter from the frontend.

If the Python interfaces to your camera and stage use different naming conventions or units, you should write some wrapping classes (see below).

Calibration data needed:
 - How many nanometers each camera pixel represents in the X and Y directions.
 - How the Z reflection spot moves when the Z position changes:
   - The direction (angle between the X axis and the line where the spot moves)
   - The number of pixels that the spot moves for each nanometers
 
The software provides a procedure to obtain the calibration data. Nevertheless, see below for calibration pitfalls.

The module communicates with other modules using callbacks. The callback procedure should do its job (put the data in a queue, etc.) fast, so the latency between position corrections is short. Check the PyQt example for a basic idea.


### Stabilization strategies

The software is designed to be used with different control systems. There are currently two included stabilization strategies:
  - Basic PI controller.
  - PI controller with outliers rejection, either using Z-score (MAD based) or standard deviation.

Both strategies perform well under normal scenarios. Nevertheless, it is easy to implement your own strategies with basic programming skills. A full example of how to implement a controller can be found in the samples folder. Morover, the implementations found in the `controllers` module are easy to understand. For the imaptient, a brief explanation follows.

#### Implementing response functions.

Response functions are implemented as methods of “Controller” objects. The file `base\_classes.py` provides an abstract base class that can be used as a base to ensure consistency. A controller object must expose three methods:  
`reset_x`: This method is called when XY stabilization is engaged. It is called with the number of XY ROIs (as an int) as its only parameter. The controller should initialize or reset its internal data, preparing for beginning stabilization on the XY axes.  
`reset_z` : This method is called when Z stabilization is engaged. It is called without parameters. The controller should initialize or reset its internal data, preparing for beginning stabilization on the Z axis.
`response`: This method is called on each stabilization cycle. it should return a 3-item tuple representing the response in X, Y and Z. Please note that a 3-tuple must be returned even if any stabilization is disabled. It receives 3 parameters:
  - A float `t`, the timestamp of the localizations. It is the time as provided by Python’s `time.time()` call.  
  - `xy_shifts`, an array of shape `(n_ROIS, 2)`. Each item of the array is the measured X and Y shift for each of the XY ROIS. If the localization of a particle failed (for example when the fitting procedure does not converge), the corresponding shifts will be `numpy.nan`, therefore the implementations must properly handle `nan` values. If XY stabilization is disabled, this parameter is `None.`  
  - `z_shift`, a float: The shift on the Z axis.

Note that the frontend provided is designed around the provided PI controller, and therefore expects the controllers to expose two extra methods: `set_Kp` and `set_Ki`. Both methods receive either a single float or a collection (tuple, list or array) of three floats. These methods are called when the corresponding values are set. If a single value is received, the same value should be used for all the axis. If a collection is received, the elements correspond to the value of the constant for the X, Y and Z  in order


### Adapting cameras and piezos.

A piezo and a camera Abstract Base Class (ABC) are provided in the `base_classes` module. Use them to ensure that the method signatures are adecuate.

Assume you have a piezo controller API that exposes functions that are called `move_to` that takes as parameter the axis as a string (`'X'`, `'Y'` or `'Z'`) and the new position in µm, and `get_pos` instead of `get_position`, that returns the values in micrometers. You can use a wrapper like this one:
```python
import my_piezo_driver
from takyaq.base_classes import BasePiezo

class WrappedPiezo(Baseiezo):
   def __init__(self):
       # Do whatever your driver needs you to do
       self._motor = my_piezodriver.get_motor()
       self._motor.init()

    def close(self):
        self._motor.shutdown()

    def set_position_xy(self, x: float, y: float):
        self._motor.move_to('X', x / 1E3)
        self._motor.move_to('Y', y / 1E3)

    def set_position_z(self, z: float):
        self._motor.move_to('Z', z / 1E3)

    def get_position(self) -> tuple[float, float, float]:
        return tuple(p * 1E3 for p in self._motor.get_pos())


piezo = WrappedPiezo()     
# Use `piezo` with takyaq 
...

piezo.close()


```
Making a [context manager] is recommended (it is simple and very convenient).

### PyQt frontend

The provided PyQt frontend is a fully functional example of how to use the stabilization module. For most purposes, you can use it _as is_. You must provide:
 - A camera object.
 - A piezo object.
 - Calibration values (as a `info_types.CameraInfo` object). If you do not know these values, just use a value of 1 for each one, and perform the calibration procedure reported below.

Once you run the program, perform the following steps:
 - Press the `Show options window` button and set a very low `Kp` value for all axis, and set all `Ki` to 0
 - Manually find the focus.
 - Create the Z ROI and move it to encompass the Z spot reflection.
 - You can check the _Z_ tracking checkbox right away.
 - If you have provided calibration values, you can also mark the _Z_ lock checkbox to stabilize the focus.
 - Create as many _XY_ ROIs as you feel. Move and adjust their size to encompass one fiducial mark each.
 - You can check the _XY_ tracking checkbox.
 - If you have provided calibration values, you can also check the _XY_ lock checkbox to stabilize the position.
 - Once tracking is active (both for _XY_ and for _Z_), changing the corresponding ROIs will have no effect. The ROIs positions that were selected when the tracking was started are remembered until tracking is disabled.

Calibration:
 - You must select the ROIs for the calibration you want to perform (XY or Z).
 - Be sure to have started tracking (locking is not neccesary).
 - Open the options window and press the Desired "Calibrate" button. The calibration data will be logged to screen. A nicer report will be implemented on the near future.
 - You might want to repeat and average many calibrations.


### Camera and piezo orientation

  Depending on the assembly, the camera X-Y axis can be switched, or one of them flipped. Correct this in the `camera` wrapper 


### Pixel sizes

Be careful since the pixel sizes used and determined using the provided calibration are only self-consistent. If the stage calibration is incorrect, the program will report valid but incorrect values for the stabilization precision.


## Some design comments and pitfalls

 - We try to keep compatibility with Python 3.7+, to support legacy setups.
 - Times are reported in seconds since the Epoch, as provided by Python's `time.time()`, since it is the most convenient (not the best) way of having the same time reference between different programs.
 - The stabilization loop runs on a different thread. It could run on a different process for efficiency (to avoid GIL issues), but most applications need to be able to move the piezo from other modules. Managing the same stage from two different processes is usually much harder than from different threads on the same process.
 - Fitting of the fiducial beads is done in multiple processes to achieve higer correction frequencies. This implies that the many processes are spawned. Objects created on module load are created on each process and might collide (open ports, etc.)
 - More information can be extracted from the calibrations (like coupling between axes). This kind on analyses might be implemented in the future.


## Some development comments
As we don't want to worry about formatting, from time to time we let [black] do its job.
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Upstream development takes place in <https://github.com/azelcer/takyaq>. Please report bugss issues and requests there.


## Citing

If you use this module or a derivation, please cite [article], and drop us a line.


## License

This module is distributed under GNU Affero General Public License v3.0 or later.


   [SciPy]: <https://scipy.org/>
   [Poetry]: <http://angularjs.org>
   [NumPy]: <https://numpy.org/>
   [black]: <https://black.readthedocs.io/en/stable/>
   [article]: <https://dx.doi.org/10.21203/rs.3.rs-6131181/v1>
   [context manager]: <https://docs.python.org/3/reference/datamodel.html#context-managers>