# -*- coding: utf-8 -*-
"""
Child Class 'LaserBeam', for holding laser atributes
"""
from typing import Callable

import numpy as np
import matplotlib.pyplot as plt
from Particle_System_Simulator.particleSystem.SystemObject import SystemObject


class LaserBeam(SystemObject):
    """
    Holds information about the laserbeam

    Attributes
    ---------
    intensity_profile : Callable[[float, float], float]
        Maps (x, y) to the scalar intensity profile of the beam.
    polarization_map : Callable[[float, float], np.ndarray]
        Maps (x, y) to the polarization profile of the beam represented as a Jones vector.

    """

    def __init__(
        self,
        intensity_profile: Callable[[float, float], float],
        polarization_map: Callable[[float, float], list[np.complex128, np.complex128]],
    ):
        """
        Initializes a laserbeam based on input parameters

        The polarization profile represents the Jones vector. Its datatype is
        allowed to be complex in order to capture both linear and circular
        polarization states.


        Parameters
        ----------
        intensity_profile : Callable[[float, float], float]
            A numpy compatible function that maps x, y to the scalar intensity profile
            of the beam. [W/m^2]
        polarization_map : Callable[[float, float], np.ndarray]
            A numpy compatible function that maps x, y to the polarization profile of the beam. [-]
            The polarisation vector should be a unit vector!
        Returns
        -------
        None.

        """
        self.intensity_profile = intensity_profile
        self.polarization_map = polarization_map

    def __str__(self):
        print("LaserBeam instantiated with attributes:")
        print(f"polarisation_map: {self.polarization_map}")
        print(f"intensity_profile: {self.intensity_profile}")
        return ""

    def plot(
        self,
        ax=None,
        x_range=(-1, 1),
        y_range=(-1, 1),
        z_scale=1,
        number_of_points=121,
        arrow_length=0.1,
    ):
        """
        plotting function to display shape and polarisation of LaserBeam object.

        Parameters
        ----------
        ax : matplotlib.Axes, optional
            Allows feeding in axes to plot over existing axes. The default is None.
        x_range : list[float,float], optional
            range in x over which to pol function. The default is [-1,1].
        y_range : list[float,float], optional
            range in y over which to pol function. The default is [-1,1].
        z_scale : float, optional
            amount by which to scale the z axis up or down.
        number_of_points : int, optional
            number of points to plot over. The default is 100.
        arrow_length : float, optional
            sets the length of the polarisation arrows

        Returns
        -------
        ax : matplitlib.Axes
            Returns axes for further work.

        """
        n_sqrt = int(np.sqrt(number_of_points))
        x_range = np.linspace(x_range[0], x_range[1], n_sqrt)
        y_range = np.linspace(y_range[0], y_range[1], n_sqrt)
        x_range, y_range = np.meshgrid(x_range, y_range)

        intensity_profile = self.intensity_profile(x_range, y_range)
        polarization_map = self.polarization_map(x_range, y_range)

        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(projection="3d")

        ax.plot_surface(
            x_range,
            y_range,
            intensity_profile / z_scale,
            label="Intensity",
            alpha=0.5,
            color="lightblue",
        )
        ax.quiver(
            x_range.ravel(),
            y_range.ravel(),
            (intensity_profile / z_scale).ravel(),
            polarization_map.T[0],
            polarization_map.T[1],
            np.zeros(polarization_map.T[0].shape),
            length=arrow_length,
            color="r",
            label="Polarization",
        )

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        plt.legend()

        return ax


if __name__ == "__main__":
    mu = 0
    sigma = 0.5
    LB = LaserBeam(
        lambda x, y: np.exp(
            -1 / 2 * ((x - mu) / sigma) ** 2 - 1 / 2 * ((y - mu) / sigma) ** 2
        ),
        lambda x, y: np.array([1 + 0j, 0 + 0j]),
    )
    LB.plot()
