"""Coordinates and reference frames.

This module contains functions and classes for transforming
orbital state vectors between different reference frames.
"""
from dataclasses import dataclass, astuple
from datetime import datetime
from math import sqrt, cos, sin, tan, atan2, radians, degrees, hypot
from typing import Tuple, Sequence

import astropy.time
import numpy as np

from doptrack.constants import a_earth, f_earth, omega_earth

__all__ = [
    # Vectors
    'PositionVector',
    'PositionGeodetic',
    'PositionAER',
    'PositionENU',
    'PositionUVW',
    'VelocityVector',
    'StateVector',

    # Transforms
    'ecef2aer',
    'aer2ecef',
    'ecef2enu',
    'enu2ecef',
    'uvw2enu',
    'enu2uvw',
    'aer2enu',
    'enu2aer',
    'geodetic2enu',
    'enu2geodetic',
    'geodetic2aer',
    'aer2geodetic',
    'geodetic2ecef',
    'ecef2geodetic',
    'teme2ecef',
    'gmst1982',

    # Other
    'EarthEllipsoid',
]


@dataclass(frozen=True)
class Vector(Sequence):
    def __getitem__(self, key):
        return astuple(self)[key]

    def __len__(self):
        return len(astuple(self))


@dataclass(frozen=True)
class PositionVector(Vector):
    x: float
    y: float
    z: float


@dataclass(frozen=True)
class PositionGeodetic(Vector):
    latitude: float
    longitude: float
    altitude: float


@dataclass(frozen=True)
class PositionAER(Vector):
    azimuth: float
    elevation: float
    range: float


@dataclass(frozen=True)
class PositionENU(Vector):
    east: float
    north: float
    up: float


@dataclass(frozen=True)
class PositionUVW(Vector):
    u: float
    v: float
    w: float


@dataclass(frozen=True)
class VelocityVector(Vector):
    vx: float
    vy: float
    vz: float


@dataclass(frozen=True)
class StateVector(Vector):
    x: float
    y: float
    z: float
    vx: float
    vy: float
    vz: float

    @property
    def position(self):
        return PositionVector(self.x, self.y, self.z)

    @property
    def velocity(self):
        return VelocityVector(self.vx, self.vy, self.vz)


@dataclass
class EarthEllipsoid:
    """
    Class for storing earth ellipsoid information.

    The default values are for the WGS84 ellipsoid.

    """
    a: float = a_earth
    """Semi-major axis [m]."""
    f: float = f_earth
    """Flattening of ellipsoid."""

    def __post_init__(self):
        self.b: float = self.a * (1 - self.f)
        """Semi-minor axis [m]."""

    @property
    def e(self) -> float:
        """Eccentricity of ellipsoid."""
        return sqrt(2 * self.f - self.f ** 2)


def ecef2aer(
        x: float, y: float, z: float,
        lat0: float, lon0: float, alt0: float,
        ellipsoid: EarthEllipsoid = EarthEllipsoid(),
) -> PositionAER:
    enu = ecef2enu(x, y, z, lat0, lon0, alt0, ellipsoid=ellipsoid)
    aer = enu2aer(*enu)
    return PositionAER(*aer)


def aer2ecef(
        azimuth: float, elevation: float, slantrange: float,
        lat0: float, lon0: float, alt0: float,
        ellipsoid: EarthEllipsoid = EarthEllipsoid(),
) -> PositionVector:
    enu = aer2enu(azimuth, elevation, slantrange)
    ecef = enu2ecef(*enu, lat0, lon0, alt0, ellipsoid=ellipsoid)
    return PositionVector(*ecef)


def ecef2enu(
        x: float, y: float, z: float,
        lat0: float, lon0: float, alt0: float,
        ellipsoid: EarthEllipsoid = EarthEllipsoid(),
) -> PositionENU:
    x0, y0, z0 = geodetic2ecef(lat0, lon0, alt0, ellipsoid)
    enu = uvw2enu(x - x0, y - y0, z - z0, lat0, lon0)
    return PositionENU(*enu)


def enu2ecef(
        east: float, north: float, up: float,
        lat0: float, lon0: float, alt0: float,
        ellipsoid: EarthEllipsoid = EarthEllipsoid(),
) -> PositionVector:
    x0, y0, z0 = geodetic2ecef(lat0, lon0, alt0, ellipsoid=ellipsoid)
    dx, dy, dz = enu2uvw(east, north, up, lat0, lon0)
    return PositionVector(x0 + dx, y0 + dy, z0 + dz)


def uvw2enu(u: float, v: float, w: float, lat0: float, lon0: float) -> PositionENU:
    lat0 = radians(lat0)
    lon0 = radians(lon0)
    t = cos(lon0) * u + sin(lon0) * v
    east = -sin(lon0) * u + cos(lon0) * v
    north = -sin(lat0) * t + cos(lat0) * w
    up = cos(lat0) * t + sin(lat0) * w
    return PositionENU(east, north, up)


def enu2uvw(east: float, north: float, up: float, lat0: float, lon0: float) -> PositionUVW:
    lat0 = radians(lat0)
    lon0 = radians(lon0)
    t = cos(lat0) * up - sin(lat0) * north
    w = sin(lat0) * up + cos(lat0) * north
    u = cos(lon0) * t - sin(lon0) * east
    v = sin(lon0) * t + cos(lon0) * east
    return PositionUVW(u, v, w)


def aer2enu(azimuth: float, elevation: float, slantrange: float) -> PositionENU:
    elevation = radians(elevation)
    azimuth = radians(azimuth)
    r = slantrange * cos(elevation)
    east = r * sin(azimuth)
    north = r * cos(azimuth)
    up = slantrange * sin(elevation)
    return PositionENU(east, north, up)


def enu2aer(east: float, north: float, up: float) -> PositionAER:
    r = hypot(east, north)
    slantrange = hypot(r, up)
    elevation = atan2(up, r)
    azimuth = np.mod(atan2(east, north), 2 * atan2(0, -1))
    return PositionAER(degrees(azimuth), degrees(elevation), slantrange)


def geodetic2enu(
        lat: float, lon: float, alt: float,
        lat0: float, lon0: float, alt0: float,
        ellipsoid: EarthEllipsoid = EarthEllipsoid(),
) -> PositionENU:
    x, y, z = geodetic2ecef(lat, lon, alt, ellipsoid=ellipsoid)
    x0, y0, z0 = geodetic2ecef(lat0, lon0, alt0, ellipsoid=ellipsoid)
    dx, dy, dz = x - x0, y - y0, z - z0
    enu = uvw2enu(dx, dy, dz, lat0, lon0)
    return PositionENU(*enu)


def enu2geodetic(
        east: float, north: float, up: float,
        lat0: float, lon0: float, alt0: float,
        ellipsoid: EarthEllipsoid = EarthEllipsoid(),
) -> PositionGeodetic:
    ecef = enu2ecef(east, north, up, lat0, lon0, alt0, ellipsoid=ellipsoid)
    geodetic = ecef2geodetic(*ecef, ellipsoid=ellipsoid)
    return PositionGeodetic(*geodetic)


def geodetic2aer(
        lat: float, lon: float, alt: float,
        lat0: float, lon0: float, alt0: float,
        ellipsoid: EarthEllipsoid = EarthEllipsoid(),
) -> PositionAER:
    east, north, up = geodetic2enu(lat, lon, alt, lat0, lon0, alt0, ellipsoid=ellipsoid)
    enu = enu2aer(east, north, up)
    return PositionAER(*enu)


def aer2geodetic(
        azimuth: float, elevation: float, slantrange: float,
        lat0: float, lon0: float, alt0: float
) -> PositionGeodetic:
    x, y, z = aer2ecef(azimuth, elevation, slantrange, lat0, lon0, alt0)
    geodetic = ecef2geodetic(x, y, z)
    return PositionGeodetic(*geodetic)


def geodetic2ecef(
        lat: float, lon: float, alt: float,
        ellipsoid: EarthEllipsoid = EarthEllipsoid(),
) -> PositionVector:
    """Reference frame transformation from Geodetic to ECEF.

    Parameters
    ----------
    lat
        Latitude in degrees.
    lon
        Longitude in degrees.
    alt: float
        Altitude in [m].
    ellipsoid
        An earth ellipsoid. If not given
        the standard WGS84 ellipsoid is used.

    Returns
    -------
    tuple
        Position vector in the ECEF frame in [m].

    See Also
    --------
    ecef2geodetic : Transform from ECEF to Geodetic

    References
    ----------
    .. [1] Montenbruck, O. and Gill, E.,
        "Satellite Orbits",
        1st Edition,
        p. 187-189, 2005.

    Examples
    --------
    >>> geodetic2ecef(-7.26654999, 72.36312094, -63.667)
    (1917032.190, 6029782.349, -801376.113)
    """
    lat = radians(lat)
    lon = radians(lon)

    n = ellipsoid.a / np.sqrt(1 - (ellipsoid.e * np.sin(lat)) ** 2)
    x = (n + alt) * cos(lat) * cos(lon)
    y = (n + alt) * cos(lat) * sin(lon)
    z = (n * (ellipsoid.b / ellipsoid.a) ** 2 + alt) * sin(lat)

    return PositionVector(x, y, z)


def ecef2geodetic(
        x: float, y: float, z: float,
        ellipsoid: EarthEllipsoid = EarthEllipsoid(),
) -> PositionGeodetic:
    """https://www.astro.uni.torun.pl/~kb/Papers/geod/Geod-BG.htm"""
    r = hypot(x, y)
    omega = atan2(ellipsoid.b * z, r * ellipsoid.a)
    c = (ellipsoid.a ** 2 - ellipsoid.b ** 2) / hypot(r * ellipsoid.a, z * ellipsoid.b)

    psi = atan2(z * ellipsoid.a, r * ellipsoid.b)
    psi_prev, count = 0, 0
    # Apply Newton's method
    while psi_prev != psi and count < 5:
        psi_prev = psi
        f = 2 * sin(psi_prev - omega) - c * sin(2 * psi_prev)
        f_deriv = 2 * (cos(psi_prev - omega) - c * cos(2 * psi_prev))
        psi = psi_prev - (f / f_deriv)
        count += 1

    lat = atan2(ellipsoid.a * tan(psi), ellipsoid.b)
    lon = atan2(y, x)
    alt = (r - ellipsoid.a * cos(psi)) * cos(lat) + \
          (z - ellipsoid.b * sin(psi)) * sin(lat)

    lon = degrees(lon)
    lat = degrees(lat)

    return PositionGeodetic(lat, lon, alt)


def teme2ecef(
        time: datetime,
        x: float, y: float, z: float,
        u: float, v: float, w: float,
        polarmotion: Tuple[float, float] = None,
        length_of_day: float = None,
) -> StateVector:
    """Reference frame transformation from TEME to ECEF.

    Transforms position and velocity vectors from the idiosyncratic
    true equator mean equinox (TEME) frame to an earth centered
    earth fixed (ECEF/ITRF) frame, taking into account sidereal
    time, and optionally polar motion.

    Parameters
    ----------
    time :
        Time in UTC.
    x :
        Position x-component in TEME frame.
    y :
        Position y-component in TEME frame.
    z :
        Position z-component in TEME frame.
    u :
        Velocity u-component in TEME frame.
    v :
        Velocity v-component in TEME frame.
    w :
        Velocity w-component in TEME frame.
    polarmotion :
        Polar motion parameters resulting in an accurate transform
        from Pseudo Earth Fixed (PEF) frame to ECEF frame. If no
        values are given it is assumed that PEF = ECEF.
    length_of_day :
        Length of day parameter.

    Returns
    -------
    :
        State vector in ECEF frame.

    Notes
    -----
    The units of the position and velocity vectors should be at same scale
    and the resulting vectors will have the same units.
    Taking into account changes in length of day should have near zero influence
    on results for a LEO satellite, even at extreme accuracy.
    Taking into account polar motion will result in about 10s of meters difference
    in position for a LEO satellite.

    This function is adapted from the ``teme2ecef.m`` script published by Vallado.
    Vallado's algorithm adds kinematic terms based on the equation of equinoxes to
    the calculation of GMST if time is after 1997. No other source for this has been
    found and the influence on the result is on the order of centimeters.
    This correction is therefore not implemented in this algorithm.

    Kelso [1]_ provides an example of the complete transformation between TEME
    and ECEF.

    The current implementation uses the astropy library to calculate UT1.
    Profiling shows that ~80% of the time is spent calculating this value,
    so this would be a good place to start if better performance is needed.

    References
    ----------
    .. [1] Kelso et al.,
        "Revisiting Spacetrack Report #3",
        Appendix C, 2006.

    """
    r_teme = (x, y, z)
    v_teme = (u, v, w)
    time = astropy.time.Time(time, scale='utc')  # See comment about performance in docstring.

    theta = gmst1982(time.ut1.jd)
    theta_dot = omega_earth if not length_of_day else omega_earth * (1 - length_of_day / 86400)
    angular_velocity = np.array([0, 0, -theta_dot])

    sidereal_matrix = _rot_z(theta)
    r_pef = sidereal_matrix.dot(r_teme)
    v_pef = sidereal_matrix.dot(v_teme) + _cross_product_3d(tuple(angular_velocity), r_pef)

    if not polarmotion:
        r_ecef = r_pef
        v_ecef = v_pef
    else:
        xp, yp = polarmotion
        polar_matrix = _rot_y(-xp) @ _rot_x(-yp)
        r_ecef = polar_matrix.dot(r_pef)
        v_ecef = polar_matrix.dot(v_pef)

    x_ecef, y_ecef, z_ecef = r_ecef
    u_ecef, v_ecef, w_ecef = v_ecef
    return StateVector(x_ecef, y_ecef, z_ecef, u_ecef, v_ecef, w_ecef)


def gmst1982(ut1_jd: float) -> float:
    """Deprecated version of Greenwich Mean Sidereal Time (GMST).

    Parameters
    ----------
    ut1_jd :
        UT1 in julian date.


    Warnings
    --------
    Should only be used in transformations to and from
    the TEME reference frame.

    Notes
    -----
    Originally defined by Aoki [1]_. Used by McCarthy [2]_ and Kelso [3]_ for
    transformations to and from the TEME reference frame.

    References
    ----------
    .. [1] Aoki et al., "The New Definition of Universal Time", eq. 14, 1982.
    .. [2] McCarthy, "IERS Technical Note 13", p. 30, 1992.
    .. [3] Kelso et al., "Revisiting Spacetrack Report #3", eq. C-5, 2006.
    """
    ut1_jc = (ut1_jd - 2451545) / 36525
    gmst = (67310.54841
            + (876600.0 * 3600.0 + 8640184.812866) * ut1_jc
            + 0.093104 * ut1_jc ** 2
            - 6.2e-6 * ut1_jc ** 3) * 360 / 86400
    return radians(gmst % 360)


def _rot_x(theta: float) -> np.ndarray:
    return np.array(
        [[1, 0, 0],
         [0, cos(theta), sin(theta)],
         [0, -sin(theta), cos(theta)]])


def _rot_y(theta: float) -> np.ndarray:
    return np.array(
        [[cos(theta), 0, -sin(theta)],
         [0, 1, 0],
         [sin(theta), 0, cos(theta)]])


def _rot_z(theta: float) -> np.ndarray:
    return np.array(
        [[cos(theta), sin(theta), 0],
         [-sin(theta), cos(theta), 0],
         [0, 0, 1]])


def _cross_product_3d(v1: Tuple[float, float, float], v2: Tuple[float, float, float]) -> np.ndarray:
    """A fast version since `np.cross` is slow for 3D vectors."""
    return np.array([
        v1[1] * v2[2] - v2[1] * v1[2],
        v1[2] * v2[0] - v2[2] * v1[0],
        v1[0] * v2[1] - v2[0] * v1[1]
    ])
