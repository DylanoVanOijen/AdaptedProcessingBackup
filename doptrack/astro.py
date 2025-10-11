from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from math import sqrt
from typing import Optional, Union

import numpy as np
import ratelimit
import requests
from astropy import units
from astropy.coordinates import get_sun
from astropy.time import Time
from scipy.optimize import brentq
from sgp4.earth_gravity import wgs84
from sgp4.io import twoline2rv

import doptrack.coordinates as coordinates
from doptrack.constants import a_earth

logger = logging.getLogger(__name__)


@dataclass
class TLE:
    line1: str
    line2: str

    def __post_init__(self):
        if not (isinstance(self.line1, str) and isinstance(self.line2, str)):
            raise TypeError(f'TLE lines must be strings, not: {type(self.line1)}, {type(self.line2)}')
        twoline2rv(*self, wgs84)  # Uses sgp4 to validate the TLE lines

    def __iter__(self):
        return iter([self.line1, self.line2])

    @property
    def noradid(self) -> str:
        return self.line1[2:7]  # TODO Check if we need to strip whitespace etc


@dataclass
class GroundStation:
    position: Union[coordinates.PositionGeodetic, tuple[float, float, float]]

    def __post_init__(self):
        if not isinstance(self.position, coordinates.PositionGeodetic):
            self.position = coordinates.PositionGeodetic(*self.position)

    def aer(self, time: datetime, satellite: TLESatellite) -> coordinates:
        satpos = satellite.state_ecef(time).position
        return coordinates.ecef2aer(*satpos, *self.position)

    def rangerate(self, time: datetime, satellite: TLESatellite) -> float:
        sat_state_ecef = satellite.state_ecef(time)
        station_position_ecef = coordinates.geodetic2ecef(*self.position)
        range_vector = np.subtract(sat_state_ecef.position, station_position_ecef)
        return _projection_length(sat_state_ecef.velocity, range_vector)


@dataclass
class TLESatellite:
    tle: Union[TLE, tuple[float, float]]

    def __post_init__(self):
        self.tle = TLE(*self.tle)
        self._satellite_sgp4 = twoline2rv(*self.tle, wgs84)

    def state_teme(self, time: datetime) -> coordinates.StateVector:
        time_nums = [float(t) for t in time.strftime('%Y %m %d %H %M %S.%f').split()]
        position_km, velocity_kmps = self._satellite_sgp4.propagate(*time_nums)
        position, velocity = np.array(position_km) * 1000, np.array(velocity_kmps) * 1000
        return coordinates.StateVector(*position, *velocity)

    def state_ecef(self, time: datetime) -> coordinates.StateVector:
        return coordinates.teme2ecef(time, *self.state_teme(time))

    def is_in_eclipse(self, time: datetime) -> bool:
        position_ecef = self.state_ecef(time)[:3]
        time = Time(time, scale='utc')
        r_sun = get_sun(time).itrs
        r_sun = np.array([
            r_sun.x.to(units.m).value,
            r_sun.y.to(units.m).value,
            r_sun.z.to(units.m).value,
        ])

        cosgamma = np.dot(r_sun, position_ecef) / (np.linalg.norm(r_sun) * np.linalg.norm(position_ecef))
        gamma = np.arccos(cosgamma)
        first_condition = cosgamma < 0
        rper = np.linalg.norm(position_ecef) * np.sin(gamma)
        second_condition = rper < a_earth
        return first_condition and second_condition


def _dot_product_3d(v1: tuple[float, float, float], v2: tuple[float, float, float]) -> float:
    """A fast version since `np.inner` is slow for 3D vectors."""
    return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]


def _projection_length(v1: tuple[float, float, float], v2: tuple[float, float, float]) -> float:
    """Returns the length of the projection of v1 unto v2."""
    return _dot_product_3d(v1, v2) / sqrt(_dot_product_3d(v2, v2))


@dataclass
class SatellitePass:
    time_start: datetime
    time_stop: datetime
    tca: Optional[datetime]
    azimuth_start: float
    azimuth_stop: float
    azimuth_tca: Optional[float]
    max_elevation: float

    def __post_init__(self):
        self.duration = (self.time_stop - self.time_start).total_seconds()


def predict_next_pass(
        station: GroundStation,
        satellite: TLESatellite,
        from_time: Optional[datetime] = None,
        dt_rough: int = 8 * 60,
        dt_fine: int = 15
) -> SatellitePass:
    time = from_time or datetime.utcnow().replace(microsecond=0)
    rough_time = _find_next_time_satellite_is_in_view(station, satellite, time, dt=dt_rough)
    start_time = _find_time_edge(station, satellite, rough_time, dt=-dt_fine)
    stop_time = _find_time_edge(station, satellite, rough_time, dt=dt_fine)
    tca = _find_time_of_closest_approach(station, satellite, start_time, stop_time)
    logger.debug(f'Predicted satellite pass: {start_time} to {stop_time}')
    return SatellitePass(
        time_start=start_time,
        time_stop=stop_time,
        tca=tca,
        azimuth_start=station.aer(start_time, satellite).azimuth,
        azimuth_stop=station.aer(stop_time, satellite).azimuth,
        azimuth_tca=station.aer(tca, satellite).azimuth,
        max_elevation=station.aer(tca, satellite).elevation,
    )


def predict_passes_for_upcoming_days(
        station: GroundStation,
        satellite: TLESatellite,
        from_time: Optional[datetime] = None,
        days=2,
        dt_rough: int = 8 * 60,
        dt_fine: int = 15
) -> list[SatellitePass]:
    time = from_time or datetime.now().replace(microsecond=0)
    break_time = time + timedelta(days=days)
    passes = []
    while True:
        next_pass = predict_next_pass(station, satellite, time, dt_rough, dt_fine)
        if next_pass.time_start > break_time:
            break
        passes.append(next_pass)
        time = next_pass.time_stop + timedelta(seconds=dt_rough)  # Move forward by dt_rough to ensure sat pass is done
    logger.debug(f'Found {len(passes)} passes in the upcoming {days} days')
    return passes


def _find_next_time_satellite_is_in_view(
        station: GroundStation,
        satellite: TLESatellite,
        from_time: datetime,
        dt: int
) -> datetime:
    """Find approximate time of when satellite is in view."""
    time = from_time
    while not station.aer(time, satellite).elevation > 0:
        time += timedelta(seconds=dt)
    logger.debug(f'Found time where satellite is in view at: {time}')
    return time


def _find_time_edge(station: GroundStation, satellite: TLESatellite, rough_time: datetime, dt: int) -> datetime:
    """Find time at which satellite goes from in view to not in view."""
    time_guess = rough_time
    while station.aer(time_guess, satellite).elevation > 0:
        time_guess = time_guess + timedelta(seconds=dt)
    time = time_guess - timedelta(seconds=dt)
    logger.debug(f'Found time edge at: {time}')
    return time


def _find_time_of_closest_approach(station: GroundStation, satellite: TLESatellite, start_time: datetime,
                                   stop_time: datetime) -> datetime:
    """Find time of the closest approach (TCA) during a satellite pass from start_time to stop_time."""
    rangerate_start = station.rangerate(start_time, satellite)
    rangerate_stop = station.rangerate(stop_time, satellite)
    if rangerate_start > 0 or rangerate_stop < 0:
        raise RuntimeError(
            f'rangerate at beginning and end should be: {int(rangerate_start)} > 0 and {int(rangerate_stop)} < 0'
        )
    duration = (stop_time - start_time).total_seconds()
    initial_guess = (0, duration)
    tca_seconds: float = brentq(lambda s: station.rangerate(start_time + timedelta(seconds=s), satellite),
                                *initial_guess)
    tca = start_time + timedelta(seconds=tca_seconds)
    logger.debug(f'Found TCA at: {tca}')
    return tca


class SpacetrackAPI:
    """
    An adapter for calling the space-track.org API.

    Parameters
    ----------
    username :
        Username of the API user.
    password :
        Password of the API user.

    Warnings
    --------
    Calls to the API are rate limited to 30 calls per minute and 300 calls per hour.
    If the one-minute rate limit is reached the call will sleep and retry
    when allowed. If the one-hour rate limit is reached an exception will
    be thrown.

    """

    RATELIMIT_HOUR = 300
    RATELIMIT_MINUTE = 30

    def __init__(self, username: str, password: str):
        self.username = username
        self.password = password

    def get_latest_tle(self, norad_id: str) -> TLE:
        """
        Get latest TLE.

        Parameters
        ----------
        norad_id :
            NORAD ID of the satellite.

        Returns
        -------
        :
            Two-line element.

        Raises
        ------
        RateLimitException
            If the API is called too many times within a certain time span.

        """
        logger.info(f"Getting latest TLE: NORAD ID {norad_id}")
        query = f'class/tle_latest/ORDINAL/1/NORAD_CAT_ID/{norad_id}/format/tle'
        tles = self.query(query)
        assert len(tles) == 1
        logger.info('TLE downloaded from space-track.org')
        return tles[0]

    def get_latest_tles(self, norad_id: str, n: int) -> list[TLE]:
        """
        Get the n latest TLE's.

        Parameters
        ----------
        norad_id :
            NORAD ID of the satellite.
        n :
            The number of TLE's to get.

        Returns
        -------
        :
            Two-line elements.

        Raises
        ------
        RateLimitException
            If the API is called too many times within a certain time span.

        """
        logger.info(f"Getting {n} latest TLE's: NORAD ID {norad_id}")
        query = f'class/tle/format/tle/NORAD_CAT_ID/{norad_id}/orderby/EPOCH desc/limit/{n}'
        tles = self.query(query)
        assert len(tles) == n
        logger.info(f"{n} TLE's downloaded from space-track.org")
        return tles

    def get_latest_days_of_tles(self, norad_id: str, days: int) -> list[TLE]:
        """
        Get all TLE's from the last n days.

        Parameters
        ----------
        norad_id :
            NORAD ID of the satellite.
        days :
            Number of days.

        Returns
        -------
        :
            Two-line elements.

        Raises
        ------
        RateLimitException
            If the API is called too many times within a certain time span.

        """
        logger.info(f"Getting {days} latest TLE's: NORAD ID {norad_id}")
        query = f'class/tle/format/tle/NORAD_CAT_ID/{norad_id}/orderby/EPOCH desc/EPOCH/>now-{days}'
        tles = self.query(query)
        logger.info(f"{days} TLE's downloaded from space-track.org")
        return tles

    @ratelimit.limits(calls=RATELIMIT_HOUR, period=60 * 60)
    @ratelimit.sleep_and_retry
    @ratelimit.limits(calls=RATELIMIT_MINUTE, period=60)
    def query(self, query: str) -> list[TLE]:
        """
        Send a query to the space-track.org API.

        The documentation to the space-track API can be found at
        https://www.space-track.org/documentation.

        Parameters
        ----------
        query :
            Space-track query.

        Returns
        -------
        :
            Two-line elements.

        Raises
        ------
        RateLimitException
            If the API is called too many times within a certain time span.

        """
        payload = {'identity': self.username,
                   'password': self.password}
        with requests.Session() as session:
            login_response = session.post('https://www.space-track.org/ajaxauth/login', data=payload)
            if login_response.status_code != 200:
                raise RuntimeError('Login failed. TLE not downloaded.', login_response)
            else:
                api_url = 'https://www.space-track.org/basicspacedata/query/' + query
                response = session.get(api_url)
                lines = response.text.split('\r\n')[:-1]
                return [TLE(*tle) for tle in zip(*[iter(lines)] * 2)]
