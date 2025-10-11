from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Callable

import dacite
import numpy as np
from scipy.optimize import brentq, bisect
from scipy.stats import linregress

import doptrack.io
from doptrack import constants as constants
from doptrack.astro import GroundStation, TLESatellite
from doptrack.recording import Metadata, Recording
from doptrack.signals import extract_s_curve, Spectrogram, create_spectrogram
from doptrack.utils import ArrayComparisonMixin, FilePath


@dataclass(frozen=True, eq=False)
class SpectrogramProduct(ArrayComparisonMixin, Spectrogram):
    meta: Metadata
    epoch: datetime

    def save(self, metafile: FilePath, datafile: FilePath) -> None:
        """Save the date to files."""
        doptrack.io.write_metadata_to_yml(Path(metafile), self)
        doptrack.io.write_arraydata_to_npz(Path(datafile), self)

    @classmethod
    def load(cls, metafile: FilePath, datafile: FilePath) -> 'SpectrogramProduct':
        """Load the date from files."""
        metadict = doptrack.io.read_metadata_from_yml(Path(metafile))
        arraydict = doptrack.io.read_arraydata_from_npz(Path(datafile))
        spetrogram_metadata = metadict.pop('spectrogram')
        return dacite.from_dict(
            cls, dict(meta=metadict, **spetrogram_metadata, **arraydict),
        )


def create_spectrogram_product(recording: Recording, dt: float, center_frequency: int = 0) -> SpectrogramProduct:
    """
    Creates a spectrogram from chunked recording data.

    The default spectrogram window width should be wide enough to
    show the whole S-curve for LEO satellites.
    """
    spectrogram = create_spectrogram(
        data=recording.load_data(dt),
        sampling_rate=recording.sampling_rate,
        dt=dt,
        mask_width=20000,  # Should be wide enough to show the whole S-curve for LEO satellites.
        mask_center=center_frequency - recording.tuning_frequency,
    )

    return SpectrogramProduct(
        meta=recording.meta,
        epoch=recording.time_start,
        dt=dt,
        df=spectrogram.df,
        image=spectrogram.image.astype(np.float16),
        time=spectrogram.time,
        frequency=spectrogram.frequency + recording.tuning_frequency,
        noise=spectrogram.noise,
        signal_average=spectrogram.signal_average,
        signal_maximum=spectrogram.signal_maximum,
        signal_median=spectrogram.signal_median, 
    )


@dataclass(frozen=True, eq=False)
class TrackingProduct(ArrayComparisonMixin):
    meta: Metadata
    epoch: datetime
    time: np.ndarray
    frequency: np.ndarray
    rangerate: np.ndarray
    tca: datetime
    tca_error: float
    fca: float
    fca_error: float

    def save(self, metafile: FilePath, datafile: FilePath) -> None:
        doptrack.io.write_metadata_to_yml(Path(metafile), self)
        doptrack.io.write_arraydata_to_csv(Path(datafile), self)

    @classmethod
    def load(cls, metafile: FilePath, datafile: FilePath) -> 'TrackingProduct':
        metadict = doptrack.io.read_metadata_from_yml(Path(metafile))
        arraydict = doptrack.io.read_arraydata_from_csv(Path(datafile))
        tracking_metadata = metadict.pop('tracking')
        return dacite.from_dict(
            cls, dict(meta=metadict, **tracking_metadata, **arraydict),
        )


def create_tracking_product(spectrogram: SpectrogramProduct, sidelobe_distance: int) -> TrackingProduct:
    s_curve = extract_s_curve(spectrogram, sidelobe_distance=sidelobe_distance)
    rangerate = (1 - (s_curve.frequency / s_curve.fca)) * constants.c
    return TrackingProduct(
        meta=spectrogram.meta,
        epoch=spectrogram.epoch,
        time=s_curve.time,
        frequency=s_curve.frequency,
        rangerate=rangerate,
        tca=spectrogram.epoch + timedelta(seconds=s_curve.tca),
        tca_error=float(s_curve.tca_error),  # Call float() since yaml does not like the np.float type
        fca=float(s_curve.fca),
        fca_error=float(s_curve.fca_error),
    )


@dataclass(frozen=True, eq=False)
class ValidationProduct(ArrayComparisonMixin):
    meta: Metadata
    epoch: datetime
    time: np.ndarray
    frequency: np.ndarray
    rangerate: np.ndarray
    rangerate_tle: np.ndarray
    first_residual: np.ndarray
    second_residual: np.ndarray

    tca: datetime
    tca_error: float
    tca_tle: datetime
    dtca: float

    fca: float
    fca_error: float

    max_elevation: float
    time_since_eclipse: Optional[float]  # Optional since some satellites might never be in eclipse.
    first_residual_fit_slope: float
    second_residual_rmse: float

    def save(self, metafile: FilePath, datafile: FilePath) -> None:
        doptrack.io.write_metadata_to_yml(Path(metafile), self)
        doptrack.io.write_arraydata_to_csv(Path(datafile), self)

    @classmethod
    def load(cls, metafile: FilePath, datafile: FilePath) -> 'ValidationProduct':
        metadict = doptrack.io.read_metadata_from_yml(Path(metafile))
        arraydict = doptrack.io.read_arraydata_from_csv(Path(datafile))
        validation_metadata = metadict.pop('validation')
        return dacite.from_dict(
            cls, dict(meta=metadict, **validation_metadata, **arraydict),
        )


def create_validation_product(data: TrackingProduct, station: GroundStation, satellite: TLESatellite) -> ValidationProduct:
    times = [data.epoch + timedelta(seconds=t) for t in data.time]
    rangerate_tle = np.array([station.rangerate(t, satellite) for t in times])
    first_residual = data.rangerate - rangerate_tle
    fit = linregress(data.time, first_residual)
    second_residual = first_residual - (fit.slope * data.time + fit.intercept)

    second_residual_rmse = np.sqrt(
        sum(second_residual ** 2) / len(second_residual)
    )

    def _calculate_rangerate(t: float) -> float:
        time = data.epoch + timedelta(seconds=t)
        return station.rangerate(time, satellite)

    tca_tle_relative: float = brentq(_calculate_rangerate, 0, data.time[-1] + 100)
    tca_tle = data.epoch + timedelta(seconds=tca_tle_relative)
    max_elevation = station.aer(tca_tle, satellite).elevation
    dtca = (data.tca - tca_tle).total_seconds()

    time_since_eclipse = _find_time_since_last_eclipse(satellite, tca_tle)

    return ValidationProduct(
        meta=data.meta,
        epoch=data.epoch,
        tca=data.tca,
        tca_error=data.tca_error,
        fca=data.fca,
        fca_error=data.fca_error,
        time=data.time,
        frequency=data.frequency,
        rangerate=data.rangerate,
        rangerate_tle=rangerate_tle,
        first_residual=first_residual,
        second_residual=second_residual,
        max_elevation=max_elevation,
        tca_tle=tca_tle,
        dtca=dtca,
        time_since_eclipse=time_since_eclipse,
        first_residual_fit_slope=float(fit.slope),
        second_residual_rmse=float(second_residual_rmse),
    )


def _find_time_since_last_eclipse(satellite: TLESatellite, tca: datetime) -> Optional[int]:
    """Accurate only to 1 second."""

    if satellite.is_in_eclipse(tca):
        return 0

    increment = 15 * 60  # Needs to be shorter than the duration a satellite is in eclipse
    time_a = 0
    while time_a > -100 * 60:
        time_a, time_b = time_a - increment, time_a
        if satellite.is_in_eclipse(tca + timedelta(seconds=time_a)):
            break
    else:
        # If the satellite is in a sun-synchronous orbit it might never be in eclipse.
        return None

    time = _boolean_bisect(
        lambda t: satellite.is_in_eclipse(tca + timedelta(seconds=t)),
        time_a,
        time_b,
        xtol=0.5,
    )
    return round(time)


def _boolean_bisect(func: Callable[[float], bool], a: float, b: float, xtol: float) -> float:
    def transfer(x: float) -> int:
        if func(x):
            return 1
        else:
            return -1

    result: float = bisect(transfer, a, b, xtol=xtol)
    return result
