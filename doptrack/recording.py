from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Generator

import dacite
import numpy as np

import doptrack.io
from doptrack.astro import SatellitePass, TLESatellite, GroundStation
from doptrack.utils import FilePath


__all__ = ['MetadataSatellite', 'MetadataGroundStation', 'Metadata', 'Recording']


@dataclass
class MetadataSatellite(TLESatellite):
    name: str
    noradid: str


@dataclass
class MetadataGroundStation(GroundStation):
    name: str


@dataclass(frozen=True)
class Metadata:
    station: MetadataGroundStation
    satellite: MetadataSatellite
    prediction: SatellitePass


@dataclass(frozen=True)
class Recording:
    meta: Metadata
    time_init: datetime
    duration: int
    sampling_rate: int
    n_samples: int
    tuning_frequency: int

    # Post-recording values
    time_start: datetime
    time_stop: datetime

    path_fc32: Path

    def save(self, metafile: FilePath, datafile: FilePath) -> None:
        raise NotImplementedError()

    @classmethod
    def load(cls, metafile: FilePath, datafile: FilePath) -> 'Recording':
        metafile = Path(metafile)
        datafile = Path(datafile)

        metadata = doptrack.io.read_metadata_from_yml(metafile)
        if not datafile.exists():
            raise FileNotFoundError(f'Recording data file does not exist: {datafile}')

        recording_metadata = metadata.pop('recording')
        return dacite.from_dict(
            cls, dict(meta=metadata, **recording_metadata, path_fc32=datafile),
        )

    def load_data(self, dt: float) -> Generator[np.ndarray, None, None]:
        n_bins = int(self.duration / dt)
        bin_size = int(self.n_samples / n_bins)
        yield from doptrack.io.read_arraydata_from_fc32(self.path_fc32, n_bins, bin_size)
