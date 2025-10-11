import logging
import os
from dataclasses import fields, is_dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Type, Generator

import numpy as np
import yaml

from doptrack.base import DoptrackError

logger = logging.getLogger(__name__)


__all__ = [
    'EmptyRecordingError',
    'write_metadata_to_yml',
    'read_metadata_from_yml',
    'read_arraydata_from_fc32',
    'write_arraydata_to_csv',
    'read_arraydata_from_csv',
    'write_arraydata_to_npz',
    'read_arraydata_from_npz',
]


class EmptyRecordingError(DoptrackError):
    pass


def write_metadata_to_yml(metafile: Path, data: Any) -> None:
    assert is_dataclass(data)
    meta_fields = [f.name for f in fields(data) if f.type != np.ndarray]
    all_data = asdict(data)
    metadata = {f: all_data[f] for f in meta_fields}
    general_metadata = metadata.pop('meta')
    datakey = _data_key(type(data))
    final = {**general_metadata, datakey: metadata}
    with metafile.open('w') as file:
        yaml.dump(data=final, stream=file)
    logger.debug(f'Metadata written to: {metafile}')


def read_metadata_from_yml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f'Metafile does not exist: {path}')
    with open(path, 'r') as metafile:
        metadata = yaml.load(metafile, Loader=yaml.FullLoader)

    if 'Sat' in metadata:
        metadata = old_to_new_metadata(metadata)

    logger.debug(f'Metafile read from: {path}')
    return metadata


def read_arraydata_from_fc32(path: Path, n_bins: int, bin_size: int) -> Generator[np.ndarray, None, None]:
    if os.stat(path).st_size < 10e7:  # Check if file is less than 10 mb
        raise EmptyRecordingError(f'Recording data file is empty or too small: {path}')

    count = 2 * bin_size
    with path.open('r') as file:
        array = np.fromfile(file, dtype=np.float32, count=count)
        if not array.any():
            raise EmptyRecordingError(f'First chunck of recording data file contains all zeros')

    with path.open('r') as file:
        for i in range(n_bins):
            array = np.fromfile(file, dtype=np.float32, count=count)
            complex_array = np.zeros(int(len(array) / 2), dtype=complex)
            complex_array.real = array[::2]
            #complex_array.imag = -array[1::2]

            # Swapping minus sign as test
            complex_array.imag = array[1::2]
            yield complex_array




def old_to_new_metadata(data: dict) -> dict:
    """An adapter function for the old doptrack metadata api to new recording api.

    TODO Old metadata files should prefereably be converted to new format at some point.
    """

    time_start_prediction: datetime = data['Sat']['Record']['time1 UTC']
    time_start_prediction = time_start_prediction.replace(second=0, microsecond=0)

    time_init_local = datetime.strptime(str(data['Sat']['Record']['Start of recording']), '%Y%m%d%H%M')
    time_init = datetime.utcfromtimestamp(time_init_local.timestamp())

    # TODO this is a temp time correction until yml files and recording have been reorganised
    # If recording includes header file, and this header data has been added to yml file, then include this
    if 'uhd' in data['Sat']:
        uhd = dict(
            rx_freq=data['Sat']['uhd']['rx_freq'],
            rx_gain=data['Sat']['uhd']['rx_gain'],
            rx_rate=data['Sat']['uhd']['rx_rate'],
            rx_time=data['Sat']['uhd']['rx_time'],
        )
        time_pps = data['Sat']['Record']['time pps']
        # if we have uhd data then change start time to more accurate version
        time_start = time_pps + timedelta(seconds=uhd['rx_time'])
    else:
        uhd = None
        time_pps = None
        time_start = data['Sat']['Record']['time1 UTC']
    duration = int(data['Sat']['Predict']['Length of pass'])
    time_stop = time_start + timedelta(seconds=duration)

    # Hotfix for 20kHz wrong tuning frequency in yml files before 2018-02-24
    # TODO fix values in yml files
    tuning_frequency = int(data['Sat']['State']['Tuning Frequency'])
    if time_start < datetime(2018, 2, 24):
        tuning_frequency -= 20_000

    return dict(
        station=dict(
            name=data['Sat']['Station']['Name'],
            # TODO Fix station position in yaml files
            position=dict(
                latitude = data['Sat']['Station']['Lat'],
                longitude = data['Sat']['Station']['Lon'],
                altitude = data['Sat']['Station']['Height'],
            ),
        ),
        satellite=dict(
            name=data['Sat']['State']['Name'],
            noradid=data['Sat']['State']['NORADID'],
            tle=dict(
                line1=data['Sat']['Predict']['used TLE line1'],
                line2=data['Sat']['Predict']['used TLE line2'],
            ),
        ),
        prediction=dict(
            time_start=time_start_prediction,
            time_stop=time_start_prediction + timedelta(seconds=int(data['Sat']['Predict']['Length of pass'])),
            tca=None,  # TODO Maybe add TCA retroactively to old metafiles
            azimuth_start=int(data['Sat']['Predict']['SAzimuth']),
            azimuth_stop=int(data['Sat']['Predict']['EAzimuth']),
            azimuth_tca=None,  # TODO Maybe add azimuth at TCA retroactively to old metafiles
            max_elevation=int(data['Sat']['Predict']['Elevation']),
        ),
        recording=dict(
            time_init=time_init,
            time_start=time_start,
            time_stop=time_stop,
            duration=duration,
            sampling_rate=int(data['Sat']['Record']['sample_rate']),
            n_samples=int(data['Sat']['Record']['num_sample']),
            tuning_frequency=tuning_frequency,
        ),
    )


def write_arraydata_to_csv(path: Path, data: Any) -> None:
    assert is_dataclass(data)
    array_fields = [f.name for f in fields(data) if f.type == np.ndarray]
    arraydata = {k: getattr(data, k) for k in array_fields}

    # Ensure that arrays are suitable for csv format
    all_arrays_are_1d = all(len(a.shape) == 1 for a in arraydata.values())
    all_arrays_are_same_shape = len(set(a.shape for a in arraydata.values())) == 1
    assert all_arrays_are_1d
    assert all_arrays_are_same_shape

    columns = np.column_stack(list(arraydata.values()))
    np.savetxt(path, columns, delimiter=',', header=','.join(array_fields))
    logger.debug(f'Array data written to: {path}')


def read_arraydata_from_csv(datafile: Path) -> dict[str, np.ndarray]:
    if not datafile.exists():
        raise FileNotFoundError(f'Datafile does not exist: {datafile}')
    with open(datafile) as file:
        column_names = file.readline().strip('#\n ').split(',')
    arrays = np.loadtxt(str(datafile), delimiter=',', skiprows=1, unpack=True)
    datadict = {k: v for k, v in zip(column_names, arrays)}
    logger.debug(f'Array data read from: {datafile}')
    return datadict


def write_arraydata_to_npz(datafile: Path, data: Any) -> None:
    assert is_dataclass(data)
    array_fields = [f.name for f in fields(data) if f.type == np.ndarray]
    arraydata = {k: getattr(data, k) for k in array_fields}
    np.savez(datafile, **arraydata)
    logger.debug(f'Array data written to: {datafile}')


def read_arraydata_from_npz(datafile: Path) -> dict[str, np.ndarray]:
    if not datafile.exists():
        raise FileNotFoundError(f'Datafile does not exist: {datafile}')
    with np.load(str(datafile)) as arrays:
        datadict = dict(arrays)
    logger.debug(f'Array data read from: {datafile}')
    return datadict


def _data_key(data: Type):
    class_name = data.__name__
    return class_name.lower().removesuffix('product')
