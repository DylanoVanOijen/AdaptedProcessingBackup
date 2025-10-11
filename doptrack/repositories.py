import os
import shutil
import warnings
from collections import defaultdict
from dataclasses import dataclass, is_dataclass, fields
from datetime import datetime
from pathlib import Path
from typing import TypeVar, Generic, Generator, Type, get_args, get_type_hints, Iterable, Union

import pandas as pd
from matplotlib import pyplot as plt

from doptrack.base import DoptrackError, DataID
from doptrack.processing import SpectrogramProduct, TrackingProduct, ValidationProduct
from doptrack.recording import Recording


class DataNotFoundError(DoptrackError):
    pass


T = TypeVar('T', Recording, SpectrogramProduct, TrackingProduct, ValidationProduct)


@dataclass
class DataRepository(Generic[T]):
    path: Path
    meta_suffix: str
    data_suffix: str

    # TODO Determine whether to handle auxilliary files, e.g. .hdr

    def __post_init__(self):
        self.path = self.path.expanduser().resolve()
        if not (self.meta_suffix.startswith('.') and self.data_suffix.startswith('.')):
            raise RuntimeError('Meta and data suffixes must start with a dot.')

    def __contains__(self, dataid) -> bool:
        """
        Instead of just checking if dataid is in repo.list() we narrow the search
        down to the specific subfolder. This results in a huge performance improvement,
        especially if the repo contains hundreds of thousands of files.
        """
        dataid = DataID(dataid)
        subfolder = self.path / dataid.satname / str(dataid.datetime.year)
        dataids = self._get_dataids_from_tree(subfolder)
        meta_dataids = dataids[self.meta_suffix]
        data_dataids = dataids[self.data_suffix]
        #print (subfolder)
        #print (meta_dataids)
        #print (data_dataids)
        return dataid in meta_dataids and dataid in data_dataids

    def list(self) -> set[DataID]:
        dataids = self._get_dataids_from_tree(self.path)
        meta_dataids = dataids[self.meta_suffix]
        data_dataids = dataids[self.data_suffix]
        incomplete_dataids = set.symmetric_difference(meta_dataids, data_dataids)
        if incomplete_dataids:
            warnings.warn(f'{len(incomplete_dataids)} incomplete records repository: {self.path}')
        return {DataID(s) for s in set.intersection(meta_dataids, data_dataids)}

    def save(self, dataid: str, data: T) -> None:
        assert isinstance(data, self._data_type)
        dataid = DataID(dataid)
        folder = self.path / dataid.satname / str(dataid.datetime.year)
        metafile = folder / f'{dataid}{self.meta_suffix}'
        datafile = folder / f'{dataid}{self.data_suffix}'
        folder.mkdir(parents=True, exist_ok=True)
        data.save(metafile, datafile)

    def load(self, dataid: str) -> T:
        dataid = DataID(dataid)
        if dataid not in self:
            raise DataNotFoundError(f'No data in repository to load: {dataid}')
        folder = self.path / dataid.satname / str(dataid.datetime.year)
        metafile = folder / f'{dataid}{self.meta_suffix}'
        datafile = folder / f'{dataid}{self.data_suffix}'
        return self._data_type.load(metafile, datafile)

    def delete(self, dataid: str) -> None:
        dataid = DataID(dataid)
        if dataid not in self:
            raise DataNotFoundError(f'No data in repository to delete: {dataid}')
        folder = self.path / dataid.satname / str(dataid.datetime.year)
        for file in folder.rglob(f'{dataid}.*'):
            file.unlink()

    def delete_all(self) -> None:
        shutil.rmtree(self.path)
        self.path.mkdir()

    def import_data(self, folder: os.PathLike, overwrite: bool = False):
        """Imports all data files from a folder into the repository.

        Imported files are removed from the original folder.
        The import is non-recursive and skips all subfolders.  # TODO Is this still true?
        If overwrite is true then existing recordings in the
        repository with the same dataid are overwritten.
        """
        folder = Path(folder)
        dataids = self._get_dataids_from_tree(folder)
        meta_dataids = dataids[self.meta_suffix]
        data_dataids = dataids[self.data_suffix]
        dataids_to_import = set.intersection(meta_dataids, data_dataids)
        for dataid in dataids_to_import:
            metafiles = list(folder.rglob(f'{dataid}{self.meta_suffix}'))
            datafiles = list(folder.rglob(f'{dataid}{self.data_suffix}'))
            if len(metafiles) == 1 and len(datafiles) == 1:
                metafile = metafiles[0]
                datafile = datafiles[0]
                new_folder = self.path / dataid.satname / str(dataid.datetime.year)
                new_metafile = new_folder / f'{dataid}{self.meta_suffix}'
                new_datafile = new_folder / f'{dataid}{self.data_suffix}'
                if not overwrite and (new_metafile.exists() or new_datafile.exists()):
                    warnings.warn(f'Data already in repository so skipping import: {dataid}')
                else:
                    new_folder.mkdir(parents=True, exist_ok=True)
                    metafile.replace(new_metafile)
                    datafile.replace(new_datafile)
            else:
                warnings.warn(f'Found incorrect number of files for dataid: {dataid}')
                warnings.warn(f'{len(metafiles)} meta files and {len(datafiles)} data files')

    @classmethod
    def _get_dataids_from_tree(cls, path: Path) -> dict[str, set[DataID]]:
        dataids: dict[str, set[DataID]] = defaultdict(set)
        for filename in cls._get_filenames_from_tree(path):
            elements = filename.split('.')
            if len(elements) != 2:
                continue
            stem, dotless_suffix = elements
            suffix = '.' + dotless_suffix
            try:
                dataids[suffix].add(DataID(stem))
            except ValueError:
                pass
        return dataids

    @staticmethod
    def _get_filenames_from_tree(path) -> Generator[str, None, None]:
        """os.walk is about 3 times faster than Path.rglob"""
        for _, _, filenames in os.walk(path):
            yield from filenames

    @property
    def _data_type(self) -> Type[T]:
        """
        This property abuses a typing implementation detail to get access
        to the data type T during runtime. This allows the repository
        to dynamically find the correct load() method.

        It is not pretty and might break if the typing implementation changes.
        """
        try:
            data_type = get_args(self.__orig_class__)[0]  # type: ignore
        except AttributeError:
            raise RuntimeError(
                f'Repository was not given a data type: {self.__class__.__name__}[DATATYPE](*args, **kwargs)'
            )
        return data_type


class PlottingRepository:

    def __init__(self, path: Path):
        self.path = path.expanduser().resolve()

    def save(self, title: str, figure: plt.Figure):
        filepath = self.path / f'{title}.png'
        filepath.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(filepath, dpi=150)


class ProcessingSummary:

    def __init__(self, path: Path):
        self.path = path.expanduser().resolve()

    def add(self, dataid: DataID, data: ValidationProduct) -> None:
        df = self.load()

        type_hints = get_type_hints(data)
        new_data = dict(
            satellite_name=data.meta.satellite.name,
            dataid=dataid,
            **{name: getattr(data, name) for name, type_ in type_hints.items() if type_ in (float, int, datetime)}
        )
        for name, type_ in type_hints.items():
            value = getattr(data, name)
            if is_dataclass(value) and isinstance(value, Iterable):
                for subfield in fields(value):
                    new_data[f'{name}_{subfield.name}'] = getattr(value, subfield.name)

        new_row = pd.DataFrame(new_data, index=[0])
        new_row.set_index(['satellite_name', 'dataid'], inplace=True)
        df = pd.concat([df, new_row])
        df = df[~df.index.duplicated(keep='last')]
        df = df.sort_index()
        df.to_csv(self.path)

    def load(self) -> pd.DataFrame:
        type_hints = get_type_hints(ValidationProduct)
        datetime_fields = [name for name, type_ in type_hints.items() if type_ == datetime]
        if self.path.exists():
            df = pd.read_csv(self.path, parse_dates=datetime_fields)
        else:
            columns = [
                'satellite_name', 'dataid',
                *[name for name, type_ in type_hints.items() if type_ in (float, int, datetime)]
            ]
            df = pd.DataFrame(columns=columns)
        return df.set_index(['satellite_name', 'dataid'])

    def list(self) -> set[DataID]:
        df = self.load()
        return {DataID(dataid) for dataid in df.index.get_level_values('dataid')}

    def delete(self, dataid: Union[str, Iterable[str]]):
        df = self.load()
        df = df.drop(dataid, level='dataid', errors='ignore')
        df.to_csv(self.path)

    def delete_all(self):
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass

    def __contains__(self, dataid: str) -> bool:
        return dataid in self.list()


@dataclass
class ProcessingRepository:
    spectrogram: DataRepository[SpectrogramProduct]
    tracking: DataRepository[TrackingProduct]
    validation: DataRepository[ValidationProduct]
    plotting: PlottingRepository
    summary: ProcessingSummary

    def list(self):
        return set.union(
            self.spectrogram.list(),
            self.tracking.list(),
            self.validation.list(),
        )
