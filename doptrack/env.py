import os
from dataclasses import InitVar, dataclass, asdict
from logging.config import dictConfig
from operator import attrgetter
from pathlib import Path
from typing import List, Union

import dacite
import yaml

from doptrack.base import DOPTRACK_PATH

CONFIG_PATH = DOPTRACK_PATH / 'config.yml'
LOGS_PATH = DOPTRACK_PATH / 'logs'
PROCESSING_AGENDA_PATH = DOPTRACK_PATH / 'agenda.csv'
PROCESSING_LOGBOOK_PATH = DOPTRACK_PATH / 'logbook.csv'
DEFAULT_RECORDING_REPOSITORY_PATH = DOPTRACK_PATH / 'recording'
DEFAULT_PROCESSING_REPOSITORY_PATH = DOPTRACK_PATH / 'processing'


@dataclass
class Credentials:
    username: str = ''
    password: str = ''


@dataclass
class RecordingSettings:
    path: str = ''


@dataclass
class ProcessingSettings:
    path: str = ''
    dt: float = 0.5


@dataclass
class StationSettings:
    name: str = 'TU Delft EWI'
    latitude: float = 51.9989
    longitude: float = 4.3733585
    altitude: float = 95


@dataclass
class CredentialsGroup:
    spacetrack: Credentials = Credentials()


@dataclass
class Config:
    path: InitVar[Union[str, os.PathLike]]
    credentials: CredentialsGroup = CredentialsGroup()
    recording: RecordingSettings = RecordingSettings()
    processing: ProcessingSettings = ProcessingSettings()
    station: StationSettings = StationSettings()

    def __post_init__(self, path: Union[str, os.PathLike]):
        self._path: Path = Path(path)

    def get_expanded_keys(self) -> List[str]:
        return self._get_expanded_keys_from_dict(asdict(self))

    def get_value_by_expanded_key(self, key: str):
        if key not in self.get_expanded_keys():
            raise AttributeError(f'Config does not have key: {key}')
        return attrgetter(key)(self)

    def set_value_by_expanded_key(self, key: str, value):
        if key not in self.get_expanded_keys():
            raise AttributeError(f'Config does not have key: {key}')
        childkey = key.split('.')[-1]
        parentkey = key[:-(len(childkey) + 1)]
        parent = attrgetter(parentkey)(self)
        setattr(parent, childkey, value)

    @classmethod
    def load(cls, path: os.PathLike = CONFIG_PATH) -> 'Config':
        path = Path(path)
        if not path.exists():
            return Config(path)
        else:
            with open(path, 'r') as file:
                data = dict(path=path, **yaml.safe_load(file))
            return dacite.from_dict(Config, data)

    def save(self):
        data = asdict(self)
        with open(self._path, 'w') as file:
            yaml.dump(data, file)

    def _get_expanded_keys_from_dict(self, x: dict, prefix='') -> List[str]:
        expanded_keys = []
        for k, v in x.items():
            new_key = f'{prefix}.{k}' if prefix else k
            if isinstance(v, dict):
                expanded_keys.extend(self._get_expanded_keys_from_dict(v, prefix=new_key))
            else:
                expanded_keys.append(new_key)
        return expanded_keys



class ApplicationEnvironment:

    def __init__(self, config_path: os.PathLike = CONFIG_PATH, agenda_path: os.PathLike = PROCESSING_AGENDA_PATH,
                 logbook_path: os.PathLike = PROCESSING_LOGBOOK_PATH):

        from doptrack.coordinates import PositionGeodetic
        from doptrack.processing import SpectrogramProduct, TrackingProduct, ValidationProduct
        from doptrack.recording import Recording, MetadataGroundStation
        from doptrack.automation import ProcessingAgenda, ProcessingLogbook
        from doptrack.repositories import ProcessingRepository
        from doptrack.repositories import ProcessingSummary
        from doptrack.repositories import DataRepository, PlottingRepository

        config = Config.load(config_path)
        processing_agenda = ProcessingAgenda(agenda_path)
        processing_logbook = ProcessingLogbook(logbook_path)

        station = MetadataGroundStation(
            name=config.station.name,
            position=PositionGeodetic(
                latitude=config.station.latitude,
                longitude=config.station.longitude,
                altitude=config.station.altitude,
            ),
        )

        if config.recording.path:
            recording_path = Path(config.recording.path)
        else:
            recording_path = DEFAULT_RECORDING_REPOSITORY_PATH

        recording_repository = DataRepository[Recording](
            recording_path, meta_suffix='.yml', data_suffix='.32fc'
        )
        # TODO Decide on whether to use default repos or make repos optional
        if config.processing.path:
            processing_path = Path(config.processing.path)
        else:
            processing_path = DEFAULT_PROCESSING_REPOSITORY_PATH

        processing_repository = ProcessingRepository(
            spectrogram=DataRepository[SpectrogramProduct](
                processing_path / 'spectrogram', meta_suffix='.yml', data_suffix='.npz'
            ),
            tracking=DataRepository[TrackingProduct](
                processing_path / 'processing', meta_suffix='.yml', data_suffix='.csv'
            ),
            validation=DataRepository[ValidationProduct](
                processing_path / 'validation', meta_suffix='.yml', data_suffix='.csv'
            ),
            plotting=PlottingRepository(processing_path / "plots"),
            summary=ProcessingSummary(processing_path / 'summary.csv'),
        )

        self.credentials = config.credentials
        self.recording_settings = config.recording
        self.processing_settings = config.processing
        self.station = station
        self.processing_agenda = processing_agenda
        self.processing_logbook = processing_logbook
        self.recording_repository = recording_repository
        self.processing_repository = processing_repository

    def print_status(self):
        from doptrack.automation import ProcessingStatus

        n_recording_dataids = len(self.recording_repository.list()) if self.recording_repository else 0
        n_processing_dataids = len(self.processing_repository.list()) if self.processing_repository else 0

        n_waiting = n_recording_dataids - len(self.processing_logbook.list_all())
        n_success = len(self.processing_logbook.list_by_status(ProcessingStatus.SUCCESS))
        n_nopass = len(self.processing_logbook.list_by_status(ProcessingStatus.NOPASS))
        n_failed = len(set.union(
            self.processing_logbook.list_by_status(ProcessingStatus.WARNING),
            self.processing_logbook.list_by_status(ProcessingStatus.ERROR),
        ))

        return f"""
        Datasets in system:
            Recording repository:  {n_recording_dataids}
            Processing repository: {n_processing_dataids}

        Processing status of datasets:
            - {n_waiting} waiting to be processed
            - {n_success} successful
            - {n_nopass} no satellite pass found
            - {n_failed} failed processing
        """


logging_config = {
    'version': 1,
    'formatters': {
        'short': {
            'format': '%(asctime)s - %(levelname)-7s - %(name)-s - %(message)-s',
            'datefmt': '%H:%M:%S',
        },
        'long': {
            'format': '%(asctime)s - %(levelname)-7s - %(name)-s - %(message)-s',
            'datefmt': '%Y-%m-%d %H:%M:%S',
        },
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'level': 'DEBUG',
            'formatter': 'short',
            'stream': 'ext://sys.stdout',
        },
        'file': {
            'class': 'logging.handlers.TimedRotatingFileHandler',
            'level': 'INFO',
            'formatter': 'long',
            'filename': LOGS_PATH / 'doptrack.log',
            'when': 'd',
            'interval': 1,
            'backupCount': 3,
        },
        'file_error': {
            'class': 'logging.FileHandler',
            'level': 'ERROR',
            'formatter': 'long',
            'filename': LOGS_PATH / 'doptrack_error.log',
        },
    },
    'loggers': {
        'doptrack': {
            'level': 'INFO',
            'handlers': ['console', 'file', 'file_error'],
            'propagate': False,
        },
    },
    'root': {
        'level': 'INFO',
        'handlers': ['console'],
    },
}


def setup_logging():
    LOGS_PATH.mkdir(parents=True, exist_ok=True)
    dictConfig(logging_config)
