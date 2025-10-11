from datetime import datetime
from pathlib import Path
from typing import Tuple


class DoptrackError(Exception):
    pass


class ApplicationException(Exception):
    pass


DOPTRACK_PATH = Path.home() / '.doptrack'


class DataID(str):
    """An ID used for datasets in the DopTrack system.

    The format of the ID has to be:
        [SATNAME]_[NORADID]_[TIMESTAMP]
        or
        [SATNAME]_[NORADID]_[TIMESTAMP]_[SUFFIX]

    For example:
        Delfi-C3_32789_201602122224_SOMESUFFIX

    No spaces are allowed in the ID and underscores are
    only permitted at the designated spaces.
    """

    def __new__(cls, value: str):
        cls._validate(value)
        return str.__new__(cls, value)

    @property
    def satname(self) -> str:
        return self.split('_')[0]

    @property
    def noradid(self) -> str:
        return self.split('_')[1]

    @property
    def timestamp(self) -> str:
        return self.split('_')[2]

    @property
    def suffix(self):
        try:
            return self.split('_')[3]
        except IndexError:
            return ''

    @property
    def datetime(self):
        return datetime.strptime(self.timestamp, '%Y%m%d%H%M')

    @staticmethod
    def _validate(string) -> Tuple[str, str, str]:
        # TODO improve validation to raise more understandable errors
        try:
            parts = string.split('_')
            assert len(parts) == 3 or len(parts) == 4
            noradid, strtimestamp = parts[1], parts[2]
            assert len(noradid) == 5
            assert len(strtimestamp) == 12
            return parts
        except AssertionError:
            raise ValueError(f'Invalid dataid format: "{string}"')
